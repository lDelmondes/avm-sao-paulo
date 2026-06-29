# -*- coding: utf-8 -*-
"""
Núcleo de inferência do AVM de apartamentos de São Paulo — Fase 6, Prompt 1.

FONTE DA VERDADE: docs/SPEC_CONTRATO.md (verificada com diff=0 contra o notebook 03).
Este módulo NÃO reinventa nenhum cálculo: cada derivação de feature copia o método
documentado na spec, com referência à seção (§) correspondente. Qualquer divergência
faria o modelo prever lixo silenciosamente — por isso a fidelidade aqui é inegociável.

Escopo deste módulo (Prompt 1): SÓ o predictor. Sem UI, sem geocodificação de CEP
(esses são os próximos prompts). A entrada é a coordenada (lat/long em EPSG:4326) já
resolvida + os atributos do imóvel.

Estrutura cacheável: os artefatos pesados (modelos, parquets) são carregados UMA vez por
mercado via `carregar_artefatos`, memoizado com `functools.lru_cache`. O Streamlit deve
envolver `carregar_artefatos` com `@st.cache_resource` (ou simplesmente reusar o cache
deste módulo) — ver nota em `carregar_artefatos`.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Constantes do contrato (todas ancoradas na spec)
# ---------------------------------------------------------------------------
_RAIZ = Path(__file__).resolve().parent.parent          # raiz do repositório
_DIR_MODELOS = _RAIZ / "models"                          # modelos versionados
_DIR_DADOS = _RAIZ / "app" / "data"                      # pacote de dados de produção

MERCADOS = ("venda", "aluguel")

K_VIZINHOS = 3                  # spatial_lag: k=3 (spec §2.3 / §7-2)
CRS_GEO = "EPSG:4326"           # lat/long de entrada (spec §7-5)
CRS_UTM = 31983                 # UTM SIRGAS 2000 fuso 23S, métrico (spec §7-5)
RENDA_COL = "ponderation_area_average_household_income"  # coluna de renda (spec §2.5)

# Atributos físicos que entram DIRETO do input (spec §2.1). Ordem só ilustrativa:
# a ordem real da matriz vem de modelo.feature_names_in_ (spec §1).
ATRIBUTOS_FISICOS = [
    "Rooms", "Toilets", "Suites", "Parking",
    "Elevator", "Furnished", "Swimming Pool", "New",
]

# MAPE do TEST FINAL — XGBoost campeão (notebook 03, célula 67; spec §6).
# Usado para a faixa de incerteza (estimativa ± MAPE).
MAPE_TESTE = {"venda": 0.1404, "aluguel": 0.2029}


# ---------------------------------------------------------------------------
# Artefatos por mercado (carregados uma vez, reusados em toda predição)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Artefatos:
    """Tudo que uma predição precisa, pré-carregado e pré-processado para um mercado."""
    mercado: str
    modelo: object                       # XGBRegressor (sklearn API)
    features: list                       # modelo.feature_names_in_ (ordem canônica)
    # Referência de treino para o spatial_lag e o district (spec §2.3 / passo 5):
    nn_treino: NearestNeighbors          # k-NN fitado nas coords UTM do treino_ref
    log_precos_treino: np.ndarray        # log1p(Price) do treino_ref, alinhado ao k-NN
    districts_treino: np.ndarray         # district de cada imóvel de treino, idem
    # Estações (spec §2.4) e áreas de ponderação (spec §2.5), já em geopandas:
    estacoes_utm: gpd.GeoDataFrame       # estações em UTM 31983 (distância em metros)
    areas_geo: gpd.GeoDataFrame          # áreas em EPSG:4326 (point-in-polygon)
    areas_utm: gpd.GeoDataFrame          # áreas em UTM 31983 (fallback nearest)


@functools.lru_cache(maxsize=len(MERCADOS))
def carregar_artefatos(mercado: str) -> Artefatos:
    """
    Carrega e prepara os artefatos de um mercado UMA vez (memoizado).

    No Streamlit, envolva com `@st.cache_resource`:

        @st.cache_resource
        def _art(mercado): return carregar_artefatos(mercado)

    ou apenas chame esta função — o `lru_cache` já evita recarga. Os objetos são
    imutáveis (`frozen=True`), seguros para compartilhar entre sessões.
    """
    if mercado not in MERCADOS:
        raise ValueError(f"mercado deve ser um de {MERCADOS}, recebido {mercado!r}")

    # --- 1. Modelo do mercado (spec §4: models/modelo_campeao_<mercado>.joblib) ---
    modelo = joblib.load(_DIR_MODELOS / f"modelo_campeao_{mercado}.joblib")
    features = list(modelo.feature_names_in_)   # ordem EXATA da matriz (spec §1, §7-1)

    # --- 2. Referência de treino p/ spatial_lag (spec §2.3) e district (passo 5) ---
    # treino_ref tem Latitude, Longitude, Price (CRU) e district do split de treino.
    ref = pd.read_parquet(_DIR_DADOS / f"treino_ref_{mercado}.parquet")
    ref_utm = (
        gpd.GeoDataFrame(
            ref,
            geometry=gpd.points_from_xy(ref["Longitude"], ref["Latitude"]),
            crs=CRS_GEO,
        )
        .to_crs(epsg=CRS_UTM)
    )
    coords_treino = np.c_[ref_utm.geometry.x.values, ref_utm.geometry.y.values]
    # k-NN único, reusado para o lag (k vizinhos) e o district (vizinho 0). spec §2.3:
    # ramo "TESTE" — o imóvel novo NÃO está no treino, então NÃO há +1/descarte.
    nn_treino = NearestNeighbors(n_neighbors=K_VIZINHOS).fit(coords_treino)
    log_precos_treino = np.log1p(ref["Price"].to_numpy(dtype=float))  # spec §2.3 passo 2
    districts_treino = ref["district"].to_numpy()

    # --- 3. Estações em UTM (spec §2.4): distância em metros, CRU ---
    estacoes = gpd.read_parquet(_DIR_DADOS / "estacoes_2019.parquet")
    estacoes_utm = estacoes.to_crs(epsg=CRS_UTM)

    # --- 4. Áreas de ponderação (spec §2.5): point-in-polygon em 4326, fallback UTM ---
    areas_geo = gpd.read_parquet(_DIR_DADOS / "areas_renda.parquet")
    if areas_geo.crs is None:                       # spec §2.5: declarar CRS 4326
        areas_geo = areas_geo.set_crs(CRS_GEO)
    areas_utm = areas_geo.to_crs(epsg=CRS_UTM)

    return Artefatos(
        mercado=mercado,
        modelo=modelo,
        features=features,
        nn_treino=nn_treino,
        log_precos_treino=log_precos_treino,
        districts_treino=districts_treino,
        estacoes_utm=estacoes_utm,
        areas_geo=areas_geo,
        areas_utm=areas_utm,
    )


# ---------------------------------------------------------------------------
# Helpers geométricos
# ---------------------------------------------------------------------------
def _ponto_utm(lat: float, lon: float) -> gpd.GeoSeries:
    """Converte uma coordenada (EPSG:4326) num ponto único em UTM 31983 (spec §7-5)."""
    return gpd.GeoSeries(
        gpd.points_from_xy([lon], [lat]), crs=CRS_GEO
    ).to_crs(epsg=CRS_UTM)


# ---------------------------------------------------------------------------
# Derivações de feature (uma por seção da spec)
# ---------------------------------------------------------------------------
def derivar_vizinhos(art: Artefatos, lat: float, lon: float):
    """
    Busca os k=3 vizinhos de TREINO mais próximos da coordenada, em UTM 31983.

    Reusado por `derivar_spatial_lag` e `derivar_district` (ambos precisam dos mesmos
    vizinhos). spec §2.3, ramo "TESTE": sem +1, sem descarte do próprio — o imóvel novo
    não pertence ao treino.

    Retorna os índices POSICIONAIS no treino_ref (array de tamanho k).
    """
    pt = _ponto_utm(lat, lon)
    q = np.c_[pt.x.values, pt.y.values]
    _, idx = art.nn_treino.kneighbors(q)
    return idx[0]


def derivar_spatial_lag(art: Artefatos, idx_vizinhos: np.ndarray) -> float:
    """
    spatial_lag = média do log1p(Price) dos k=3 vizinhos de treino (spec §2.3, passo 5).

    A escala é log-preço; é exatamente o que o modelo recebeu no treino. NÃO reverter.
    """
    return float(art.log_precos_treino[idx_vizinhos].mean())


def derivar_district(art: Artefatos, idx_vizinhos: np.ndarray) -> str:
    """
    District derivado da coordenada = district do VIZINHO de treino mais próximo
    (passo 5 do Prompt 1). NÃO se usa nome de bairro digitado: a grafia do treino_ref
    bate com as dummies `District_*` do modelo por construção (mesma fonte de dados),
    eliminando risco de divergência de grafia/acentuação.

    O vizinho mais próximo é a 1ª posição da busca k-NN (idx_vizinhos[0]).
    """
    return str(art.districts_treino[idx_vizinhos[0]])


def derivar_distancia_estacao(art: Artefatos, lat: float, lon: float) -> float:
    """
    distancia_estacao = distância (METROS, CRUA) até a estação de metrô/trem mais
    próxima, medida em UTM 31983 (spec §2.4). Equivale ao `sjoin_nearest` do nb02:
    a menor distância ponto-a-ponto. NÃO aplicar log (as árvores usam o valor cru).
    """
    pt = _ponto_utm(lat, lon).iloc[0]
    return float(art.estacoes_utm.geometry.distance(pt).min())


def derivar_renda_area(art: Artefatos, lat: float, lon: float):
    """
    renda_area = log1p(renda média domiciliar da área de ponderação que contém o ponto),
    point-in-polygon em EPSG:4326 (spec §2.5).

    Fallback (imóvel órfão): se NENHUM polígono contém o ponto, usa a área de ponderação
    MAIS PRÓXIMA em UTM 31983 (espelha `preencher_renda_orfaos`, spec §2.5).

    Retorna (renda_log, orfao: bool). `renda_log` já é log1p — pronto p/ a matriz.
    """
    ponto_geo = gpd.points_from_xy([lon], [lat], crs=CRS_GEO)[0]
    contidos = art.areas_geo[art.areas_geo.geometry.contains(ponto_geo)]

    if len(contidos) > 0:
        renda_crua = float(contidos.iloc[0][RENDA_COL])     # within: 1ª área que contém
        orfao = False
    else:
        # órfão: nenhuma área contém → área mais próxima em métrico (UTM)
        pt_utm = _ponto_utm(lat, lon).iloc[0]
        i = art.areas_utm.geometry.distance(pt_utm).idxmin()
        renda_crua = float(art.areas_utm.loc[i, RENDA_COL])
        orfao = True

    return float(np.log1p(renda_crua)), orfao


# ---------------------------------------------------------------------------
# Resultado da predição
# ---------------------------------------------------------------------------
@dataclass
class ResultadoPredicao:
    """Saída de `prever`: preço, incerteza e as features derivadas (p/ a UI explicar)."""
    mercado: str
    preco_estimado: float                 # em reais, já revertido com expm1
    preco_min: float                      # estimativa * (1 - MAPE)
    preco_max: float                      # estimativa * (1 + MAPE)
    mape: float                           # MAPE do test final do mercado
    pred_log: float                       # predição crua do modelo (log1p) — diagnóstico
    features_derivadas: dict              # spatial_lag, distancia_estacao, renda_area
    district_usado: str                   # district que alimentou as dummies
    district_reconhecido: bool            # o district casou com alguma dummy do modelo?
    renda_orfa: bool                      # a renda veio do fallback nearest?


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------
def prever(
    mercado: str,
    lat: float,
    lon: float,
    *,
    rooms: float,
    toilets: float,
    suites: float,
    parking: float,
    elevator: int,
    furnished: int,
    swimming_pool: int,
    new: int,
    size_m2: float,
    district: Optional[str] = None,
) -> ResultadoPredicao:
    """
    Prediz o preço de um apartamento e devolve a faixa de incerteza + features derivadas.

    Parâmetros
    ----------
    mercado : "venda" | "aluguel"
    lat, lon : coordenada do imóvel em EPSG:4326 (graus).
    rooms, toilets, suites, parking : contagens (atributos físicos, spec §2.1).
    elevator, furnished, swimming_pool, new : binários 0/1 (spec §2.1).
    size_m2 : área em m² — entra como log1p(size_m2) (spec §2.1).
    district : opcional. Por padrão (None) o district é DERIVADO da coordenada pelo
        vizinho de treino mais próximo (passo 5) — caminho de produção recomendado,
        pois a grafia bate com as dummies por construção. Se passado, sobrepõe a
        derivação (deve estar no formato do modelo, ex.: "Moema/São Paulo").

    Retorna
    -------
    ResultadoPredicao

    Passos (todos conforme a spec):
      1. Carrega artefatos do mercado (cache).            spec §4
      2. spatial_lag: k=3 vizinhos de treino, log1p, UTM. spec §2.3
      3. distancia_estacao: estação mais próxima, metros. spec §2.4
      4. renda_area: point-in-polygon + log1p, fallback.  spec §2.5
      5. district: vizinho mais próximo no treino_ref.    passo 5
      6. Monta a matriz na ordem de feature_names_in_.    spec §1, §5
      7. Prediz e reverte com expm1.                      spec §3
    """
    art = carregar_artefatos(mercado)            # passo 1

    # --- passos 2 e 5: uma única busca k-NN serve para o lag e para o district ---
    idx_vizinhos = derivar_vizinhos(art, lat, lon)
    spatial_lag = derivar_spatial_lag(art, idx_vizinhos)            # passo 2
    district_derivado = derivar_district(art, idx_vizinhos)         # passo 5

    # --- passos 3 e 4 ---
    distancia_estacao = derivar_distancia_estacao(art, lat, lon)    # passo 3
    renda_area, renda_orfa = derivar_renda_area(art, lat, lon)      # passo 4

    # district final: input do usuário sobrepõe a derivação, se fornecido
    district_usado = district if district is not None else district_derivado

    # --- passo 6: matriz na ordem EXATA de feature_names_in_ (spec §1, §5, §7-1) ---
    # Começa zerada: dummies de District inéditas/ausentes ficam 0 (handle_unknown
    # "ignore", spec §2.2). NÃO reordenar nada.
    vec = pd.Series(0.0, index=art.features)

    # atributos físicos diretos (spec §2.1)
    vec["Rooms"] = rooms
    vec["Toilets"] = toilets
    vec["Suites"] = suites
    vec["Parking"] = parking
    vec["Elevator"] = elevator
    vec["Furnished"] = furnished
    vec["Swimming Pool"] = swimming_pool
    vec["New"] = new
    vec["log_Size"] = np.log1p(size_m2)            # único físico transformado (spec §2.1)

    # District: casa por nome (spec §2.2 / §5). Bairro inédito → todas as dummies 0.
    col_district = f"District_{district_usado}"
    district_reconhecido = col_district in vec.index
    if district_reconhecido:
        vec[col_district] = 1.0

    # 3 espaciais, sempre as 3 últimas, nesta ordem (spec §1, §2.6)
    vec["spatial_lag"] = spatial_lag
    vec["distancia_estacao"] = distancia_estacao
    vec["renda_area"] = renda_area

    # --- passo 7: prediz (log) e reverte com expm1 (spec §3) ---
    X = pd.DataFrame([vec.to_numpy()], columns=art.features)
    pred_log = float(art.modelo.predict(X)[0])
    preco = float(np.expm1(pred_log))

    # faixa de incerteza: estimativa ± MAPE do mercado (spec §6)
    mape = MAPE_TESTE[mercado]
    preco_min = preco * (1.0 - mape)
    preco_max = preco * (1.0 + mape)

    return ResultadoPredicao(
        mercado=mercado,
        preco_estimado=preco,
        preco_min=preco_min,
        preco_max=preco_max,
        mape=mape,
        pred_log=pred_log,
        features_derivadas={
            "spatial_lag": spatial_lag,
            "distancia_estacao": distancia_estacao,
            "renda_area": renda_area,
        },
        district_usado=district_usado,
        district_reconhecido=district_reconhecido,
        renda_orfa=renda_orfa,
    )
