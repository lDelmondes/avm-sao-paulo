# -*- coding: utf-8 -*-
"""
Geocodificação CEP → coordenada — AVM São Paulo, Fase 6, Prompt 2.

INDEPENDENTE do predictor: não toca no contrato do modelo. Recebe um CEP e devolve uma
coordenada validada (EPSG:4326) ou um erro tratado, num resultado estruturado que a UI
(Prompt 3) consome para decidir a mensagem.

FLUXO (cascata de custo zero, sem chave de API):
  1. CEP → endereço estruturado via ViaCEP (logradouro, bairro, localidade, uf, ibge).
  2. Endereço → coordenada via Nominatim (OpenStreetMap).
  3. Validação em 2 camadas (ver `validar_municipio_sp` e `validar_cobertura`).

POR QUE 2 CAMADAS DE VALIDAÇÃO (decisão de robustez):
  - Camada 1 — MUNICÍPIO (ViaCEP): `uf == "SP"` e código IBGE do município == 3550308
    (São Paulo capital). É a resposta AUTORITATIVA para "este CEP é da cidade de SP?",
    pois é exatamente o que a base dos Correios codifica. Roda ANTES do Nominatim, então
    um CEP do RJ (ou de Guarulhos/Osasco) é rejeitado sem gastar requisição.
  - Camada 2 — COBERTURA (coordenada): distância ao imóvel de treino mais próximo, com
    limiar GENEROSO (15 km). NÃO serve para checar município (um bounding box de SP pega
    Guarulhos; e a distância não separa Parelheiros/SP — 8 km do treino — de Guarulhos —
    3,8 km). Serve para pegar ERRO GROSSEIRO de geocodificação (Nominatim devolvendo
    ponto no oceano/outro estado). 15 km acomoda o maior vão interno do treino (10,4 km)
    e zonas esparsas legítimas de SP, e ainda rejeita um geocode a centenas de km.

REDE / ToS:
  - Nominatim exige User-Agent identificável e rate limit de 1 req/s — ambos respeitados.
  - Cache: CEPs repetidos não geram nova requisição (sucessos e erros determinísticos são
    memoizados; erros transitórios — timeout/falha de geocode — NÃO são cacheados, para
    permitir nova tentativa). Estruturado para o Streamlit envolver com @st.cache_data.

Dependências: usa apenas `requests` (já no projeto) — geopy NÃO é necessário.
"""
from __future__ import annotations

import os
import re
import time
import threading
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
VIACEP_URL = "https://viacep.com.br/ws/{cep}/json/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Nominatim exige apenas que o User-Agent IDENTIFIQUE a aplicação (não exige e-mail
# pessoal). O padrão é um identificador genérico de projeto. Para fornecer um contato
# real (recomendado em produção de alto volume), defina a env var AVM_GEOCODE_CONTATO
# — ela NÃO é versionada, mantendo dados pessoais fora do repositório.
CONTATO = os.environ.get(
    "AVM_GEOCODE_CONTATO",
    "github.com/avm-sao-paulo",
)
USER_AGENT = f"avm-sao-paulo/1.0 (+{CONTATO})"

TIMEOUT = 10                    # segundos por requisição HTTP
NOMINATIM_INTERVALO = 1.0       # rate limit: >= 1 req/segundo (ToS do Nominatim)

IBGE_SAO_PAULO = "3550308"      # código IBGE do município de São Paulo (capital)
COBERTURA_MAX_M = 15_000        # limiar DURO: acima disso é geocode grosseiro → recusa
COBERTURA_FRACA_M = 2_000       # limiar SUAVE: acima disso é zona esparsa de SP (jornada 3
                                # do PRD) → estima MAS sinaliza cobertura fraca à UI.
                                # Calibrado: p99,9 da densidade normal do treino ≈ 1,7 km.

CRS_GEO = "EPSG:4326"
CRS_UTM = 31983
_DIR_DADOS = Path(__file__).resolve().parent / "data"


# Tipos de erro (a UI mapeia para a mensagem final — jornadas 2 e 4 do PRD).
class TipoErro:
    CEP_MALFORMADO = "cep_malformado"
    CEP_NAO_ENCONTRADO = "cep_nao_encontrado"
    FORA_DE_SP = "fora_de_sp"
    ENDERECO_NAO_GEOCODIFICADO = "endereco_nao_geocodificado"
    SERVICO_INDISPONIVEL = "servico_indisponivel"


# ---------------------------------------------------------------------------
# Resultado estruturado
# ---------------------------------------------------------------------------
@dataclass
class ResultadoGeocode:
    """Saída de `geocodificar`. A UI decide a mensagem a partir de `tipo_erro`."""
    sucesso: bool
    cep: Optional[str] = None                 # CEP normalizado (8 dígitos)
    lat: Optional[float] = None               # EPSG:4326
    lon: Optional[float] = None
    endereco_formatado: Optional[str] = None  # p/ o usuário confirmar (jornada 1)
    logradouro: Optional[str] = None
    bairro: Optional[str] = None
    cidade: Optional[str] = None
    uf: Optional[str] = None
    distancia_cobertura_m: Optional[float] = None  # dist. ao treino mais próximo
    fora_de_sp: bool = False                  # sinal p/ a UI (jornada 2)
    cobertura_fraca: bool = False             # SP, mas zona esparsa (jornada 3): estima c/ aviso
    tipo_erro: Optional[str] = None           # um de TipoErro.*
    detalhe: Optional[str] = None             # diagnóstico interno (log/debug)


# ---------------------------------------------------------------------------
# Referência de cobertura (camada 2) — carregada uma vez
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _referencia_cobertura() -> NearestNeighbors:
    """
    k-NN (k=1) sobre a UNIÃO das coordenadas de treino dos dois mercados, em UTM 31983.

    Mede a distância de uma coordenada à área onde o AVM tem dados (guarda de cobertura).
    Usa os mesmos `treino_ref_*` do predictor, mas só as coordenadas — sem acoplar ao
    contrato do modelo.
    """
    refs = []
    for mercado in ("venda", "aluguel"):
        r = pd.read_parquet(_DIR_DADOS / f"treino_ref_{mercado}.parquet")
        refs.append(r[["Latitude", "Longitude"]])
    ref = pd.concat(refs, ignore_index=True)
    utm = gpd.GeoDataFrame(
        ref, geometry=gpd.points_from_xy(ref["Longitude"], ref["Latitude"]), crs=CRS_GEO
    ).to_crs(epsg=CRS_UTM)
    coords = np.c_[utm.geometry.x.values, utm.geometry.y.values]
    return NearestNeighbors(n_neighbors=1).fit(coords)


def validar_cobertura(lat: float, lon: float):
    """Retorna (dentro_da_cobertura: bool, distancia_m: float) para uma coordenada."""
    nn = _referencia_cobertura()
    pt = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=CRS_GEO).to_crs(epsg=CRS_UTM)
    d, _ = nn.kneighbors(np.c_[pt.x.values, pt.y.values])
    dist_m = float(d[0, 0])
    return dist_m <= COBERTURA_MAX_M, dist_m


def validar_municipio_sp(dados_viacep: dict) -> bool:
    """Camada 1: o CEP pertence ao município de São Paulo? (autoritativo, via ViaCEP)."""
    uf = (dados_viacep.get("uf") or "").strip().upper()
    ibge = (dados_viacep.get("ibge") or "").strip()
    localidade = (dados_viacep.get("localidade") or "").strip().lower()
    # IBGE é o sinal mais forte; localidade+uf é o fallback se o IBGE vier vazio.
    if ibge:
        return ibge == IBGE_SAO_PAULO
    return uf == "SP" and localidade == "são paulo"


# ---------------------------------------------------------------------------
# Etapa 1 — ViaCEP
# ---------------------------------------------------------------------------
def normalizar_cep(cep: str) -> Optional[str]:
    """Remove tudo que não é dígito; retorna 8 dígitos ou None se malformado."""
    digitos = re.sub(r"\D", "", cep or "")
    return digitos if len(digitos) == 8 else None


def consultar_viacep(cep8: str) -> dict:
    """
    Consulta o ViaCEP. Retorna o dict de dados.
    Levanta `LookupError` se o CEP não existe, `ConnectionError` se o serviço falha.
    """
    try:
        resp = requests.get(VIACEP_URL.format(cep=cep8), timeout=TIMEOUT)
        resp.raise_for_status()
        dados = resp.json()
    except (requests.Timeout, requests.ConnectionError) as e:
        raise ConnectionError("ViaCEP indisponível") from e
    except (requests.HTTPError, ValueError) as e:
        raise ConnectionError("ViaCEP respondeu inválido") from e
    # ViaCEP devolve 200 com {"erro": true} quando o CEP não existe.
    if isinstance(dados, dict) and dados.get("erro"):
        raise LookupError("CEP não localizado")
    return dados


def _formatar_endereco(d: dict, cep8: str) -> str:
    """Monta um endereço legível a partir dos campos do ViaCEP (jornada 1)."""
    cep_fmt = f"{cep8[:5]}-{cep8[5:]}"
    partes = [p for p in (d.get("logradouro"), d.get("bairro")) if p]
    cidade_uf = f"{d.get('localidade', '')} - {d.get('uf', '')}".strip(" -")
    cabeca = ", ".join(partes)
    return f"{cabeca}, {cidade_uf}, {cep_fmt}" if cabeca else f"{cidade_uf}, {cep_fmt}"


# ---------------------------------------------------------------------------
# Etapa 2 — Nominatim (com rate limit)
# ---------------------------------------------------------------------------
_lock_nominatim = threading.Lock()
_ultimo_nominatim = [0.0]   # timestamp da última chamada (lista p/ mutabilidade)


def _nominatim_query(query: str) -> Optional[tuple]:
    """
    Geocodifica uma query no Nominatim, respeitando o rate limit de 1 req/s.
    Retorna (lat, lon) ou None se não geocodificou. Levanta ConnectionError em falha.
    """
    with _lock_nominatim:
        espera = NOMINATIM_INTERVALO - (time.monotonic() - _ultimo_nominatim[0])
        if espera > 0:
            time.sleep(espera)
        try:
            resp = requests.get(
                NOMINATIM_URL,
                params={"q": query, "format": "json", "limit": 1, "countrycodes": "br"},
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            resultados = resp.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            raise ConnectionError("Nominatim indisponível") from e
        except (requests.HTTPError, ValueError) as e:
            raise ConnectionError("Nominatim respondeu inválido") from e
        finally:
            _ultimo_nominatim[0] = time.monotonic()

    if not resultados:
        return None
    return float(resultados[0]["lat"]), float(resultados[0]["lon"])


def geocodificar_endereco(d: dict) -> Optional[tuple]:
    """
    Endereço estruturado → (lat, lon), com fallback.
    1ª tentativa: logradouro + bairro + cidade. 2ª: bairro + cidade (menos preciso).
    Retorna None só se ambas falharem (sem resultado).
    """
    cidade = d.get("localidade", "São Paulo")
    logradouro = (d.get("logradouro") or "").strip()
    bairro = (d.get("bairro") or "").strip()

    tentativas = []
    if logradouro:
        tentativas.append(", ".join(p for p in (logradouro, bairro, cidade, "SP, Brasil") if p))
    if bairro:                                  # fallback: bairro + cidade
        tentativas.append(", ".join(p for p in (bairro, cidade, "SP, Brasil") if p))
    if not tentativas:                          # CEP de cidade inteira (sem logr./bairro)
        tentativas.append(f"{cidade}, SP, Brasil")

    for q in tentativas:
        coord = _nominatim_query(q)
        if coord is not None:
            return coord
    return None


# ---------------------------------------------------------------------------
# Função principal (com cache inteligente)
# ---------------------------------------------------------------------------
# Cacheia só sucessos e erros DETERMINÍSTICOS (malformado, não-encontrado, fora-de-SP).
# Erros transitórios (serviço/geocode) ficam de fora p/ permitir nova tentativa.
_cache: dict[str, ResultadoGeocode] = {}
_ERROS_CACHEAVEIS = {
    TipoErro.CEP_MALFORMADO, TipoErro.CEP_NAO_ENCONTRADO, TipoErro.FORA_DE_SP,
}


def geocodificar(cep: str) -> ResultadoGeocode:
    """
    CEP (str, com ou sem máscara) → ResultadoGeocode validado.

    Sucessos e erros determinísticos são memoizados (sem nova requisição). No Streamlit,
    pode-se envolver com @st.cache_data; o cache interno já evita recarga e respeita o
    rate limit do Nominatim em CEPs repetidos.
    """
    # --- normalização (erro determinístico, não consulta rede) ---
    cep8 = normalizar_cep(cep)
    if cep8 is None:
        return ResultadoGeocode(
            sucesso=False, tipo_erro=TipoErro.CEP_MALFORMADO,
            detalhe="CEP deve ter 8 dígitos.",
        )

    if cep8 in _cache:
        return _cache[cep8]

    res = _geocodificar_sem_cache(cep8)
    if res.sucesso or res.tipo_erro in _ERROS_CACHEAVEIS:
        _cache[cep8] = res
    return res


def _geocodificar_sem_cache(cep8: str) -> ResultadoGeocode:
    cep_fmt = f"{cep8[:5]}-{cep8[5:]}"

    # --- etapa 1: ViaCEP ---
    try:
        d = consultar_viacep(cep8)
    except LookupError:
        return ResultadoGeocode(sucesso=False, cep=cep_fmt,
                                tipo_erro=TipoErro.CEP_NAO_ENCONTRADO,
                                detalhe="ViaCEP não localizou o CEP.")
    except ConnectionError as e:
        return ResultadoGeocode(sucesso=False, cep=cep_fmt,
                                tipo_erro=TipoErro.SERVICO_INDISPONIVEL,
                                detalhe=str(e))

    endereco = _formatar_endereco(d, cep8)
    base = dict(
        cep=cep_fmt, endereco_formatado=endereco,
        logradouro=d.get("logradouro") or None, bairro=d.get("bairro") or None,
        cidade=d.get("localidade") or None, uf=d.get("uf") or None,
    )

    # --- camada 1: município de SP (antes do Nominatim) ---
    if not validar_municipio_sp(d):
        return ResultadoGeocode(sucesso=False, fora_de_sp=True,
                                tipo_erro=TipoErro.FORA_DE_SP,
                                detalhe=f"CEP é de {d.get('localidade')}/{d.get('uf')}.",
                                **base)

    # --- etapa 2: Nominatim (endereço → coordenada) ---
    try:
        coord = geocodificar_endereco(d)
    except ConnectionError as e:
        return ResultadoGeocode(sucesso=False, tipo_erro=TipoErro.SERVICO_INDISPONIVEL,
                                detalhe=str(e), **base)
    if coord is None:
        return ResultadoGeocode(sucesso=False,
                                tipo_erro=TipoErro.ENDERECO_NAO_GEOCODIFICADO,
                                detalhe="Nominatim não encontrou o endereço.", **base)

    lat, lon = coord

    # --- camada 2: cobertura (guarda contra geocode grosseiro) ---
    dentro, dist_m = validar_cobertura(lat, lon)
    if not dentro:
        return ResultadoGeocode(sucesso=False, lat=lat, lon=lon, fora_de_sp=True,
                                distancia_cobertura_m=dist_m,
                                tipo_erro=TipoErro.FORA_DE_SP,
                                detalhe=f"Coordenada a {dist_m/1000:.1f} km do treino "
                                        f"(> {COBERTURA_MAX_M/1000:.0f} km): geocode fora "
                                        f"da cobertura.", **base)

    # --- sucesso (com possível aviso de cobertura fraca — jornada 3) ---
    return ResultadoGeocode(sucesso=True, lat=lat, lon=lon,
                            distancia_cobertura_m=dist_m,
                            cobertura_fraca=dist_m > COBERTURA_FRACA_M, **base)
