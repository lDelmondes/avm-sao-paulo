# -*- coding: utf-8 -*-
"""
AVM São Paulo — interface Streamlit (Fase 6, Prompt 3 + ajustes de design).

Costura os módulos já verificados:
  - predictor.py  → prever(mercado, lat, lon, *, atributos...) → ResultadoPredicao
  - geocode.py    → geocodificar(cep) → ResultadoGeocode

Fontes da verdade: docs/PRD_JORNADAS.md (jornadas, erros, explicação heurística) e
docs/one_pager_prd.md (tom, escopo). NÃO reinventa derivação de feature — tudo vem do
predictor. A explicação é heurística de NEGÓCIO (sem SHAP, sem nome técnico de feature),
conforme a tabela da seção "A explicação" do PRD.

Design: paleta monocromática AZUL / preto / cinza / branco (tema em .streamlit/config.toml).
Sem emojis — ícones via Material Symbols nativos (:material/...:). Campos numéricos começam
VAZIOS (placeholder) e o botão "Estimar" só habilita quando os obrigatórios estão preenchidos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium

# Garante que os módulos irmãos (predictor, geocode) sejam importáveis no deploy.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor import prever, carregar_artefatos          # noqa: E402
from geocode import geocodificar, TipoErro                 # noqa: E402

_DIR_DADOS = Path(__file__).resolve().parent / "data"

# Paleta (azul / cinza / preto / branco)
_AZUL = "#1d4ed8"          # azul principal (destaque, ↑)
_AZUL_ESCURO = "#1e3a8a"
_CINZA = "#64748b"         # cinza (↓, textos secundários)
_PRETO = "#0f172a"

# ===========================================================================
# Configuração da página + CSS (paleta azul/cinza/preto/branco)
# ===========================================================================
st.set_page_config(
    page_title="Avaliação de Apartamentos · São Paulo",
    page_icon=":material/apartment:",   # Material Symbol (sem emoji)
    layout="centered",          # centered empilha bem no mobile e não estica no desktop
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
      /* preço-herói */
      .preco-hero {{ font-size: 2.7rem; font-weight: 800; line-height: 1.1;
                    color: {_AZUL}; margin: 0.1rem 0; }}
      .preco-faixa {{ font-size: 1.0rem; color: #475569; margin-bottom: 0.2rem; }}
      .preco-rotulo {{ font-size: 0.85rem; color: {_CINZA}; text-transform: uppercase;
                      letter-spacing: 0.05em; margin-bottom: 0.2rem; }}
      /* fatores da explicação: azul para ↑, cinza para ↓ (sem verde/vermelho) */
      .fator {{ font-size: 1.02rem; padding: 0.18rem 0; }}
      .fator .up   {{ color: {_AZUL}; font-weight: 800; }}
      .fator .down {{ color: {_CINZA}; font-weight: 800; }}
      /* alertas (st.success/info/warning/error): força paleta azul/cinza —
         remove verde/vermelho/amarelo nativos; a semântica fica nos ícones Material. */
      div[data-testid="stAlert"], div[data-testid="stAlert"] > div {{
          background-color: #f1f5f9 !important;       /* cinza-azulado */
          border: 1px solid #cbd5e1 !important;
          border-left: 4px solid {_AZUL} !important;  /* acento azul */
          border-radius: 6px;
      }}
      div[data-testid="stAlert"] * {{ color: {_PRETO} !important; }}
      /* compacta um pouco no mobile */
      @media (max-width: 640px) {{ .preco-hero {{ font-size: 2.2rem; }} }}
      div[data-testid="stMetricValue"] {{ font-size: 1.4rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# Cache de recursos / dados
# ===========================================================================
@st.cache_resource(show_spinner="Carregando modelo…")
def _artefatos(mercado: str):
    """Artefatos pesados do predictor (modelo + parquets): carregados uma vez por processo."""
    return carregar_artefatos(mercado)


@st.cache_data(show_spinner=False)
def _referencias() -> dict:
    """
    Referências da região para a explicação heurística, computadas dos artefatos de
    produção:
      - treino_ref → mediana de PREÇO por mercado (localização) e mediana de ÁREA por
        DISTRITO e por mercado (área típica do bairro — dado real dos vizinhos);
      - areas_renda → quartis de renda (perfil socioeconômico).
    Tudo é informativo (só para a narrativa) — nunca entra na predição. A coluna `Size`
    do treino_ref é usada AQUI e só aqui; o predictor não a usa (matriz segue 107/105).
    """
    ref = {}
    for m in ("venda", "aluguel"):
        tr = pd.read_parquet(_DIR_DADOS / f"treino_ref_{m}.parquet")
        ref[f"preco_mediana_{m}"] = float(tr["Price"].median())
        ref[f"size_mediana_{m}"] = float(tr["Size"].median())            # fallback global
        # área típica POR DISTRITO (mediana dos imóveis daquele bairro no treino):
        ref[f"size_distrito_{m}"] = tr.groupby("district")["Size"].median().to_dict()
    renda = gpd.read_parquet(_DIR_DADOS / "areas_renda.parquet")[
        "ponderation_area_average_household_income"
    ]
    ref["renda_mediana"] = float(renda.median())
    ref["renda_p75"] = float(renda.quantile(0.75))
    ref["renda_p25"] = float(renda.quantile(0.25))
    return ref


# NOTA SOBRE CACHE DE GEOCODE: não envolvemos `geocodificar` com @st.cache_data de
# propósito. O módulo geocode já tem cache interno INTELIGENTE (memoiza sucessos e erros
# determinísticos, mas NÃO erros transitórios — timeout/serviço fora). Envolver com
# cache_data cachearia também os erros transitórios e impediria o usuário de tentar de
# novo. Então chamamos `geocodificar` direto; o cache certo já acontece no módulo.


# ===========================================================================
# Helpers de formatação
# ===========================================================================
def reais(v: float) -> str:
    """Formata em reais no padrão pt-BR (R$ 780.000), sem centavos."""
    return "R$ " + f"{v:,.0f}".replace(",", ".")


def _sufixo(mercado: str) -> str:
    return " /mês" if mercado == "aluguel" else ""


# ===========================================================================
# Explicação heurística (linguagem de negócio — PRD seção "A explicação")
# ===========================================================================
def explicar(mercado: str, r, attrs: dict, ref: dict) -> list:
    """
    Traduz o imóvel + features derivadas em 3–5 fatores de NEGÓCIO com direção (↑/↓).
    Compara valores do imóvel com referências da região. Sem SHAP, sem nome de feature.
    Cada item: {"texto": str, "dir": "up"|"down"}. Ordenado por relevância (prioridade).
    """
    feats = r.features_derivadas
    preco_med = ref[f"preco_mediana_{mercado}"]
    fatores = []  # (prioridade, dir, texto)

    # 1. Localização (spatial_lag → preço típico da vizinhança). Fator dominante.
    preco_vizinhanca = float(np.expm1(feats["spatial_lag"]))
    if preco_vizinhanca >= 1.25 * preco_med:
        fatores.append((1, "up", "Localização valorizada — vizinhança de alto valor"))
    elif preco_vizinhanca <= 0.80 * preco_med:
        fatores.append((1, "down", "Localização de menor valor de mercado"))

    # 2. Área — comparada à ÁREA TÍPICA DO BAIRRO (dado real: mediana dos imóveis do
    #    mesmo distrito no treino, pelo district derivado da coordenada). Banda ±20% para
    #    não sinalizar diferenças irrelevantes. Fallback: mediana do mercado (raro).
    area = attrs["size_m2"]
    tipica = ref[f"size_distrito_{mercado}"].get(
        r.district_usado, ref[f"size_mediana_{mercado}"]
    )
    if area >= 1.20 * tipica:
        fatores.append((2, "up",
            f"Área de {area:.0f} m² — acima da típica do bairro (~{tipica:.0f} m²)"))
    elif area <= 0.80 * tipica:
        fatores.append((2, "down",
            f"Área de {area:.0f} m² — compacta para o bairro (~{tipica:.0f} m²)"))

    # 3. Renda da região (perfil socioeconômico do entorno)
    renda = float(np.expm1(feats["renda_area"]))
    if renda >= ref["renda_p75"]:
        fatores.append((3, "up", "Entorno de alto padrão socioeconômico"))
    elif renda <= ref["renda_p25"]:
        fatores.append((3, "down", "Entorno de menor renda média"))

    # 4. Vagas de garagem
    vagas = int(attrs["parking"])
    if vagas >= 2:
        fatores.append((4, "up", f"{vagas} vagas de garagem"))
    elif vagas == 0:
        fatores.append((4, "down", "Sem vaga de garagem"))

    # 5. Proximidade de transporte sobre trilhos (metrô/trem)
    dist = feats["distancia_estacao"]
    if dist <= 800:
        fatores.append((5, "up", "Próximo de metrô/trem"))
    elif dist >= 2500:
        fatores.append((5, "down", "Distante de transporte sobre trilhos"))

    # 6. Suíte(s)
    suites = int(attrs["suites"])
    if suites >= 1:
        fatores.append((6, "up", f"{suites} suíte" + ("s" if suites > 1 else "")))

    # 7/8. Mobília e "novo" — pesam mais no aluguel
    if attrs["furnished"]:
        fatores.append((6 if mercado == "aluguel" else 8, "up", "Imóvel mobiliado"))
    if attrs["new"]:
        fatores.append((7 if mercado == "aluguel" else 8, "up", "Imóvel novo"))

    fatores.sort(key=lambda t: t[0])
    selecionados = [{"texto": t[2], "dir": t[1]} for t in fatores[:5]]
    if not selecionados:  # tudo neutro: dá um fator-âncora honesto
        selecionados = [{"texto": "Características em linha com a média da região",
                         "dir": "up"}]
    return selecionados


# ===========================================================================
# Estado da sessão
# ===========================================================================
ss = st.session_state
ss.setdefault("mercado", "venda")
ss.setdefault("geo", None)          # ResultadoGeocode confirmável
ss.setdefault("confirmado", False)  # usuário confirmou o endereço?
ss.setdefault("resultado", None)    # ResultadoPredicao


def _resetar_tudo():
    ss.geo = None
    ss.confirmado = False
    ss.resultado = None


# ===========================================================================
# CABEÇALHO
# ===========================================================================
st.title("Avaliação de Apartamentos · São Paulo")
st.caption(
    "Estimativa de **apoio** para apartamentos no **município de São Paulo** — não é "
    "laudo de avaliação (NBR/CREA). Informe o CEP e as características para uma estimativa "
    "com faixa de incerteza."
)

# --- Toggle Venda | Aluguel ---
_OPCOES = {"Venda": "venda", "Aluguel": "aluguel"}
if hasattr(st, "segmented_control"):
    escolha = st.segmented_control(
        "Tipo de avaliação", list(_OPCOES), default="Venda", key="seg_mercado"
    )
    escolha = escolha or "Venda"
else:
    escolha = st.radio(
        "Tipo de avaliação", list(_OPCOES), horizontal=True, key="seg_mercado"
    )
mercado_novo = _OPCOES[escolha]
if mercado_novo != ss.mercado:           # trocou de mercado → resultado anterior é stale
    ss.mercado = mercado_novo
    ss.resultado = None
mercado = ss.mercado

st.divider()

# ===========================================================================
# ETAPA 1 — LOCALIZAÇÃO (CEP → geocode → confirmar)
# ===========================================================================
st.subheader("1 · Localização")

with st.form("form_cep", clear_on_submit=False):
    col_cep, col_btn = st.columns([3, 1])
    with col_cep:
        cep_input = st.text_input(
            "CEP do imóvel", placeholder="01310-100", label_visibility="collapsed"
        )
    with col_btn:
        localizar = st.form_submit_button(
            "Localizar", icon=":material/search:", use_container_width=True
        )

if localizar:
    with st.spinner("Localizando o CEP…"):
        ss.geo = geocodificar(cep_input)
    ss.confirmado = False
    ss.resultado = None

geo = ss.geo

# --- Tratamento dos retornos do geocode ---
if geo is not None and not ss.confirmado:
    if geo.sucesso:
        st.success(f"**{geo.endereco_formatado}**", icon=":material/check_circle:")
        # Mapa Folium (via streamlit-folium): marcador com tamanho de TELA fixo —
        # CircleMarker (raio em px, não escala com o zoom como o st.map fazia).
        # Borda azul sólida + miolo translúcido; tile claro neutro (combina c/ tema light).
        mapa = folium.Map(
            location=[geo.lat, geo.lon], zoom_start=16,   # nível rua/quarteirão
            tiles="CartoDB positron", control_scale=False, zoom_control=True,
        )
        folium.CircleMarker(
            location=[geo.lat, geo.lon],
            radius=32,                      # raio FIXO em pixels (tamanho de tela)
            color=_AZUL, weight=3,          # borda azul SÓLIDA
            fill=True, fill_color=_AZUL, fill_opacity=0.30,   # miolo azul translúcido
            opacity=1.0,
        ).add_to(mapa)
        # returned_objects=[] → não devolve interação (zoom/pan) → não dispara reruns;
        # key estável mantém o mapa entre reruns. Modo leitura: só visualizar/confirmar.
        st_folium(mapa, height=320, use_container_width=True,
                  returned_objects=[], key="mapa_confirmacao")
        if geo.cobertura_fraca:                       # jornada 3
            st.warning(
                "Esta região tem **poucos imóveis na base de referência**. A estimativa "
                "é menos precisa do que em áreas com mais dados.",
                icon=":material/warning:",
            )
        st.caption("Confira se o endereço está correto antes de continuar.")
        if st.button("Confirmar endereço e continuar", type="primary",
                     icon=":material/check:", use_container_width=True):
            ss.confirmado = True
            st.rerun()

    elif geo.tipo_erro == TipoErro.FORA_DE_SP:        # jornada 2
        if geo.cidade:
            st.error(
                f"Este modelo cobre apenas apartamentos no **município de São Paulo**. "
                f"O CEP informado parece estar em **{geo.cidade}/{geo.uf}**. Verifique o "
                f"CEP ou consulte uma ferramenta adequada para essa região.",
                icon=":material/wrong_location:",
            )
        else:
            st.error(
                "Este modelo cobre apenas apartamentos no **município de São Paulo**. O "
                "CEP informado parece estar em outra localidade.",
                icon=":material/wrong_location:",
            )

    elif geo.tipo_erro == TipoErro.CEP_MALFORMADO:    # jornada 4
        st.warning("CEP incompleto. Digite os **8 dígitos** (ex.: 01310-100).",
                   icon=":material/warning:")

    elif geo.tipo_erro == TipoErro.CEP_NAO_ENCONTRADO:  # jornada 4
        st.error("Não foi possível localizar este CEP. Verifique se está correto.",
                 icon=":material/error:")

    elif geo.tipo_erro == TipoErro.ENDERECO_NAO_GEOCODIFICADO:  # jornada 4
        st.error(
            "Encontramos o CEP, mas não conseguimos posicioná-lo no mapa. Tente um CEP "
            "próximo ou confira o endereço.",
            icon=":material/error:",
        )

    elif geo.tipo_erro == TipoErro.SERVICO_INDISPONIVEL:  # jornada 4 (nota técnica)
        st.error(
            "Serviço de localização temporariamente indisponível. Tente novamente em "
            "instantes.",
            icon=":material/cloud_off:",
        )

# ===========================================================================
# ETAPA 2 — CARACTERÍSTICAS (só após confirmar o endereço)
# Campos numéricos começam VAZIOS (value=None + placeholder); o botão "Estimar"
# só habilita quando os 5 obrigatórios estão preenchidos. Sem st.form aqui — fora
# de form os widgets reavaliam a cada mudança, permitindo habilitar o botão ao vivo.
# ===========================================================================
if ss.confirmado and geo is not None and geo.sucesso:
    st.divider()
    st.subheader("2 · Características do imóvel")
    st.caption(f":material/location_on: {geo.endereco_formatado}")

    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input("Área útil (m²)", min_value=1, max_value=2000,
                               value=None, step=1, placeholder="Ex.: 75",
                               help="Tipicamente 20–500 m².")
        quartos = st.number_input("Quartos", min_value=0, max_value=15,
                                  value=None, step=1, placeholder="Ex.: 2")
        banheiros = st.number_input("Banheiros", min_value=0, max_value=15,
                                    value=None, step=1, placeholder="Ex.: 2")
        suites = st.number_input("Suítes", min_value=0, max_value=15,
                                 value=None, step=1, placeholder="Ex.: 1")
    with c2:
        vagas = st.number_input("Vagas de garagem", min_value=0, max_value=15,
                                value=None, step=1, placeholder="Ex.: 1")
        elevador = st.toggle("Prédio com elevador", value=False)
        mobiliado = st.toggle("Mobiliado", value=False)
        piscina = st.toggle("Piscina no condomínio", value=False)
        novo = st.toggle("Imóvel novo", value=False)

    # campos obrigatórios = os 5 numéricos preenchidos (None = ainda vazio)
    obrigatorios = [area, quartos, banheiros, suites, vagas]
    campos_ok = all(v is not None for v in obrigatorios)
    if not campos_ok:
        st.caption(
            ":material/info: Preencha área, quartos, banheiros, suítes e vagas para "
            "habilitar a estimativa."
        )

    estimar = st.button(
        f"Estimar preço de {'aluguel' if mercado == 'aluguel' else 'venda'}",
        type="primary", icon=":material/calculate:",
        use_container_width=True, disabled=not campos_ok,
    )

    if estimar and campos_ok:
        # validação leve (jornada 5): campo vazio NÃO chega aqui (botão desabilitado);
        # aqui só alertamos valores preenchidos porém implausíveis — sem bloquear.
        avisos = []
        if area < 15 or area > 800:
            avisos.append(f"Área de {area:.0f} m²? Confirme se está correta.")
        if quartos > 8:
            avisos.append(f"{quartos} quartos? Confirme se está correto.")
        if banheiros > 10:
            avisos.append(f"{banheiros} banheiros? Confirme se está correto.")
        for a in avisos:
            st.warning(a, icon=":material/warning:")

        with st.spinner("Calculando a estimativa…"):
            _artefatos(mercado)  # garante modelo em cache (st.cache_resource)
            resultado = prever(
                mercado, geo.lat, geo.lon,
                rooms=quartos, toilets=banheiros, suites=suites, parking=vagas,
                elevator=int(elevador), furnished=int(mobiliado),
                swimming_pool=int(piscina), new=int(novo), size_m2=float(area),
                district=None,                 # derivado da coordenada pelo predictor
            )
        ss.resultado = resultado
        ss.attrs = dict(size_m2=float(area), parking=vagas, suites=suites,
                        furnished=int(mobiliado), new=int(novo))

# ===========================================================================
# ETAPA 3 — RESULTADO
# ===========================================================================
if ss.resultado is not None:
    r = ss.resultado
    st.divider()
    st.subheader("3 · Estimativa")

    with st.container(border=True):
        st.markdown('<div class="preco-rotulo">Preço estimado</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="preco-hero">{reais(r.preco_estimado)}{_sufixo(mercado)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="preco-faixa">Faixa provável: <b>{reais(r.preco_min)}</b> a '
            f'<b>{reais(r.preco_max)}</b>{_sufixo(mercado)} '
            f'<span style="color:#94a3b8">(± {r.mape:.0%}, erro típico do modelo)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- Fatores que mais pesaram (↑ azul, ↓ cinza) ---
    st.markdown("##### Fatores que mais pesaram")
    fatores = explicar(mercado, r, ss.attrs, _referencias())
    for f in fatores:
        seta = "↑" if f["dir"] == "up" else "↓"
        classe = "up" if f["dir"] == "up" else "down"
        st.markdown(
            f'<div class="fator"><span class="{classe}">{seta}</span> {f["texto"]}</div>',
            unsafe_allow_html=True,
        )

    # --- Avisos de menor precisão (jornada 3) ---
    avisos_precisao = []
    if geo is not None and geo.cobertura_fraca:
        avisos_precisao.append("região com poucos imóveis na base de referência")
    if not r.district_reconhecido:
        avisos_precisao.append("bairro pouco representado no modelo")
    if r.renda_orfa:
        avisos_precisao.append("dado socioeconômico aproximado para esta coordenada")
    if avisos_precisao:
        st.info("Estimativa menos precisa: " + "; ".join(avisos_precisao) + ".",
                icon=":material/info:")

    st.caption(
        "Estimativa de apoio baseada em modelo estatístico sobre anúncios — não substitui "
        "avaliação formal. A faixa reflete o erro típico medido do modelo."
    )

    if st.button("Nova estimativa", icon=":material/refresh:",
                 use_container_width=True):
        _resetar_tudo()
        st.rerun()
