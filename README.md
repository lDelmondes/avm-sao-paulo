# AVM — Avaliação de Apartamentos em São Paulo

Modelo de avaliação automatizada (*Automated Valuation Model*) para apartamentos no
**município de São Paulo**, exposto como aplicação web. A partir do **CEP** e das
**características** do imóvel, estima o preço de **venda ou aluguel** com faixa de
incerteza e uma explicação dos fatores que mais pesaram.

> **É um AVM-produto (estimativa de apoio), não um laudo.** A estimativa serve como
> apoio à decisão e defesa de valor — **não substitui** avaliação formal com validade
> jurídica (NBR 14653 / CREA).

Projeto desenvolvido no contexto do MBA em Data Science, IA e Analytics (USP/ESALQ).

## O que faz
- Modelos campeões **XGBoost** (venda e aluguel), com **MAPE ~14% / ~20%** no teste.
- Deriva da coordenada do CEP os **fatores espaciais** (preço do entorno via *spatial
  lag*, proximidade de transporte, perfil socioeconômico) — replicando fielmente o
  pipeline de treino (ver `docs/SPEC_CONTRATO.md`).
- Interface Streamlit guiada: localização (CEP) → características → estimativa.

## Stack
Python · pandas · scikit-learn · XGBoost · geopandas · Streamlit · folium

## Como rodar o app localmente
```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```
O app já inclui o pacote de dados de produção em `app/data/` e os modelos em `models/` —
não é preciso baixar nada para rodar a estimativa. (A base bruta original, usada só para
re-treinar, é a [São Paulo Real Estate · Sale/Rent · April 2019](https://www.kaggle.com/datasets/argonalyst/sao-paulo-real-estate-sale-rent-april-2019/data),
não versionada.)

## Estrutura
- `app/` — aplicação: `app.py` (UI), `predictor.py` (inferência), `geocode.py` (CEP →
  coordenada), `data/` (dados de produção versionados).
- `models/` — modelos campeões (`.joblib`).
- `docs/` — contrato modelo↔app (`SPEC_CONTRATO.md`), jornadas e one-pager.
- `01_auditoria.ipynb`, `02_preparacao.ipynb`, `03_modelagem.ipynb` — pipeline de dados
  e modelagem. Visão geral em `PROJETO.md`.

## Deploy
Streamlit Community Cloud (arquivo principal: `app/app.py`). Dependências pinadas em
`requirements.txt`; tema em `.streamlit/config.toml`.

## Status
MVP funcional (Fase 6) — venda e aluguel, com estimativa, faixa de incerteza e explicação.
