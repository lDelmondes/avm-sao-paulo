# AVM — Precificação de Imóveis em São Paulo

Modelo de avaliação automatizada (Automated Valuation Model) para apartamentos
em São Paulo, comparando regressão linear múltipla (baseline normativo, NBR 14653)
com técnicas modernas de machine learning, exposto via aplicação web.

Projeto desenvolvido no contexto do MBA em Data Science, IA e Analytics (USP/ESALQ).

## Objetivo
Estimar o preço de anúncio de apartamentos (venda e locação) a partir de suas
características, com recomendação de valor interpretável.

## Stack
Python · pandas · scikit-learn · XGBoost · análise espacial · Streamlit

## Como reproduzir
1. Clone o repositório
2. Crie e ative um ambiente virtual: `python -m venv venv`
3. Instale as dependências: `pip install -r requirements.txt`
4. Baixe a base no [Kaggle São Paulo Real Estate, Sale/Rent, April 2019](https://www.kaggle.com/datasets/argonalyst/sao-paulo-real-estate-sale-rent-april-2019/data) e coloque em `data/`

## Planejamento
O plano completo do projeto está em [plano_projeto_avm_sao_paulo.md](plano_projeto_avm_sao_paulo.md).

## Status
Em desenvolvimento — Fase 0 (fundação).