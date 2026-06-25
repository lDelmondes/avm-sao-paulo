# Projeto AVM — Precificação de Imóveis em São Paulo

> **Documento central de referência do projeto.**
> Modelo de avaliação automatizada (*Automated Valuation Model*) para apartamentos em São Paulo, construído como projeto pessoal/acadêmico no contexto do MBA em Data Science, IA e Analytics (USP/ESALQ).
>
> *Última atualização: 25/06/2026 — versão 1.0*

---

## Índice

1. [Visão geral](#1-visão-geral)
2. [Objetivos](#2-objetivos)
3. [A base de dados](#3-a-base-de-dados)
4. [Decisões de escopo (fechadas)](#4-decisões-de-escopo-fechadas)
5. [Métricas de sucesso](#5-métricas-de-sucesso)
6. [Stack técnica e ambiente](#6-stack-técnica-e-ambiente)
7. [Princípios técnicos inegociáveis](#7-princípios-técnicos-inegociáveis)
8. [Roadmap por fases](#8-roadmap-por-fases)
9. [Abordagem de modelagem](#9-abordagem-de-modelagem)
10. [Embasamento (normativo, acadêmico, mercado)](#10-embasamento-normativo-acadêmico-mercado)
11. [Conexão com a ementa do MBA](#11-conexão-com-a-ementa-do-mba)
12. [Glossário](#12-glossário)

---

## 1. Visão geral

Construir um **AVM (Automated Valuation Model)** que estima o preço de apartamentos em São Paulo a partir de suas características, e expô-lo como uma **aplicação web** onde qualquer pessoa cadastra um imóvel e recebe uma **recomendação de valor**.

O projeto segue um arco deliberado: começa por um **baseline de regressão linear múltipla** (alinhado à tradição normativa brasileira de avaliação de imóveis), evolui para **técnicas modernas de machine learning**, **compara** as duas abordagens, e termina num **produto utilizável**.

A natureza do que estamos construindo é um **AVM-produto** (uma *sugestão* de preço, como a "Calculadora" da Loft ou o "Preço Certo" do QuintoAndar) — **não** um AVM-laudo (avaliação com validade jurídica, que exigiria conformidade plena com a NBR 14653 e assinatura de profissional CREA/CAU).

---

## 2. Objetivos

### Objetivo técnico do projeto

Entregar um modelo de precificação funcional e uma aplicação web pública que gera recomendações de preço para venda e para locação.

---

## 3. A base de dados

**Fonte escolhida:** dataset público do Kaggle — *São Paulo Real Estate, Sale / Rent* (~13.000 imóveis, coletados de classificados imobiliários, **abril de 2019**).

**Variáveis:** `Price`, `Condo`, `Size`, `Rooms`, `Toilets`, `Suites`, `Parking`, `Elevator`, `Furnished`, `Swimming Pool`, `New`, `District`, `Negotiation Type`, `Property Type`, `Latitude`, `Longitude`.

**Por que esta base (e não a alternativa de 2.499 imóveis):**
- ~5× maior → melhor generalização e viabiliza boosting sem overfitting imediato.
- **Sem feature derivada do target** (a base menor tinha `media_bairro`, com risco grave de *leakage*).
- Estrutura limpa: cada coluna é um atributo honesto do imóvel.
- Tem o eixo **venda vs. aluguel** (`Negotiation Type`) → dois modelos pelo preço de um.
- Tem **latitude/longitude** → habilita toda a frente espacial (o MUST do projeto).
- Espelha a estrutura da base real da empresa (preço de **anúncio** + atributos) → metodologia transferível.

**Limitações assumidas conscientemente:**
- É **preço de anúncio**, não de transação (anúncio é sistematicamente acima do valor de fechamento).
- É de **abril/2019** → não serve para precificar imóveis *hoje* (irrelevante para aprender e para o POC; o método é o mesmo). **Toda variável de enriquecimento deve respeitar essa janela temporal.**
- Escopo: **apenas apartamentos** (não generaliza para casas — declarar como limitação).

---

## 4. Decisões de escopo (fechadas)

| # | Decisão | Detalhe |
|---|---------|---------|
| 1 | **Venda E aluguel** | **Dois modelos separados**, não um modelo único com `Negotiation Type` como feature. Motivo: os preços implícitos dos atributos são estruturalmente diferentes entre comprar e alugar (escala e elasticidade distintas). Bônus: comparar os coeficientes vira insight de produto. |
| 2 | **Frente espacial = MUST** | **Spatial lag + autocorrelação espacial** são obrigatórios (a base real da empresa tem 95% de geolocalização preenchida — esta é a competência mais estratégica a desenvolver). Complementos confirmados: **distância à estação de metrô/trem mais próxima** e **bloco socioeconômico** (IDH, renda, Gini). |
| 3 | **Aplicação = Streamlit** | **Streamlit + Streamlit Community Cloud** (hospedagem pública gratuita). *Não* Vercel (Vercel hospeda front-end JS, não roda o modelo Python). Objetivo: link público para portfólio, qualquer pessoa testa. |
| 4 | **Ambiente local** | VSCode local + ambiente virtual Python (`venv`). Sem nuvem para processamento — a máquina dá conta deste volume. |
| 5 | **Ritmo livre** | Sem prazo. Postura: **caprichar, não cortar**. Profundidade na frente espacial, pausas para praticar quando algo não assentar. Horizonte estimado: 6–10 semanas a ~15h/semana. |
| 6 | **Claude Code: depois** | As fases iniciais (auditoria, EDA, baseline) são feitas **no chat**, devagar, para maximizar aprendizado. Migração para Claude Code quando houver fluência e o gargalo virar produtividade (provavelmente na fase de boosting ou da app). |

---

## 5. Métricas de sucesso

| Métrica | Papel | Por quê |
|---------|-------|---------|
| **MAPE** (erro percentual absoluto médio) | **Primária** | Trata o erro proporcionalmente — R$ 50 mil é grave num imóvel de R$ 300 mil e trivial num de R$ 3 mi. É a métrica que o mercado de AVM usa. |
| **MAE** (erro absoluto médio) | Apoio | Erro em reais, interpretável diretamente. |
| **R²** | Apoio | Variância explicada; é a referência da tradição normativa brasileira. |

**Âncoras de expectativa (para não frustrar nem iludir):**
- R² acima de **0,70** é a referência de modelo decente na tradição NBR.
- MAPE realista para o nosso contexto (anúncio, ~13k linhas, 2019): faixa de **15–25%** no baseline, melhorando com boosting.
- AVMs de ponta (Zillow, Loft) atingem MAPE de 5–10%, mas com **milhões** de transações reais. **Não comparar nosso resultado com o deles.**

**Critério de comparação entre modelos:** o modelo vencedor é o de menor MAPE no conjunto de **teste** (nunca no treino), com R² e MAE como desempate/contexto.

---

## 6. Stack técnica e ambiente

**Ambiente:** VSCode local + `venv` (ambiente virtual isolado por projeto).

**Bibliotecas por fase (instalar quando a fase pedir, não tudo de uma vez):**

| Fase | Bibliotecas |
|------|-------------|
| Base / EDA | `pandas`, `numpy`, `matplotlib`, `seaborn`, `jupyter` |
| Modelagem clássica | `scikit-learn`, `statsmodels` (para diagnósticos da regressão) |
| Boosting | `xgboost` (e/ou `lightgbm`) |
| Espacial | a definir na Fase 3 (ex.: `geopandas`, `libpysal`/`esda`, `scikit-learn` para k-NN) |
| Interpretabilidade | `shap` |
| Aplicação | `streamlit` |

**Estrutura de pastas:**
```
avm-sao-paulo/
├── venv/                  # ambiente virtual (não versionar)
├── data/                  # dados brutos
│   └── sao-paulo-properties-april-2019.csv
├── notebooks/             # análises (.ipynb)
├── src/                   # código reutilizável (.py) — surge mais adiante
└── app/                   # aplicação Streamlit — fase final
```

---

## 7. Princípios técnicos inegociáveis

Estes são os pontos onde projetos de AVM costumam morrer. Revisitar sempre.

- **Anti-leakage (o mais crítico):** o **train/test split é feito CEDO**, antes de qualquer transformação que aprenda com os dados. Toda estatística — média para preencher missing, parâmetros de encoding, e **o próprio spatial lag** — é calculada **só no treino** e aplicada no teste. Calcular qualquer média usando o dataset inteiro antes de dividir = vazamento.
- **Endogeneidade:** `Condo` (condomínio) é consequência do mesmo padrão construtivo que determina o preço, não causa dele. Ajuda a predição, mas contamina interpretação causal e infla a sensação de acurácia. Decisão consciente de incluir/excluir, com olhos abertos. (Mesmo raciocínio valeria para IPTU na base real.)
- **Multicolinearidade:** atributos de tamanho (`Size`, `Rooms`, `Toilets`, `Suites`) são correlacionados; o bloco socioeconômico (IDH, renda, Gini) é quase um fator latente único. Para a regressão, checar **VIF**; considerar reduzir o bloco socioeconômico (escolher um, ou PCA).
- **Contemporaneidade temporal:** todo enriquecimento deve refletir a realidade de **2019**. Censo a usar = **2010** (era o vigente em 2019). Estações de metrô/trem = só as que **já existiam em jan/2019** (não dar ao modelo uma amenidade que o comprador da época não tinha).
- **Interpretabilidade:** se o modelo black-box (XGBoost) vencer, abrir a caixa com **SHAP** — a recomendação na app precisa ser explicável ("R$ 600 mil, puxado por localização e metragem"), não um número solto. Também honra a exigência de interpretabilidade da tradição normativa brasileira.

---

## 8. Roadmap por fases

### Fase 0 — Fundação
- [ ] Criar pasta do projeto e abrir no VSCode
- [ ] Criar e ativar o ambiente virtual (`venv`)
- [ ] Instalar bibliotecas da fase (pandas, numpy, matplotlib, seaborn, jupyter)
- [ ] Colocar o CSV em `data/`
- [ ] Criar o primeiro notebook e conectar o kernel ao `venv`
- [ ] Carregar a base e rodar primeira inspeção (`shape`, `head()`, `info()`)
- [x] Definir as métricas de sucesso (MAPE / MAE / R²)

### Fase 1 — Auditoria e EDA
- [ ] Distribuição do `Price` (provável cauda à direita → log)
- [ ] Investigar `Condo` (endogeneidade; quão colado está no preço?)
- [ ] Identificar e tratar outliers e valores faltantes (missing)
- [ ] Validar `Latitude`/`Longitude` (coordenadas plausíveis para SP?)
- [ ] Esclarecer colunas ambíguas (ex.: o que é cada categoria de `Property Type`)
- [ ] Separar os dados em **venda** e **aluguel**
- [ ] Documentar achados da auditoria

### Fase 2 — Split e tratamento
- [ ] **Train/test split** (feito antes de qualquer transformação que aprenda dos dados)
- [ ] Transformações log (preço e, provavelmente, área)
- [ ] Encoding do `District` (decidir estratégia: one-hot, target encoding, ou via geografia)
- [ ] Estratégia de validação cruzada definida

### Fase 3 — Enriquecimento espacial e socioeconômico (frente nobre)
- [ ] Distância de cada imóvel à estação de metrô/trem mais próxima **existente em jan/2019**
- [ ] Cruzar com dados do **Censo 2010** (IDH, renda, Gini por região)
- [ ] Construir a **matriz de pesos espaciais** (W) — contiguidade ou k-NN
- [ ] Calcular o **spatial lag** (preço médio dos vizinhos, só no treino, excluindo a própria observação)
- [ ] Análise de **autocorrelação espacial** (Moran's I / LISA)

### Fase 4 — Modelagem (×2: venda e aluguel)
- [ ] Baseline: **regressão linear múltipla** com diagnósticos (normalidade dos resíduos, homocedasticidade, VIF) — referência NBR
- [ ] **Random Forest** (bagging)
- [ ] **XGBoost** (boosting — candidato a campeão, segundo a literatura BR)

### Fase 5 — Comparação e interpretabilidade
- [ ] Tabela comparativa de MAPE / MAE / R² (no teste), por modelo e por tipo de negociação
- [ ] **SHAP** no modelo campeão (importância e direção das features)
- [ ] Narrativa: o que cada modelo ganha/perde; trade-off interpretabilidade × acurácia
- [ ] Comparação de coeficientes venda × aluguel (insight de produto)

### Fase 6 — Aplicação Streamlit
- [ ] Interface de cadastro de imóvel (campos dos atributos)
- [ ] Integração do modelo treinado → recomendação de preço
- [ ] Exibir a recomendação com explicação (o "porquê" do valor)
- [ ] Deploy no Streamlit Community Cloud (link público)
- [ ] Documentação final + seção de privacidade/LGPD (ponte para a base real)

---

## 9. Abordagem de modelagem

**Escada de três níveis, de interpretável a preciso:**

1. **Regressão linear múltipla (baseline normativo).** Método que a NBR 14653 consagra. Trabalha com **log do preço** (relação hedônica é multiplicativa; resíduos do preço bruto são feios). Box-Cox e diagnósticos formais (normalidade, homocedasticidade, multicolinearidade) fazem parte. É o terreno econométrico forte do autor, agora em Python.

2. **Random Forest (bagging).** Robusto, pouca configuração. Segundo degrau; tolera melhor a colinearidade.

3. **XGBoost (boosting).** Estado da arte para dados tabulares. A literatura brasileira inteira aponta boosting como vencedor em acurácia. Candidato a campeão.

*Redes neurais ficam fora deste projeto* — para dados tabulares deste tamanho raramente batem boosting e adicionam complexidade sem retorno.

**A comparação entre os três níveis É a narrativa central do projeto:** quando vale o modelo auditável e interpretável (regressão) vs. quando vale o modelo preciso e black-box (boosting). Esse contraste é o argumento de Data Product Manager.

---

## 10. Embasamento (normativo, acadêmico, mercado)

### Normativo
- **ABNT NBR 14653** (Avaliação de bens) — Parte 1 (procedimentos gerais) e Parte 2 (imóveis urbanos). Consagra a **regressão linear múltipla** com requisitos estatísticos formais (Anexo A): significância dos coeficientes, R², análise de resíduos. Formaliza **graus de fundamentação e precisão**. Permite técnicas modernas (redes neurais, regressão espacial) "desde que devidamente justificadas".
- **IBAPE** (Instituto Brasileiro de Avaliações e Perícias de Engenharia) — material técnico de inferência estatística aplicada.

### Acadêmico (referências brasileiras de ponta)
- **SEMEAD/USP** — comparação de 6 modelos hedônicos com dados de ITBI (XGBoost e SVM linear venceram).
- **Revista do Depto. de Geografia/USP** — mass appraisal em Florianópolis com Random Forest e Gradient Boosting.
- **UNIFESP** — preço de aluguel em SP (XGBoost melhor; tamanho, localização e lazer dominam).
- **UNESP (São José do Rio Preto)** — distância a POIs + clusters geográficos; árvores superam regressão.
- **EMBRAESP / CEM-USP** — base georreferenciada clássica de SP, rica em features e setor censitário.

**Padrão consolidado:** (1) área e localização dominam; (2) boosting bate regressão em acurácia; (3) tratamento espacial explícito agrega; (4) lacuna pouco explorada = features de texto/foto do anúncio.

### Mercado (benchmark de produto)
- **Loft** — "Calculadora Loft": AVM com ML sobre ~10 milhões de anúncios; modelo de casas com 100+ variáveis.
- **QuintoAndar** — "Calculadora" e "Preço Certo": +16% de demanda de visitas em teste; 85% dos usuários aproveitam a sugestão.
- Referência de impacto (KPMG, 2024): AVMs reduzem custo de avaliação em até 70% e tempo em 90%+.

---

## 11. Conexão com a ementa do MBA

| Fase / tema do projeto | Disciplina da ementa |
|------------------------|----------------------|
| Manipulação de dados, Jupyter, GitHub | Data Wrangling |
| Regressão baseline + diagnósticos | Supervised ML: Regressão Simples e Múltipla |
| Random Forest, XGBoost, validação | Árvores, Redes e Ensemble Models |
| Spatial lag, autocorrelação, shapefiles | Análise Estatística Espacial |
| PCA do bloco socioeconômico (se usado) | Unsupervised ML: Análise Fatorial e PCA |
| Features de texto do anúncio (futuro) | Text Mining, Sentiment Analysis e NLP |
| Aplicação web / encapsulamento do modelo | Big Data e Deployment de Modelos |
| Tratamento de dado pessoal (base real) | Legislação no Ambiente Digital (LGPD); Analytics e Gestão de Riscos |

---

## 12. Glossário

- **AVM (Automated Valuation Model):** modelo que estima o valor de um imóvel automaticamente a partir de suas características.
- **AVM-produto vs. AVM-laudo:** *produto* = sugestão de preço (sem validade jurídica, foco em acurácia); *laudo* = avaliação oficial conforme NBR, auditável, assinada por engenheiro/arquiteto.
- **Hedonic pricing (preço hedônico):** abordagem econômica em que o preço de um bem é a soma dos preços implícitos de seus atributos.
- **MAPE:** erro percentual absoluto médio — média de quanto o modelo erra, em %, em relação ao valor real.
- **Leakage (vazamento):** quando informação do conjunto de teste (ou do target) contamina o treino, inflando o desempenho de forma enganosa.
- **Endogeneidade:** quando uma variável explicativa é, na verdade, consequência (e não causa) do que se quer prever.
- **Multicolinearidade:** quando variáveis explicativas são fortemente correlacionadas entre si, prejudicando a interpretação dos coeficientes. Medida pelo **VIF** (Variance Inflation Factor).
- **Spatial lag (defasagem espacial):** feature que traz, para cada imóvel, o preço médio (ou mediana) dos imóveis vizinhos — capturando determinantes de localização não observados. Análogo espacial do lag temporal de séries.
- **Matriz de pesos espaciais (W):** define quem é "vizinho" de quem (por contiguidade ou por distância/k-NN). É o operador que torna o spatial lag possível.
- **Autocorrelação espacial / Moran's I / LISA:** medidas de quanto valores próximos no espaço se parecem entre si.
- **SHAP:** técnica que explica a contribuição de cada feature para uma previsão individual de um modelo (abre a "caixa-preta").
- **Bagging / Boosting:** famílias de ensemble. *Bagging* (ex.: Random Forest) treina árvores em paralelo e faz média; *boosting* (ex.: XGBoost) treina árvores em sequência, cada uma corrigindo o erro da anterior.