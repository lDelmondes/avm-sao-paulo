# Projeto AVM — Precificação de Imóveis em São Paulo

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
11. [Áreas técnicas cobertas pelo projeto](#11-áreas-técnicas-cobertas-pelo-projeto)
12. [Glossário](#12-glossário)

---

## 1. Visão geral

Construir um **AVM (Automated Valuation Model)** que estima o preço de apartamentos em São Paulo a partir de suas características, e expô-lo como uma **aplicação web** onde qualquer pessoa cadastra um imóvel e recebe uma **recomendação de valor**.

O projeto segue um arco deliberado: começa por um **baseline de regressão linear múltipla** (alinhado à tradição normativa brasileira de avaliação de imóveis), evolui para **técnicas modernas de machine learning**, **compara** as duas abordagens, e termina num **produto utilizável**.

A natureza do que estamos construindo é um **AVM-produto** (uma *sugestão* de preço, como a "Calculadora" da Loft ou o "Preço Certo" do QuintoAndar) — **não** um AVM-laudo (avaliação com validade jurídica, que exigiria conformidade plena com a NBR 14653 e assinatura de profissional CREA/CAU).

---

## 2. Objetivos

**Objetivo principal.** Construir um modelo de avaliação automatizada (AVM) que estime o preço de anúncio de apartamentos em São Paulo a partir de suas características — um modelo para venda e outro para locação —, tendo o **MAPE** como métrica primária de sucesso.

**Objetivos específicos.**
- Estabelecer um baseline de **regressão linear múltipla** e compará-lo a modelos de **machine learning** (Random Forest, XGBoost), quantificando o trade-off entre interpretabilidade e acurácia.
- Incorporar a **dimensão espacial** (distância a estações de transporte e *spatial lag*) como componente central do modelo.
- Entregar uma **aplicação web** que recebe os atributos de um imóvel e retorna uma recomendação de valor explicável.

**Critério de conclusão.** Os dois modelos treinados e avaliados no conjunto de teste, a comparação entre metodologias documentada, e a aplicação publicada e funcional.

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

**Limitações assumidas conscientemente:**
- É **preço de anúncio**, não de transação (anúncio é sistematicamente acima do valor de fechamento).
- É de **abril/2019** → não serve para precificar imóveis *hoje* (irrelevante para o objetivo do projeto; o método é o mesmo). **Toda variável de enriquecimento deve respeitar essa janela temporal.**
- Escopo: **apenas apartamentos** (confirmado via `Property Type` — 100% apartamentos; não generaliza para casas).

**Estado após a auditoria:**
- Partiu de **13.640** registros brutos → **12.759** após tratamento.
- **17** coordenadas invertidas (lat/lon trocadas) corrigidas por troca de campo.
- **881** registros com coordenada (0, 0) removidos (geolocalização ausente, distribuída de forma difusa por 83 bairros → remoção sem viés).
- **1.887** valores de `Condo = 0` convertidos em `NaN` (missing disfarçado de zero). Posteriormente, `Condo` foi **descartado do modelo** (missing não-aleatório + multicolinearidade com as comodidades).
- Base tratada salva em `data/processed/imoveis_tratados.parquet`; o `df` original permanece intocado (reprodutibilidade).

---

## 4. Decisões de escopo (fechadas)

| # | Decisão | Detalhe |
|---|---------|---------|
| 1 | **Venda E aluguel** | **Dois modelos separados**, não um modelo único com `Negotiation Type` como feature. Motivo: os preços implícitos dos atributos são estruturalmente diferentes entre comprar e alugar (escala e elasticidade distintas). Bônus: comparar como cada mercado precifica os atributos (via SHAP) vira insight de produto. |
| 2 | **Frente espacial = MUST** | **Spatial lag + autocorrelação espacial** são componentes obrigatórios do modelo: a localização é o principal determinante de preço imobiliário, e o *spatial lag* captura efeitos locais não observados pelas demais variáveis. Complementos: **distância à estação de metrô/trem mais próxima** e **bloco socioeconômico do entorno** (IDH, renda, Gini). |
| 3 | **Aplicação = Streamlit** | **Streamlit + Streamlit Community Cloud** (hospedagem pública gratuita). *Não* Vercel (Vercel hospeda front-end JS, não roda o modelo Python). Objetivo: link público, qualquer pessoa testa. |
| 4 | **Ambiente local** | VSCode local + ambiente virtual Python (`venv`). Sem nuvem para processamento — a máquina dá conta deste volume. |
| 5 | **Profundidade, não atalhos** | Diretriz de escopo: completar bem cada frente — em especial a espacial — em vez de cortar caminho. Sem reduções de escopo por pressa. |

---

## 5. Métricas de sucesso

| Métrica | Papel | Por quê |
|---------|-------|---------|
| **MAPE** (erro percentual absoluto médio) | **Primária** | Trata o erro proporcionalmente — R$ 50 mil é grave num imóvel de R$ 300 mil e trivial num de R$ 3 mi. É a métrica que o mercado de AVM usa. |
| **MAE** (erro absoluto médio) | Apoio | Erro em reais, interpretável diretamente. |
| **R²** | Apoio | Variância explicada; é a referência da tradição normativa brasileira. |

**Âncoras de expectativa (para não frustrar nem iludir):**
- R² acima de **0,70** é a referência de modelo decente na tradição NBR.
- MAPE realista para este contexto (anúncio, ~13k linhas, 2019): faixa de **15–25%** no baseline, melhorando com boosting.
- AVMs de ponta (Zillow, Loft) atingem MAPE de 5–10%, mas com **milhões** de transações reais. Não comparar o resultado deste projeto com o deles.

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
├── venv/                       # ambiente virtual (não versionar)
├── data/                       # dados brutos (não versionar)
│   └── processed/              # artefatos tratados, ponte entre notebooks
├── app/                        # aplicação Streamlit (Fase 6)
│   ├── app.py                  # UI
│   ├── predictor.py            # inferência fiel ao modelo
│   ├── geocode.py              # CEP → coordenada (ViaCEP + Nominatim)
│   └── data/                   # pacote de dados de produção (VERSIONADO)
├── models/                     # modelos campeões (.joblib) — VERSIONAR
├── docs/                       # SPEC_CONTRATO, one-pager, PRD de jornadas
├── outputs/                    # gráficos (SHAP, diagnósticos) e screenshot
├── .streamlit/config.toml      # tema do app
├── 01_auditoria.ipynb          # Fase 1
├── 02_preparacao.ipynb         # Fases 2 e 3 (registro histórico)
├── 03_modelagem.ipynb          # Fases 4 e 5
├── PROJETO.md                  # este documento
├── requirements.txt
└── README.md
```

---

## 7. Princípios técnicos inegociáveis

Estes são os pontos onde projetos de AVM costumam morrer. Revisitar sempre.

- **Anti-leakage (o mais crítico):** o **train/test split é feito CEDO**, antes de qualquer transformação que aprenda com os dados. Toda estatística — média para preencher missing, parâmetros de encoding, e **o próprio spatial lag** — é calculada **só no treino** e aplicada no teste. Calcular qualquer média usando o dataset inteiro antes de dividir = vazamento.
    - **Lógica para validação cruzada:** features que aprendem dos dados — incluindo o **spatial lag** — devem ser **reconstruídas dentro de cada fold**. Calcular o lag no treino inteiro e *depois* fatiar em folds vaza informação entre folds: um ponto do fold-validação acabaria usando como vizinhos pontos do seu próprio fold, indisponíveis em produção.  
    Casos com split aninhado (ex.: early stopping do XGBoost, que exige sub-treino/sub-validação dentro de cada fold) reconstroem o lag em cada nível. Target encoding segue a mesma regra: média do bairro out-of-fold, reconstruída por fold.
- **Endogeneidade:** `Condo` (condomínio) é consequência do mesmo padrão construtivo que determina o preço, não causa dele. Ajuda a predição, mas contamina interpretação causal e infla a sensação de acurácia. Decisão consciente de incluir/excluir, com olhos abertos.
- **Multicolinearidade:** atributos de tamanho (`Size`, `Rooms`, `Toilets`, `Suites`) são correlacionados; o bloco socioeconômico (IDH, renda, Gini) é quase um fator latente único. Para a regressão, checar **VIF**; considerar reduzir o bloco socioeconômico (escolher um, ou PCA).
    > *Resultado: o bloco socioeconômico reduziu-se na prática a uma feature (`renda_area`); não houve fator latente múltiplo a reduzir por PCA. A renda mostrou-se redundante com o `District`/`spatial_lag` na regressão, mas útil nas árvores.*
- **Contemporaneidade temporal:** todo enriquecimento deve refletir a realidade de **2019**. Censo a usar = **2010** (era o vigente em 2019). Estações de metrô/trem = só as que **já existiam em jan/2019** (não dar ao modelo uma amenidade que o comprador da época não tinha).
- **Interpretabilidade:** se o modelo black-box (XGBoost) vencer, abrir a caixa com **SHAP** — a recomendação na app precisa ser explicável ("R$ 600 mil, puxado por localização e metragem"), não um número solto. Também honra a exigência de interpretabilidade da tradição normativa brasileira.

---

## 8. Roadmap por fases

**Fase ≠ notebook.** *Fase* é uma divisão conceitual do trabalho (as etapas lógicas abaixo); *notebook* é a divisão física dos arquivos. Não há correspondência 1:1 — um notebook agrupa as fases que formam um fluxo contínuo de ponta a ponta e produzem um artefato. O mapeamento:

| Notebook / arquivo | Fases | Artefato produzido |
|--------------------|-------|--------------------|
| `01_auditoria.ipynb` | Fase 1 (auditoria + tratamento) | `imoveis_tratados.parquet` |
| `02_preparacao.ipynb` | Fase 2 (split, log, encoding) + Fase 3 Bloco 1 (construção e auditoria da feature de distância a estações; limpeza de coordenadas) | `venda_espacial.parquet`, `aluguel_espacial.parquet` |
| `03_modelagem.ipynb` | Remodelagem sobre a base limpa: preparação, **baseline definitivo**, demais features espaciais (Blocos 2 e 3), RF, XGBoost, comparação, SHAP (Fases 4 e 5) | modelos treinados e avaliados |
| `app/` (`.py`) | Fase 6 | aplicação Streamlit publicada |

Nota sobre a evolução real do projeto: a Fase 3 (espacial) revelou coordenadas corrompidas e imóveis fora de escopo (Jundiaí), o que obrigou a re-limpar a base e, por consequência, a refazer a preparação e o baseline sobre a base limpa. Por isso o `02` acumulou a auditoria espacial (Bloco 1) e passou a ser o **registro histórico** dessa construção (não é mais editado); a remodelagem a partir da base limpa — preparação, baseline definitivo e os blocos espaciais seguintes — vive no `03`. O baseline preliminar (16,5% / 22,4%) foi calculado antes dessa limpeza e **já foi recalculado** no `03` como marco-zero definitivo (17,1% / 22,3% na base limpa).

**Progresso atual:** Fases 0–4 concluídas. Baseline oficial na base limpa (venda 17,1% / aluguel 22,3%). Fase 3 (espacial): k=3, spatial lag confirmado como o ganho espacial central; renda e distância redundantes na regressão linear, reservadas para teste nas árvores. Fase 4 (modelagem) **concluída** — RF e XGBoost treinados, selecionados por CV (lag reconstruído por fold) e avaliados no test final. Renda e distância, inúteis na regressão, **agregam nas árvores** (não-linearidade) e entram na matriz de árvore. **Resultado central: XGBoost campeão nos dois mercados — venda 14,04% / aluguel 20,29% de MAPE.** H1 confirmada (ML supera a regressão). **Fase 5 (interpretabilidade) concluída** — SHAP no campeão, diagnósticos da regressão, comparação consolidada. **Fase 6 (aplicação) concluída** — app Streamlit no ar (https://avm-sao-paulo.streamlit.app/), com inferência fiel ao notebook (diff ~1e-11), geocodificação por CEP e explicação dos fatores. **Projeto completo: do dado cru ao produto público.**

### Fase 0 — Fundação ✅
- [x] Criar pasta do projeto e abrir no VSCode
- [x] Criar e ativar o ambiente virtual (`venv`)
- [x] Instalar bibliotecas da fase (pandas, numpy, matplotlib, seaborn, jupyter)
- [x] Colocar o CSV em `data/`
- [x] Criar o primeiro notebook e conectar o kernel ao `venv`
- [x] Carregar a base e rodar primeira inspeção (`shape`, `head()`, `info()`)
- [x] Definir as métricas de sucesso (MAPE / MAE / R²)
- [x] Repositório criado e publicado no GitHub

### Fase 1 — Auditoria e EDA ✅
- [x] Distribuição do `Price` (cauda à direita confirmada → log na modelagem)
- [x] Investigar `Condo` (`Condo = 0` identificado como missing não-aleatório → descartado)
- [x] Identificar e tratar outliers e valores faltantes (missing)
- [x] Validar `Latitude`/`Longitude` (881 zerados removidos, 17 invertidos corrigidos)
- [x] Esclarecer colunas ambíguas (`Property Type` = 100% apartamentos)
- [x] Separar os dados em **venda** (6.014) e **aluguel** (6.745)
- [x] Documentar achados da auditoria (padrão de 3 tempos: abre → código → fecha)

### Fase 2 — Split e tratamento ✅
- [x] **Train/test split** (80/20, semente fixa, feito antes de qualquer transformação)
- [x] Transformações log (`Price` e `Size`, via `log1p`)
- [x] Encoding do `District` (one-hot via `OneHotEncoder`, `fit` só no treino, `handle_unknown='ignore'`)
- [x] Estratégia de validação cruzada definida (por ora, holdout 80/20 simples; CV a definir para a fase de ML)

### Fase 3 — Enriquecimento espacial e socioeconômico (frente nobre)
- [x] Distância de cada imóvel à estação de metrô/trem mais próxima **existente em jan/2019** (Bloco 1 — construída, auditada; ganho nulo no baseline, feature reservada para os modelos de árvore)
- [x] Cruzar com dados do **Censo 2010** — renda média domiciliar por área de ponderação (Bloco 2 — join espacial point-in-polygon; redundante com o spatial lag na regressão (~0,1 p.p.), mas **agrega nas árvores** — reservada para a fase de ML, em log)
- [x] Construir a **matriz de pesos espaciais** (W) — contiguidade ou k-NN (Bloco 3)
- [x] Calcular o **spatial lag** (preço médio dos vizinhos, só no treino, excluindo a própria observação)
- [x] Análise de **autocorrelação espacial** (Moran's I / LISA)

### Fase 4 — Modelagem (×2: venda e aluguel)
- [x] Baseline: **regressão linear múltipla** com diagnósticos (VIF, normalidade dos resíduos, homocedasticidade) — referência NBR
- [x] **Avaliar a configuração espacial final** da regressão (físicas + District + `spatial_lag` k=3; sem renda, sem distância — ambas redundantes na forma linear) no **test set 20% intocado** → venda 15,98% / aluguel 21,67%. Régua que RF e XGBoost superaram.
- [x] **Random Forest** (bagging)
- [x] **XGBoost** (boosting — candidato a campeão, segundo a literatura BR)

> **Test final — comparação dos três modelos (test 20% intocado):**
>
> | Modelo | Venda MAPE | Venda R² | Aluguel MAPE | Aluguel R² |
> |--------|-----------|----------|--------------|------------|
> | Regressão linear | 15,98% | 0,909 | 21,67% | 0,741 |
> | Random Forest | 14,38% | 0,897 | 19,91% | 0,699 |
> | **XGBoost** | **14,04%** | **0,928** | 20,29% | 0,740 |
>
> **H1 confirmada:** ML supera a regressão nos dois mercados. **H3 confirmada:** comportamento difere por mercado (na venda o XGB domina os três indicadores; no aluguel RF e XGB empatam tecnicamente — RF 0,38 p.p. à frente no test, indistinguíveis na CV). **Campeão: XGBoost nos dois** — na venda por mérito claro; no aluguel por consistência de produto (empate técnico com RF, mesmo algoritmo simplifica deploy e SHAP).
>
> **Matriz diferenciada por modelo:** regressão = físicas + District + spatial_lag (k=3). Árvores = físicas + District + spatial_lag + distancia_estacao + renda_area (log), 107 features (venda) / 105 (aluguel).
>
> Achados de método (Fase 4): `distancia_estacao` E `renda_area`, ambas nulas/redundantes na regressão, **agregam nas árvores** (que captam a não-linearidade da relação localização↔preço) → ambas entram na matriz de árvore. Renda: ganho de ~0,8 p.p. (venda) / ~0,4 p.p. (aluguel) nas árvores, contra ~0,1 p.p. na regressão. Target encoding do District testado contra one-hot (cada um no seu ótimo) — one-hot venceu nos 4 casos, mantido.

### Fase 5 — Comparação e interpretabilidade
- [x] Tabela comparativa de MAPE / MAE / R² (no teste), por modelo e por tipo de negociação
- [x] **SHAP** no modelo campeão (importância e direção das features)
- [x] Narrativa: o que cada modelo ganha/perde; trade-off interpretabilidade × acurácia
- [x] ~~Comparação de coeficientes venda × aluguel~~ → **substituída por SHAP.** Coeficientes da regressão são inviáveis de interpretar (pressupostos violados + multicolinearidade VIF 5-7 nos atributos de tamanho); o insight de produto vem do SHAP

> **Achados de interpretabilidade (Fase 5):**
>
> **O que dirige o preço (SHAP):** tamanho (`log_Size`) domina, seguido de `spatial_lag`
> (2º — valida a frente espacial: o lag é o 2º motor de preço e tem VIF ~2, informação
> não-redundante) e `renda_area` (3º). District agregado, distância e atributos pesam menos.
>
> **Achados de H3 (venda × aluguel divergem em O QUE dirige o preço):**
> - **Distância à estação inverte de sinal:** na venda, longe da estação → preço SOBE
>   (bairros nobres de SP são orientados ao carro); no aluguel, longe → preço CAI (o
>   locatário depende mais de transporte público). Mesma feature, direção oposta.
> - **Furnished e New importam só no aluguel** (mobília +18% vs +4%; novo +25% vs −2%):
>   o locatário paga por pronto-para-morar, o comprador por patrimônio.
> - **Tamanho domina menos no aluguel** (elasticidade 0,45 vs 0,74 na venda).
> - **Renda e spatial_lag pesam igual nos dois** — localização e entorno são universais.
>
> A inversão da distância valida retroativamente a decisão de escopo nº 1 (modelos
> separados): um modelo único mediaria os dois efeitos opostos e perderia ambos.
>
> **Diagnósticos da regressão (6.1):** normalidade e homocedasticidade rejeitadas
> formalmente (p≈0), mas violação moderada à inspeção visual (resíduos quase normais no
> centro, problema nas caudas; curtose 9 na venda / 5 no aluguel). Multicolinearidade
> moderada em Toilets/Suites (VIF 5-7). Conclusão: inviável interpretar coeficientes
> individuais → interpretabilidade via SHAP; a violação dos pressupostos justifica o ML.
>
> **Artefatos visuais** (em `outputs/`): `shap_beeswarm_{venda,aluguel}.png`,
> `shap_bar_{venda,aluguel}.png`, `diagnostico_residuos_{venda,aluguel}.png`

### Fase 6 — Aplicação Streamlit
- [x] Interface de cadastro de imóvel (campos dos atributos)
- [x] Integração do modelo treinado → recomendação de preço
- [x] Exibir a recomendação com explicação (o "porquê" do valor)
- [x] Deploy no Streamlit Community Cloud (link público)
- [x] Documentação final

---

## 9. Abordagem de modelagem

**Escada de três níveis, de interpretável a preciso:**

1. **Regressão linear múltipla (baseline normativo).** Método que a NBR 14653 consagra. Trabalha com **log do preço** (relação hedônica é multiplicativa; resíduos do preço bruto são feios). Box-Cox e diagnósticos formais (normalidade, homocedasticidade, multicolinearidade) fazem parte.   
    - Os diagnósticos da versão final (Seção 6.1) rejeitam os pressupostos — violação moderada, mas suficiente para inviabilizar inferência sobre coeficientes

2. **Random Forest (bagging).** Robusto, pouca configuração. Segundo degrau; tolera melhor a colinearidade.

3. **XGBoost (boosting).** Estado da arte para dados tabulares. A literatura brasileira inteira aponta boosting como vencedor em acurácia. Candidato a campeão.

*Redes neurais ficam fora deste projeto* — para dados tabulares deste tamanho raramente batem boosting e adicionam complexidade sem retorno.

**A comparação entre os três níveis é a questão central do projeto:** quando vale o modelo auditável e interpretável (regressão) vs. quando vale o modelo preciso e black-box (boosting). Esse contraste — interpretabilidade contra acurácia — é o que o projeto investiga.

---

## 10. Embasamento (normativo, acadêmico, mercado)

### Normativo
- **ABNT NBR 14653** (Avaliação de bens) — Parte 1 (procedimentos gerais) e Parte 2 (imóveis urbanos). Consagra a **regressão linear múltipla** com requisitos estatísticos formais (Anexo A): significância dos coeficientes, R², análise de resíduos. Formaliza **graus de fundamentação e precisão**. Permite técnicas modernas (redes neurais, regressão espacial) "desde que devidamente justificadas".
- **IBAPE** (Instituto Brasileiro de Avaliações e Perícias de Engenharia) — material técnico de inferência estatística aplicada.

### Acadêmico (referências brasileiras de ponta)
- **SEMEAD / UFPR** — comparação de 6 modelos hedônicos com dados de ITBI de **Belo Horizonte** (transação real); XGBoost e SVM linear venceram. Variáveis: bairro, ano de construção, padrão de acabamento, zona de uso (poucas, pois o ITBI é pobre em atributos).
- **CEFET-RJ / Vetor (Nova Friburgo)** — web scraping do VivaReal (anúncio), regressão linear; área é a variável dominante, ~25% de MAPE. Usou Random Forest para seleção de variáveis e bairro como dummies. **O mais próximo da nossa abordagem.**
- **Medium / Ulisses (EMBRAESP, SP)** — base CEM/USP, 85 colunas; removeu nº de banheiros por VIF alto; variáveis do censo não ajudaram (sinal amarelo para o nosso bloco socioeconômico).
- **UNIFESP** — preço de aluguel em SP (XGBoost melhor; tamanho, localização e lazer dominam).
- **EMBRAESP / CEM-USP** — base georreferenciada clássica de SP, rica em features e setor censitário.

**Padrão consolidado:** (1) área e localização dominam; (2) boosting bate regressão em acurácia; (3) tratamento espacial explícito agrega; (4) restringir a apartamentos melhora o modelo; (5) lacuna pouco explorada = features espaciais finas (spatial lag) e de texto/foto do anúncio.

### Mercado (benchmark de produto)
- **Loft** — "Calculadora Loft": AVM com ML sobre ~10 milhões de anúncios; modelo de casas com 100+ variáveis.
- **QuintoAndar** — "Calculadora" e "Preço Certo": +16% de demanda de visitas em teste; 85% dos usuários aproveitam a sugestão.
- Referência de impacto (KPMG, 2024): AVMs reduzem custo de avaliação em até 70% e tempo em 90%+.

---

## 11. Áreas técnicas cobertas pelo projeto

| Fase / tema do projeto | Área técnica |
|------------------------|--------------|
| Manipulação e limpeza de dados | Data wrangling |
| Regressão baseline + diagnósticos | Regressão linear múltipla / inferência estatística |
| Random Forest, XGBoost, validação | Ensembles (bagging / boosting) |
| Spatial lag, autocorrelação, distância a POIs | Estatística espacial |
| Encapsulamento e publicação do modelo | Deployment de modelos |
| Geocodificação e derivação de features em produção | Engenharia de features / deploy |
| Aplicação web interativa (Streamlit) | Produto / deployment |

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