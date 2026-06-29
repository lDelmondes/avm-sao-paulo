# SPEC DE CONTRATO MODELO ↔ APP

**Fonte da verdade para a Fase 6 (app Streamlit).**
Garante fidelidade entre o pipeline de treino (`03_modelagem.ipynb`, que parte de
`02_preparacao.ipynb`) e a derivação de features em produção (o app).

> **Regra de ouro:** qualquer divergência entre como uma feature é construída no treino
> e como o app a constrói faz o modelo prever lixo *silenciosamente*. Toda derivação
> abaixo foi **verificada empiricamente** contra o valor que o modelo realmente recebeu
> no notebook (ver §6). Diferença observada: **0,0** em todos os casos testados.

Modelos: `XGBRegressor` (sklearn API), salvos em `models/`:
- `modelo_campeao_venda.joblib` — **107 features**
- `modelo_campeao_aluguel.joblib` — **105 features**

Ambos predizem em **log** (`log1p` do `Price`); o app reverte com **`expm1`** (ver §3).

---

## 1. Lista exata e ordenada das features

A ordem foi extraída de `modelo.feature_names_in_` (a ordem **importa** — XGBoost casa
colunas por posição). Estrutura idêntica nos dois modelos:

| Bloco | Posições (venda) | Posições (aluguel) | Conteúdo |
|---|---|---|---|
| **Atributos físicos** | 0–8 (9 features) | 0–8 (9 features) | `Rooms`, `Toilets`, `Suites`, `Parking`, `Elevator`, `Furnished`, `Swimming Pool`, `New`, `log_Size` |
| **Dummies District** | 9–103 (**95** dummies) | 9–101 (**93** dummies) | `District_<Bairro>/São Paulo`, em ordem `np.sort` (ver nota) |
| **Espaciais** | 104, 105, 106 | 102, 103, 104 | `spatial_lag`, `distancia_estacao`, `renda_area` (nesta ordem) |

**Atributos físicos (ordem exata, idêntica nos 2 modelos):**

```
0  Rooms
1  Toilets
2  Suites
3  Parking
4  Elevator
5  Furnished
6  Swimming Pool      <- nome com espaço, exatamente assim
7  New
8  log_Size           <- log1p(Size); NÃO existe coluna "Size" no modelo
```

**Dummies District:**
- Formato exato do nome: `District_<Bairro>/São Paulo` — ex.: `District_Pinheiros/São Paulo`,
  `District_Moema/São Paulo`, `District_Água Rasa/São Paulo`.
- Venda tem **95** dummies; aluguel tem **93**. O conjunto de aluguel é **subconjunto** do
  de venda. Os 2 bairros presentes só em venda: **`Perus/São Paulo`** e **`São Domingos/São Paulo`**
  (não havia imóveis desses bairros no treino de aluguel).
- **Ordenação:** a ordem é a do `OneHotEncoder.categories_`, que é `np.sort` das strings
  únicas. Em ordem de *code point* Unicode — por isso `District_Água Rasa/São Paulo`
  (`Á` = U+00C1) aparece **por último**, depois de `District_Vila Sônia/São Paulo`.
  `sorted(...)` do Python produz a mesma ordem (verificado).
- ⚠️ **Recomendação ao app:** **não re-fite o encoder nem re-ordene à mão.** Leia os nomes
  `District_*` direto de `modelo.feature_names_in_` e monte o vetor casando por nome
  (ver §5 e §7). Isso elimina qualquer risco de ordenação divergente.

**Espaciais (sempre as 3 últimas, nesta ordem):**
```
spatial_lag         <- média do log-preço dos k=3 vizinhos de TREINO
distancia_estacao   <- metros até a estação mais próxima (CRU, não log)
renda_area          <- log1p(renda média domiciliar da área de ponderação)
```

> A lista completa e numerada das 107 / 105 colunas está no **Apêndice A**.

---

## 2. Como cada feature é construída

Referências de célula apontam para `03_modelagem.ipynb`, salvo quando indicado `nb02`
(`02_preparacao.ipynb`), onde `distancia_estacao` é pré-calculada.

### 2.1 Atributos físicos (célula 3 e 5)

Oito atributos entram **direto do input do usuário**, sem transformação:
`Rooms`, `Toilets`, `Suites`, `Parking`, `Elevator`, `Furnished`, `Swimming Pool`, `New`.
(`Elevator`, `Furnished`, `Swimming Pool`, `New` são binários 0/1; os demais, contagens.)

Apenas **`log_Size` é transformado**. A área (`Size`, em m²) entra como `log1p(Size)` e a
coluna `Size` é descartada. Transcrição literal (célula 5):

```python
# --- log da área: cria log_Size, remove Size ---
for X in (X_train, X_test):
    X["log_Size"] = np.log1p(X["Size"])
    X.drop(columns="Size", inplace=True)
```

✅ **Confirmado:** `log_Size = np.log1p(Size)` (logaritmo natural de `1 + Size`).

### 2.2 District → dummies (célula 5)

O `District` categórico vira dummies via `OneHotEncoder`, **fitado só no treino**.
Transcrição literal (célula 5):

```python
from sklearn.preprocessing import OneHotEncoder
...
# --- one-hot do District: fit só no treino ---
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
enc.fit(X_train[["District"]])
nomes = enc.get_feature_names_out(["District"])

def aplicar_encoding(X):
    dummies = pd.DataFrame(enc.transform(X[["District"]]), columns=nomes, index=X.index)
    return pd.concat([X.drop(columns="District"), dummies], axis=1)
```

Parâmetros: **`handle_unknown="ignore"`** (bairro inédito → todas as dummies = 0, sem erro),
**`sparse_output=False`**, **sem `drop`** (nenhuma categoria de referência removida → uma
dummy por bairro do treino).

**O encoder NÃO foi salvo em disco.** Só os dois `.joblib` dos modelos existem em `models/`.
O app **não precisa** do objeto encoder: como não há `drop` e o esquema é "1 na coluna do
bairro, 0 nas demais", o app reproduz isso identicamente lendo os nomes `District_*` de
`feature_names_in_` e setando `1.0` na coluna `f"District_{district_input}"` se ela existir
(bairro inédito → todas 0, espelhando `handle_unknown="ignore"`). Ver §5/§7.

### 2.3 `spatial_lag` — A MAIS CRÍTICA (célula 28)

Definição: **média do log-preço (`log1p(Price)`) dos `k=3` vizinhos mais próximos**,
**vindos sempre do conjunto de TREINO**, com distância medida em **UTM EPSG:31983**
(métrico). Busca por `sklearn.neighbors.NearestNeighbors`.

- **k = 3** (fixado por CV na célula 33/34; `k=3` foi o escolhido para os modelos finais —
  ver chamadas em células 67 e 75 que passam `k=3`).
- **Transformação do preço:** os preços usados são `y_train_log = np.log1p(Price)`
  (célula 5). O lag é a média desses log-preços. ✅ Confirmado `log1p`.
- **CRS:** lat/long (EPSG:4326) → **UTM EPSG:31983** antes do k-NN.
- **Anti-leakage:** no treino busca-se `k+1` vizinhos e descarta-se o 1º (o próprio
  imóvel, distância 0). **Em produção isso NÃO se aplica:** um imóvel novo não está no
  treino, então busca-se exatamente `k=3` vizinhos de treino (sem `+1`, sem descarte).
  Esse é o caminho `idx_test = idx_test[:, :k]` da função.

**Transcrição literal de `construir_spatial_lag` (célula 28) — copie fielmente:**

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def construir_spatial_lag(df, X_train, X_test, y_train_log, k):
    """
    Calcula o spatial lag (média do log-preço dos k vizinhos) para treino e teste,
    com vizinhos vindos SEMPRE do treino.
    """
    # --- Coordenadas em UTM (métrico) para medir vizinhança corretamente ---
    # Converte lat/long -> UTM uma vez, para treino e teste
    import geopandas as gpd
    def coords_utm(indices):
        sub = df.loc[indices]
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(sub["Longitude"], sub["Latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=31983)
        return np.c_[gdf.geometry.x, gdf.geometry.y]

    coords_train = coords_utm(X_train.index)
    coords_test  = coords_utm(X_test.index)
    precos_train = y_train_log.values  # log-preço, só do treino — é a fonte dos vizinhos

    # ===== TREINO =====
    # Buscamos k+1 vizinhos porque o mais próximo de cada imóvel é ELE MESMO (distância 0).
    # Depois descartamos a primeira coluna (o próprio imóvel) para não vazar seu preço.
    nn_train = NearestNeighbors(n_neighbors=k + 1).fit(coords_train)
    _, idx_train = nn_train.kneighbors(coords_train)
    idx_train = idx_train[:, 1:]  # remove a 1ª coluna (o próprio imóvel)

    # spatial lag do treino = média do log-preço dos k vizinhos (todos do treino)
    lag_train = precos_train[idx_train].mean(axis=1)

    # ===== TESTE =====
    # Para o teste, buscamos os k vizinhos DENTRO DO TREINO (modelo já treinado em coords_train).
    # Aqui NÃO há +1: o imóvel de teste não está no conjunto de treino, então não é seu próprio vizinho.
    _, idx_test = nn_train.kneighbors(coords_test)  # usa o MESMO nn_train (vizinhos do treino)
    idx_test = idx_test[:, :k]  # os k vizinhos de treino mais próximos de cada imóvel de teste

    lag_test = precos_train[idx_test].mean(axis=1)

    return lag_train, lag_test
```

**Em produção** o app reproduz o ramo "TESTE" (o imóvel do usuário = um "imóvel de teste"):

1. Carrega `app/data/treino_ref_<mercado>.parquet` → tem `Latitude`, `Longitude`, `Price`
   (cru) dos imóveis de **treino** (o split 80% com `random_state=42`).
2. `precos_train = np.log1p(ref["Price"])`.
3. Converte `(Longitude, Latitude)` do treino para UTM EPSG:31983; idem a coordenada do
   imóvel novo.
4. `NearestNeighbors(n_neighbors=3).fit(coords_treino_utm)`; `kneighbors(coord_nova_utm)`.
5. `spatial_lag = precos_train[idx[0]].mean()`.

> **Os `*_espacial.parquet` (em `data/processed/`) têm `Price` CRU e lat/long em GRAUS,
> SEM `spatial_lag` e SEM log** — ✅ confirmado (colunas: `Price, Condo, Size, Rooms, ...,
> Latitude, Longitude, distancia_estacao`; nenhuma coluna logada nem `spatial_lag`). Por
> isso o app **tem que** aplicar `log1p` e converter para UTM antes do k-NN, replicando
> `construir_spatial_lag`. O `treino_ref_*.parquet` é a fatia de treino desses arquivos
> (ver §4).

### 2.4 `distancia_estacao` (nb02, célula 38)

Distância (em **metros**, **CRU** — não log) do imóvel até a **estação de metrô ou trem
mais próxima**, medida em **UTM EPSG:31983**. Pré-calculada no nb02 e gravada nos
`*_espacial.parquet`. **Em produção é direta — NÃO depende do conjunto de treino.**

- ✅ **Está em metros e entra CRUA nas árvores.** Em `treinar_e_avaliar_final` (célula 67):
  `Xtr["distancia_estacao"] = df.loc[Xtr.index, "distancia_estacao"]` — sem `log1p`.
  (Na regressão linear chegou-se a testar `log` (célula 8), mas a regressão final
  **descartou** a distância; as **árvores usam o valor cru**.)
- **Estações:** união de metrô + trem, **filtrando as inauguradas após abr/2019**.
  O filtro (nb02 célula 34) remove por nome 9 estações (6 metrô + 3 trem). Diagnóstico
  do casamento nome a nome contra o GeoJSON: **8 dos 9 nomes casaram e removeram a
  estação corretamente** — incluindo a **`VILA SÔNIA`** (metrô, Linha 4-Amarela,
  dez/2021), que **FOI removida** (a grafia na lista é byte-a-byte idêntica à do GeoJSON).
  O único nome que **não casou foi `JARDIM COLONIAL`**, e isso é **inócuo**: a Jardim
  Colonial (metrô, inaugurada em 2021) **nunca esteve no GeoJSON** — não há nenhuma
  estação contendo "COLONIAL" na base —, então tentar removê-la é um no-op e não afeta a
  contagem. Contagem efetiva: **metrô 94→89** (5 removidas) + **trem 109→106**
  (3 removidas) = **195 estações**. **O filtro está completo e correto**: verificação nas
  fontes oficiais confirmou que **nenhuma estação remanescente é posterior a abril/2019**
  (as candidatas de borda — Campo Belo/Linha 5 e Vila União, Vila Tolstói, Camilo
  Haddad/Linha 15-Prata — são de abr/2018 ou caem dentro de abril/2019, dentro da regra
  do projeto de remover apenas o que é *posterior* a abr/2019). O conjunto exato está
  salvo em `app/data/estacoes_2019.parquet` (195 pontos) e foi verificado contra a
  `distancia_estacao` do parquet com diferença 0,0.
- **Anomalia de dados (inócua, documentada):** a estação **`VILA MARIANA` (Linha 1-Azul)
  aparece duplicada** no GeoJSON do metrô — por isso há **195 registros, mas 194 estações
  únicas**. Sem efeito sobre `distancia_estacao`: um ponto duplicado exatamente no mesmo
  local não altera a distância mínima até a estação mais próxima. Fica registrado caso
  alguma contagem futura de estações precise considerar unicidade.

**Transcrição literal de `adicionar_distancia_estacao` (nb02 célula 38):**

```python
import geopandas as gpd

def adicionar_distancia_estacao(df, estacoes_latlong):
    """Calcula, para cada imóvel, a distância (m) até a estação mais próxima."""
    # 1. Transforma o df de imóveis em GeoDataFrame, criando a geometria a partir de lat/long
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326"  # lat/long, mesmo CRS das estações já convertidas
    )

    # 2. Reprojeta imóveis E estações para UTM (métrico) para medir em metros
    gdf_utm = gdf.to_crs(epsg=31983)
    estacoes_utm = estacoes_latlong.to_crs(epsg=31983)

    # 3. Junção espacial pelo vizinho mais próximo: liga cada imóvel à estação mais perto
    #    O sjoin_nearest cria a coluna 'distancia_estacao' com a distância em metros
    juntado = gpd.sjoin_nearest(
        gdf_utm,
        estacoes_utm[["nm_estacao_metro_trem", "geometry"]],
        how="left",
        distance_col="distancia_estacao"
    )

    # 4. sjoin_nearest pode duplicar linhas em caso de empate (imóvel equidistante de 2
    #    estações). Removemos duplicatas mantendo a primeira, pelo índice original.
    juntado = juntado[~juntado.index.duplicated(keep="first")]

    # Devolve só a coluna de distância, alinhada ao índice original do df
    return juntado["distancia_estacao"]
```

**Em produção:** carrega `app/data/estacoes_2019.parquet` (já em lat/long EPSG:4326),
converte para UTM 31983, mede a menor distância ponto-a-ponto até o imóvel (equivalente ao
`sjoin_nearest`). Valor em **metros, cru**.

### 2.5 `renda_area` (células 15 e 17)

Renda média domiciliar da **área de ponderação** que contém o imóvel (point-in-polygon),
transformada em **`log1p`**. Coluna de origem:
**`ponderation_area_average_household_income`** de `tb_area_of_ponderation.parquet`.

- O parquet de áreas **não traz CRS**; o notebook o declara como **EPSG:4326**
  (`gdf_ap.set_crs("EPSG:4326")`, célula 15).
- ✅ **A renda entra como `log1p(renda)`** nas árvores. Em `treinar_e_avaliar_final`
  (célula 67): `Xtr["renda_area"] = np.log1p(df.loc[Xtr.index, "renda_area"])`. A coluna
  `renda_area` guardada em `df` é a renda **crua** em R$; o `log1p` é aplicado na montagem
  da matriz.
- **Fallback (imóveis órfãos):** coordenadas que **não caem em nenhum polígono** recebem a
  renda da **área de ponderação mais próxima** (medida em UTM 31983 via `sjoin_nearest`).
  No treino: **12 órfãos (venda) + 17 (aluguel) = 29** (bate com os "29" citados).

**Transcrição literal — atribuição por contenção (`adicionar_renda`, célula 15):**

```python
RENDA = "ponderation_area_average_household_income"

# 1. Declara o CRS correto nas áreas de ponderação
gdf_ap = gdf_ap.set_crs("EPSG:4326")

def adicionar_renda(df, nome):
    # 2. Transforma os imóveis em pontos geográficos (lat/long)
    gdf_imoveis = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326"
    )

    # 3. Join espacial: cada imóvel recebe os dados da área de ponderação que o contém
    juntado = gpd.sjoin(
        gdf_imoveis,
        gdf_ap[[RENDA, "geometry"]],
        how="left",
        predicate="within"
    )

    # 4. Remove duplicatas (imóvel em cima de divisa pode casar com 2 áreas)
    juntado = juntado[~juntado.index.duplicated(keep="first")]

    # 5. Renomeia a coluna de renda e devolve só ela, alinhada ao índice original
    return juntado[RENDA].rename("renda_area")
```

**Transcrição literal — fallback órfãos (`preencher_renda_orfaos`, célula 17):**

```python
def preencher_renda_orfaos(df, nome):
    # Identifica os órfãos (renda NaN após o join within)
    orfaos = df[df["renda_area"].isna()].copy()
    if len(orfaos) == 0:
        print(f"{nome}: sem órfãos.")
        return df["renda_area"]

    # Pontos dos órfãos e polígonos, ambos em UTM (métrico) para medir distância correta
    gdf_orfaos = gpd.GeoDataFrame(
        orfaos,
        geometry=gpd.points_from_xy(orfaos["Longitude"], orfaos["Latitude"]),
        crs="EPSG:4326"
    ).to_crs(epsg=31983)
    ap_utm = gdf_ap[[RENDA, "geometry"]].to_crs(epsg=31983)

    # Join pela área de ponderação MAIS PRÓXIMA (não a que contém, que não existe p/ órfãos)
    preenchidos = gpd.sjoin_nearest(gdf_orfaos, ap_utm, how="left")
    preenchidos = preenchidos[~preenchidos.index.duplicated(keep="first")]

    # Atualiza a coluna renda_area só nos índices órfãos
    renda = df["renda_area"].copy()
    renda.loc[orfaos.index] = preenchidos[RENDA]
    print(f"{nome}: {len(orfaos)} órfãos preenchidos pela área mais próxima.")
    return renda
```

**Em produção, para a coordenada do imóvel:**
1. Carrega `app/data/areas_renda.parquet` (EPSG:4326, com a coluna de renda e a geometria).
2. Acha o polígono que **contém** o ponto. Se houver → `renda = log1p(renda_do_poligono)`.
3. Se **nenhum** polígono contém (órfão) → converte para UTM 31983, pega a área **mais
   próxima**, `renda = log1p(renda_dessa_area)`.

### 2.6 Montagem da matriz final (célula 67) — a ordem de concatenação

Como as 3 espaciais são **anexadas ao final**, na ordem `spatial_lag`,
`distancia_estacao`, `renda_area`. Transcrição literal de `treinar_e_avaliar_final`
(célula 67) — define a ordem que o XGBoost campeão recebeu (`usa_distancia=True`,
`usa_renda=True`):

```python
def treinar_e_avaliar_final(df, X_train, X_test, y_train_log, y_test_real,
                            k, modelo, usa_distancia, rotulo, usa_renda=False):
    """Treina no treino completo, avalia uma vez no test. Matriz conforme o modelo."""
    lag_tr, lag_te = construir_spatial_lag(df, X_train, X_test, y_train_log, k)
    Xtr, Xte = X_train.copy(), X_test.copy()
    Xtr["spatial_lag"], Xte["spatial_lag"] = lag_tr, lag_te
    if usa_distancia:
        Xtr["distancia_estacao"] = df.loc[Xtr.index, "distancia_estacao"]
        Xte["distancia_estacao"] = df.loc[Xte.index, "distancia_estacao"]
    if usa_renda:
        Xtr["renda_area"] = np.log1p(df.loc[Xtr.index, "renda_area"])
        Xte["renda_area"] = np.log1p(df.loc[Xte.index, "renda_area"])

    modelo.fit(Xtr, y_train_log)
    pred = np.expm1(modelo.predict(Xte))
    ...
```

`X_train`/`X_test` aqui já contêm `[8 atributos, log_Size, District_*]` (físicos + dummies),
após `consolidar` ter removido `distancia_estacao` (célula 21). As 3 espaciais são as 3
últimas colunas. ✅ Bate com `feature_names_in_` (Apêndice A).

---

## 3. Alvo e sua reversão

- O alvo é **`y_log = np.log1p(Price)`** (célula 5). O modelo é treinado e **prediz em log**.
- ✅ **Reversão: `Price = np.expm1(pred_log)`** (células 7, 67, etc.:
  `pred = np.expm1(modelo.predict(Xte))`).

O app **deve** aplicar `np.expm1` à saída de `modelo.predict(...)` para obter o preço em reais.

---

## 4. Artefatos de dados necessários para o app

`data/` está no `.gitignore` (não versionado). Por isso já existe um **pacote de dados de
produção enxuto** em **`app/data/`** (versionável, colunas reduzidas), que o app deve
carregar. Todos verificados com diferença 0,0 contra o pipeline de treino (§6).

| Arquivo (app/data/) | Tamanho | CRS | Linhas | Colunas | O que o app extrai |
|---|---|---|---|---|---|
| `treino_ref_venda.parquet` | ~91 KB | EPSG:4326 (graus) | 4748 | `Latitude, Longitude, Price, district` | Vizinhos do `spatial_lag` de **venda** (k-NN sobre log1p(Price)) |
| `treino_ref_aluguel.parquet` | ~96 KB | EPSG:4326 | 5349 | `Latitude, Longitude, Price, district` | Vizinhos do `spatial_lag` de **aluguel** |
| `estacoes_2019.parquet` | ~13 KB | EPSG:4326 | 195 | `nm_estacao_metro_trem, geometry` | `distancia_estacao` (estação mais próxima, em UTM) |
| `areas_renda.parquet` | ~1,9 MB | EPSG:4326 | 310 | `ponderation_area_average_household_income, geometry` | `renda_area` (point-in-polygon + fallback) |

Notas:
- `treino_ref_*` contém **`Price` CRU** (o app aplica `log1p`) e lat/long em **graus** (o
  app converte para UTM 31983 antes do k-NN). Linhas = exatamente o split de treino 80%
  (`train_test_split(..., test_size=0.20, random_state=42)`). 5936×0,8 ≈ 4748 (venda);
  6687×0,8 ≈ 5349 (aluguel). ✅ verificado: produz `spatial_lag` idêntico ao do notebook.
- `areas_renda.parquet` é o `tb_area_of_ponderation.parquet` enxugado para 2 colunas
  (renda + geometria), ainda 1,9 MB por causa das geometrias dos 310 polígonos. Se for
  preciso encolher para o Streamlit Cloud, pode-se simplificar a geometria
  (`.simplify(tol)`) **com cautela** — alterar fronteiras muda o resultado do
  point-in-polygon e quebra a fidelidade. **Não simplificar sem revalidar contra §6.**
- A coluna `district` nos `treino_ref_*` é informativa (não é usada no cálculo do lag).

**Fontes originais (não versionadas, em `data/`), caso seja preciso regenerar o pacote:**
`data/processed/venda_espacial.parquet`, `data/processed/aluguel_espacial.parquet`,
`data/raw/estacoes_metro.geojson`, `data/raw/estacoes_trem.geojson`,
`data/raw/tb_area_of_ponderation.parquet`. Os modelos ficam em `models/*.joblib`
(esses **são** versionados; ~2 MB cada).

---

## 5. Como o app monta a matriz (resumo operacional)

Para cada mercado (`venda`/`aluguel`), dado o input do usuário + coordenada (lat/long):

```text
feats = list(modelo.feature_names_in_)        # ordem canônica — NÃO reordenar
vec   = pd.Series(0.0, index=feats)

# físicos (direto do input)
vec["Rooms"], vec["Toilets"], vec["Suites"], vec["Parking"] = ...
vec["Elevator"], vec["Furnished"], vec["Swimming Pool"], vec["New"] = ...   # binários 0/1
vec["log_Size"] = np.log1p(size_m2)

# District (casa por nome; bairro inédito -> todas as dummies ficam 0)
col = f"District_{district_input}"
if col in feats:
    vec[col] = 1.0

# espaciais (derivadas da coordenada)
vec["spatial_lag"]       = prod_spatial_lag(mercado, lat, lon, k=3)   # média log1p dos 3 vizinhos de treino, UTM
vec["distancia_estacao"] = prod_distancia(lat, lon)                   # metros, CRU
vec["renda_area"]        = np.log1p(renda_point_in_polygon(lat, lon)) # log1p, com fallback nearest

X = pd.DataFrame([vec.values], columns=feats)
preco_reais = np.expm1(modelo.predict(X)[0])
```

(Pseudocódigo da spec — o app real implementa as funções `prod_*` conforme §2.3–2.5. **Não
escrever o app aqui.**)

---

## 6. Verificação empírica

Método: replicou-se o pipeline do notebook (verdade-terreno: o valor que o modelo
**realmente recebeu**) e, separadamente, derivou-se cada feature **só pelos artefatos de
produção** (`app/data/*`), comparando os dois. Testados 3 imóveis do conjunto de teste por
mercado × 3 features espaciais.

**Resultado: diferença = 0,0 em TODOS os 18 casos** (`|diff| = 0.00e+00`).

```
MERCADO: VENDA   (12 órfãos de renda)
  idx=7301  spatial_lag  treino=13.71288347  prod=13.71288347  |diff|=0.0
            distancia    treino=880.27715665 prod=880.27715665 |diff|=0.0
            renda_area   treino= 9.28814928  prod= 9.28814928  |diff|=0.0
  idx=6874  spatial_lag  treino=12.53065428  prod=12.53065428  |diff|=0.0   (+ dist, renda 0.0)
  idx=8543  spatial_lag  treino=13.31710672  prod=13.31710672  |diff|=0.0   (+ dist, renda 0.0)

MERCADO: ALUGUEL (17 órfãos de renda)
  idx=247   spatial_lag  treino= 6.48867244  prod= 6.48867244  |diff|=0.0   (+ dist, renda 0.0)
  idx=1760  spatial_lag  treino= 7.25031999  prod= 7.25031999  |diff|=0.0   (+ dist, renda 0.0)
  idx=3352  spatial_lag  treino= 7.93656526  prod= 7.93656526  |diff|=0.0   (+ dist, renda 0.0)
```

**Teste end-to-end (matriz 107 completa, montada 100% por métodos de produção vs. matriz do
notebook):** predição idêntica — `pred(notebook) = pred(produção) = 14.7738704681`,
`|diff| = 0.0`, matriz byte-a-byte idêntica (`np.allclose == True`). A reversão `expm1`
devolve um preço em reais coerente. (Num imóvel específico o modelo erra — ex.: previu
R$ 2,6 M vs. R$ 1,6 M real — o que é esperado a 14% de MAPE; isso é erro do **modelo**, não
da montagem da matriz, que está provada idêntica.)

→ **A documentação dos métodos de §2 está correta.** Os artefatos em `app/data/` reproduzem
o pipeline de treino exatamente.

---

## 7. Pontos críticos (resumo)

1. **Ordem das colunas é sagrada.** Sempre construir a matriz na ordem de
   `modelo.feature_names_in_`. Não reordenar, não confiar em ordem alfabética "natural"
   (`Água Rasa` vem por último, não no início).
2. **`spatial_lag` em produção usa o ramo "TESTE": k=3 vizinhos de treino, SEM `+1`.** Fonte
   = `treino_ref_<mercado>.parquet`, `log1p(Price)`, distância em **UTM 31983**.
3. **`log` onde é log, cru onde é cru:** `log_Size = log1p(Size)`; `spatial_lag` é média de
   `log1p(Price)`; `renda_area = log1p(renda)`; **`distancia_estacao` é metros CRUS.**
4. **Dois modelos têm conjuntos de District diferentes** (venda 95, aluguel 93). Casar por
   nome contra `feature_names_in_` do **modelo daquele mercado**.
5. **CRS:** lat/long de entrada = **EPSG:4326**; toda medição métrica (k-NN do lag,
   distância à estação, fallback de renda) em **EPSG:31983**. Point-in-polygon da renda em
   4326.
6. **Reversão final: `expm1`.**
7. **`areas_renda.parquet` (1,9 MB)** é o maior artefato. Se precisar reduzir para deploy,
   não simplificar geometria sem revalidar (§6) — fronteiras alteradas mudam o
   point-in-polygon.

## 8. A CONFIRMAR

Nenhum item da derivação ficou em aberto — todas as features foram confirmadas no código
**e** verificadas empiricamente com diferença 0,0. Pontos de atenção residuais (não são
bloqueios, mas o app deve tratá-los conscientemente):

- **(edge case) Coordenada idêntica a um imóvel de treino.** Se o usuário inserir uma
  coordenada que coincide exatamente com um imóvel do `treino_ref`, o k-NN de produção
  incluirá esse imóvel entre os 3 vizinhos (sem o descarte do "próprio", que no treino só
  existe para os pontos do próprio treino). Para um imóvel genuinamente novo isso não
  ocorre. **A CONFIRMAR com o dono do produto:** se isso é aceitável (provavelmente sim, é
  o comportamento correto para um ponto novo que por acaso coincide).
- **Geocodificação CEP → (lat, lon).** A spec cobre o contrato da coordenada para frente;
  **a fonte/serviço de geocodding do CEP não está definido no notebook** e precisa ser
  escolhido na Fase 6. A precisão da coordenada afeta diretamente as 3 features espaciais.
  **A CONFIRMAR:** qual geocoder usar (ex.: ViaCEP só dá bairro; precisa de um que devolva
  lat/long).
- **Bairro inédito / fora de SP.** Com `handle_unknown="ignore"`, um bairro não visto vira
  todas as dummies 0 (o modelo cai no efeito médio + espaciais). É o comportamento de
  treino; o app pode querer **avisar o usuário** em vez de prever silenciosamente.

---

## Apêndice A — Lista completa e ordenada (venda, 107 features)

```
  0  Rooms
  1  Toilets
  2  Suites
  3  Parking
  4  Elevator
  5  Furnished
  6  Swimming Pool
  7  New
  8  log_Size
  9  District_Alto de Pinheiros/São Paulo
 10  District_Anhanguera/São Paulo
 11  District_Aricanduva/São Paulo
 12  District_Artur Alvim/São Paulo
 13  District_Barra Funda/São Paulo
 14  District_Bela Vista/São Paulo
 15  District_Belém/São Paulo
 16  District_Bom Retiro/São Paulo
 17  District_Brasilândia/São Paulo
 18  District_Brooklin/São Paulo
 19  District_Brás/São Paulo
 20  District_Butantã/São Paulo
 21  District_Cachoeirinha/São Paulo
 22  District_Cambuci/São Paulo
 23  District_Campo Belo/São Paulo
 24  District_Campo Grande/São Paulo
 25  District_Campo Limpo/São Paulo
 26  District_Cangaíba/São Paulo
 27  District_Capão Redondo/São Paulo
 28  District_Carrão/São Paulo
 29  District_Casa Verde/São Paulo
 30  District_Cidade Ademar/São Paulo
 31  District_Cidade Dutra/São Paulo
 32  District_Cidade Líder/São Paulo
 33  District_Cidade Tiradentes/São Paulo
 34  District_Consolação/São Paulo
 35  District_Cursino/São Paulo
 36  District_Ermelino Matarazzo/São Paulo
 37  District_Freguesia do Ó/São Paulo
 38  District_Grajaú/São Paulo
 39  District_Guaianazes/São Paulo
 40  District_Iguatemi/São Paulo
 41  District_Ipiranga/São Paulo
 42  District_Itaim Bibi/São Paulo
 43  District_Itaim Paulista/São Paulo
 44  District_Itaquera/São Paulo
 45  District_Jabaquara/São Paulo
 46  District_Jaguaré/São Paulo
 47  District_Jaraguá/São Paulo
 48  District_Jardim Helena/São Paulo
 49  District_Jardim Paulista/São Paulo
 50  District_Jardim São Luis/São Paulo
 51  District_Jardim Ângela/São Paulo
 52  District_Jaçanã/São Paulo
 53  District_José Bonifácio/São Paulo
 54  District_Lajeado/São Paulo
 55  District_Lapa/São Paulo
 56  District_Liberdade/São Paulo
 57  District_Limão/São Paulo
 58  District_Mandaqui/São Paulo
 59  District_Moema/São Paulo
 60  District_Mooca/São Paulo
 61  District_Morumbi/São Paulo
 62  District_Pari/São Paulo
 63  District_Parque do Carmo/São Paulo
 64  District_Pedreira/São Paulo
 65  District_Penha/São Paulo
 66  District_Perdizes/São Paulo
 67  District_Perus/São Paulo            <- só em VENDA
 68  District_Pinheiros/São Paulo
 69  District_Pirituba/São Paulo
 70  District_Ponte Rasa/São Paulo
 71  District_Raposo Tavares/São Paulo
 72  District_República/São Paulo
 73  District_Rio Pequeno/São Paulo
 74  District_Sacomã/São Paulo
 75  District_Santa Cecília/São Paulo
 76  District_Santana/São Paulo
 77  District_Santo Amaro/São Paulo
 78  District_Sapopemba/São Paulo
 79  District_Saúde/São Paulo
 80  District_Socorro/São Paulo
 81  District_São Domingos/São Paulo     <- só em VENDA
 82  District_São Lucas/São Paulo
 83  District_São Mateus/São Paulo
 84  District_São Miguel/São Paulo
 85  District_São Rafael/São Paulo
 86  District_Sé/São Paulo
 87  District_Tatuapé/São Paulo
 88  District_Tremembé/São Paulo
 89  District_Tucuruvi/São Paulo
 90  District_Vila Andrade/São Paulo
 91  District_Vila Curuçá/São Paulo
 92  District_Vila Formosa/São Paulo
 93  District_Vila Guilherme/São Paulo
 94  District_Vila Jacuí/São Paulo
 95  District_Vila Leopoldina/São Paulo
 96  District_Vila Madalena/São Paulo
 97  District_Vila Maria/São Paulo
 98  District_Vila Mariana/São Paulo
 99  District_Vila Matilde/São Paulo
100  District_Vila Olimpia/São Paulo
101  District_Vila Prudente/São Paulo
102  District_Vila Sônia/São Paulo
103  District_Água Rasa/São Paulo
104  spatial_lag
105  distancia_estacao
106  renda_area
```

**Aluguel (105 features):** idêntica à lista acima, **removendo** `District_Perus/São Paulo`
(67) e `District_São Domingos/São Paulo` (81) e re-indexando. As 3 espaciais ficam nas
posições 102, 103, 104. Sempre obter a lista real de
`joblib.load("models/modelo_campeao_aluguel.joblib").feature_names_in_`.
