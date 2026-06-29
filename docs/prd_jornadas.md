# PRD — Jornadas de Usuário · AVM São Paulo

> Documento de jornadas para guiar a construção do app Streamlit. Acompanha o
> ONE_PAGER.md (visão) e o SPEC_CONTRATO.md (contrato técnico modelo↔app). Foco aqui:
> **o que o usuário faz e vê**, incluindo todos os estados de erro.

---

## Princípios de UX (público profissional)

1. **Velocidade acima de tudo** — o profissional usa repetidamente; o caminho do CEP ao
   resultado deve ser o mais curto possível, sem telas intermediárias desnecessárias.
2. **Honestidade sobre incerteza** — sempre faixa, nunca número falsamente preciso;
   avisos claros quando o modelo está em terreno fraco. É o que ganha a confiança de um
   usuário cético.
3. **Explicabilidade como munição** — a explicação não é enfeite; é o que o profissional
   usa para defender o valor ao cliente. Deve ser clara e citável.
4. **Escopo explícito** — "apartamentos em São Paulo" visível o tempo todo; o usuário
   nunca se surpreende com uma recusa.

---

## Estrutura geral da tela (single-page)

O app é uma **página única** (coerente com o MVP enxuto). Layout de cima para baixo:

```
┌─────────────────────────────────────────────┐
│  CABEÇALHO                                    │
│  Título: "Avaliação de Apartamentos · SP"     │
│  Subtítulo: escopo + natureza (apoio, não     │
│  laudo)                                        │
├─────────────────────────────────────────────┤
│  [ Toggle: VENDA | ALUGUEL ]                  │
├─────────────────────────────────────────────┤
│  ENTRADA                                      │
│  - Campo CEP                                   │
│  - (após CEP válido) endereço confirmado      │
│  - Características do imóvel (área, quartos,   │
│    suítes, vaga, andar, mobília, etc.)        │
│  - Botão "Estimar preço"                       │
├─────────────────────────────────────────────┤
│  RESULTADO (após estimar)                     │
│  - Preço estimado (número grande) + faixa     │
│  - Fatores que mais pesaram (explicação)      │
│  - Avisos (se região mal coberta)             │
└─────────────────────────────────────────────┘
```

---

## Jornada 1 — Caminho feliz (estimativa bem-sucedida)

**Persona:** corretora avaliando um apartamento em Pinheiros para orientar um cliente
vendedor.

**Passos:**

1. **Abre o app.** Vê o título "Avaliação de Apartamentos · São Paulo" e o subtítulo
   deixando claro: estimativa de apoio para apartamentos na cidade de SP, não um laudo.
2. **Escolhe o tipo** no toggle: **Venda** (padrão) ou Aluguel.
3. **Digita o CEP** do imóvel (ex.: 05422-010).
4. **O app valida e geocodifica o CEP** → mostra o endereço/bairro identificado para a
   profissional **confirmar** que é o lugar certo ("Rua dos Pinheiros, Pinheiros — está
   correto?"). Isso dá controle sem exigir mapa.
5. **Preenche as características** do imóvel:
   - Área (m²) — campo numérico
   - Quartos, Suítes, Banheiros, Vagas de garagem — campos numéricos
   - Andar / tem elevador — conforme o modelo precisa
   - Mobiliado? (sim/não), Tem piscina no condomínio? (sim/não), É novo? (sim/não)
6. **Clica em "Estimar preço".**
7. **Vê o resultado:**
   - **Preço estimado em destaque:** "R$ 780.000"
   - **Faixa de incerteza:** "entre R$ 670.000 e R$ 890.000" (derivada do erro do modelo)
   - **Fatores que mais pesaram:** lista curta e legível, ex.:
     - "Localização valorizada — imóveis vizinhos têm preço alto (↑)"
     - "Área de 95m² acima da média da região (↑)"
     - "2 vagas de garagem (↑)"
   - Cada fator com direção (↑ puxa preço para cima, ↓ para baixo).
8. **Ajusta e re-estima** se quiser (muda uma característica, clica de novo) — fluxo
   rápido para iterar.

**Requisito-chave:** do CEP confirmado ao resultado, o mínimo de fricção. A profissional
deve conseguir estimar vários imóveis em sequência sem recarregar nada.

---

## Jornada 2 — CEP fora de São Paulo

**Situação:** o usuário digita um CEP de Campinas, Guarulhos, ou qualquer lugar fora do
município de SP.

**Comportamento:**

1. O app geocodifica o CEP → obtém a coordenada.
2. **Valida se a coordenada está dentro do município de São Paulo** (bounding box ou
   distância às referências de treino).
3. Se **fora**: o app **não tenta estimar**. Mostra uma mensagem clara e respeitosa:
   > "Este modelo cobre apenas apartamentos no **município de São Paulo**. O CEP
   > informado parece estar em outra localidade. Verifique o CEP ou consulte uma
   > ferramenta adequada para essa região."
4. O campo de características pode ficar **desabilitado** até um CEP válido de SP ser
   informado — evita que o usuário preencha tudo e só then descubra a recusa.

**Princípio:** falhar cedo e com clareza. Não frustrar o usuário deixando-o preencher
tudo para recusar no final.

---

## Jornada 3 — CEP em SP, mas região mal coberta pelo modelo

**Situação:** o CEP é válido e está em SP, mas cai num distrito que o modelo viu pouco
ou não viu (ex.: distritos periféricos com poucos imóveis na base; ou Perus/São Domingos,
que não existem no modelo de aluguel).

**Comportamento:**

1. O app deriva o distrito (pelo imóvel de treino mais próximo) e as features espaciais.
2. Detecta que a região tem **cobertura fraca** (distrito desconhecido pelo modelo, ou
   poucos vizinhos próximos no treino).
3. **Ainda estima** (o modelo degrada graciosamente), mas **avisa**:
   > "⚠️ Esta região tem poucos imóveis na base de referência. A estimativa é menos
   > precisa do que em áreas com mais dados."
4. Mostra a estimativa normalmente, mas o aviso calibra a confiança do profissional.

**Princípio:** transparência sobre a qualidade da estimativa. Um profissional cético
confia mais numa ferramenta que admite seus limites do que numa que finge precisão
uniforme.

---

## Jornada 4 — CEP inválido ou não encontrado

**Situação:** o usuário digita um CEP que não existe, está incompleto, ou o serviço de
geocodificação não localiza.

**Comportamento:**

1. O app tenta validar/geocodificar.
2. Se o CEP é malformado (menos de 8 dígitos): valida no campo, pede correção.
3. Se o serviço não encontra o CEP: mensagem clara:
   > "Não foi possível localizar este CEP. Verifique se está correto."
4. Não trava o app — o usuário corrige e tenta de novo.

**Nota técnica:** a geocodificação depende de serviço externo (ViaCEP + geocoder). Tratar
o caso de o serviço estar **indisponível** (timeout) com mensagem específica:
> "Serviço de localização temporariamente indisponível. Tente novamente em instantes."

---

## Jornada 5 — Entrada de características incompleta ou inválida

**Situação:** o usuário clica em "Estimar" sem preencher campos obrigatórios, ou põe
valores absurdos (área = 0, 50 quartos).

**Comportamento:**

1. **Campos obrigatórios vazios:** o botão "Estimar" fica desabilitado até o mínimo ser
   preenchido, ou aponta o que falta.
2. **Valores fora de faixa plausível:** validação leve com aviso (não bloqueio rígido,
   mas alerta): "Área de 5m²? Confirme se está correto." — porque um valor absurdo gera
   estimativa absurda, e o profissional pode ter errado a digitação.
3. Faixas sugeridas nos campos (ex.: área tipicamente 20–500m²) ajudam a evitar erros.

---

## Estados do resultado — resumo

| Situação | O que o usuário vê |
|----------|--------------------|
| Sucesso, região bem coberta | Preço + faixa + fatores |
| Sucesso, região mal coberta | Preço + faixa + fatores + **aviso de baixa precisão** |
| CEP fora de SP | **Recusa** educada, sem estimativa |
| CEP inválido/não encontrado | Pedido de correção |
| Serviço de geocode fora | Mensagem de indisponibilidade temporária |
| Características inválidas | Validação/alerta antes de estimar |

---

## A faixa de incerteza — requisito técnico

O modelo (XGBoost) prediz um **número único**. A faixa mostrada ("entre X e Y") deve ser
derivada do **erro conhecido do modelo** (MAPE no test: ~14% venda / ~20% aluguel):

- Faixa = estimativa ± (MAPE do modelo correspondente).
- Ex.: venda, estimativa R$ 780k, MAPE 14% → faixa ≈ R$ 670k–890k.

**Justificativa:** a faixa reflete a incerteza *real e medida* do modelo, não um chute.
É honesto e defensável. (Detalhar na implementação; alternativas mais sofisticadas —
intervalos de predição por quantile regression — ficam fora do MVP.)

---

## A explicação — requisito de produto

"Fatores que mais pesaram" deve ser:

- **Curta** (3–5 fatores) — o profissional escaneia rápido.
- **Em linguagem de negócio**, não de modelo — "localização valorizada", não
  "spatial_lag = 13.2".
- **Com direção** (↑/↓) — puxou o preço para cima ou para baixo.
- **Específica àquele imóvel** — baseada nos valores que o usuário informou, não genérica.

**Decisão (fixada): a explicação NÃO usa SHAP.** O usuário é um profissional de mercado,
não de dados — para ele, o valor da explicação está na **plausibilidade da narrativa que
ele repete ao cliente**, não na exatidão da atribuição. SHAP otimiza uma precisão técnica
que esse usuário não usa e exigiria tradução. A explicação prioriza **linguagem de negócio
e clareza** sobre exatidão matemática.

**Como é construída:** combina a importância das features (do modelo) com os **valores do
imóvel específico**, traduzindo cada fator relevante para uma frase de negócio com direção:

| Condição (valor do imóvel vs. referência da região) | Frase ao usuário |
|------------------------------------------------------|------------------|
| Área acima da média da região | "Área generosa para a região (↑)" |
| `spatial_lag` alto (vizinhos caros) | "Localização valorizada (↑)" |
| `spatial_lag` baixo (vizinhos baratos) | "Localização de menor valor de mercado (↓)" |
| 2+ vagas de garagem | "Vagas de garagem (↑)" |
| Renda da região alta | "Entorno de alto padrão socioeconômico (↑)" |
| Mobiliado (no aluguel) | "Imóvel mobiliado (↑)" |
| Novo (no aluguel) | "Imóvel novo (↑)" |

A direção (↑/↓) sai de comparar o valor do imóvel com a média/referência da região. Mostra
os 3–5 fatores mais relevantes para aquele imóvel.

**Sobre a precisão da heurística:** ela pode, em casos de borda, errar a direção de um fator
*pouco relevante* — mas justamente por ser pouco relevante, isso não afeta a narrativa que o
profissional usa. Os fatores que dominam (tamanho, localização claramente cara/barata) a
heurística acerta com facilidade. O trade-off é aceitável e favorável para este usuário.

---

## Fora do MVP (roadmap futuro)

- Comparar 2–3 imóveis lado a lado.
- Exportar a estimativa (PDF para anexar à proposta ao cliente).
- Histórico de consultas da sessão.
- Ajuste fino da localização por mapa (arrastar pin) — se a geocodificação por CEP se
  mostrar imprecisa demais para o *spatial lag*.