# One-Pager — AVM São Paulo (Calculadora de Preço de Apartamentos)

## O produto em uma frase

Uma ferramenta web onde um corretor ou avaliador, a partir do CEP e das características
de um apartamento em São Paulo, obtém em segundos uma estimativa de preço (venda ou
aluguel) com faixa de incerteza e uma explicação dos fatores que mais pesaram — para
usar como apoio e defesa de valor no trabalho dele.

---

## O problema

Profissionais do mercado imobiliário (corretores, avaliadores) precisam estimar o preço
de apartamentos rapidamente e **defender esse número** para clientes. Hoje isso depende
de:

- **Intuição e experiência** — subjetiva, difícil de justificar, varia entre profissionais.
- **Análise comparativa manual** (buscar anúncios parecidos) — demorada e imprecisa.
- **Avaliação formal** (laudo NBR) — cara, lenta, desproporcional para uma consulta rápida.

Falta uma ferramenta intermediária: rápida como a intuição, mas **objetiva e explicável**
como uma análise — para o dia a dia de quem precisa cravar um valor com segurança.

---

## A solução

Um AVM (Automated Valuation Model) acessível via web, que:

1. Recebe o **CEP** do imóvel e suas **características** (área, quartos, suítes, vaga,
   andar, mobília, etc.).
2. Deriva automaticamente os **fatores de localização** (preço do entorno, proximidade
   de transporte, perfil socioeconômico da região) a partir da coordenada.
3. Retorna uma **estimativa de preço** (número + faixa de incerteza) para **venda ou
   aluguel**.
4. Mostra **quais fatores mais pesaram** na estimativa — a munição que o profissional
   usa para defender o valor.

**Natureza:** AVM-produto (sugestão de preço, como a Calculadora da Loft ou o Preço
Certo do QuintoAndar) — **não** um laudo com validade jurídica.

---

## Quem usa

**Usuário-alvo principal:** corretor ou avaliador imobiliário, usando como ferramenta
de apoio.

**Características desse usuário que moldam o produto:**
- **Quer o "porquê", não só o número** — vai defender a estimativa para o cliente dele.
- **Usa repetidamente** — fluxo precisa ser rápido (estimar, ajustar, estimar de novo).
- **É cético com modelo** — conhece o mercado; se a estimativa for absurda ou opaca,
  descarta a ferramenta. Confiança se ganha com honestidade sobre incerteza e
  explicabilidade.

---

## Escopo (e seus limites, assumidos conscientemente)

| Dentro do escopo | Fora do escopo |
|------------------|----------------|
| Apartamentos | Casas, terrenos, comercial |
| Município de São Paulo | Outras cidades / região metropolitana |
| Estimativa de preço de **anuncio** | Preço de transação / valor de fechamento |
| Venda e aluguel | Financiamento, ITBI, custos de transação |
| Sugestão de apoio | Laudo com validade jurídica (NBR/CREA) |

O escopo "apartamentos em São Paulo" é **explícito no nome e na interface** — o usuário
sabe o contexto antes de usar.

---

## Proposta de valor (por que usariam)

- **Velocidade:** estimativa em segundos, contra horas de análise comparativa manual.
- **Objetividade:** número derivado de um modelo treinado sobre ~13 mil imóveis, não de
  intuição.
- **Explicabilidade:** mostra os fatores que pesaram — defensável perante o cliente.
- **Honestidade:** entrega faixa de incerteza, não um número falsamente preciso; avisa
  quando a região é mal coberta.
- **Diferencial técnico:** incorpora **fatores espaciais finos** (preço do entorno
  imediato via *spatial lag*) que análises comparativas manuais não capturam.

---

## Métrica de sucesso do modelo (já alcançada)

- **MAPE de ~14% (venda) / ~20% (aluguel)** no conjunto de teste — competitivo com a
  literatura brasileira de AVMs sobre dados de anúncio.
- Modelo campeão: **XGBoost**, validado contra regressão linear (baseline normativo) e
  Random Forest.

---

## O que o MVP entrega

- [x] Modelo treinado e validado (venda + aluguel)
- [ ] Entrada por CEP + características do imóvel
- [ ] Toggle venda/aluguel na mesma tela
- [ ] Estimativa: número + faixa de incerteza
- [ ] Explicação: fatores que mais pesaram
- [ ] Aviso quando a localização é mal coberta pelo modelo
- [ ] Deploy público (Streamlit Community Cloud)

**Fora do MVP (futuro):** comparação de múltiplos imóveis, exportação em PDF, histórico
de consultas, ajuste fino de localização por mapa.

---

## Stack

- **Frontend/app:** Streamlit (visual nativo, deploy gratuito no Streamlit Community Cloud)
- **Modelo:** XGBoost (`.joblib`)
- **Feature engineering em produção:** derivação das features espaciais a partir da
  coordenada do CEP, replicando fielmente o pipeline de treino (ver SPEC_CONTRATO.md)
- **Dados:** referências de treino (para o *spatial lag*), estações de transporte,
  áreas de ponderação (renda) — empacotados em `app/data/`