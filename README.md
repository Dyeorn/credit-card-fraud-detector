# Credit Card Fraud Detection

Projeto de Machine Learning para detecção de fraude em transações de cartão de crédito — AV2 da disciplina de Machine Learning (2026).

## Problema

Classificar transações de cartão de crédito como **legítimas ou fraudulentas** a partir de um conjunto de features anonimizadas. O dataset é altamente desbalanceado (~0.17% de fraudes), tornando necessário o uso de métricas e estratégias apropriadas para dados com desequilíbrio de classes.

**Variável-alvo:** `Class` (0 = legítima, 1 = fraude)

## Dataset

- **Fonte:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Tamanho:** 284.807 transações, 31 colunas
- **Features:** `Time`, `Amount` e V1–V28 (componentes PCA por razões de confidencialidade)
- **Desbalanceamento:** 284.315 legítimas (99.83%) vs. 492 fraudes (0.17%)

> O arquivo `creditcard.csv` deve ser colocado em `data/raw/creditcard.csv`.
> Por questões de tamanho, o dataset não é versionado neste repositório.

## Instalação

Requer [uv](https://docs.astral.sh/uv/) instalado.

```bash
uv sync
```

## Execução

```bash
uv run python main.py
```

O pipeline completo executa em sequência:

1. **Preparação dos dados** — split estratificado 80/20, salva em `data/processed/`
2. **Treinamento** — 3 modelos com GridSearchCV + validação cruzada (5 folds)
3. **Avaliação** — métricas no conjunto de teste, teste McNemar
4. **Visualizações** — figuras salvas em `article/figures/`

## Modelos

| Modelo | Estratégia para Desbalanceamento |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` |

## Métricas

**Primária:** AUC-PR (Average Precision) — mais informativa que AUC-ROC para dados desbalanceados.  
**Secundárias:** F1-score, AUC-ROC, Precision, Recall.

## Resultados

| Modelo | AUC-PR | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0,7205 | 0,1139 | 0,9721 |
| Random Forest | 0,8567 | **0,8495** | 0,9561 |
| **XGBoost** | **0,8652** | 0,8079 | **0,9747** |

McNemar test (XGBoost vs RF): p = 0,027 — diferença estatisticamente significativa.  
Ver `experiments/experiments.csv` e `article/tables/model_comparison.md` para detalhes completos.

## Estrutura do Repositório

```
credit-card-fraud-detector/
├── data/raw/           # Dataset original (não versionado)
├── data/processed/     # Train/test splits (gerado pelo pipeline)
├── notebooks/          # EDA interativa
├── src/                # Código-fonte modular
│   ├── data/           # Preparação de dados
│   ├── features/       # Feature engineering
│   ├── models/         # Treinamento
│   ├── evaluation/     # Métricas e testes estatísticos
│   └── visualization/  # Geração de figuras
├── experiments/        # Log de experimentos e modelos serializados
├── article/            # Artigo técnico e figuras
└── docs/               # Documentação técnica
```

## Limitações

- Features V1–V28 são anonimizadas via PCA — interpretabilidade intrínseca limitada
- Dataset coletado em dois dias de setembro de 2013 — validade temporal restrita
- Ausência de features contextuais (localização, histórico do cliente)

## Reprodutibilidade

`random_state=42` fixado em todos os componentes estocásticos. Python e versões de bibliotecas em `requirements.txt`.
