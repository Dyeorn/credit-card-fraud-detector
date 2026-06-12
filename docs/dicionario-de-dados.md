# Dicionário de Dados — creditcard.csv

**Fonte:** ULB Machine Learning Group — [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Período:** Setembro de 2013, titulares europeus de cartão de crédito (48 horas)  
**Tamanho:** 284.807 linhas × 31 colunas  
**Classe alvo:** `Class` (binário: 0 = legítima, 1 = fraude)

---

## Features

| Feature | Tipo | Descrição |
|---|---|---|
| `Time` | float64 | Segundos decorridos entre esta transação e a primeira do dataset. Varia de 0 a ~172.792 (≈48h). |
| `V1`–`V28` | float64 | Componentes principais (PCA) das features originais. Identidades ocultadas por confidencialidade. Já normalizados pela transformação PCA. |
| `Amount` | float64 | Valor monetário da transação em EUR. Varia de 0 a 25.691,16. Fortemente assimétrico à direita. |
| `Class` | int64 | **Variável alvo.** 0 = transação legítima, 1 = fraude. |

---

## Estatísticas Descritivas

| Feature | Min | Mediana | Média | Max | Nota |
|---|---|---|---|---|---|
| `Time` | 0 | 84.692 | 94.814 | 172.792 | Segundos |
| `Amount` | 0 | 22,00 | 88,35 | 25.691,16 | EUR, assimétrico |
| `Class` | 0 | 0 | 0,00173 | 1 | 492 fraudes (0,173%) |

---

## Distribuição da Classe Alvo

| Classe | Contagem | Proporção |
|---|---|---|
| 0 (Legítima) | 284.315 | 99,827% |
| 1 (Fraude) | 492 | 0,173% |

---

## Notas sobre Qualidade

- **Valores nulos:** Nenhum (dataset completo)
- **Tipos:** Todos numéricos (float64/int64), sem features categóricas
- **Features V:** Por serem PCA, são ortogonais entre si (correlação próxima de zero entre pares V_i, V_j)
- **Limitação principal:** Features originais são desconhecidas, impedindo interpretação de negócio direta
- **Viés temporal:** Dataset de 2013 pode não refletir padrões de fraude modernos
