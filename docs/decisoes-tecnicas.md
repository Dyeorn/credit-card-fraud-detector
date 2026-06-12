# Decisões Técnicas e Justificativas Metodológicas

## 1. Métrica Primária: AUC-PR (Average Precision)

**Decisão:** AUC-PR como métrica primária, não AUC-ROC.

**Justificativa:** Com 0.17% de fraudes, a classe negativa (legítima) domina completamente as métricas baseadas em TN. AUC-ROC pode ser artificialmente alta (> 0.95) mesmo para modelos ruins, pois a grande quantidade de verdadeiros negativos infla o TPR. A curva Precision-Recall foca no comportamento do modelo na classe minoritária, que é exatamente onde o custo do erro é maior.

## 2. Tratamento do Desbalanceamento: class_weight vs SMOTE

**Decisão:** `class_weight='balanced'` para LR e RF; `scale_pos_weight` para XGBoost. SMOTE não utilizado.

**Justificativa:** 
- SMOTE gera amostras sintéticas que precisam ser criadas *dentro* de cada fold de validação cruzada para evitar vazamento de dados. Isso torna o pipeline mais complexo (necessidade de `imbalanced-learn Pipeline`) sem garantia de melhoria significativa em dados com PCA já aplicado.
- `class_weight='balanced'` ajusta o peso das classes diretamente na função de perda, sem alterar a distribuição dos dados de validação.
- Referência: He & Garcia (2009) mostram que re-amostragem dentro de CV é obrigatória para evitar estimativas otimistas.

## 3. Prevenção de Vazamento de Dados

**Decisão:** `StandardScaler` embutido em `sklearn.Pipeline` + `ColumnTransformer`.

**Justificativa:** Ajustar o scaler em todos os dados antes do split introduziria vazamento de informação estatística do conjunto de teste para o treino. O pipeline do sklearn garante que `fit()` ocorre apenas nos dados de treino de cada fold.

**Implementação:**
- `train_test_split` estratificado ocorre antes de qualquer transformação
- `GridSearchCV` usa apenas o conjunto de treino
- Conjunto de teste usado uma única vez para avaliação final

## 4. Validação: StratifiedKFold (k=5)

**Decisão:** 5-fold estratificado com `shuffle=True`.

**Justificativa:** Com apenas 492 fraudes no dataset completo (≈394 no treino), cada fold precisa conter exemplos positivos suficientes para estimativa confiável. A estratificação garante proporção idêntica de fraudes em todos os folds. K=5 é o padrão recomendado por Kohavi (1995) como equilíbrio entre viés e variância na estimativa.

## 5. Modelos Escolhidos

| Modelo | Justificativa |
|---|---|
| Logistic Regression | Baseline linear, interpretabilidade total via coeficientes, rápido |
| Random Forest | Ensemble não-linear, feature importance nativa, robusto a outliers |
| XGBoost | Estado-da-arte em dados tabulares, gradient boosting com regularização |

**Trade-off Interpretabilidade vs. Desempenho:** LR é o mais interpretável mas possivelmente o menos poderoso. XGBoost é o mais poderoso mas requer análise de importância de features para interpretação. RF equilibra os dois.

## 6. Features: V1–V28 sem Transformação

**Decisão:** Features V1–V28 usadas diretamente sem transformação adicional.

**Justificativa:** São componentes de PCA já normalizados. Aplicar StandardScaler sobre features PCA é redundante. Apenas `Amount` e `Time` recebem StandardScaler por terem escalas originais muito diferentes das demais.

## 7. Teste Estatístico: McNemar

**Decisão:** Teste de McNemar entre os dois melhores modelos.

**Justificativa:** Para comparar classificadores no *mesmo* conjunto de teste, o teste de McNemar é o mais adequado pois considera as predições corretas/incorretas de cada par (teste pareado). É mais poderoso que t-test em classificação porque considera a variância das predições individuais. Recomendado por Dietterich (1998).
