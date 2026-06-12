"""
Pipeline completo de detecção de fraude em cartões de crédito.

Uso:
    uv run python main.py
"""
import pandas as pd
from src.data.prepare import run as prepare_data
from src.models.train import train_all, update_test_metrics
from src.evaluation.metrics import run_evaluation
from src.visualization.plots import generate_all
from src.features.build import get_X_y


def main() -> None:
    print("=" * 55)
    print("  Credit Card Fraud Detection — ML Pipeline")
    print("=" * 55)

    # 1. Split e salvar dados processados
    train_df, test_df = prepare_data()

    # 2. Treinar modelos com GridSearchCV + CV
    models = train_all(train_df)

    # 3. Avaliar no conjunto de teste (usado apenas aqui)
    results = run_evaluation(models, test_df)

    # 4. Atualizar métricas de teste no log de experimentos
    for name, metrics in results.items():
        update_test_metrics(name, {
            "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"],
            "auc_roc": metrics["auc_roc"],
        })

    # 5. Gerar figuras para o artigo
    raw_df = pd.read_csv("data/raw/creditcard.csv")
    _, y_test = get_X_y(test_df)
    generate_all(raw_df, models, results, y_test)

    print("\n" + "=" * 55)
    print("  Pipeline concluído com sucesso!")
    print("  Experimentos: experiments/experiments.csv")
    print("  Figuras:      article/figures/")
    print("  Tabelas:      article/tables/")
    print("=" * 55)


if __name__ == "__main__":
    main()
