"""
Evaluate trained models on the held-out test set.
Computes F1, AUC-PR (primary), AUC-ROC, Precision, Recall.
Runs McNemar test between the two best models.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from src.features.build import get_X_y

TABLES_DIR = Path("article/tables")


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "auc_pr": average_precision_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict:
    """McNemar test between two classifiers (Dietterich, 1998)."""
    from scipy.stats import binom, chi2

    b = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    c = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "significant": False}

    if b + c < 25:
        # Exact two-sided binomial test
        stat = float(min(b, c))
        p_value = float(min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5)))
    else:
        # Chi-squared approximation with continuity correction
        stat = float((abs(b - c) - 1) ** 2 / (b + c))
        p_value = float(1 - chi2.cdf(stat, df=1))

    return {
        "b": b,
        "c": c,
        "statistic": round(stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }


def print_comparison_table(results: dict[str, dict]) -> None:
    print("\n=== Test Set Results ===")
    header = f"{'Model':<22} {'AUC-PR':>8} {'F1':>8} {'AUC-ROC':>9} {'Precision':>10} {'Recall':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<22} {m['auc_pr']:>8.4f} {m['f1']:>8.4f} "
            f"{m['auc_roc']:>9.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}"
        )


def save_comparison_table(results: dict[str, dict]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "AUC-PR": round(m["auc_pr"], 4),
            "F1": round(m["f1"], 4),
            "AUC-ROC": round(m["auc_roc"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    # Write markdown manually — no tabulate dependency needed
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    (TABLES_DIR / "model_comparison.md").write_text("\n".join([header, sep] + rows) + "\n")
    print(f"\nComparison table saved to {TABLES_DIR}/")


def run_evaluation(models: dict, test_df: pd.DataFrame) -> dict[str, dict]:
    print("\n=== Test Set Evaluation ===")
    X_test, y_test = get_X_y(test_df)
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)
        print(f"\n{name}:")
        print(classification_report(y_test, results[name]["y_pred"], digits=4))

    print_comparison_table(results)
    save_comparison_table(results)

    # McNemar between best two models ranked by AUC-PR
    sorted_models = sorted(results.items(), key=lambda x: x[1]["auc_pr"], reverse=True)
    if len(sorted_models) >= 2:
        name_a, res_a = sorted_models[0]
        name_b, res_b = sorted_models[1]
        mc = mcnemar_test(y_test.values, res_a["y_pred"], res_b["y_pred"])
        print(f"\n=== McNemar Test: {name_a} vs {name_b} ===")
        print(f"  b={mc['b']}, c={mc['c']}, stat={mc['statistic']}, p={mc['p_value']}")
        print(f"  Significant difference: {mc['significant']}")

        mc_df = pd.DataFrame([{
            "model_a": name_a,
            "model_b": name_b,
            **mc,
        }])
        mc_df.to_csv(TABLES_DIR / "mcnemar_test.csv", index=False)

    return results
