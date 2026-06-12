"""
Generate and save all figures for the article.
All figures saved to article/figures/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    ConfusionMatrixDisplay,
)

matplotlib.use("Agg")  # non-interactive backend for script execution

FIGURES_DIR = Path("article/figures")
STYLE = "seaborn-v0_8-whitegrid"
DPI = 150
PALETTE = {"0": "#4878CF", "1": "#D65F5F"}


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["Class"].value_counts()
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            ["Legítima", "Fraude"],
            counts.values,
            color=["#4878CF", "#D65F5F"],
            edgecolor="white",
        )
        ax.set_title("Distribuição das Classes", fontsize=13)
        ax.set_ylabel("Número de Transações")
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1000,
                f"{val:,}\n({val/len(df)*100:.2f}%)",
                ha="center", fontsize=10,
            )
    _save(fig, "class_distribution.png")


def plot_amount_distribution(df: pd.DataFrame) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        for cls, label, color in [(0, "Legítima", "#4878CF"), (1, "Fraude", "#D65F5F")]:
            vals = df.loc[df["Class"] == cls, "Amount"]
            ax.hist(np.log1p(vals), bins=50, alpha=0.6, label=label, color=color)
        ax.set_title("Distribuição de log(Amount+1) por Classe", fontsize=13)
        ax.set_xlabel("log(Amount + 1)")
        ax.set_ylabel("Frequência")
        ax.legend()
    _save(fig, "amount_distribution.png")


def plot_time_distribution(df: pd.DataFrame) -> None:
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, cls, label, color in [
            (axes[0], 0, "Legítima", "#4878CF"),
            (axes[1], 1, "Fraude", "#D65F5F"),
        ]:
            vals = df.loc[df["Class"] == cls, "Time"]
            ax.hist(vals / 3600, bins=48, color=color, alpha=0.8)
            ax.set_title(f"Time — {label}", fontsize=11)
            ax.set_xlabel("Horas desde primeira transação")
            ax.set_ylabel("Frequência")
        fig.suptitle("Distribuição Temporal por Classe", fontsize=13)
        fig.tight_layout()
    _save(fig, "time_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    v_cols = [f"V{i}" for i in range(1, 29)]
    corr_with_class = (
        df[v_cols + ["Class"]].corr()["Class"]
        .drop("Class")
        .abs()
        .sort_values(ascending=False)
    )
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        corr_with_class.head(15).plot(
            kind="barh", ax=ax, color="#4878CF", edgecolor="white"
        )
        ax.invert_yaxis()
        ax.set_title("Top 15 Features por Correlação Absoluta com Fraude", fontsize=13)
        ax.set_xlabel("|Correlação de Pearson| com Class")
    _save(fig, "feature_correlation.png")


def plot_precision_recall_curves(results: dict, y_test: pd.Series) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#4878CF", "#D65F5F", "#6ACC65"]
        for (name, res), color in zip(results.items(), colors):
            prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
            ap = res["auc_pr"]
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, lw=2)
        baseline = y_test.mean()
        ax.axhline(baseline, linestyle="--", color="gray", alpha=0.7, label=f"Baseline ({baseline:.3f})")
        ax.set_title("Curvas Precision-Recall (Conjunto de Teste)", fontsize=13)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
    _save(fig, "precision_recall_curves.png")


def plot_roc_curves(results: dict, y_test: pd.Series) -> None:
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ["#4878CF", "#D65F5F", "#6ACC65"]
        for (name, res), color in zip(results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            auc = res["auc_roc"]
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatório")
        ax.set_title("Curvas ROC (Conjunto de Teste)", fontsize=13)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
    _save(fig, "roc_curves.png")


def plot_feature_importance(models: dict) -> None:
    for name in ["RandomForest", "XGBoost"]:
        if name not in models:
            continue
        model = models[name]
        clf = model.named_steps["classifier"]
        importances = clf.feature_importances_

        from src.features.build import SCALE_FEATURES, V_FEATURES
        all_features = SCALE_FEATURES + V_FEATURES

        feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(8, 5))
            feat_imp.head(15).plot(kind="barh", ax=ax, color="#4878CF", edgecolor="white")
            ax.invert_yaxis()
            ax.set_title(f"Top 15 Features — {name}", fontsize=13)
            ax.set_xlabel("Importância")
        _save(fig, f"feature_importance_{name.lower()}.png")


def plot_confusion_matrices(results: dict) -> None:
    n = len(results)
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, (name, res) in zip(axes, results.items()):
            disp = ConfusionMatrixDisplay(
                confusion_matrix=res["confusion_matrix"],
                display_labels=["Legítima", "Fraude"],
            )
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(name, fontsize=11)
        fig.suptitle("Matrizes de Confusão (Conjunto de Teste)", fontsize=13, y=1.02)
        fig.tight_layout()
    _save(fig, "confusion_matrices.png")


def generate_all(
    raw_df: pd.DataFrame,
    models: dict,
    results: dict,
    y_test: pd.Series,
) -> None:
    print("\n=== Generating Figures ===")
    plot_class_distribution(raw_df)
    plot_amount_distribution(raw_df)
    plot_time_distribution(raw_df)
    plot_correlation_heatmap(raw_df)
    plot_precision_recall_curves(results, y_test)
    plot_roc_curves(results, y_test)
    plot_feature_importance(models)
    plot_confusion_matrices(results)
    print("All figures generated.")
