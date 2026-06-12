"""
Train 3 models with GridSearchCV + StratifiedKFold.
Results logged to experiments/experiments.csv.
Models serialized to experiments/<model_name>.joblib.
"""
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, average_precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from src.features.build import build_pipeline, get_X_y

RANDOM_STATE = 42
N_SPLITS = 5
EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_CSV = EXPERIMENTS_DIR / "experiments.csv"

SCORING = {
    "auc_pr": make_scorer(average_precision_score, response_method="predict_proba"),
    "f1": make_scorer(f1_score, zero_division=0),
    "auc_roc": make_scorer(roc_auc_score, response_method="predict_proba"),
}

CV = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def _fraud_ratio(y_train: pd.Series) -> float:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return neg / pos


def _model_configs(scale_pos_weight: float) -> list[dict]:
    return [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(
                class_weight="balanced",
                max_iter=200,
                random_state=RANDOM_STATE,
                solver="liblinear",
            ),
            "param_grid": {
                "classifier__C": [0.1, 1, 10],
            },
        },
        {
            "name": "RandomForest",
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_grid": {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [10, 20],
            },
        },
        {
            "name": "XGBoost",
            "estimator": XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                eval_metric="aucpr",
                verbosity=0,
                n_estimators=100,
            ),
            "param_grid": {
                "classifier__max_depth": [3, 6],
                "classifier__learning_rate": [0.05, 0.1],
            },
        },
    ]


def _log_experiment(row: dict) -> None:
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    df_new = pd.DataFrame([row])
    if EXPERIMENTS_CSV.exists():
        df_existing = pd.read_csv(EXPERIMENTS_CSV)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(EXPERIMENTS_CSV, index=False)


def train_all(
    train_df: pd.DataFrame,
) -> dict[str, GridSearchCV]:
    print("\n=== Model Training ===")
    X_train, y_train = get_X_y(train_df)
    scale_pos_weight = _fraud_ratio(y_train)
    configs = _model_configs(scale_pos_weight)

    fitted_models: dict[str, GridSearchCV] = {}

    for config in configs:
        name = config["name"]
        print(f"\n--- {name} ---")

        pipeline = build_pipeline(config["estimator"])

        search = GridSearchCV(
            pipeline,
            param_grid=config["param_grid"],
            scoring="average_precision",
            cv=CV,
            refit=True,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        best = search.best_estimator_

        cv_results = cross_validate(
            best,
            X_train,
            y_train,
            cv=CV,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=-1,
        )

        auc_pr_mean = cv_results["test_auc_pr"].mean()
        auc_pr_std = cv_results["test_auc_pr"].std()
        f1_mean = cv_results["test_f1"].mean()
        f1_std = cv_results["test_f1"].std()
        auc_roc_mean = cv_results["test_auc_roc"].mean()

        print(f"  Best params:   {search.best_params_}")
        print(f"  CV AUC-PR:     {auc_pr_mean:.4f} ± {auc_pr_std:.4f}")
        print(f"  CV F1:         {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"  CV AUC-ROC:    {auc_roc_mean:.4f}")

        model_path = EXPERIMENTS_DIR / f"{name}.joblib"
        joblib.dump(best, model_path)

        _log_experiment({
            "date": datetime.now().isoformat(timespec="seconds"),
            "model": name,
            "hyperparameters": json.dumps(search.best_params_),
            "cv_auc_pr_mean": round(auc_pr_mean, 6),
            "cv_auc_pr_std": round(auc_pr_std, 6),
            "cv_f1_mean": round(f1_mean, 6),
            "cv_f1_std": round(f1_std, 6),
            "cv_auc_roc_mean": round(auc_roc_mean, 6),
            "test_auc_pr": "",
            "test_f1": "",
            "test_auc_roc": "",
            "observations": "GridSearchCV best params",
        })

        fitted_models[name] = best

    return fitted_models


def update_test_metrics(name: str, metrics: dict) -> None:
    """Update test metrics for a model entry in experiments.csv."""
    if not EXPERIMENTS_CSV.exists():
        return
    df = pd.read_csv(EXPERIMENTS_CSV)
    mask = (df["model"] == name) & (df["test_auc_pr"] == "")
    if mask.any():
        idx = df[mask].index[-1]
        df.loc[idx, "test_auc_pr"] = round(metrics["auc_pr"], 6)
        df.loc[idx, "test_f1"] = round(metrics["f1"], 6)
        df.loc[idx, "test_auc_roc"] = round(metrics["auc_roc"], 6)
        df.to_csv(EXPERIMENTS_CSV, index=False)
