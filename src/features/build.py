"""
Build sklearn Pipelines with preprocessing.
StandardScaler applied only to Amount and Time (V1-V28 are already PCA-scaled).
Fit happens only on training data — no leakage.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

SCALE_FEATURES = ["Amount", "Time"]
V_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = V_FEATURES + SCALE_FEATURES
TARGET_COL = "Class"


def get_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: scale Amount/Time, pass-through V1-V28."""
    return ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), SCALE_FEATURES),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_pipeline(classifier) -> Pipeline:
    """Wrap a classifier in a preprocessing pipeline."""
    return Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", classifier),
    ])


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLS], df[TARGET_COL]


def feature_names_after_transform() -> list[str]:
    """Column order after ColumnTransformer (scaler cols first, then passthrough)."""
    return SCALE_FEATURES + V_FEATURES
