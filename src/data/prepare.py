"""
Load raw creditcard.csv, perform stratified train/test split, save to data/processed/.
Raw data is never modified.
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DIR = Path("data/processed")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


def split_and_save(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    train_df = X_train.copy()
    train_df["Class"] = y_train

    test_df = X_test.copy()
    test_df["Class"] = y_test

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(
        f"Train: {len(train_df)} rows | "
        f"Fraud: {y_train.sum()} ({y_train.mean()*100:.3f}%)"
    )
    print(
        f"Test:  {len(test_df)} rows | "
        f"Fraud: {y_test.sum()} ({y_test.mean()*100:.3f}%)"
    )

    return train_df, test_df


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=== Data Preparation ===")
    df = load_raw()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return split_and_save(df)


if __name__ == "__main__":
    run()
