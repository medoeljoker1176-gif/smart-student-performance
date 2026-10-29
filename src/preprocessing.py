"""
preprocessing.py
----------------
Loading, cleaning, data-quality report and the scikit-learn preprocessor.

Data-leakage rule: everything that LEARNS from data (imputation values, scaling
mean/std, one-hot categories) lives inside a Pipeline that is fitted on the
training rows only. `clean_data` only does row-wise fixes that need no statistics.
"""
import warnings
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (CATEGORICAL_FEATURES, DATA_PATH, FEATURES, INPUT_LIMITS, NUMERIC_FEATURES,
                        POSITIVE_LABEL, TARGET)
from src.validation import DataError, validate_dataframe

# Some datasets (UCI) have no sleep/participation data; the imputer then skips those columns on purpose.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and check its structure. Raises DataError with a friendly message."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"Dataset not found at '{path}'. Run `python generate_dataset.py` first.")
    try:
        df = pd.read_csv(path)
    except Exception as exc:                      # unreadable / corrupt file
        raise DataError(f"Could not read the dataset: {exc}") from exc
    problems = validate_dataframe(df)
    if problems:
        raise DataError("Dataset problem(s): " + " | ".join(problems))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, rows without a label and impossible values.

    Missing *feature* values are NOT filled here; the Pipeline does it after the
    train/test split so the test set never influences the fill values.
    """
    df = df.drop_duplicates().dropna(subset=[TARGET]).copy()

    # Do not silently clip impossible values: that changes the meaning of the
    # source data. Drop rows outside the documented hard limits instead.
    valid = pd.Series(True, index=df.index)
    for col, (low, high) in INPUT_LIMITS.items():
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid &= numeric.isna() | numeric.between(low, high)
    study = pd.to_numeric(df["Study_Hours"], errors="coerce")
    sleep = pd.to_numeric(df["Sleep_Hours"], errors="coerce")
    valid &= study.isna() | sleep.isna() | ((study + sleep) <= 24)
    return df.loc[valid].reset_index(drop=True)


def split_xy(df: pd.DataFrame):
    """Features X and binary target y (Pass = 1, Fail = 0)."""
    return df[FEATURES].copy(), (df[TARGET] == POSITIVE_LABEL).astype(int)


def build_preprocessor() -> ColumnTransformer:
    """numeric: median-impute -> standardise | categorical: mode-impute -> one-hot."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])


def data_quality_report(df: pd.DataFrame) -> dict:
    """Numbers shown on the Data Quality tab (computed from the raw file)."""
    missing = df[FEATURES + [TARGET]].isna().sum()
    n = len(df)
    return {
        "n_records": n,
        "n_features": len(FEATURES),
        "missing_total": int(missing.sum()),
        "missing_by_column": missing[missing > 0].to_dict(),
        "missing_pct": float(missing.sum() / (n * len(FEATURES)) * 100) if n else 0.0,
        "rows_with_missing": int(df[FEATURES].isna().any(axis=1).sum()),
        "duplicates": int(df.duplicated().sum()),
        "class_counts": df[TARGET].value_counts().to_dict(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
