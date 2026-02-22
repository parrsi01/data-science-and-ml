"""Feature engineering and preprocessing builders for ML core pipeline."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic engineered features and clip extreme outliers.

    Engineered features:
    - ratio_ab = metric_a / (metric_b + 1e-9)
    - rolling_mean_a (sorted by timestamp then id)
    """

    required = {"id", "timestamp", "metric_a", "metric_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {sorted(missing)}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values(["timestamp", "id"], kind="mergesort").reset_index(drop=True)

    out["ratio_ab"] = out["metric_a"] / (out["metric_b"] + 1e-9)
    out["rolling_mean_a"] = (
        out["metric_a"].rolling(window=10, min_periods=1).mean().astype(float)
    )

    # Simple winsorize-style clipping using deterministic quantiles.
    for column in ["metric_a", "metric_b", "ratio_ab", "rolling_mean_a"]:
        lower = float(out[column].quantile(0.01))
        upper = float(out[column].quantile(0.99))
        out[column] = out[column].clip(lower=lower, upper=upper)

    return out


def build_preprocessor(
    categorical_cols: Iterable[str],
    numerical_cols: Iterable[str],
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for categorical + numerical features."""

    categorical_cols = list(categorical_cols)
    numerical_cols = list(numerical_cols)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_cols),
            ("numerical", numerical_pipeline, numerical_cols),
        ],
        remainder="drop",
    )
    return preprocessor
