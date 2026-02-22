"""Advanced feature engineering for the institutional ML pipeline."""

from __future__ import annotations

import pandas as pd


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic advanced features for modeling.

    Features added:
    - lag_metric_a: previous metric_a by stable time ordering
    - rolling_std_a: rolling standard deviation of metric_a
    - interaction_a_b: metric_a * metric_b
    - winsorized clipping on numerical columns
    """

    required = {"id", "timestamp", "metric_a", "metric_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values(["timestamp", "id"], kind="mergesort").reset_index(drop=True)

    out["lag_metric_a"] = out["metric_a"].shift(1)
    out["lag_metric_a"] = out["lag_metric_a"].fillna(out["metric_a"].iloc[0])

    out["rolling_std_a"] = (
        out["metric_a"].rolling(window=10, min_periods=2).std().fillna(0.0).astype(float)
    )

    out["interaction_a_b"] = out["metric_a"] * out["metric_b"]

    # Clip extremes on engineered + base numeric columns deterministically.
    for col in ["metric_a", "metric_b", "ratio_ab", "rolling_mean_a", "lag_metric_a", "rolling_std_a", "interaction_a_b"]:
        if col not in out.columns:
            continue
        lower = float(out[col].quantile(0.01))
        upper = float(out[col].quantile(0.99))
        out[col] = out[col].clip(lower=lower, upper=upper)

    return out
