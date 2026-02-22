"""Operational drift snapshot monitor reusing evaluation drift utilities."""

from __future__ import annotations

from pathlib import Path
import json
import logging
from typing import Any

import pandas as pd

try:
    from src.data_quality.logging_config import get_logger, log_event  # type: ignore[import-not-found]
    from src.evaluation.drift import categorical_drift_report, numeric_drift_report  # type: ignore[import-not-found]
    from src.ml_advanced.features import add_advanced_features  # type: ignore[import-not-found]
    from src.ml_core.preprocess import add_feature_engineering  # type: ignore[import-not-found]
except Exception:
    from data_quality.logging_config import get_logger, log_event
    from evaluation.drift import categorical_drift_report, numeric_drift_report
    from ml_advanced.features import add_advanced_features
    from ml_core.preprocess import add_feature_engineering


LOGGER = get_logger("ops_anomaly.drift")


def _load_training_reference(limit_rows: int) -> pd.DataFrame:
    path = Path("datasets/ml_core_synth.csv")
    if not path.exists():
        raise FileNotFoundError(f"Training reference dataset not found: {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = add_feature_engineering(df)
    df = add_advanced_features(df)
    return df.tail(limit_rows).reset_index(drop=True)


def run_drift_monitor(
    recent_engineered_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compare recent batch to training reference and flag drift."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_df = _load_training_reference(limit_rows=max(len(recent_engineered_df), 500))

    numeric_cols = [
        "metric_a",
        "metric_b",
        "ratio_ab",
        "rolling_mean_a",
        "lag_metric_a",
        "rolling_std_a",
        "interaction_a_b",
    ]
    cat_cols = ["category"]
    report = {}
    report.update(numeric_drift_report(reference_df, recent_engineered_df, numeric_cols))
    report.update(categorical_drift_report(reference_df, recent_engineered_df, cat_cols))

    max_ks = 0.0
    max_ks_feature = None
    for col, payload in report.get("numeric", {}).items():
        ks = float(payload.get("ks_statistic", 0.0))
        if ks > max_ks:
            max_ks = ks
            max_ks_feature = col
    threshold = float(config.get("drift", {}).get("threshold_ks_stat", 0.1))
    drift_flag = max_ks > threshold

    snapshot = {
        "drift_flag": bool(drift_flag),
        "threshold_ks_stat": threshold,
        "max_ks_statistic": float(max_ks),
        "max_ks_feature": max_ks_feature,
        "reference_rows": int(len(reference_df)),
        "recent_rows": int(len(recent_engineered_df)),
        "report": report,
    }
    path = output_dir / "drift_snapshot.json"
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

    log_event(
        LOGGER,
        logging.INFO if not drift_flag else logging.WARNING,
        event="drift_snapshot",
        message=f"drift_flag={drift_flag} max_ks={max_ks:.4f} feature={max_ks_feature}",
        dataset_name="flights",
        row_count=int(len(recent_engineered_df)),
        error_code="DRIFT_FLAG" if drift_flag else None,
    )
    return snapshot
