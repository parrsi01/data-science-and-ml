"""Anomaly inference using the institutional advanced XGBoost model bundle."""

from __future__ import annotations

from pathlib import Path
import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from src.data_quality.logging_config import get_logger, log_event  # type: ignore[import-not-found]
    from src.ml_advanced.features import add_advanced_features  # type: ignore[import-not-found]
    from src.ml_core.preprocess import add_feature_engineering  # type: ignore[import-not-found]
except Exception:
    from data_quality.logging_config import get_logger, log_event
    from ml_advanced.features import add_advanced_features
    from ml_core.preprocess import add_feature_engineering


LOGGER = get_logger("ops_anomaly.inference")


def _load_threshold(default: float = 0.5) -> float:
    summary_path = Path("reports/evaluation/evaluation_summary.json")
    if not summary_path.exists():
        return default
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return float(payload.get("best_threshold", default))
    except Exception:
        return default


def _prepare_features_for_model(model_input_df: pd.DataFrame) -> pd.DataFrame:
    base = add_feature_engineering(model_input_df)
    advanced = add_advanced_features(base)
    return advanced


def run_inference(
    raw_flights_df: pd.DataFrame,
    model_input_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run anomaly inference and persist predictions + summary metrics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(config["model"]["model_path"])
    bundle = joblib.load(model_path)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    categorical_cols = list(bundle["categorical_cols"])
    numerical_cols = list(bundle["numerical_cols"])

    engineered = _prepare_features_for_model(model_input_df)
    X = engineered[categorical_cols + numerical_cols].copy()
    X_trans = preprocessor.transform(X)
    y_proba = model.predict_proba(X_trans)[:, 1]
    threshold = _load_threshold(default=0.5)
    y_pred = (y_proba >= threshold).astype(int)

    output_df = raw_flights_df.copy().reset_index(drop=True)
    output_df["anomaly_probability"] = y_proba.astype(float)
    output_df["anomaly_label"] = y_pred.astype(int)
    output_df["model_threshold"] = float(threshold)
    output_df = output_df.sort_values(
        ["anomaly_probability", "scheduled_dep", "flight_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    anomaly_rate = float(output_df["anomaly_label"].mean()) if len(output_df) else 0.0
    top_anomalies = output_df.head(10).copy()

    csv_path = output_dir / "anomaly_results.csv"
    output_df.to_csv(csv_path, index=False)
    top_json = output_dir / "top_anomalies.json"
    top_json.write_text(
        json.dumps(top_anomalies.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )

    metrics = {
        "rows_scored": int(len(output_df)),
        "anomaly_rate": anomaly_rate,
        "threshold": float(threshold),
        "model_path": str(model_path),
        "top_anomalies_count": int(len(top_anomalies)),
        "artifacts": {
            "anomaly_results_csv": str(csv_path),
            "top_anomalies_json": str(top_json),
        },
    }
    metrics_path = output_dir / "inference_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    metrics["artifacts"]["inference_metrics_json"] = str(metrics_path)

    log_event(
        LOGGER,
        logging.INFO,
        event="inference_complete",
        message=f"rows_scored={len(output_df)} anomaly_rate={anomaly_rate:.4f} threshold={threshold:.2f}",
        dataset_name="flights",
        row_count=int(len(output_df)),
    )
    return {
        "metrics": metrics,
        "predictions_df": output_df,
        "top_anomalies_df": top_anomalies,
        "engineered_features_df": engineered,
    }
