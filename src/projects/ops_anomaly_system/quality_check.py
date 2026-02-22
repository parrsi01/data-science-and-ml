"""Operational quality checks reusing institutional data_quality modules."""

from __future__ import annotations

from pathlib import Path
import json
import logging
from typing import Any

import pandas as pd

try:
    from src.data_quality.logging_config import get_logger, log_event  # type: ignore[import-not-found]
    from src.data_quality.quality_metrics import compute_quality_metrics  # type: ignore[import-not-found]
    from src.data_quality.reporting import write_invalid_rows_csv  # type: ignore[import-not-found]
    from src.data_quality.schemas import FlightRow  # type: ignore[import-not-found]
    from src.data_quality.validators import enforce_domain_rules, validate_dataframe  # type: ignore[import-not-found]
except Exception:
    from data_quality.logging_config import get_logger, log_event
    from data_quality.quality_metrics import compute_quality_metrics
    from data_quality.reporting import write_invalid_rows_csv
    from data_quality.schemas import FlightRow
    from data_quality.validators import enforce_domain_rules, validate_dataframe


LOGGER = get_logger("ops_anomaly.quality")


def run_quality_check(
    raw_flights_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate flights batch and write operational quality snapshot."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(config.get("data_source", {}).get("table", "flights"))

    valid_schema_df, schema_invalid_df = validate_dataframe(raw_flights_df, FlightRow, dataset_name)
    valid_domain_df, domain_invalid_df = enforce_domain_rules(valid_schema_df, dataset_name)

    metrics = compute_quality_metrics(
        dataset_name=dataset_name,
        raw_df=raw_flights_df,
        valid_df=valid_domain_df,
        schema_invalid_df=schema_invalid_df,
        domain_invalid_df=domain_invalid_df,
        primary_id_col="flight_id",
    )

    max_missing_rate = max(metrics["missing_rate_per_column"].values(), default=0.0)
    metrics["max_missing_rate"] = float(max_missing_rate)
    thresholds = config.get("quality", {})
    schema_threshold = float(thresholds.get("max_schema_violation_rate", 0.01))
    missing_threshold = float(thresholds.get("max_missing_rate", 0.05))
    quality_pass = (
        metrics["schema_violation_rate"] <= schema_threshold and max_missing_rate <= missing_threshold
    )

    invalid_combined = pd.concat(
        [pd.DataFrame(schema_invalid_df), pd.DataFrame(domain_invalid_df)],
        ignore_index=True,
    )
    invalid_csv = output_dir / "quality_invalid_rows.csv"
    write_invalid_rows_csv(invalid_combined, str(invalid_csv))

    snapshot = {
        "quality_pass": bool(quality_pass),
        "thresholds": {
            "max_schema_violation_rate": schema_threshold,
            "max_missing_rate": missing_threshold,
        },
        "metrics": metrics,
        "artifacts": {"invalid_rows_csv": str(invalid_csv)},
    }
    snapshot_path = output_dir / "quality_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    snapshot["artifacts"]["quality_snapshot_json"] = str(snapshot_path)

    level = logging.INFO if quality_pass else logging.ERROR
    log_event(
        LOGGER,
        level,
        event="quality_snapshot",
        message=(
            f"quality_pass={quality_pass} schema_violation_rate={metrics['schema_violation_rate']:.4f} "
            f"max_missing_rate={max_missing_rate:.4f}"
        ),
        dataset_name=dataset_name,
        row_count=int(metrics["row_counts"]["total"]),
        error_code=None if quality_pass else "QUALITY_FAIL",
    )
    return snapshot
