"""Run institutional data quality gates with structured logs and reports."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import logging
import sys
from typing import Any

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - main path here
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:  # type: ignore[no-redef]
        return False

from data_engineering.ingest import (
    generate_cern_events,
    generate_flights,
    generate_humanitarian_shipments,
)
from data_quality.logging_config import get_logger, log_event
from data_quality.quality_metrics import compute_quality_metrics
from data_quality.reporting import REPORT_DIR, write_dataset_artifacts, write_summary_quality_report
from data_quality.schemas import CERNEventRow, FlightRow, HumanitarianShipmentRow
from data_quality.validators import enforce_domain_rules, validate_dataframe


LOGGER = get_logger("data_quality.runner")
SCHEMA_MODELS = {
    "flights": FlightRow,
    "humanitarian_shipments": HumanitarianShipmentRow,
    "cern_events": CERNEventRow,
}
PRIMARY_KEYS = {
    "flights": "flight_id",
    "humanitarian_shipments": "shipment_id",
    "cern_events": "event_id",
}
SCHEMA_VIOLATION_THRESHOLD = 0.01
DUPLICATE_RATE_THRESHOLD = 0.001


def _to_records(df: Any) -> list[dict[str, Any]]:
    if hasattr(df, "to_dict") and df.__class__.__name__ == "DataFrame":
        return [dict(row) for row in df.to_dict(orient="records")]
    if isinstance(df, list):
        return [dict(row) for row in df]
    raise TypeError("Expected pandas DataFrame or list of dictionaries")


def _concat_rows(*dfs: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for df in dfs:
        rows.extend(_to_records(df))
    return rows


def load_synthetic_datasets() -> dict[str, list[dict[str, Any]]]:
    """Load fresh synthetic datasets using Prompt 4 ingestion generators."""

    return {
        "flights": generate_flights(count=1000),
        "humanitarian_shipments": generate_humanitarian_shipments(count=1000),
        "cern_events": generate_cern_events(count=1000),
    }


def run_dataset_quality_checks(dataset_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run schema + domain validation, metrics, logs, and artifact generation."""

    raw_df = deepcopy(rows)
    schema_model = SCHEMA_MODELS[dataset_name]
    primary_id = PRIMARY_KEYS[dataset_name]

    valid_schema_df, schema_invalid_df = validate_dataframe(raw_df, schema_model, dataset_name)
    valid_domain_df, domain_invalid_df = enforce_domain_rules(valid_schema_df, dataset_name)

    metrics = compute_quality_metrics(
        dataset_name=dataset_name,
        raw_df=raw_df,
        valid_df=valid_domain_df,
        schema_invalid_df=schema_invalid_df,
        domain_invalid_df=domain_invalid_df,
        primary_id_col=primary_id,
    )

    invalid_combined = _concat_rows(schema_invalid_df, domain_invalid_df)
    artifacts = write_dataset_artifacts(
        dataset_name=dataset_name,
        metrics=metrics,
        invalid_rows_df=invalid_combined,
    )
    metrics["artifacts"] = artifacts

    log_event(
        LOGGER,
        logging.INFO,
        event="quality_metrics_computed",
        message=(
            f"schema_violation_rate={metrics['schema_violation_rate']:.6f} "
            f"duplicate_rate={metrics['duplicate_rate']:.6f}"
        ),
        dataset_name=dataset_name,
        row_count=metrics["row_counts"]["total"],
    )
    return metrics


def _gate_failed(metrics: dict[str, Any]) -> bool:
    return (
        metrics["schema_violation_rate"] > SCHEMA_VIOLATION_THRESHOLD
        or metrics["duplicate_rate"] > DUPLICATE_RATE_THRESHOLD
    )


def evaluate_quality_gate(
    datasets: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run the quality gate across datasets and return the aggregated summary."""

    load_dotenv()
    dataset_rows = datasets or load_synthetic_datasets()
    log_event(
        LOGGER,
        logging.INFO,
        event="quality_gate_start",
        message="Starting quality gate run",
        dataset_name="all",
        row_count=sum(len(rows) for rows in dataset_rows.values()),
    )

    per_dataset: dict[str, Any] = {}
    failed_datasets: list[str] = []
    for dataset_name, rows in dataset_rows.items():
        metrics = run_dataset_quality_checks(dataset_name, rows)
        per_dataset[dataset_name] = metrics
        if _gate_failed(metrics):
            failed_datasets.append(dataset_name)

    summary = {
        "gate_thresholds": {
            "schema_violation_rate": SCHEMA_VIOLATION_THRESHOLD,
            "duplicate_rate": DUPLICATE_RATE_THRESHOLD,
        },
        "datasets": per_dataset,
        "failed_datasets": failed_datasets,
        "passed": not failed_datasets,
    }

    summary_path = write_summary_quality_report(summary)
    summary["summary_report_path"] = summary_path
    log_event(
        LOGGER,
        logging.INFO if summary["passed"] else logging.ERROR,
        event="quality_gate_complete",
        message="Quality gate passed" if summary["passed"] else "Quality gate failed",
        dataset_name="all",
        row_count=sum(m["row_counts"]["total"] for m in per_dataset.values()),
        error_code=None if summary["passed"] else "QUALITY_GATE_FAIL",
    )
    return summary


def print_quality_summary(summary: dict[str, Any]) -> None:
    """Print a readable PASS/FAIL summary using Rich when available."""

    try:  # pragma: no cover - UI path
        from rich.console import Console  # type: ignore[import-not-found]
        from rich.table import Table  # type: ignore[import-not-found]

        console = Console()
        title = "[bold green]QUALITY GATE PASS[/bold green]" if summary["passed"] else "[bold red]QUALITY GATE FAIL[/bold red]"
        console.print(title)
        table = Table(title="Dataset Quality Summary")
        table.add_column("Dataset")
        table.add_column("Total")
        table.add_column("Valid")
        table.add_column("Schema Rate")
        table.add_column("Duplicate Rate")
        for name, metrics in summary["datasets"].items():
            counts = metrics["row_counts"]
            table.add_row(
                name,
                str(counts["total"]),
                str(counts["valid"]),
                f"{metrics['schema_violation_rate']:.4f}",
                f"{metrics['duplicate_rate']:.4f}",
            )
        console.print(table)
        console.print(f"Summary report: {summary['summary_report_path']}")
    except Exception:
        print("QUALITY GATE PASS" if summary["passed"] else "QUALITY GATE FAIL")
        for name, metrics in summary["datasets"].items():
            counts = metrics["row_counts"]
            print(
                f"{name}: total={counts['total']} valid={counts['valid']} "
                f"schema_rate={metrics['schema_violation_rate']:.4f} "
                f"duplicate_rate={metrics['duplicate_rate']:.4f}"
            )
        print(f"Summary report: {summary['summary_report_path']}")


def main() -> int:
    """CLI entry point for ``make quality``."""

    summary = evaluate_quality_gate()
    print_quality_summary(summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
