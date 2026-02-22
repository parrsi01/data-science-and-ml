"""Report artifact generation for institutional data quality runs."""

from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"


def _to_records(df: Any) -> list[dict[str, Any]]:
    if hasattr(df, "to_dict") and df.__class__.__name__ == "DataFrame":
        return [dict(row) for row in df.to_dict(orient="records")]
    if isinstance(df, list):
        return [dict(row) for row in df]
    raise TypeError("Expected pandas DataFrame or list of dictionaries")


def write_quality_report(metrics: dict[str, Any], output_path: str) -> str:
    """Write a JSON quality report artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return str(path)


def write_invalid_rows_csv(invalid_df: Any, output_path: str) -> str:
    """Write invalid rows to CSV for human review and auditability."""

    rows = _to_records(invalid_df)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # Keep artifact reproducible even when empty.
        path.write_text("", encoding="utf-8")
        return str(path)

    fieldnames: list[str] = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return str(path)


def write_dataset_artifacts(
    *,
    dataset_name: str,
    metrics: dict[str, Any],
    invalid_rows_df: Any,
    report_dir: Path | None = None,
) -> dict[str, str]:
    """Write per-dataset JSON and invalid-row CSV artifacts."""

    base_dir = report_dir or REPORT_DIR
    json_path = base_dir / f"{dataset_name}_quality_report.json"
    csv_path = base_dir / f"{dataset_name}_invalid_rows.csv"
    return {
        "quality_report_json": write_quality_report(metrics, str(json_path)),
        "invalid_rows_csv": write_invalid_rows_csv(invalid_rows_df, str(csv_path)),
    }


def write_summary_quality_report(summary_metrics: dict[str, Any], output_path: str | None = None) -> str:
    """Write aggregated quality report JSON."""

    path = Path(output_path) if output_path else (REPORT_DIR / "summary_quality_report.json")
    return write_quality_report(summary_metrics, str(path))

