"""Chunked processing demos for out-of-core style workflows.

Prefers pandas when available, but remains runnable with pure-Python list-of-dict
records in offline environments.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import csv
import random
from typing import Any, Callable, Iterable

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - main path in this VM
    pd = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV_PATH = PROJECT_ROOT / "datasets" / "large_synthetic.csv"
PROCESSED_CSV_PATH = PROJECT_ROOT / "datasets" / "large_synthetic_processed.csv"


def _records_to_like(original: Any, records: list[dict[str, Any]]) -> Any:
    if pd is not None and original is not None and original.__class__.__name__ == "DataFrame":
        return pd.DataFrame(records)
    if pd is not None and original == "pandas":
        return pd.DataFrame(records)
    return records


def _to_records(data: Any) -> list[dict[str, Any]]:
    if pd is not None and data.__class__.__name__ == "DataFrame":
        return [dict(row) for row in data.to_dict(orient="records")]
    if isinstance(data, list):
        return [dict(row) for row in data]
    raise TypeError("Expected pandas DataFrame or list of dictionaries")


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def read_csv_records(path: str | Path) -> list[dict[str, Any]]:
    """Read CSV rows and coerce common numeric/boolean columns."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            out: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    out[key] = None
                    continue
                if key in {"id"}:
                    out[key] = int(value)
                elif key in {"metric_a", "metric_b", "ratio_a_b", "rolling_mean_metric_a"}:
                    out[key] = float(value)
                elif key in {"label_rare_event", "anomaly_flag"}:
                    out[key] = value.lower() in {"1", "true", "t", "yes"}
                else:
                    out[key] = value
            rows.append(out)
    return rows


def generate_large_synthetic_dataset(
    n_rows: int = 20_000,
    seed: int = 42,
    save_csv_path: str | Path | None = None,
) -> Any:
    """Generate a synthetic institutional-style dataset and save it to CSV.

    Columns: ``id``, ``timestamp``, ``metric_a``, ``metric_b``, ``category``,
    ``label_rare_event``.

    Returns a pandas DataFrame when pandas is available, otherwise a list of
    dictionaries.
    """

    rng = random.Random(seed)
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    categories = ["aviation", "humanitarian", "scientific", "supply_chain"]
    records: list[dict[str, Any]] = []
    for idx in range(n_rows):
        metric_a = rng.gauss(100.0, 15.0)
        metric_b = rng.gauss(50.0, 8.0)
        rare = rng.random() < 0.02
        if rare:
            metric_a += rng.uniform(30.0, 80.0)
        records.append(
            {
                "id": idx + 1,
                "timestamp": (base_time + timedelta(seconds=idx * 15)).isoformat(),
                "metric_a": round(metric_a, 4),
                "metric_b": round(max(0.01, metric_b), 4),
                "category": rng.choice(categories),
                "label_rare_event": bool(rare),
            }
        )

    output_path = Path(save_csv_path) if save_csv_path else RAW_CSV_PATH
    _write_csv(records, output_path)
    return _records_to_like("pandas" if pd is not None else None, records)


def add_features(chunk: Any) -> Any:
    """Example chunk transform adding rolling mean, ratio metrics, and anomalies."""

    records = _to_records(chunk)
    transformed: list[dict[str, Any]] = []
    rolling_window: list[float] = []

    for row in records:
        metric_a = float(row["metric_a"])
        metric_b = max(0.0001, float(row["metric_b"]))
        rolling_window.append(metric_a)
        if len(rolling_window) > 5:
            rolling_window.pop(0)
        rolling_mean = sum(rolling_window) / len(rolling_window)
        ratio = metric_a / metric_b
        anomaly_flag = bool(ratio > 3.0 or metric_a > rolling_mean + 30.0)

        new_row = dict(row)
        new_row["rolling_mean_metric_a"] = round(rolling_mean, 4)
        new_row["ratio_a_b"] = round(ratio, 6)
        new_row["anomaly_flag"] = anomaly_flag
        transformed.append(new_row)

    return _records_to_like(chunk, transformed)


def process_in_chunks(df: Any, chunk_size: int, fn: Callable[[Any], Any]) -> Any:
    """Apply ``fn`` to each chunk and concatenate the results."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    records = _to_records(df)
    output_records: list[dict[str, Any]] = []
    for start in range(0, len(records), chunk_size):
        chunk_records = records[start : start + chunk_size]
        chunk_like = _records_to_like(df, chunk_records)
        processed_chunk = fn(chunk_like)
        output_records.extend(_to_records(processed_chunk))
    return _records_to_like(df, output_records)


def save_processed_output(processed: Any, path: str | Path | None = None) -> Path:
    """Save processed chunk output to CSV."""

    target = Path(path) if path else PROCESSED_CSV_PATH
    _write_csv(_to_records(processed), target)
    return target


def run_chunking_pipeline(
    *,
    n_rows: int = 20_000,
    seed: int = 42,
    chunk_size: int = 1000,
    raw_csv_path: str | Path | None = None,
    processed_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate dataset, process in chunks, save raw/processed CSV, and summarize."""

    df = generate_large_synthetic_dataset(n_rows=n_rows, seed=seed, save_csv_path=raw_csv_path)
    processed = process_in_chunks(df, chunk_size=chunk_size, fn=add_features)
    processed_path = save_processed_output(processed, processed_csv_path)
    raw_path = Path(raw_csv_path) if raw_csv_path else RAW_CSV_PATH
    row_count = len(_to_records(processed))
    return {
        "backend": "pandas" if pd is not None else "python-fallback",
        "raw_csv": str(raw_path),
        "processed_csv": str(processed_path),
        "row_count": row_count,
        "chunk_size": chunk_size,
    }


def main() -> None:
    """CLI entry point for ``make scale-generate``."""

    summary = run_chunking_pipeline()
    print(summary)


if __name__ == "__main__":
    main()
