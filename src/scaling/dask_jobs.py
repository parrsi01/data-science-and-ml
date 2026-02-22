"""Dask parallelism demo with offline-safe fallback behavior."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Any

try:  # pragma: no cover - optional dependency
    import dask.dataframe as dd  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - main path in this VM
    dd = None  # type: ignore[assignment]

from scaling.chunking import add_features, read_csv_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET_OUTPUT_PATH = PROJECT_ROOT / "datasets" / "large_synthetic_processed.parquet"


def parquet_vs_csv_explanation() -> dict[str, str]:
    """Explain Parquet vs CSV usage in simple institutional terms."""

    return {
        "why_parquet": "Parquet is a columnar format that is typically smaller and faster for analytics, especially when reading only some columns.",
        "when_prefer_parquet": "Prefer Parquet for repeated analytical queries, large datasets, and typed column storage; prefer CSV for quick interchange and manual inspection.",
    }


def _write_fallback_parquet_placeholder(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a deterministic JSON placeholder when Dask/Parquet support is absent."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    payload = {
        "format": "fallback-json-records-stored-with-parquet-extension",
        "note": "Dask/Parquet dependencies unavailable; this is a reproducible offline placeholder.",
        "row_count": len(rows),
        "preview": rows[:5],
        "parquet_vs_csv": parquet_vs_csv_explanation(),
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def dask_process_csv(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Process a CSV with Dask and write Parquet, or fallback if Dask unavailable."""

    output_path = Path(output_csv) if output_csv else PARQUET_OUTPUT_PATH

    if dd is None:
        rows = read_csv_records(input_csv)
        processed = add_features(rows)
        processed_rows = processed if isinstance(processed, list) else read_csv_records(input_csv)
        _write_fallback_parquet_placeholder(processed_rows, output_path)
        return {
            "method": "dask-fallback",
            "backend": "python-fallback",
            "output_parquet": str(output_path),
            "row_count": len(processed_rows),
            **parquet_vs_csv_explanation(),
        }

    # Dask path (requires dask + pandas + parquet engine like pyarrow/fastparquet).
    # Use partition-local feature engineering to avoid unsupported global rolling ops.
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    ddf = dd.read_csv(str(input_csv))

    def _partition_features(pdf):
        processed = add_features(pdf)
        return processed

    meta = ddf._meta.copy()
    meta["rolling_mean_metric_a"] = meta["metric_a"].astype("float64")
    meta["ratio_a_b"] = meta["metric_a"].astype("float64")
    meta["anomaly_flag"] = False
    ddf = ddf.map_partitions(_partition_features, meta=meta)
    try:
        ddf.to_parquet(str(output_path), write_index=False, overwrite=True)
        row_count = int(ddf.shape[0].compute())
        method = "dask"
    except Exception as exc:
        # If a parquet engine is missing, store a deterministic placeholder instead.
        rows = [dict(row) for row in ddf.head(1000, compute=True).to_dict(orient="records")]
        _write_fallback_parquet_placeholder(rows, output_path)
        row_count = len(rows)
        method = "dask-fallback"
        fallback_error = str(exc)

    result = {
        "method": method,
        "backend": "dask",
        "output_parquet": str(output_path),
        "row_count": row_count,
        **parquet_vs_csv_explanation(),
    }
    if method != "dask":
        result["fallback_error"] = fallback_error
    return result


def main() -> None:
    from scaling.chunking import RAW_CSV_PATH

    print(dask_process_csv(RAW_CSV_PATH, PARQUET_OUTPUT_PATH))


if __name__ == "__main__":
    main()
