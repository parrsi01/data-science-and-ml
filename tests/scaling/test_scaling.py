from __future__ import annotations

from pathlib import Path
import json

from scaling.benchmark import run_and_report
from scaling.chunking import (
    add_features,
    generate_large_synthetic_dataset,
    process_in_chunks,
    read_csv_records,
    save_processed_output,
)
from scaling.dask_jobs import dask_process_csv
from scaling.multiprocessing_jobs import parallel_process_csv


def test_chunk_processing_preserves_row_count(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    processed_csv = tmp_path / "processed.csv"
    df = generate_large_synthetic_dataset(n_rows=200, seed=1, save_csv_path=raw_csv)
    processed = process_in_chunks(df, chunk_size=25, fn=add_features)
    save_processed_output(processed, processed_csv)
    assert len(read_csv_records(raw_csv)) == len(read_csv_records(processed_csv))


def test_multiprocessing_output_exists_and_expected_cols(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    mp_csv = tmp_path / "processed_mp.csv"
    generate_large_synthetic_dataset(n_rows=120, seed=2, save_csv_path=raw_csv)
    result = parallel_process_csv(raw_csv, mp_csv, chunk_size=20, n_workers=2)
    assert result["rows_processed"] == 120
    rows = read_csv_records(mp_csv)
    assert rows
    assert {
        "rolling_mean_metric_a",
        "ratio_a_b",
        "anomaly_flag",
    } <= set(rows[0].keys())


def test_dask_parquet_exists(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    parquet_path = tmp_path / "out.parquet"
    generate_large_synthetic_dataset(n_rows=100, seed=3, save_csv_path=raw_csv)
    result = dask_process_csv(raw_csv, parquet_path)
    assert Path(result["output_parquet"]).exists()
    assert Path(result["output_parquet"]).stat().st_size > 0


def test_benchmark_json_contains_required_keys(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    results = run_and_report(n_rows=300, chunk_size=50, n_workers=2)
    # Ensure required keys exist and report was written.
    assert {"system_info", "parameters", "methods", "artifacts", "report_paths"} <= set(
        results.keys()
    )
    json_path = Path(results["report_paths"]["json"])
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert {"chunking", "multiprocessing", "dask"} <= set(payload["methods"].keys())
