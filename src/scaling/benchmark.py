"""Benchmark runner for chunking, multiprocessing, and Dask processing demos."""

from __future__ import annotations

from pathlib import Path
import json
import os
import platform
import sys
from typing import Any

from scaling.chunking import RAW_CSV_PATH, PROCESSED_CSV_PATH, run_chunking_pipeline, read_csv_records
from scaling.dask_jobs import PARQUET_OUTPUT_PATH, dask_process_csv
from scaling.multiprocessing_jobs import parallel_process_csv
from scaling.profiling import memory_snapshot, time_it


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "scaling"
BENCH_JSON = REPORT_DIR / "benchmark_results.json"
BENCH_MD = REPORT_DIR / "benchmark_results.md"


def _count_csv_rows(path: str | Path) -> int:
    rows = read_csv_records(path)
    return len(rows)


def _count_parquet_fallback_rows(path: str | Path) -> int:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return int(payload.get("row_count", 0))
    except Exception:
        return 0


def _system_info() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_cores": os.cpu_count(),
    }


def _run_single_benchmark(label: str, fn) -> dict[str, Any]:
    before_mem = memory_snapshot()
    result, seconds = time_it(label, fn)
    after_mem = memory_snapshot()
    return {
        "label": label,
        "wall_time_seconds": seconds,
        "memory_before": before_mem,
        "memory_after": after_mem,
        "result": result,
    }


def run_benchmark(
    *,
    n_rows: int = 20_000,
    chunk_size: int = 1000,
    n_workers: int = 2,
    raw_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run chunking, multiprocessing, and Dask/fallback benchmarks."""

    raw_path = Path(raw_csv_path) if raw_csv_path else RAW_CSV_PATH
    mp_out = raw_path.with_name("large_synthetic_processed_mp.csv")
    dask_out = raw_path.with_name("large_synthetic_processed.parquet")

    chunk_bench = _run_single_benchmark(
        "chunking",
        lambda: run_chunking_pipeline(
            n_rows=n_rows,
            chunk_size=chunk_size,
            raw_csv_path=raw_path,
            processed_csv_path=PROCESSED_CSV_PATH,
        ),
    )
    mp_bench = _run_single_benchmark(
        "multiprocessing",
        lambda: parallel_process_csv(raw_path, mp_out, chunk_size=chunk_size, n_workers=n_workers),
    )
    dask_bench = _run_single_benchmark(
        "dask",
        lambda: dask_process_csv(raw_path, dask_out),
    )

    results = {
        "system_info": _system_info(),
        "parameters": {
            "n_rows": n_rows,
            "chunk_size": chunk_size,
            "n_workers": n_workers,
        },
        "methods": {
            "chunking": {
                **chunk_bench,
                "output_row_count": _count_csv_rows(PROCESSED_CSV_PATH),
            },
            "multiprocessing": {
                **mp_bench,
                "output_row_count": _count_csv_rows(mp_out),
            },
            "dask": {
                **dask_bench,
                "output_row_count": _count_parquet_fallback_rows(dask_out),
            },
        },
        "artifacts": {
            "raw_csv": str(raw_path),
            "chunked_csv": str(PROCESSED_CSV_PATH),
            "multiprocessing_csv": str(mp_out),
            "dask_parquet": str(dask_out),
        },
    }
    return results


def write_benchmark_reports(results: dict[str, Any]) -> dict[str, str]:
    """Write benchmark JSON and Markdown reports."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_JSON.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    methods = results["methods"]
    lines = [
        "# Scaling Benchmark Results",
        "",
        "## System Info",
        "",
        f"- Python version: `{results['system_info']['python_version']}`",
        f"- Platform: `{results['system_info']['platform']}`",
        f"- CPU cores: `{results['system_info']['cpu_cores']}`",
        "",
        "## Results",
        "",
        "| Method | Wall Time (s) | Output Rows | Backend/Notes |",
        "|---|---:|---:|---|",
    ]
    for name in ["chunking", "multiprocessing", "dask"]:
        method = methods[name]
        result = method["result"]
        backend_note = result.get("backend") or result.get("method") or "n/a"
        lines.append(
            f"| {name} | {method['wall_time_seconds']:.6f} | {method['output_row_count']} | {backend_note} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Raw CSV: `{results['artifacts']['raw_csv']}`",
            f"- Chunked CSV: `{results['artifacts']['chunked_csv']}`",
            f"- Multiprocessing CSV: `{results['artifacts']['multiprocessing_csv']}`",
            f"- Dask/Parquet output: `{results['artifacts']['dask_parquet']}`",
            "",
            "Note: If Dask/Parquet dependencies are unavailable, the `.parquet` file may be an offline-safe placeholder artifact with benchmark metadata and preview rows.",
        ]
    )
    BENCH_MD.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(BENCH_JSON), "md": str(BENCH_MD)}


def run_and_report(
    *,
    n_rows: int = 20_000,
    chunk_size: int = 1000,
    n_workers: int = 2,
) -> dict[str, Any]:
    """Run benchmark and write JSON/Markdown reports."""

    results = run_benchmark(n_rows=n_rows, chunk_size=chunk_size, n_workers=n_workers)
    results["report_paths"] = write_benchmark_reports(results)
    return results


def main() -> None:
    """CLI entry point for ``make scale-bench``."""

    results = run_and_report()
    for name in ["chunking", "multiprocessing", "dask"]:
        method = results["methods"][name]
        print(
            f"{name}: {method['wall_time_seconds']:.4f}s, rows={method['output_row_count']}, "
            f"backend={method['result'].get('backend', method['result'].get('method'))}"
        )
    print(results["report_paths"])


if __name__ == "__main__":
    main()
