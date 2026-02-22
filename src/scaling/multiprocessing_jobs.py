"""Multiprocessing chunk processing for deterministic CSV transforms."""

from __future__ import annotations

from multiprocessing import Pool, cpu_count
from pathlib import Path
import logging
import time
from typing import Any

from scaling.chunking import add_features, read_csv_records


LOGGER = logging.getLogger("scaling.multiprocessing")


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    from scaling.chunking import _write_csv as write_csv_impl

    write_csv_impl(records, path)


def _process_chunk_task(task: tuple[int, list[dict[str, Any]]]) -> tuple[int, list[dict[str, Any]]]:
    """Worker task: process one chunk and return indexed results."""

    chunk_index, rows = task
    processed = add_features(rows)
    return chunk_index, processed


def _chunk_records(rows: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]


def parallel_process_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    chunk_size: int = 1000,
    n_workers: int = 2,
) -> dict[str, Any]:
    """Process CSV chunks in parallel using ``multiprocessing.Pool``.

    Determinism:
    - Stable chunk indices are preserved
    - Results are sorted by chunk index before writing
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")

    rows = read_csv_records(input_csv)
    tasks = [(idx, chunk) for idx, chunk in enumerate(_chunk_records(rows, chunk_size))]
    start = time.perf_counter()

    worker_count = min(n_workers, max(1, cpu_count()), max(1, len(tasks)))
    execution_mode = "multiprocessing"
    if tasks:
        try:
            with Pool(processes=worker_count) as pool:
                chunk_results = pool.map(_process_chunk_task, tasks)
        except (PermissionError, OSError):
            # Some sandboxed environments block multiprocessing semaphores.
            execution_mode = "sequential-fallback"
            chunk_results = [_process_chunk_task(task) for task in tasks]
            worker_count = 1

        chunk_results.sort(key=lambda x: x[0])
        processed_rows = [row for _, chunk in chunk_results for row in chunk]
    else:
        processed_rows = []

    elapsed = time.perf_counter() - start
    output_path = Path(output_csv)
    _write_csv(processed_rows, output_path)

    result = {
        "method": "multiprocessing",
        "elapsed_seconds": elapsed,
        "rows_processed": len(processed_rows),
        "workers_used": worker_count,
        "execution_mode": execution_mode,
        "input_csv": str(input_csv),
        "output_csv": str(output_path),
    }
    LOGGER.info("multiprocessing_complete %s", result)
    return result


def main() -> None:
    from scaling.chunking import RAW_CSV_PATH

    output = RAW_CSV_PATH.with_name("large_synthetic_processed_mp.csv")
    print(parallel_process_csv(RAW_CSV_PATH, output))


if __name__ == "__main__":
    main()
