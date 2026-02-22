"""Numerical computing foundations demos with offline-safe fallbacks.

The preferred implementation uses NumPy and Pandas. When those libraries are
unavailable (for example in a restricted offline environment), this module
falls back to pure-Python routines so the educational workflow remains runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import random
import sys
import time
from typing import Any, Iterable

try:
    import numpy as np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency path
    np = None  # type: ignore[assignment]

try:
    import pandas as pd  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency path
    pd = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_EXPORT_PATH = PROJECT_ROOT / "datasets" / "processed_aviation_sample.csv"


@dataclass(slots=True)
class MatrixMultiplicationResult:
    """Container for matrix multiplication demo results."""

    backend: str
    left_shape: tuple[int, int]
    right_shape: tuple[int, int]
    output_shape: tuple[int, int]
    timing_seconds: float


def _python_vectorized_equivalent(values: list[float]) -> list[float]:
    """Pure-Python numeric transform used as a fallback benchmark."""

    return [value * 1.8 + 32.0 for value in values]


def numpy_vectorization_demo(n: int = 50_000) -> dict[str, float | int | str | bool]:
    """Compare numeric transformations using a loop/list approach vs vectorization.

    If NumPy is unavailable, the function returns a pure-Python fallback result
    and marks the backend accordingly.
    """

    values = [float(i) for i in range(n)]

    loop_start = time.perf_counter()
    loop_result = []
    for value in values:
        loop_result.append(value * 1.8 + 32.0)
    loop_time = time.perf_counter() - loop_start

    if np is not None:
        array = np.array(values, dtype=float)
        vec_start = time.perf_counter()
        vectorized = array * 1.8 + 32.0
        vec_time = time.perf_counter() - vec_start
        result_equal = bool(np.allclose(vectorized, loop_result))
        backend = "numpy"
    else:
        vec_start = time.perf_counter()
        vectorized = _python_vectorized_equivalent(values)
        vec_time = time.perf_counter() - vec_start
        result_equal = vectorized == loop_result
        backend = "python-fallback"

    return {
        "backend": backend,
        "n": n,
        "loop_seconds": loop_time,
        "vectorized_seconds": vec_time,
        "speedup_ratio_loop_over_vectorized": (loop_time / vec_time) if vec_time else float("inf"),
        "results_equal": result_equal,
    }


def matrix_multiplication_demo() -> MatrixMultiplicationResult:
    """Demonstrate matrix multiplication and return dimension metadata."""

    left = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    right = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    start = time.perf_counter()
    if np is not None:
        product = np.matmul(np.array(left), np.array(right))
        output_shape = tuple(int(x) for x in product.shape)
        backend = "numpy"
    else:
        rows = len(left)
        cols = len(right[0])
        inner = len(right)
        product: list[list[float]] = []
        for i in range(rows):
            row: list[float] = []
            for j in range(cols):
                total = 0.0
                for k in range(inner):
                    total += left[i][k] * right[k][j]
                row.append(total)
            product.append(row)
        output_shape = (len(product), len(product[0]) if product else 0)
        backend = "python-fallback"
    elapsed = time.perf_counter() - start

    return MatrixMultiplicationResult(
        backend=backend,
        left_shape=(2, 3),
        right_shape=(3, 2),
        output_shape=output_shape,
        timing_seconds=elapsed,
    )


def broadcasting_demo() -> dict[str, Any]:
    """Show a simple broadcasting example (or equivalent fallback)."""

    if np is not None:
        matrix = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        adjustment = np.array([1.0, -1.0, 0.5])
        result = matrix + adjustment
        return {
            "backend": "numpy",
            "input_shape": tuple(int(v) for v in matrix.shape),
            "adjustment_shape": tuple(int(v) for v in adjustment.shape),
            "result_preview": result.tolist(),
        }

    matrix = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    adjustment = [1.0, -1.0, 0.5]
    result = [[value + adjustment[idx] for idx, value in enumerate(row)] for row in matrix]
    return {
        "backend": "python-fallback",
        "input_shape": (2, 3),
        "adjustment_shape": (3,),
        "result_preview": result,
    }


def _synthetic_aviation_rows(n: int = 24, seed: int = 42) -> list[dict[str, Any]]:
    """Create a deterministic synthetic aviation-style dataset."""

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for idx in range(1, n + 1):
        delay_value: float | None
        fuel_value: float | None
        passengers_value: int | None

        delay_value = float(rng.randint(-5, 180))
        fuel_value = round(1800.0 + rng.random() * 700.0, 2)
        passengers_value = rng.randint(70, 220)

        if idx % 7 == 0:
            delay_value = None
        if idx % 11 == 0:
            fuel_value = None
        if idx % 13 == 0:
            passengers_value = None

        rows.append(
            {
                "flight_id": f"FL{idx:04d}",
                "delay_minutes": delay_value,
                "fuel_consumption": fuel_value,
                "passenger_count": passengers_value,
            }
        )
    return rows


def _approx_rows_memory_bytes(rows: Iterable[dict[str, Any]]) -> int:
    """Approximate memory usage of row dictionaries using ``sys.getsizeof``."""

    total = 0
    for row in rows:
        total += sys.getsizeof(row)
        for key, value in row.items():
            total += sys.getsizeof(key)
            total += sys.getsizeof(value)
    return total


def _process_with_pandas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the requested data-cleaning and aggregation workflow with Pandas."""

    assert pd is not None

    df = pd.DataFrame(rows)
    memory_before = int(df.memory_usage(deep=True).sum())

    df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce")
    df["fuel_consumption"] = pd.to_numeric(df["fuel_consumption"], errors="coerce")
    df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")

    # Missing value handling and cleaning.
    df["delay_minutes"] = df["delay_minutes"].fillna(df["delay_minutes"].median())
    df["fuel_consumption"] = df["fuel_consumption"].fillna(df["fuel_consumption"].mean())
    df["passenger_count"] = df["passenger_count"].fillna(
        df["passenger_count"].median()
    )

    # Feature creation.
    df["fuel_per_passenger"] = (
        df["fuel_consumption"] / df["passenger_count"].replace(0, math.nan)
    )
    df["delay_flag"] = (df["delay_minutes"] > 15).astype("int8")

    # dtype optimization.
    df["flight_id"] = df["flight_id"].astype("string")
    df["delay_minutes"] = df["delay_minutes"].astype("float32")
    df["fuel_consumption"] = df["fuel_consumption"].astype("float32")
    df["passenger_count"] = df["passenger_count"].astype("int16")
    df["fuel_per_passenger"] = df["fuel_per_passenger"].astype("float32")

    memory_after = int(df.memory_usage(deep=True).sum())

    grouped = (
        df.assign(route_group=df["flight_id"].str.slice(0, 4))
        .groupby("route_group", as_index=False)
        .agg(
            avg_delay_minutes=("delay_minutes", "mean"),
            avg_fuel_consumption=("fuel_consumption", "mean"),
            total_passengers=("passenger_count", "sum"),
        )
    )

    DATASET_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_EXPORT_PATH, index=False)

    return {
        "backend": "pandas",
        "dataframe": df,
        "head": df.head().to_dict(orient="records"),
        "groupby_summary": grouped.to_dict(orient="records"),
        "memory_before_bytes": memory_before,
        "memory_after_bytes": memory_after,
        "export_path": str(DATASET_EXPORT_PATH),
    }


def _process_with_python_fallback(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run a Pandas-like workflow using pure Python for offline environments."""

    memory_before = _approx_rows_memory_bytes(rows)

    cleaned_rows: list[dict[str, Any]] = []
    observed_delays = [r["delay_minutes"] for r in rows if r["delay_minutes"] is not None]
    observed_fuels = [r["fuel_consumption"] for r in rows if r["fuel_consumption"] is not None]
    observed_passengers = [
        r["passenger_count"] for r in rows if r["passenger_count"] is not None
    ]

    sorted_delays = sorted(float(v) for v in observed_delays)
    delay_median = sorted_delays[len(sorted_delays) // 2]
    fuel_mean = sum(float(v) for v in observed_fuels) / len(observed_fuels)
    sorted_passengers = sorted(int(v) for v in observed_passengers)
    passenger_median = sorted_passengers[len(sorted_passengers) // 2]

    for row in rows:
        delay = float(row["delay_minutes"]) if row["delay_minutes"] is not None else delay_median
        fuel = (
            float(row["fuel_consumption"])
            if row["fuel_consumption"] is not None
            else round(fuel_mean, 2)
        )
        passengers = (
            int(row["passenger_count"]) if row["passenger_count"] is not None else passenger_median
        )
        fuel_per_passenger = fuel / passengers if passengers else 0.0
        cleaned_rows.append(
            {
                "flight_id": str(row["flight_id"]),
                "delay_minutes": round(delay, 2),
                "fuel_consumption": round(fuel, 2),
                "passenger_count": int(passengers),
                "fuel_per_passenger": round(fuel_per_passenger, 4),
                "delay_flag": 1 if delay > 15 else 0,
            }
        )

    grouped: dict[str, dict[str, float | int]] = {}
    for row in cleaned_rows:
        group = row["flight_id"][:4]
        entry = grouped.setdefault(
            group,
            {
                "route_group": group,
                "sum_delay": 0.0,
                "sum_fuel": 0.0,
                "total_passengers": 0,
                "count": 0,
            },
        )
        entry["sum_delay"] = float(entry["sum_delay"]) + float(row["delay_minutes"])
        entry["sum_fuel"] = float(entry["sum_fuel"]) + float(row["fuel_consumption"])
        entry["total_passengers"] = int(entry["total_passengers"]) + int(row["passenger_count"])
        entry["count"] = int(entry["count"]) + 1

    groupby_summary: list[dict[str, Any]] = []
    for group, entry in grouped.items():
        count = int(entry["count"])
        groupby_summary.append(
            {
                "route_group": group,
                "avg_delay_minutes": round(float(entry["sum_delay"]) / count, 2),
                "avg_fuel_consumption": round(float(entry["sum_fuel"]) / count, 2),
                "total_passengers": int(entry["total_passengers"]),
            }
        )

    # Simulate dtype optimization by materializing a compact tuple representation.
    optimized_rows = [
        (
            row["flight_id"],
            float(row["delay_minutes"]),
            float(row["fuel_consumption"]),
            int(row["passenger_count"]),
            float(row["fuel_per_passenger"]),
            int(row["delay_flag"]),
        )
        for row in cleaned_rows
    ]
    memory_after = _approx_rows_memory_bytes(cleaned_rows)
    memory_after = min(memory_after, sys.getsizeof(optimized_rows) + sum(sys.getsizeof(t) for t in optimized_rows))

    DATASET_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_EXPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "flight_id",
                "delay_minutes",
                "fuel_consumption",
                "passenger_count",
                "fuel_per_passenger",
                "delay_flag",
            ],
        )
        writer.writeheader()
        writer.writerows(cleaned_rows)

    return {
        "backend": "python-fallback",
        "dataframe": None,
        "head": cleaned_rows[:5],
        "groupby_summary": groupby_summary,
        "memory_before_bytes": memory_before,
        "memory_after_bytes": memory_after,
        "export_path": str(DATASET_EXPORT_PATH),
    }


def pandas_aviation_dataset_demo() -> dict[str, Any]:
    """Create, clean, aggregate, optimize, and export an aviation-style dataset."""

    rows = _synthetic_aviation_rows()
    if pd is not None:
        return _process_with_pandas(rows)
    return _process_with_python_fallback(rows)


def run_numerical_foundations_demo() -> dict[str, Any]:
    """Run the full numerical foundations workflow and return structured results."""

    vectorization = numpy_vectorization_demo()
    matrix = matrix_multiplication_demo()
    broadcasting = broadcasting_demo()
    dataset = pandas_aviation_dataset_demo()

    return {
        "vectorization": vectorization,
        "matrix_multiplication": {
            "backend": matrix.backend,
            "left_shape": matrix.left_shape,
            "right_shape": matrix.right_shape,
            "output_shape": matrix.output_shape,
            "timing_seconds": matrix.timing_seconds,
        },
        "broadcasting": broadcasting,
        "dataset": dataset,
    }

