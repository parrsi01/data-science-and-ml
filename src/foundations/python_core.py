"""Python core foundations demos for institutional data systems.

This module provides small, testable demonstrations of:
- Python performance patterns (loops vs list comprehensions)
- Memory usage (list vs generator)
- OOP design for an institutional dataset object
- Structured logging to a project-local log file
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import sys
import time
from typing import Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "foundations.log"


def _build_logger() -> logging.Logger:
    """Configure and return the module logger.

    The logger writes to ``logs/foundations.log`` and uses INFO as a default
    operational level for normal events.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("foundations.python_core")
    logger.setLevel(logging.INFO)

    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == LOG_FILE
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(LOG_FILE)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


LOGGER = _build_logger()


def log_info(message: str) -> None:
    """Log an informational message for foundations workflows."""

    LOGGER.info(message)


def log_error(message: str) -> None:
    """Log an error message for foundations workflows."""

    LOGGER.error(message)


def _square_with_loop(values: Sequence[int]) -> list[int]:
    """Return squared values using an explicit for-loop."""

    result: list[int] = []
    for value in values:
        result.append(value * value)
    return result


def _square_with_comprehension(values: Sequence[int]) -> list[int]:
    """Return squared values using a list comprehension."""

    return [value * value for value in values]


def python_performance_comparison(
    n: int = 100_000,
) -> dict[str, float | int | bool]:
    """Compare loop and list-comprehension performance using the ``time`` module.

    Args:
        n: Number of integers to square.

    Returns:
        Dictionary with timings, speedup ratio, and result equality flag.
    """

    values = list(range(n))

    start_loop = time.perf_counter()
    loop_result = _square_with_loop(values)
    loop_time = time.perf_counter() - start_loop

    start_comp = time.perf_counter()
    comp_result = _square_with_comprehension(values)
    comp_time = time.perf_counter() - start_comp

    result = {
        "n": n,
        "loop_seconds": loop_time,
        "comprehension_seconds": comp_time,
        "speedup_ratio_loop_over_comp": (loop_time / comp_time) if comp_time else float("inf"),
        "results_equal": loop_result == comp_result,
    }
    log_info(f"Python performance comparison completed for n={n}")
    return result


def _number_generator(limit: int) -> Iterator[int]:
    """Yield integers from 0 to ``limit - 1``."""

    for value in range(limit):
        yield value


def memory_demonstration(limit: int = 10_000) -> dict[str, int]:
    """Compare memory footprint of a list and a generator object.

    Note:
        ``sys.getsizeof`` on a generator reports the generator object's memory,
        not the total memory of values it may produce over time.
    """

    number_list = list(range(limit))
    number_gen = _number_generator(limit)
    result = {
        "limit": limit,
        "list_size_bytes": sys.getsizeof(number_list),
        "generator_object_size_bytes": sys.getsizeof(number_gen),
    }
    log_info(f"Memory demonstration completed for limit={limit}")
    return result


@dataclass(slots=True)
class InstitutionalDataset:
    """Simple dataset metadata model for institutional workflows.

    Attributes:
        name: Human-readable dataset name.
        size: Number of records in the dataset.
        source: Data source identifier (system, team, or provider).
    """

    name: str
    size: int
    source: str

    def describe(self) -> str:
        """Return a short dataset summary string."""

        return (
            f"InstitutionalDataset(name={self.name}, size={self.size}, "
            f"source={self.source})"
        )

    def validate_schema(
        self,
        actual_columns: Iterable[str],
        required_columns: Iterable[str],
        column_types: Mapping[str, str] | None = None,
    ) -> bool:
        """Validate required columns and optional type declarations.

        Args:
            actual_columns: Columns present in the dataset.
            required_columns: Columns required by a downstream workflow.
            column_types: Optional mapping of column name to type label. If
                provided, values must be non-empty strings.

        Returns:
            ``True`` when validation passes, otherwise ``False``.
        """

        actual_set = set(actual_columns)
        required_set = set(required_columns)

        missing = sorted(required_set - actual_set)
        if missing:
            log_error(f"Schema validation failed for {self.name}; missing={missing}")
            return False

        if column_types is not None:
            for column in required_set:
                if column not in column_types or not str(column_types[column]).strip():
                    log_error(
                        f"Schema validation failed for {self.name}; invalid type for {column}"
                    )
                    return False

        log_info(f"Schema validation passed for {self.name}")
        return True


def run_python_foundations_demo() -> dict[str, object]:
    """Run all Python foundations demos and return structured results."""

    dataset = InstitutionalDataset(name="aviation_ops", size=5000, source="IATA-demo")
    schema_ok = dataset.validate_schema(
        actual_columns=["flight_id", "delay_minutes", "fuel_consumption", "passenger_count"],
        required_columns=["flight_id", "delay_minutes", "fuel_consumption", "passenger_count"],
        column_types={
            "flight_id": "str",
            "delay_minutes": "float",
            "fuel_consumption": "float",
            "passenger_count": "int",
        },
    )
    return {
        "performance": python_performance_comparison(),
        "memory": memory_demonstration(),
        "dataset_description": dataset.describe(),
        "schema_valid": schema_ok,
        "log_file": str(LOG_FILE),
    }

