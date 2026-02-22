"""Validation helpers for data ingestion pipelines."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import logging
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs" / "pipeline.log"


def _get_logger() -> logging.Logger:
    """Return a logger for pipeline validation and ingestion messages."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("data_engineering.pipeline")
    logger.setLevel(logging.INFO)
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == LOG_PATH
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(LOG_PATH)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


LOGGER = _get_logger()


class DataValidationError(ValueError):
    """Raised when incoming pipeline data fails validation."""


SCHEMAS: dict[str, dict[str, Any]] = {
    "flights": {
        "required": [
            "flight_id",
            "dep_airport",
            "arr_airport",
            "scheduled_dep",
            "actual_dep",
            "delay_minutes",
            "passenger_count",
            "fuel_consumption_kg",
        ],
        "types": {
            "flight_id": str,
            "dep_airport": str,
            "arr_airport": str,
            "scheduled_dep": datetime,
            "actual_dep": datetime,
            "delay_minutes": int,
            "passenger_count": int,
            "fuel_consumption_kg": float,
        },
        "ranges": {
            "delay_minutes": (0, None),
            "passenger_count": (0, None),
            "fuel_consumption_kg": (0.0, None),
        },
    },
    "humanitarian_shipments": {
        "required": ["shipment_id", "region", "item_type", "quantity", "priority", "status"],
        "types": {
            "shipment_id": str,
            "region": str,
            "item_type": str,
            "quantity": int,
            "priority": int,
            "status": str,
        },
        "ranges": {"quantity": (0, None), "priority": (1, 5)},
    },
    "cern_events": {
        "required": ["event_id", "detector", "energy_gev", "is_rare_event", "recorded_at"],
        "types": {
            "event_id": str,
            "detector": str,
            "energy_gev": float,
            "is_rare_event": bool,
            "recorded_at": datetime,
        },
        "ranges": {"energy_gev": (0.0, None)},
    },
}


def validate_required_columns(headers: Iterable[str], required_columns: Iterable[str]) -> None:
    """Validate that all required columns are present in a CSV header."""

    header_set = set(headers)
    required_set = set(required_columns)
    missing = sorted(required_set - header_set)
    if missing:
        LOGGER.error("Missing required columns: %s", missing)
        raise DataValidationError(f"Missing required columns: {missing}")


def validate_csv_schema(csv_path: Path, required_columns: Sequence[str]) -> None:
    """Validate CSV header columns before ingestion."""

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader, None)
    if headers is None:
        LOGGER.error("CSV file is empty: %s", csv_path)
        raise DataValidationError(f"CSV file is empty: {csv_path}")
    validate_required_columns(headers, required_columns)


def _type_ok(value: Any, expected_type: type[Any]) -> bool:
    """Check whether a value matches an expected type with numeric conveniences."""

    if value is None:
        return True
    if expected_type is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected_type)


def validate_records(table_name: str, records: Sequence[Mapping[str, Any]]) -> None:
    """Validate required columns, types, and ranges for a table payload."""

    if table_name not in SCHEMAS:
        raise DataValidationError(f"Unknown schema: {table_name}")
    if not records:
        raise DataValidationError(f"No records supplied for table: {table_name}")

    schema = SCHEMAS[table_name]
    required = schema["required"]
    type_map = schema["types"]
    range_map = schema["ranges"]

    for idx, record in enumerate(records):
        validate_required_columns(record.keys(), required)

        for field, expected_type in type_map.items():
            value = record.get(field)
            if not _type_ok(value, expected_type):
                LOGGER.error(
                    "Type validation failed for %s row %s field %s value=%r expected=%s",
                    table_name,
                    idx,
                    field,
                    value,
                    expected_type.__name__,
                )
                raise DataValidationError(
                    f"Type check failed: table={table_name} row={idx} field={field}"
                )

        for field, (min_value, max_value) in range_map.items():
            value = record.get(field)
            if value is None:
                continue
            numeric_value = float(value)
            if min_value is not None and numeric_value < float(min_value):
                LOGGER.error(
                    "Range validation failed for %s row %s field %s value=%r < %r",
                    table_name,
                    idx,
                    field,
                    value,
                    min_value,
                )
                raise DataValidationError(
                    f"Range check failed: table={table_name} row={idx} field={field}"
                )
            if max_value is not None and numeric_value > float(max_value):
                LOGGER.error(
                    "Range validation failed for %s row %s field %s value=%r > %r",
                    table_name,
                    idx,
                    field,
                    value,
                    max_value,
                )
                raise DataValidationError(
                    f"Range check failed: table={table_name} row={idx} field={field}"
                )

    LOGGER.info("Validation passed for table=%s rows=%s", table_name, len(records))
