"""Synthetic data ingestion pipeline for local PostgreSQL development."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import random
from typing import Any, Iterable, Sequence

from data_engineering.db import get_engine
from data_engineering.validate import LOGGER as VALIDATION_LOGGER
from data_engineering.validate import validate_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs" / "pipeline.log"


def _get_logger() -> logging.Logger:
    """Reuse the validation logger so all pipeline steps share one log file."""

    return VALIDATION_LOGGER


LOGGER = _get_logger()


def generate_flights(count: int = 1000, seed: int = 100) -> list[dict[str, Any]]:
    """Generate synthetic flight records for ingestion."""

    rng = random.Random(seed)
    airports = ["JFK", "LHR", "NBO", "GVA", "DOH", "DXB", "ATL", "FRA"]
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        dep = rng.choice(airports)
        arr = rng.choice([a for a in airports if a != dep])
        scheduled = base_time + timedelta(minutes=45 * idx)
        delay = max(0, int(round(rng.gauss(18, 14))))
        actual = scheduled + timedelta(minutes=delay)
        passenger_count = max(40, int(round(rng.gauss(155, 35))))
        fuel_kg = round(max(500.0, rng.gauss(6800.0, 1200.0)), 2)
        rows.append(
            {
                "flight_id": f"FL{idx + 1:05d}",
                "dep_airport": dep,
                "arr_airport": arr,
                "scheduled_dep": scheduled.replace(tzinfo=None),
                "actual_dep": actual.replace(tzinfo=None),
                "delay_minutes": delay,
                "passenger_count": passenger_count,
                "fuel_consumption_kg": float(fuel_kg),
            }
        )
    return rows


def generate_humanitarian_shipments(
    count: int = 1000, seed: int = 200
) -> list[dict[str, Any]]:
    """Generate synthetic humanitarian shipment records."""

    rng = random.Random(seed)
    regions = ["East Africa", "Levant", "Sahel", "South Asia", "Latin America"]
    item_types = ["medical_kit", "food_rations", "water_purifier", "shelter_kit"]
    statuses = ["pending", "in_transit", "delivered", "on_hold"]
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        rows.append(
            {
                "shipment_id": f"HS{idx + 1:05d}",
                "region": rng.choice(regions),
                "item_type": rng.choice(item_types),
                "quantity": rng.randint(0, 2500),
                "priority": rng.randint(1, 5),
                "status": rng.choices(statuses, weights=[3, 4, 5, 1], k=1)[0],
            }
        )
    return rows


def generate_cern_events(count: int = 1000, seed: int = 300) -> list[dict[str, Any]]:
    """Generate synthetic CERN-style detector event records."""

    rng = random.Random(seed)
    detectors = ["ATLAS", "CMS", "ALICE", "LHCb"]
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        energy = max(0.0, rng.gauss(90.0, 25.0))
        is_rare = rng.random() < 0.04
        if is_rare:
            energy += rng.uniform(60.0, 180.0)
        rows.append(
            {
                "event_id": f"CE{idx + 1:05d}",
                "detector": rng.choice(detectors),
                "energy_gev": float(round(energy, 4)),
                "is_rare_event": bool(is_rare),
                "recorded_at": (base_time + timedelta(seconds=idx * 37)).replace(
                    tzinfo=None
                ),
            }
        )
    return rows


def _batch(iterable: Sequence[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    """Yield records in fixed-size batches."""

    for start in range(0, len(iterable), batch_size):
        yield list(iterable[start : start + batch_size])


def _insert_many(table_name: str, rows: Sequence[dict[str, Any]], batch_size: int = 200) -> None:
    """Insert rows into a table in batches using parameterized SQL."""

    if not rows:
        return

    validate_records(table_name, rows)
    try:
        from sqlalchemy import text  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "SQLAlchemy is required for ingestion. Install `sqlalchemy` and `psycopg2-binary`."
        ) from exc

    insert_sql = {
        "flights": text(
            """
            INSERT INTO flights (
              flight_id, dep_airport, arr_airport, scheduled_dep, actual_dep,
              delay_minutes, passenger_count, fuel_consumption_kg
            ) VALUES (
              :flight_id, :dep_airport, :arr_airport, :scheduled_dep, :actual_dep,
              :delay_minutes, :passenger_count, :fuel_consumption_kg
            )
            ON CONFLICT (flight_id) DO NOTHING
            """
        ),
        "humanitarian_shipments": text(
            """
            INSERT INTO humanitarian_shipments (
              shipment_id, region, item_type, quantity, priority, status
            ) VALUES (
              :shipment_id, :region, :item_type, :quantity, :priority, :status
            )
            ON CONFLICT (shipment_id) DO NOTHING
            """
        ),
        "cern_events": text(
            """
            INSERT INTO cern_events (
              event_id, detector, energy_gev, is_rare_event, recorded_at
            ) VALUES (
              :event_id, :detector, :energy_gev, :is_rare_event, :recorded_at
            )
            ON CONFLICT (event_id) DO NOTHING
            """
        ),
    }[table_name]

    engine = get_engine()
    with engine.begin() as connection:
        for chunk in _batch(list(rows), batch_size):
            connection.execute(insert_sql, list(chunk))

    LOGGER.info("Ingested table=%s rows=%s", table_name, len(rows))


def fetch_row_counts() -> dict[str, int]:
    """Return row counts for the pipeline tables."""

    try:
        from sqlalchemy import text  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("SQLAlchemy is required for row-count queries.") from exc

    queries = {
        "flights": "SELECT COUNT(*) AS count FROM flights",
        "humanitarian_shipments": "SELECT COUNT(*) AS count FROM humanitarian_shipments",
        "cern_events": "SELECT COUNT(*) AS count FROM cern_events",
    }
    engine = get_engine()
    counts: dict[str, int] = {}
    with engine.connect() as connection:
        for table_name, sql in queries.items():
            count = connection.execute(text(sql)).scalar_one()
            counts[table_name] = int(count)
    return counts


def ingest_all(batch_size: int = 200) -> dict[str, int]:
    """Generate and ingest all synthetic datasets, then return row counts."""

    LOGGER.info("Starting ingestion pipeline batch_size=%s", batch_size)
    _insert_many("flights", generate_flights(count=1000), batch_size=batch_size)
    _insert_many(
        "humanitarian_shipments",
        generate_humanitarian_shipments(count=1000),
        batch_size=batch_size,
    )
    _insert_many("cern_events", generate_cern_events(count=1000), batch_size=batch_size)
    counts = fetch_row_counts()
    LOGGER.info("Completed ingestion pipeline counts=%s", counts)
    return counts


def main() -> None:
    """CLI entry point for ``make ingest``."""

    counts = ingest_all()
    for table_name, row_count in counts.items():
        print(f"{table_name}: {row_count}")


if __name__ == "__main__":
    main()
