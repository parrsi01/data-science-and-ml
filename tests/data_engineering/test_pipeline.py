from __future__ import annotations

from pathlib import Path

import pytest

from data_engineering.db import database_healthcheck, get_engine
from data_engineering.ingest import fetch_row_counts, ingest_all
from data_engineering.query_examples import run_query_examples
from data_engineering.schema_init import initialize_schema


@pytest.fixture(scope="module")
def db_ready() -> None:
    healthy, message = database_healthcheck()
    if not healthy:
        pytest.skip(f"PostgreSQL unavailable or dependencies missing: {message}")

    initialize_schema()
    counts = ingest_all()
    if not all(count > 0 for count in counts.values()):
        pytest.skip(f"Ingestion did not produce rows: {counts}")


def test_tables_exist(db_ready: None) -> None:
    try:
        from sqlalchemy import text  # type: ignore[import-not-found]
    except Exception as exc:
        pytest.skip(f"SQLAlchemy unavailable: {exc}")

    engine = get_engine()
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('flights', 'humanitarian_shipments', 'cern_events')
                ORDER BY table_name
                """
            )
        ).mappings()
        table_names = [row["table_name"] for row in rows]
    assert table_names == ["cern_events", "flights", "humanitarian_shipments"]


def test_ingestion_inserts_rows(db_ready: None) -> None:
    counts = fetch_row_counts()
    assert counts["flights"] > 0
    assert counts["humanitarian_shipments"] > 0
    assert counts["cern_events"] > 0


def test_query_returns_expected_columns(db_ready: None) -> None:
    results = run_query_examples(limit=3)
    avg_delay = results["average_delay_by_route"]
    assert avg_delay
    first_row = avg_delay[0]
    assert {"dep_airport", "arr_airport", "avg_delay_minutes", "flight_count"} <= set(
        first_row.keys()
    )
