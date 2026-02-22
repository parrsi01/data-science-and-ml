"""Example institution-style SQL queries for the PostgreSQL pipeline."""

from __future__ import annotations

from typing import Any

from data_engineering.db import get_engine


def _fetch_rows(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute a SQL query and return a list of dictionaries."""

    try:
        from sqlalchemy import text  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("SQLAlchemy is required for query examples.") from exc

    engine = get_engine()
    with engine.connect() as connection:
        result = connection.execute(text(sql), params or {})
        return [dict(row) for row in result.mappings().all()]


def run_query_examples(limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Run six institution-style queries and return structured results."""

    queries = {
        "average_delay_by_route": """
            SELECT dep_airport, arr_airport,
                   ROUND(AVG(delay_minutes)::numeric, 2) AS avg_delay_minutes,
                   COUNT(*) AS flight_count
            FROM flights
            GROUP BY dep_airport, arr_airport
            HAVING COUNT(*) > 0
            ORDER BY avg_delay_minutes DESC, dep_airport, arr_airport
            LIMIT :limit
        """,
        "top_10_delayed_flights": """
            SELECT flight_id, dep_airport, arr_airport, delay_minutes, scheduled_dep
            FROM flights
            ORDER BY delay_minutes DESC, scheduled_dep
            LIMIT 10
        """,
        "shipment_backlog_by_region_priority": """
            SELECT region, priority, COUNT(*) AS shipment_count, SUM(quantity) AS total_quantity
            FROM humanitarian_shipments
            WHERE status IN ('pending', 'on_hold', 'in_transit')
            GROUP BY region, priority
            ORDER BY region, priority
            LIMIT :limit
        """,
        "rare_event_rate_per_detector_per_day": """
            SELECT detector,
                   DATE(recorded_at) AS event_day,
                   ROUND(AVG(CASE WHEN is_rare_event THEN 1.0 ELSE 0.0 END)::numeric, 4) AS rare_event_rate,
                   COUNT(*) AS total_events
            FROM cern_events
            GROUP BY detector, DATE(recorded_at)
            ORDER BY event_day, detector
            LIMIT :limit
        """,
        "fuel_per_passenger_by_route": """
            SELECT dep_airport, arr_airport,
                   ROUND(AVG(fuel_consumption_kg / NULLIF(passenger_count, 0))::numeric, 4) AS avg_fuel_per_passenger_kg,
                   COUNT(*) AS flight_count
            FROM flights
            WHERE passenger_count > 0 AND fuel_consumption_kg IS NOT NULL
            GROUP BY dep_airport, arr_airport
            ORDER BY avg_fuel_per_passenger_kg DESC
            LIMIT :limit
        """,
        "data_quality_checks": """
            SELECT 'flights'::text AS table_name,
                   SUM(CASE WHEN actual_dep IS NULL THEN 1 ELSE 0 END) AS null_actual_dep_count,
                   SUM(CASE WHEN delay_minutes < 0 THEN 1 ELSE 0 END) AS negative_delay_count,
                   SUM(CASE WHEN passenger_count < 0 THEN 1 ELSE 0 END) AS negative_passenger_count,
                   SUM(CASE WHEN fuel_consumption_kg < 0 THEN 1 ELSE 0 END) AS negative_fuel_count
            FROM flights
        """,
    }

    return {
        name: _fetch_rows(sql, {"limit": limit}) if ":limit" in sql else _fetch_rows(sql)
        for name, sql in queries.items()
    }


def print_query_examples(limit: int = 10) -> None:
    """Run and print example query outputs in a readable CLI format."""

    results = run_query_examples(limit=limit)
    for name, rows in results.items():
        print(f"\n[{name}]")
        for row in rows[: min(limit, 5)]:
            print(row)


def main() -> None:
    """CLI entry point for ``make queries``."""

    print_query_examples(limit=10)


if __name__ == "__main__":
    main()
