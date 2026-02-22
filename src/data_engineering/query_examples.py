"""Example institution-style SQL queries for the PostgreSQL pipeline."""

from __future__ import annotations

from typing import Any

from data_engineering.db import get_engine
from data_engineering.ingest import (
    generate_cern_events,
    generate_flights,
    generate_humanitarian_shipments,
)


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


def run_query_examples_offline(limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Run Python-based query equivalents when PostgreSQL is unavailable.

    This is intended for reference work in offline environments; the canonical
    path remains the SQL/PostgreSQL implementation in ``run_query_examples``.
    """

    flights = generate_flights(count=1000)
    shipments = generate_humanitarian_shipments(count=1000)
    events = generate_cern_events(count=1000)

    route_groups: dict[tuple[str, str], dict[str, float]] = {}
    top_delayed = sorted(
        flights,
        key=lambda r: (int(r["delay_minutes"]), r["scheduled_dep"]),
        reverse=True,
    )[:10]
    for row in flights:
        key = (str(row["dep_airport"]), str(row["arr_airport"]))
        agg = route_groups.setdefault(
            key,
            {"delay_sum": 0.0, "fuel_pp_sum": 0.0, "count": 0.0},
        )
        delay = float(row["delay_minutes"] or 0)
        fuel = float(row["fuel_consumption_kg"] or 0.0)
        pax = int(row["passenger_count"] or 0)
        agg["delay_sum"] += delay
        agg["fuel_pp_sum"] += (fuel / pax) if pax else 0.0
        agg["count"] += 1.0

    average_delay_by_route = [
        {
            "dep_airport": dep,
            "arr_airport": arr,
            "avg_delay_minutes": round(v["delay_sum"] / v["count"], 2),
            "flight_count": int(v["count"]),
        }
        for (dep, arr), v in route_groups.items()
    ]
    average_delay_by_route.sort(
        key=lambda r: (-float(r["avg_delay_minutes"]), str(r["dep_airport"]), str(r["arr_airport"]))
    )

    fuel_per_passenger_by_route = [
        {
            "dep_airport": dep,
            "arr_airport": arr,
            "avg_fuel_per_passenger_kg": round(v["fuel_pp_sum"] / v["count"], 4),
            "flight_count": int(v["count"]),
        }
        for (dep, arr), v in route_groups.items()
    ]
    fuel_per_passenger_by_route.sort(key=lambda r: -float(r["avg_fuel_per_passenger_kg"]))

    backlog: dict[tuple[str, int], dict[str, int]] = {}
    for row in shipments:
        status = str(row["status"])
        if status not in {"pending", "on_hold", "in_transit"}:
            continue
        key = (str(row["region"]), int(row["priority"]))
        agg = backlog.setdefault(key, {"shipment_count": 0, "total_quantity": 0})
        agg["shipment_count"] += 1
        agg["total_quantity"] += int(row["quantity"])
    shipment_backlog = [
        {
            "region": region,
            "priority": priority,
            "shipment_count": vals["shipment_count"],
            "total_quantity": vals["total_quantity"],
        }
        for (region, priority), vals in backlog.items()
    ]
    shipment_backlog.sort(key=lambda r: (str(r["region"]), int(r["priority"])))

    rare_rates: dict[tuple[str, str], dict[str, int]] = {}
    for row in events:
        detector = str(row["detector"])
        event_day = row["recorded_at"].date().isoformat()
        key = (detector, event_day)
        agg = rare_rates.setdefault(key, {"rare": 0, "total": 0})
        agg["rare"] += 1 if bool(row["is_rare_event"]) else 0
        agg["total"] += 1
    rare_event_rate = [
        {
            "detector": det,
            "event_day": day,
            "rare_event_rate": round(vals["rare"] / vals["total"], 4),
            "total_events": vals["total"],
        }
        for (det, day), vals in rare_rates.items()
    ]
    rare_event_rate.sort(key=lambda r: (str(r["event_day"]), str(r["detector"])))

    data_quality_checks = [
        {
            "table_name": "flights",
            "null_actual_dep_count": sum(1 for row in flights if row.get("actual_dep") is None),
            "negative_delay_count": sum(1 for row in flights if int(row["delay_minutes"]) < 0),
            "negative_passenger_count": sum(1 for row in flights if int(row["passenger_count"]) < 0),
            "negative_fuel_count": sum(1 for row in flights if float(row["fuel_consumption_kg"]) < 0),
        }
    ]

    return {
        "average_delay_by_route": average_delay_by_route[:limit],
        "top_10_delayed_flights": [
            {
                "flight_id": row["flight_id"],
                "dep_airport": row["dep_airport"],
                "arr_airport": row["arr_airport"],
                "delay_minutes": row["delay_minutes"],
                "scheduled_dep": row["scheduled_dep"],
            }
            for row in top_delayed
        ],
        "shipment_backlog_by_region_priority": shipment_backlog[:limit],
        "rare_event_rate_per_detector_per_day": rare_event_rate[:limit],
        "fuel_per_passenger_by_route": fuel_per_passenger_by_route[:limit],
        "data_quality_checks": data_quality_checks,
    }


def print_query_examples_offline(limit: int = 10) -> None:
    """Print offline query-example equivalents for reference use."""

    results = run_query_examples_offline(limit=limit)
    for name, rows in results.items():
        print(f"\n[{name}]")
        for row in rows[: min(limit, 5)]:
            print(row)


def main() -> None:
    """CLI entry point for ``make queries``."""

    try:
        print_query_examples(limit=10)
    except Exception as exc:
        print(f"Database query path unavailable ({exc}); falling back to offline reference queries.")
        print_query_examples_offline(limit=10)


if __name__ == "__main__":
    main()
