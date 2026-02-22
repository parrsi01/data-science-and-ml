from __future__ import annotations

from datetime import datetime
import json

from data_quality.quality_metrics import compute_quality_metrics
from data_quality.run_quality_gate import evaluate_quality_gate
from data_quality.schemas import FlightRow
from data_quality.validators import enforce_domain_rules, validate_dataframe


def _tiny_flights() -> list[dict[str, object]]:
    return [
        {
            "flight_id": "FL001",
            "dep_airport": "JFK",
            "arr_airport": "LHR",
            "scheduled_dep": datetime(2026, 1, 1, 10, 0, 0),
            "actual_dep": datetime(2026, 1, 1, 10, 5, 0),
            "delay_minutes": 5,
            "passenger_count": 180,
            "fuel_consumption_kg": 7200.0,
        },
        {
            # Schema-invalid: bad airport length and negative passenger_count.
            "flight_id": "FL002",
            "dep_airport": "AB",
            "arr_airport": "CDG",
            "scheduled_dep": datetime(2026, 1, 1, 11, 0, 0),
            "actual_dep": datetime(2026, 1, 1, 10, 55, 0),
            "delay_minutes": 0,
            "passenger_count": -1,
            "fuel_consumption_kg": 6500.0,
        },
        {
            # Domain-invalid only: same airport.
            "flight_id": "FL003",
            "dep_airport": "FRA",
            "arr_airport": "FRA",
            "scheduled_dep": datetime(2026, 1, 1, 12, 0, 0),
            "actual_dep": datetime(2026, 1, 1, 12, 3, 0),
            "delay_minutes": 3,
            "passenger_count": 150,
            "fuel_consumption_kg": 6800.0,
        },
    ]


def test_invalid_rows_detected() -> None:
    df = _tiny_flights()
    valid_df, invalid_df = validate_dataframe(df, FlightRow, "flights")
    assert len(valid_df) == 2
    assert len(invalid_df) == 1
    valid_domain_df, domain_invalid_df = enforce_domain_rules(valid_df, "flights")
    assert len(valid_domain_df) == 1
    assert len(domain_invalid_df) == 1


def test_metrics_keys_exist() -> None:
    df = _tiny_flights()
    valid_df, schema_invalid_df = validate_dataframe(df, FlightRow, "flights")
    valid_domain_df, domain_invalid_df = enforce_domain_rules(valid_df, "flights")
    metrics = compute_quality_metrics(
        dataset_name="flights",
        raw_df=df,
        valid_df=valid_domain_df,
        schema_invalid_df=schema_invalid_df,
        domain_invalid_df=domain_invalid_df,
        primary_id_col="flight_id",
    )
    for key in [
        "missing_rate_per_column",
        "duplicate_rate",
        "outlier_rate_per_numeric_column",
        "schema_violation_rate",
        "drift_snapshot",
    ]:
        assert key in metrics
    json.dumps(metrics, default=str)


def test_gate_fails_when_violations_exceed_threshold() -> None:
    bad_flights = [_tiny_flights()[0], _tiny_flights()[0], _tiny_flights()[1]]
    datasets = {
        "flights": bad_flights,
        "humanitarian_shipments": [
            {
                "shipment_id": "HS1",
                "region": "Sahel",
                "item_type": "medical_kit",
                "quantity": 1,
                "priority": 3,
                "status": "pending",
            }
        ],
        "cern_events": [
            {
                "event_id": "CE1",
                "detector": "ATLAS",
                "energy_gev": 100.0,
                "is_rare_event": False,
                "recorded_at": datetime(2026, 1, 1, 0, 0, 0),
            }
        ],
    }
    summary = evaluate_quality_gate(datasets)
    assert summary["passed"] is False
    assert "flights" in summary["failed_datasets"]
