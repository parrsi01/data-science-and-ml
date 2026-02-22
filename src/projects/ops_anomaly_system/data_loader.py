"""Postgres-first operational data loading with offline fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any

import numpy as np
import pandas as pd

try:  # Supports `python -m src...`
    from src.data_quality.logging_config import get_logger, log_event  # type: ignore[import-not-found]
    from src.data_engineering.db import get_engine  # type: ignore[import-not-found]
except Exception:
    from data_quality.logging_config import get_logger, log_event
    from data_engineering.db import get_engine


LOGGER = get_logger("ops_anomaly.data_loader")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class LoadedBatch:
    """Operational batch with both flight-style and model-style data."""

    raw_flights_df: pd.DataFrame
    model_input_df: pd.DataFrame
    metadata: dict[str, Any]


def _airport_cycle_codes(n: int = 16) -> list[str]:
    return [
        "ATL",
        "DFW",
        "LHR",
        "DOH",
        "SIN",
        "NRT",
        "JFK",
        "CDG",
        "FRA",
        "AMS",
        "MAD",
        "DXB",
        "ORD",
        "LAX",
        "YYZ",
        "SYD",
    ][:n]


def _make_fallback_from_ml_core(limit_rows: int) -> LoadedBatch:
    """Use local ML-core dataset to simulate recent operational rows."""

    path = PROJECT_ROOT / "datasets" / "ml_core_synth.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fallback dataset missing: {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["timestamp", "id"], kind="mergesort").tail(limit_rows).reset_index(drop=True)

    categories = ["aviation", "humanitarian", "scientific", "supply_chain"]
    airport_codes = _airport_cycle_codes()
    dep_map = {
        "aviation": "ATL",
        "humanitarian": "DOH",
        "scientific": "GVA",
        "supply_chain": "FRA",
    }
    arr_map = {
        "aviation": "LHR",
        "humanitarian": "NBO",
        "scientific": "ZRH",
        "supply_chain": "SIN",
    }
    df["dep_airport"] = df["category"].map(dep_map).fillna("ATL")
    df["arr_airport"] = df["category"].map(arr_map).fillna("LHR")
    # Avoid same dep/arr in all cases; deterministic rotation.
    same_mask = df["dep_airport"] == df["arr_airport"]
    if same_mask.any():
        rotated = [airport_codes[(i + 1) % len(airport_codes)] for i in range(same_mask.sum())]
        df.loc[same_mask, "arr_airport"] = rotated

    derived_delay = np.maximum(
        0,
        np.round(
            0.25 * (df["metric_a"].to_numpy() - 90.0)
            + 0.35 * (df["metric_b"].to_numpy() - 45.0)
            + 9.0 * df["label_rare_event"].to_numpy()
        ),
    ).astype(int)
    scheduled_dep = pd.to_datetime(df["timestamp"], errors="coerce")
    actual_dep = scheduled_dep + pd.to_timedelta(derived_delay, unit="m")
    passenger_count = np.maximum(0, np.round(70 + df["metric_a"].to_numpy() * 0.9)).astype(int)
    fuel_consumption = np.maximum(
        0.0,
        passenger_count * np.maximum(5.0, df["metric_b"].to_numpy() * 0.35),
    ).astype(float)

    raw_flights_df = pd.DataFrame(
        {
            "flight_id": [f"OPS{int(i):07d}" for i in df["id"]],
            "dep_airport": df["dep_airport"].astype(str).str[:3].str.upper(),
            "arr_airport": df["arr_airport"].astype(str).str[:3].str.upper(),
            "scheduled_dep": scheduled_dep,
            "actual_dep": actual_dep,
            "delay_minutes": derived_delay.astype(int),
            "passenger_count": passenger_count.astype(int),
            "fuel_consumption_kg": fuel_consumption.astype(float),
        }
    )

    model_input_df = df[["id", "timestamp", "metric_a", "metric_b", "category"]].copy()
    metadata = {
        "source_type": "fallback_local",
        "source_name": str(path),
        "rows_loaded": int(len(raw_flights_df)),
        "reason": "PostgreSQL unavailable or table not accessible",
    }
    return LoadedBatch(raw_flights_df=raw_flights_df, model_input_df=model_input_df, metadata=metadata)


def _map_postgres_flights_to_model_inputs(raw_flights_df: pd.DataFrame) -> pd.DataFrame:
    """Create ML-core-like inputs from operational flights table rows."""

    df = raw_flights_df.copy().reset_index(drop=True)
    scheduled = pd.to_datetime(df["scheduled_dep"], errors="coerce")
    passenger = pd.to_numeric(df["passenger_count"], errors="coerce").fillna(0.0)
    fuel = pd.to_numeric(df["fuel_consumption_kg"], errors="coerce").fillna(0.0)
    delay = pd.to_numeric(df["delay_minutes"], errors="coerce").fillna(0.0)

    metric_a = np.clip(80.0 + 0.12 * passenger + 0.35 * delay, 30.0, 250.0)
    fuel_per_pax = fuel / (passenger.replace(0, 1.0))
    metric_b = np.clip(fuel_per_pax, 0.5, 120.0)

    def _route_category(dep: str, arr: str) -> str:
        key = (str(dep or "") + str(arr or "")).upper()
        classes = ["aviation", "humanitarian", "scientific", "supply_chain"]
        return classes[sum(ord(c) for c in key) % len(classes)]

    categories = [
        _route_category(dep, arr)
        for dep, arr in zip(df["dep_airport"].astype(str), df["arr_airport"].astype(str))
    ]
    return pd.DataFrame(
        {
            "id": np.arange(1, len(df) + 1, dtype=int),
            "timestamp": scheduled,
            "metric_a": metric_a.astype(float),
            "metric_b": metric_b.astype(float),
            "category": pd.Series(categories, dtype="object"),
        }
    )


def _load_from_postgres(
    *,
    table: str,
    limit_rows: int,
    window_hours: int | None,
) -> LoadedBatch:
    """Pull recent rows from Postgres and convert to dataframes."""

    engine = get_engine()
    timestamp_filter = ""
    params: dict[str, Any] = {"limit_rows": int(limit_rows)}
    if window_hours is not None:
        timestamp_filter = "WHERE scheduled_dep >= NOW() - (:window_hours || ' hours')::interval"
        params["window_hours"] = int(window_hours)

    query = f"""
        SELECT
            flight_id,
            dep_airport,
            arr_airport,
            scheduled_dep,
            actual_dep,
            delay_minutes,
            passenger_count,
            fuel_consumption_kg
        FROM {table}
        {timestamp_filter}
        ORDER BY scheduled_dep DESC, flight_id DESC
        LIMIT :limit_rows
    """
    try:
        from sqlalchemy import text  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("SQLAlchemy text() import unavailable") from exc

    with engine.connect() as conn:
        raw = pd.read_sql(text(query), conn, params=params)
    if raw.empty:
        raise RuntimeError("PostgreSQL query returned no rows")

    raw["scheduled_dep"] = pd.to_datetime(raw["scheduled_dep"], errors="coerce")
    raw["actual_dep"] = pd.to_datetime(raw["actual_dep"], errors="coerce")
    raw = raw.sort_values(["scheduled_dep", "flight_id"], kind="mergesort").reset_index(drop=True)
    model_input_df = _map_postgres_flights_to_model_inputs(raw)
    metadata = {
        "source_type": "postgres",
        "source_name": table,
        "rows_loaded": int(len(raw)),
        "window_hours": window_hours,
    }
    return LoadedBatch(raw_flights_df=raw, model_input_df=model_input_df, metadata=metadata)


def load_recent_operational_batch(
    config: dict[str, Any],
    *,
    limit_rows: int = 1000,
    window_hours: int | None = 72,
) -> LoadedBatch:
    """Load latest flights from Postgres with deterministic fallback."""

    source_cfg = config.get("data_source", {})
    table = str(source_cfg.get("table", "flights"))
    try:
        batch = _load_from_postgres(table=table, limit_rows=limit_rows, window_hours=window_hours)
        log_event(
            LOGGER,
            logging.INFO,
            event="data_load_success",
            message="Loaded recent rows from PostgreSQL",
            dataset_name=table,
            row_count=len(batch.raw_flights_df),
        )
        return batch
    except Exception as exc:
        batch = _make_fallback_from_ml_core(limit_rows=limit_rows)
        batch.metadata["fallback_error"] = str(exc)
        log_event(
            LOGGER,
            logging.WARNING,
            event="data_load_fallback",
            message=f"Falling back to local dataset: {exc}",
            dataset_name=table,
            row_count=len(batch.raw_flights_df),
            error_code="POSTGRES_FALLBACK",
        )
        return batch
