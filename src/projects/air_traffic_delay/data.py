"""Synthetic air traffic operations data simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationOutputs:
    """Paths and generated frames from simulation."""

    airports: list[str]
    routes_df: pd.DataFrame
    flights_df: pd.DataFrame
    routes_csv: str
    flights_csv: str


def simulate_airports(n_airports: int) -> list[str]:
    """Create synthetic airport identifiers like A01..A25."""

    if n_airports < 2:
        raise ValueError("n_airports must be >= 2")
    return [f"A{i:02d}" for i in range(1, n_airports + 1)]


def simulate_routes(airports: list[str], n_routes: int, seed: int) -> pd.DataFrame:
    """Generate directed routes with distance and base congestion properties."""

    if n_routes < 1:
        raise ValueError("n_routes must be >= 1")
    rng = np.random.default_rng(seed)

    all_pairs = [(dep, arr) for dep in airports for arr in airports if dep != arr]
    if n_routes > len(all_pairs):
        raise ValueError(f"n_routes={n_routes} exceeds possible directed routes={len(all_pairs)}")

    selected_idx = rng.choice(len(all_pairs), size=n_routes, replace=False)
    selected_pairs = [all_pairs[int(i)] for i in selected_idx]
    distance_km = rng.integers(250, 6000, size=n_routes)
    base_congestion = np.clip(rng.normal(loc=0.45, scale=0.18, size=n_routes), 0.05, 0.95)

    routes_df = pd.DataFrame(
        {
            "dep": [p[0] for p in selected_pairs],
            "arr": [p[1] for p in selected_pairs],
            "distance_km": distance_km.astype(int),
            "base_congestion": base_congestion.astype(float),
        }
    ).sort_values(["dep", "arr"], kind="mergesort")
    return routes_df.reset_index(drop=True)


def simulate_flights(
    routes_df: pd.DataFrame,
    n_days: int,
    flights_per_day: int,
    seed: int,
    delay_threshold_minutes: int,
) -> pd.DataFrame:
    """Generate synthetic flight operations with delay mechanisms."""

    if n_days < 1 or flights_per_day < 1:
        raise ValueError("n_days and flights_per_day must be >= 1")

    rng = np.random.default_rng(seed)
    total_flights = int(n_days * flights_per_day)
    route_choices = rng.integers(0, len(routes_df), size=total_flights)
    chosen_routes = routes_df.iloc[route_choices].reset_index(drop=True)

    start = pd.Timestamp("2026-01-01 00:00:00")
    day_offsets = np.repeat(np.arange(n_days), flights_per_day)
    minute_offsets = rng.integers(0, 24 * 60, size=total_flights)
    scheduled_time = start + pd.to_timedelta(day_offsets, unit="D") + pd.to_timedelta(
        minute_offsets, unit="m"
    )

    hour = scheduled_time.hour.to_numpy()
    dow = scheduled_time.dayofweek.to_numpy()

    # Simulate weather by day with smooth cycles + noise.
    daily_weather_base = np.clip(
        0.45
        + 0.20 * np.sin(np.linspace(0, 3 * np.pi, n_days))
        + rng.normal(0.0, 0.08, size=n_days),
        0.0,
        1.0,
    )
    weather_index = np.clip(
        daily_weather_base[day_offsets] + rng.normal(0.0, 0.07, size=total_flights), 0.0, 1.0
    )

    peak_hour = np.isin(hour, [7, 8, 9, 16, 17, 18, 19]).astype(float)
    shoulder_hour = np.isin(hour, [6, 10, 15, 20]).astype(float)
    night_penalty = np.isin(hour, [0, 1, 2, 3, 4]).astype(float)
    weekend_factor = np.isin(dow, [5, 6]).astype(float)

    congestion_index = np.clip(
        chosen_routes["base_congestion"].to_numpy(dtype=float)
        + 0.22 * peak_hour
        + 0.08 * shoulder_hour
        - 0.04 * night_penalty
        + 0.03 * weekend_factor
        + rng.normal(0.0, 0.08, size=total_flights),
        0.0,
        1.0,
    )

    distance = chosen_routes["distance_km"].to_numpy(dtype=float)
    route_base = chosen_routes["base_congestion"].to_numpy(dtype=float)

    delay_signal = (
        0.5
        + 6.0 * weather_index
        + 8.5 * congestion_index
        + 6.0 * route_base
        + 2.5 * peak_hour
        + 0.7 * shoulder_hour
        + 0.0010 * distance
        + 0.8 * weekend_factor
        + rng.normal(0.0, 7.0, size=total_flights)
    )
    delay_minutes = np.maximum(0, np.round(delay_signal)).astype(int)
    delayed = (delay_minutes >= int(delay_threshold_minutes)).astype(int)

    flights_df = pd.DataFrame(
        {
            "flight_id": [f"FLT{i:07d}" for i in range(1, total_flights + 1)],
            "dep": chosen_routes["dep"].astype(str).to_numpy(),
            "arr": chosen_routes["arr"].astype(str).to_numpy(),
            "scheduled_time": pd.to_datetime(scheduled_time),
            "weather_index": weather_index.astype(float),
            "congestion_index": congestion_index.astype(float),
            "distance_km": distance.astype(float),
            "delay_minutes": delay_minutes.astype(int),
            "delayed": delayed.astype(int),
            "hour_of_day": hour.astype(int),
            "day_of_week": dow.astype(int),
        }
    ).sort_values(["scheduled_time", "flight_id"], kind="mergesort")
    return flights_df.reset_index(drop=True)


def generate_air_traffic_datasets(
    config: dict[str, Any],
    *,
    dataset_dir: str | Path = "datasets",
) -> SimulationOutputs:
    """Run full simulation and save route + flight CSVs."""

    sim_cfg = config["simulation"]
    threshold = int(config["delay_definition"]["delay_threshold_minutes"])
    seed = int(sim_cfg["seed"])
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    airports = simulate_airports(int(sim_cfg["n_airports"]))
    routes_df = simulate_routes(airports, int(sim_cfg["n_routes"]), seed=seed)
    flights_df = simulate_flights(
        routes_df,
        n_days=int(sim_cfg["n_days"]),
        flights_per_day=int(sim_cfg["flights_per_day"]),
        seed=seed,
        delay_threshold_minutes=threshold,
    )

    routes_csv = dataset_dir / "air_traffic_routes.csv"
    flights_csv = dataset_dir / "air_traffic_flights.csv"
    routes_df.to_csv(routes_csv, index=False)
    flights_df.to_csv(flights_csv, index=False)

    return SimulationOutputs(
        airports=airports,
        routes_df=routes_df,
        flights_df=flights_df,
        routes_csv=str(routes_csv),
        flights_csv=str(flights_csv),
    )
