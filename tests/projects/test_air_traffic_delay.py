from __future__ import annotations

from pathlib import Path

import yaml

from projects.air_traffic_delay.data import generate_air_traffic_datasets
from projects.air_traffic_delay.graph_flow import build_route_graph, compute_graph_metrics
from projects.air_traffic_delay.run_project import run_air_traffic_project


def _tiny_config(output_dir: Path) -> dict:
    return {
        "simulation": {
            "seed": 11,
            "n_airports": 8,
            "n_routes": 16,
            "n_days": 14,
            "flights_per_day": 120,
        },
        "delay_definition": {"delay_threshold_minutes": 15},
        "modeling": {"task": "classification", "model": "xgboost", "test_size": 0.2},
        "forecasting": {"enabled": True, "horizon_days": 5},
        "artifacts": {"output_dir": str(output_dir)},
    }


def test_data_simulation_returns_expected_columns(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path / "reports")
    sim = generate_air_traffic_datasets(cfg, dataset_dir=tmp_path / "datasets")
    assert {"dep", "arr", "distance_km", "base_congestion"} <= set(sim.routes_df.columns)
    assert {
        "flight_id",
        "dep",
        "arr",
        "scheduled_time",
        "weather_index",
        "congestion_index",
        "distance_km",
        "delay_minutes",
        "delayed",
    } <= set(sim.flights_df.columns)


def test_graph_metrics_produced(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path / "reports")
    sim = generate_air_traffic_datasets(cfg, dataset_dir=tmp_path / "datasets")
    graph = build_route_graph(sim.routes_df)
    metrics = compute_graph_metrics(graph)
    assert {"airport", "in_degree", "out_degree", "betweenness_centrality", "pagerank", "clustering"} <= set(
        metrics.columns
    )
    assert len(metrics) > 0


def test_run_project_writes_model_and_forecast_artifacts(tmp_path: Path) -> None:
    cfg_path = tmp_path / "air_traffic.yaml"
    cfg_path.write_text(yaml.safe_dump(_tiny_config(tmp_path / "reports")), encoding="utf-8")
    result = run_air_traffic_project(
        cfg_path,
        dataset_dir=tmp_path / "datasets",
        model_dir=tmp_path / "models",
    )
    out_dir = Path(result["forecast_result"]["artifacts"]["forecast_csv"]).parent
    assert Path(result["model_result"]["metrics_json"]).exists()
    assert Path(result["graph_artifacts"]["graph_metrics_csv"]).exists()
    assert Path(result["graph_artifacts"]["route_graph_png"]).exists()
    assert Path(result["forecast_result"]["artifacts"]["forecast_csv"]).exists()
    assert Path(result["forecast_result"]["artifacts"]["forecast_png"]).exists()
    assert (out_dir / "executive_summary.md").exists()

