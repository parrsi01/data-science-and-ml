"""CLI runner for air traffic flow analytics and delay forecasting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from src.projects.air_traffic_delay.data import generate_air_traffic_datasets  # type: ignore[import-not-found]
    from src.projects.air_traffic_delay.delay_models import train_delay_model  # type: ignore[import-not-found]
    from src.projects.air_traffic_delay.forecasting import run_delay_forecast  # type: ignore[import-not-found]
    from src.projects.air_traffic_delay.graph_flow import (  # type: ignore[import-not-found]
        attach_node_metrics_to_flights,
        build_route_graph,
        compute_graph_metrics,
        save_graph_artifacts,
    )
    from src.projects.air_traffic_delay.reporting import write_executive_summary  # type: ignore[import-not-found]
except Exception:
    from projects.air_traffic_delay.data import generate_air_traffic_datasets
    from projects.air_traffic_delay.delay_models import train_delay_model
    from projects.air_traffic_delay.forecasting import run_delay_forecast
    from projects.air_traffic_delay.graph_flow import (
        attach_node_metrics_to_flights,
        build_route_graph,
        compute_graph_metrics,
        save_graph_artifacts,
    )
    from projects.air_traffic_delay.reporting import write_executive_summary


def load_config(config_path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def run_air_traffic_project(
    config_path: str | Path,
    *,
    dataset_dir: str | Path = "datasets",
    model_dir: str | Path = "models/air_traffic_delay",
) -> dict[str, Any]:
    """Run full air traffic simulation, graph analytics, modeling, and forecasting."""

    config = load_config(config_path)
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = generate_air_traffic_datasets(config, dataset_dir=dataset_dir)
    graph = build_route_graph(sim.routes_df)
    graph_metrics_df = compute_graph_metrics(graph)
    graph_artifacts = save_graph_artifacts(graph, graph_metrics_df, output_dir=output_dir)

    flights_with_graph = attach_node_metrics_to_flights(sim.flights_df, graph_metrics_df)
    model_result = train_delay_model(
        flights_with_graph,
        config,
        output_dir=output_dir,
        model_dir=model_dir,
    )

    forecasting_cfg = config.get("forecasting", {})
    forecast_result = run_delay_forecast(
        flights_with_graph,
        enabled=bool(forecasting_cfg.get("enabled", True)),
        horizon_days=int(forecasting_cfg.get("horizon_days", 14)),
        output_dir=output_dir,
    )

    reporting_result = write_executive_summary(
        graph_metrics_df=graph_metrics_df,
        model_result=model_result,
        forecast_result=forecast_result,
        output_dir=output_dir,
    )

    summary = {
        "config_path": str(config_path),
        "routes_csv": sim.routes_csv,
        "flights_csv": sim.flights_csv,
        "graph_artifacts": graph_artifacts,
        "model_result": model_result,
        "forecast_result": forecast_result,
        "reporting_result": reporting_result,
    }
    summary_json = output_dir / "run_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["run_summary_json"] = str(summary_json)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Air traffic flow & delay forecasting project")
    parser.add_argument("--config", default="configs/projects/air_traffic_delay.yaml")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = run_air_traffic_project(args.config)
    metrics = result["model_result"]["metrics"]
    task = result["model_result"]["task"]
    bottlenecks = result["reporting_result"]["bottleneck_airports"][:3]
    print("Air Traffic Delay Project Summary")
    print(f"task={task}")
    if task == "classification":
        print(
            "metrics: "
            f"accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}"
        )
    else:
        print(
            "metrics: "
            f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} r2={metrics['r2']:.4f}"
        )
    print(f"bottleneck_airports={bottlenecks}")
    if result["forecast_result"].get("enabled"):
        print(
            f"forecast: method={result['forecast_result']['method']} "
            f"trend={result['forecast_result']['trend_label']} "
            f"delta={result['forecast_result']['trend_delta_minutes']:.3f}"
        )
    print(f"run_summary={result['run_summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

