"""Operational anomaly system runner: load, quality, inference, drift, dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any

import uvicorn
import yaml

try:
    from src.projects.ops_anomaly_system.dashboard import create_app  # type: ignore[import-not-found]
    from src.projects.ops_anomaly_system.data_loader import load_recent_operational_batch  # type: ignore[import-not-found]
    from src.projects.ops_anomaly_system.drift_monitor import run_drift_monitor  # type: ignore[import-not-found]
    from src.projects.ops_anomaly_system.inference import run_inference  # type: ignore[import-not-found]
    from src.projects.ops_anomaly_system.quality_check import run_quality_check  # type: ignore[import-not-found]
except Exception:
    from projects.ops_anomaly_system.dashboard import create_app
    from projects.ops_anomaly_system.data_loader import load_recent_operational_batch
    from projects.ops_anomaly_system.drift_monitor import run_drift_monitor
    from projects.ops_anomaly_system.inference import run_inference
    from projects.ops_anomaly_system.quality_check import run_quality_check


def load_config(config_path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def run_ops_system(
    config_path: str | Path,
    *,
    limit_rows: int = 1000,
    window_hours: int | None = 72,
) -> dict[str, Any]:
    """Execute quality, inference, and drift checks and write system artifacts."""

    config = load_config(config_path)
    output_dir = Path(config["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = load_recent_operational_batch(config, limit_rows=limit_rows, window_hours=window_hours)
    quality = run_quality_check(batch.raw_flights_df, config, output_dir=output_dir)

    inference_result: dict[str, Any] | None = None
    drift_result: dict[str, Any] | None = None
    if quality["quality_pass"]:
        inference_result = run_inference(
            batch.raw_flights_df, batch.model_input_df, config, output_dir=output_dir
        )
        drift_result = run_drift_monitor(
            inference_result["engineered_features_df"], config, output_dir=output_dir
        )

    anomaly_rate = (
        float(inference_result["metrics"]["anomaly_rate"]) if inference_result is not None else 0.0
    )
    drift_flag = bool(drift_result["drift_flag"]) if drift_result is not None else False
    quality_pass = bool(quality["quality_pass"])
    rows_processed = int(len(batch.raw_flights_df))

    system_state = {
        "rows_processed": rows_processed,
        "anomaly_rate": anomaly_rate,
        "drift_flag": drift_flag,
        "quality_pass": quality_pass,
        "data_source": batch.metadata,
        "artifacts": {
            "quality_snapshot": quality["artifacts"]["quality_snapshot_json"],
            "anomaly_results_csv": (
                inference_result["metrics"]["artifacts"]["anomaly_results_csv"]
                if inference_result
                else None
            ),
            "inference_metrics_json": (
                inference_result["metrics"]["artifacts"]["inference_metrics_json"]
                if inference_result
                else None
            ),
            "drift_snapshot_json": str(output_dir / "drift_snapshot.json") if drift_result else None,
        },
    }
    system_state_path = output_dir / "system_state.json"
    system_state_path.write_text(json.dumps(system_state, indent=2, default=str), encoding="utf-8")
    system_state["artifacts"]["system_state_json"] = str(system_state_path)
    return system_state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run operational anomaly system")
    parser.add_argument("--config", default="configs/projects/ops_anomaly_system.yaml")
    parser.add_argument("--limit-rows", type=int, default=1000)
    parser.add_argument("--window-hours", type=int, default=72)
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI dashboard after processing")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    state = run_ops_system(
        args.config,
        limit_rows=int(args.limit_rows),
        window_hours=int(args.window_hours),
    )

    print("Operational Anomaly System Summary")
    print(f"rows processed: {state['rows_processed']}")
    print(f"anomaly rate: {state['anomaly_rate']:.4f}")
    print(f"drift detected? {'yes' if state['drift_flag'] else 'no'}")
    print(f"quality pass? {'yes' if state['quality_pass'] else 'no'}")
    print(f"data source: {state['data_source']['source_type']} ({state['data_source']['source_name']})")
    print(f"state file: {state['artifacts']['system_state_json']}")

    if args.serve:
        host = str(config.get("dashboard", {}).get("host", "0.0.0.0"))
        port = int(config.get("dashboard", {}).get("port", 8000))
        print(f"Launching dashboard on http://127.0.0.1:{port}")
        app = create_app(output_dir=config["artifacts"]["output_dir"])
        uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

