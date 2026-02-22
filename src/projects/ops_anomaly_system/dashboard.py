"""Minimal FastAPI dashboard for operational anomaly system metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import yaml


DEFAULT_OUTPUT_DIR = Path("reports/projects/ops_anomaly_system")


def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _load_top_anomalies_from_csv(path: Path, limit: int = 10) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append(dict(row))
    return rows


def build_dashboard_state(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    quality = _load_json(output_dir / "quality_snapshot.json", {})
    inference = _load_json(output_dir / "inference_metrics.json", {})
    drift = _load_json(output_dir / "drift_snapshot.json", {})
    system = _load_json(output_dir / "system_state.json", {})
    anomalies = _load_top_anomalies_from_csv(output_dir / "anomaly_results.csv", limit=10)
    return {
        "quality": quality,
        "inference": inference,
        "drift": drift,
        "system": system,
        "top_anomalies": anomalies,
    }


def create_app(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> FastAPI:
    """Create FastAPI app serving operational metrics and top anomalies."""

    app = FastAPI(title="Operational Anomaly System Dashboard", version="1.0.0")
    app.state.output_dir = str(output_dir)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        state = build_dashboard_state(app.state.output_dir)
        system = state.get("system", {})
        return {
            "status": "ok",
            "quality_flag": bool(system.get("quality_pass", False)),
            "drift_flag": bool(system.get("drift_flag", False)),
            "rows_processed": int(system.get("rows_processed", 0)),
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        state = build_dashboard_state(app.state.output_dir)
        system = state.get("system", {})
        inference = state.get("inference", {})
        anomaly_rate = float(inference.get("anomaly_rate", inference.get("metrics", {}).get("anomaly_rate", 0.0)))
        return {
            "anomaly_rate": anomaly_rate,
            "drift_flag": bool(system.get("drift_flag", False)),
            "quality_flag": bool(system.get("quality_pass", False)),
        }

    @app.get("/top-anomalies")
    async def top_anomalies() -> JSONResponse:
        state = build_dashboard_state(app.state.output_dir)
        return JSONResponse(content={"rows": state.get("top_anomalies", [])})

    @app.get("/", response_class=HTMLResponse)
    async def home() -> str:
        state = build_dashboard_state(app.state.output_dir)
        system = state.get("system", {})
        inference = state.get("inference", {})
        anomaly_rate = float(inference.get("anomaly_rate", inference.get("metrics", {}).get("anomaly_rate", 0.0)))
        quality_flag = bool(system.get("quality_pass", False))
        drift_flag = bool(system.get("drift_flag", False))
        rows_processed = int(system.get("rows_processed", 0))
        return f"""
        <html>
          <head><title>Operational Anomaly Dashboard</title></head>
          <body style="font-family: sans-serif; margin: 2rem;">
            <h1>Operational Anomaly Dashboard</h1>
            <p><strong>Rows Processed:</strong> {rows_processed}</p>
            <p><strong>Anomaly Rate:</strong> {anomaly_rate:.4f}</p>
            <p><strong>Quality Status:</strong> {"PASS" if quality_flag else "FAIL"}</p>
            <p><strong>Drift Status:</strong> {"DRIFT DETECTED" if drift_flag else "NO DRIFT FLAG"}</p>
            <p>Endpoints: <code>/health</code>, <code>/metrics</code>, <code>/top-anomalies</code></p>
          </body>
        </html>
        """

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ops anomaly dashboard")
    parser.add_argument("--config", default="configs/projects/ops_anomaly_system.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = args.output_dir or cfg["artifacts"]["output_dir"]
    host = str(cfg.get("dashboard", {}).get("host", "0.0.0.0"))
    port = int(cfg.get("dashboard", {}).get("port", 8000))
    app = create_app(output_dir=output_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
