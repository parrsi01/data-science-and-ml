from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json

import anyio
import httpx
import pandas as pd

from projects.ops_anomaly_system.dashboard import create_app
from projects.ops_anomaly_system.drift_monitor import run_drift_monitor
from projects.ops_anomaly_system.inference import run_inference
from projects.ops_anomaly_system.quality_check import run_quality_check


def _mock_batch(n: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_time = datetime(2026, 2, 1, 0, 0, 0)
    rows = []
    model_rows = []
    cats = ["aviation", "humanitarian", "scientific", "supply_chain"]
    for i in range(n):
        ts = base_time + timedelta(minutes=i)
        delay = max(0, int((i % 17) - 3))
        pax = 100 + (i % 80)
        fuel = float((pax * (8 + (i % 11))) + (i % 5) * 10)
        dep = ["ATL", "DOH", "GVA", "FRA"][i % 4]
        arr = ["LHR", "NBO", "ZRH", "SIN"][(i + 1) % 4]
        rows.append(
            {
                "flight_id": f"TST{i:05d}",
                "dep_airport": dep,
                "arr_airport": arr,
                "scheduled_dep": ts,
                "actual_dep": ts + timedelta(minutes=delay),
                "delay_minutes": delay,
                "passenger_count": pax,
                "fuel_consumption_kg": fuel,
            }
        )
        model_rows.append(
            {
                "id": i + 1,
                "timestamp": ts,
                "metric_a": 90 + (i % 30),
                "metric_b": 40 + (i % 12),
                "category": cats[i % len(cats)],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(model_rows)


def _config(output_dir: Path) -> dict:
    return {
        "data_source": {"type": "postgres", "table": "flights"},
        "quality": {"max_schema_violation_rate": 0.01, "max_missing_rate": 0.05},
        "model": {"type": "xgboost", "model_path": "models/ml_advanced/xgboost_tuned.joblib"},
        "drift": {"threshold_ks_stat": 0.5},
        "dashboard": {"host": "0.0.0.0", "port": 8000},
        "artifacts": {"output_dir": str(output_dir)},
    }


def test_quality_check_returns_dict(tmp_path: Path) -> None:
    raw_df, _ = _mock_batch(50)
    result = run_quality_check(raw_df, _config(tmp_path), output_dir=tmp_path)
    assert isinstance(result, dict)
    assert "quality_pass" in result
    assert Path(result["artifacts"]["quality_snapshot_json"]).exists()


def test_inference_outputs_probability_column(tmp_path: Path) -> None:
    raw_df, model_df = _mock_batch(80)
    result = run_inference(raw_df, model_df, _config(tmp_path), output_dir=tmp_path)
    preds = result["predictions_df"]
    assert "anomaly_probability" in preds.columns
    assert "anomaly_label" in preds.columns
    assert Path(result["metrics"]["artifacts"]["anomaly_results_csv"]).exists()


def test_drift_monitor_returns_expected_keys(tmp_path: Path) -> None:
    _, model_df = _mock_batch(200)
    # Reuse inference prep path to generate engineered features quickly.
    from projects.ops_anomaly_system.inference import _prepare_features_for_model

    engineered = _prepare_features_for_model(model_df)
    result = run_drift_monitor(engineered, _config(tmp_path), output_dir=tmp_path)
    assert {"drift_flag", "threshold_ks_stat", "max_ks_statistic", "report"} <= set(result.keys())
    assert Path(tmp_path / "drift_snapshot.json").exists()


def test_dashboard_endpoints_return_200(tmp_path: Path) -> None:
    (tmp_path / "system_state.json").write_text(
        json.dumps(
            {
                "rows_processed": 100,
                "anomaly_rate": 0.12,
                "drift_flag": False,
                "quality_pass": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "inference_metrics.json").write_text(
        json.dumps({"anomaly_rate": 0.12}), encoding="utf-8"
    )
    (tmp_path / "quality_snapshot.json").write_text(
        json.dumps({"quality_pass": True}), encoding="utf-8"
    )
    (tmp_path / "drift_snapshot.json").write_text(
        json.dumps({"drift_flag": False}), encoding="utf-8"
    )
    (tmp_path / "anomaly_results.csv").write_text(
        "flight_id,anomaly_probability,anomaly_label\nTST00001,0.98,1\n",
        encoding="utf-8",
    )

    app = create_app(output_dir=tmp_path)

    async def _check() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/metrics")).status_code == 200
            assert (await client.get("/top-anomalies")).status_code == 200
            assert (await client.get("/")).status_code == 200

    anyio.run(_check)
