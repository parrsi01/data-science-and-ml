from __future__ import annotations

from pathlib import Path
import json

import yaml

from algorithm_marl_xgboost.src.data import (
    enforce_minority_presence,
    load_or_generate_data,
    partition_dirichlet,
)
from algorithm_marl_xgboost.src.run_experiment import run_experiment
from algorithm_marl_xgboost.src.topologies import build_topology


def test_topology_builder_supported_shapes() -> None:
    ring = build_topology("ring", 5, seed=42)
    star = build_topology("star", 5, seed=42)
    fc = build_topology("fully_connected", 5, seed=42)
    rnd = build_topology("random", 5, p=0.3, seed=42)
    assert len(ring) == len(star) == len(fc) == len(rnd) == 5
    assert all(len(ring[i]) == 2 for i in range(5))
    assert len(star[0]) == 4
    assert all(len(fc[i]) == 4 for i in range(5))
    assert all(len(rnd[i]) >= 1 for i in range(5))


def test_partitioning_returns_n_agent_splits() -> None:
    cfg = {"data": {"source": "UNSW_NB15_or_synthetic", "seed": 1, "n_samples": 240, "n_numeric_features": 10, "anomaly_rate": 0.15}}
    X, y = load_or_generate_data(cfg)
    parts = partition_dirichlet(X, y, n_agents=6, alpha=0.5, seed=1)
    parts = enforce_minority_presence(parts, min_pos_per_agent=3, seed=2, max_attempts=2)
    assert len(parts) == 6
    assert sum(len(Xi) for Xi, _ in parts) >= len(X)


def test_one_round_experiment_end_to_end_creates_artifacts(tmp_path: Path) -> None:
    cfg = {
        "meta": {"author": "Simon Parris", "date": "2026-02-22", "purpose": "test"},
        "data": {
            "source": "UNSW_NB15_or_synthetic",
            "seed": 7,
            "n_agents": 4,
            "non_iid_alpha": 0.6,
            "focus_labels": ["normal", "anomaly"],
            "per_agent_smote": True,
            "min_pos_per_agent": 3,
            "n_samples": 320,
            "n_numeric_features": 10,
            "anomaly_rate": 0.15,
            "train_val_split": 0.8,
        },
        "marl": {
            "epsilon_start": 0.8,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.95,
            "neighbor_selection": "adaptive",
            "communication_budget": 1.0,
        },
        "xgboost": {
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
            "early_stopping_rounds": 10,
        },
        "training": {"rounds": 1, "topology": "ring", "random_graph_p": 0.25, "trust_aggregation": True},
        "metrics": {"report_per_round": True, "save_png_per_round": True, "save_txt_per_round": True},
        "traffic_energy": {
            "measure_bandwidth": True,
            "measure_latency": True,
            "measure_packet_loss": True,
            "simulate_energy": True,
        },
        "artifacts": {
            "out_dir": str(tmp_path / "reports"),
            "model_dir": str(tmp_path / "models"),
            "log_path": str(tmp_path / "logs" / "experiment.jsonl"),
        },
    }
    cfg_path = tmp_path / "experiment.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = run_experiment(cfg_path)
    assert Path(result["artifacts"]["jsonl_log"]).exists()
    assert Path(result["artifacts"]["final_summary_txt"]).exists()
    assert Path(result["artifacts"]["traffic_metrics_per_round_csv"]).exists()
    assert Path(result["artifacts"]["performance_comparison_boxplot_png"]).exists()
    round_json = Path(cfg["artifacts"]["out_dir"]) / "per_round" / "round_01_metrics.json"
    round_txt = Path(cfg["artifacts"]["out_dir"]) / "per_round" / "round_01_summary.txt"
    round_png = Path(cfg["artifacts"]["out_dir"]) / "per_round" / "round_01_plots.png"
    assert round_json.exists() and round_txt.exists() and round_png.exists()
    payload = json.loads(round_json.read_text(encoding="utf-8"))
    assert "aggregate_metrics" in payload and "traffic_metrics" in payload

