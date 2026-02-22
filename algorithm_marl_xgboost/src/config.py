"""Configuration loading and default-merging utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EXPERIMENT_CONFIG: dict[str, Any] = {
    "data": {
        "source": "UNSW_NB15_or_synthetic",
        "seed": 42,
        "n_agents": 10,
        "non_iid_alpha": 0.5,
        "focus_labels": ["normal", "anomaly"],
        "per_agent_smote": True,
        "min_pos_per_agent": 20,
        "n_samples": 2000,
        "n_numeric_features": 10,
        "anomaly_rate": 0.12,
        "train_val_split": 0.8,
    },
    "marl": {
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.98,
        "neighbor_selection": "adaptive",
        "communication_budget": 1.0,
    },
    "xgboost": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "early_stopping_rounds": 30,
    },
    "training": {
        "rounds": 5,
        "topology": "ring",
        "random_graph_p": 0.25,
        "trust_aggregation": True,
    },
    "metrics": {
        "report_per_round": True,
        "save_png_per_round": True,
        "save_txt_per_round": True,
    },
    "traffic_energy": {
        "measure_bandwidth": True,
        "measure_latency": True,
        "measure_packet_loss": True,
        "simulate_energy": True,
    },
    "artifacts": {
        "out_dir": "algorithm_marl_xgboost/reports",
        "model_dir": "algorithm_marl_xgboost/models",
        "log_path": "algorithm_marl_xgboost/logs/experiment.jsonl",
    },
}


def merge_defaults(user_config: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recursively merge user configuration over defaults."""

    base = deepcopy(defaults or DEFAULT_EXPERIMENT_CONFIG)
    for key, value in user_config.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = merge_defaults(value, base[key])
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and merge YAML config with defaults."""

    cfg_path = Path(path)
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    return merge_defaults(payload)

