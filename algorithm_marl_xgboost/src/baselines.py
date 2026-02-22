"""Baseline methods for comparison against MARL-guided decentralized aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from algorithm_marl_xgboost.src.data import apply_smote_training_only, split_train_val
from algorithm_marl_xgboost.src.decentralized_agg import uniform_aggregate
from algorithm_marl_xgboost.src.traffic_energy import (
    simulate_bandwidth,
    simulate_energy,
    simulate_latency,
    simulate_packet_loss,
    summarize_round_traffic,
)
from algorithm_marl_xgboost.src.xgb_local import train_local_xgb


def _mean_metric(local_metrics: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(m.get(key, 0.0)) for m in local_metrics])) if local_metrics else 0.0


def run_local_only_baseline(
    agent_partitions: list[tuple[pd.DataFrame, pd.Series]],
    config: dict[str, Any],
    *,
    raw_feature_order: list[str],
    seed: int,
) -> dict[str, Any]:
    """Train local XGBoost on each agent with no communication."""

    local_metrics: list[dict[str, Any]] = []
    for agent_id, (X, y) in enumerate(agent_partitions):
        X_train, y_train, X_val, y_val = split_train_val(
            X, y, train_fraction=float(config["data"].get("train_val_split", 0.8)), seed=seed + agent_id
        )
        X_train, y_train = apply_smote_training_only(
            X_train, y_train, enabled=bool(config["data"].get("per_agent_smote", True)), seed=seed + agent_id
        )
        _, metrics = train_local_xgb(
            X_train, y_train, X_val, y_val, config, seed=seed + agent_id, raw_feature_order=raw_feature_order
        )
        local_metrics.append(metrics)
    return {
        "name": "local_only_xgboost",
        "metrics": {
            "accuracy": _mean_metric(local_metrics, "accuracy"),
            "precision": _mean_metric(local_metrics, "precision"),
            "recall": _mean_metric(local_metrics, "recall"),
            "f1": _mean_metric(local_metrics, "f1"),
            "roc_auc": _mean_metric(local_metrics, "roc_auc"),
        },
        "traffic": {"bytes_sent_total": 0.0, "avg_latency_ms": 0.0, "packet_loss_mean": 0.0, "energy_total": 0.0, "n_events": 0.0},
    }


def run_naive_decentralized_baseline(
    agent_partitions: list[tuple[pd.DataFrame, pd.Series]],
    adjacency: dict[int, list[int]],
    config: dict[str, Any],
    *,
    raw_feature_order: list[str],
    seed: int,
) -> dict[str, Any]:
    """Naive decentralized baseline with uniform neighbor aggregation (no MARL)."""

    rng = np.random.default_rng(seed)
    local_metrics: list[dict[str, Any]] = []
    updates: dict[int, dict[str, Any]] = {}
    for agent_id, (X, y) in enumerate(agent_partitions):
        X_train, y_train, X_val, y_val = split_train_val(
            X, y, train_fraction=float(config["data"].get("train_val_split", 0.8)), seed=seed + 100 + agent_id
        )
        X_train, y_train = apply_smote_training_only(
            X_train, y_train, enabled=bool(config["data"].get("per_agent_smote", True)), seed=seed + 200 + agent_id
        )
        _, metrics = train_local_xgb(
            X_train, y_train, X_val, y_val, config, seed=seed + 300 + agent_id, raw_feature_order=raw_feature_order
        )
        local_metrics.append(metrics)
        updates[agent_id] = {
            "agent_id": agent_id,
            "feature_importance_vector": metrics["feature_importance_vector"],
            "local_metrics": metrics,
        }

    traffic_events: list[dict[str, Any]] = []
    for agent_id, neighbors in adjacency.items():
        received = [updates[n] for n in neighbors if n in updates]
        _ = uniform_aggregate(received)
        bytes_sent = 8 * len(raw_feature_order) * len(neighbors)
        bw = simulate_bandwidth(bytes_sent, rng)
        lat = simulate_latency(8.0, congestion_factor=min(1.0, len(neighbors) / 5.0), rng=rng)
        loss = simulate_packet_loss(0.01, rng)
        energy = simulate_energy(bw["bandwidth_score"], training_cost=len(agent_partitions[agent_id][0]))
        traffic_events.append(
            {
                "agent_id": agent_id,
                "bytes_sent": bw["bytes_sent"],
                "latency_ms": lat,
                "packet_loss": loss,
                "energy_j": energy,
            }
        )
    return {
        "name": "naive_decentralized_uniform",
        "metrics": {
            "accuracy": _mean_metric(local_metrics, "accuracy"),
            "precision": _mean_metric(local_metrics, "precision"),
            "recall": _mean_metric(local_metrics, "recall"),
            "f1": _mean_metric(local_metrics, "f1"),
            "roc_auc": _mean_metric(local_metrics, "roc_auc"),
        },
        "traffic": summarize_round_traffic(traffic_events),
    }

