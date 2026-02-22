"""Peer-to-peer update abstraction and decentralized aggregation routines."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_update(
    *,
    agent_id: int,
    feature_importance_vector: list[float],
    local_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build a stable update payload for peer exchange."""

    return {
        "agent_id": int(agent_id),
        "feature_importance_vector": [float(v) for v in feature_importance_vector],
        "local_metrics": {
            "f1": float(local_metrics.get("f1", 0.0)),
            "precision": float(local_metrics.get("precision", 0.0)),
            "recall": float(local_metrics.get("recall", 0.0)),
            "accuracy": float(local_metrics.get("accuracy", 0.0)),
        },
    }


def _stack_vectors(neighbor_updates: list[dict[str, Any]]) -> np.ndarray:
    if not neighbor_updates:
        return np.zeros((0, 0), dtype=float)
    vectors = [np.asarray(u["feature_importance_vector"], dtype=float) for u in neighbor_updates]
    return np.vstack(vectors)


def trust_weighted_aggregate(
    neighbor_updates: list[dict[str, Any]],
    trust_scores: dict[int, float],
) -> np.ndarray | None:
    """Aggregate peer updates using trust scores (no central server abstraction)."""

    if not neighbor_updates:
        return None
    matrix = _stack_vectors(neighbor_updates)
    if matrix.size == 0:
        return None
    weights = np.array(
        [max(1e-6, float(trust_scores.get(int(u["agent_id"]), 0.5))) for u in neighbor_updates],
        dtype=float,
    )
    weights = weights / weights.sum()
    aggregated = (matrix * weights[:, None]).sum(axis=0)
    return aggregated.astype(float)


def uniform_aggregate(neighbor_updates: list[dict[str, Any]]) -> np.ndarray | None:
    """Naive uniform aggregation baseline."""

    if not neighbor_updates:
        return None
    matrix = _stack_vectors(neighbor_updates)
    if matrix.size == 0:
        return None
    return matrix.mean(axis=0).astype(float)

