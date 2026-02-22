"""Reward functions for MARL communication-policy learning."""

from __future__ import annotations

from typing import Any


def compute_reward(
    *,
    f1_improvement: float,
    communication_cost: float,
    energy_cost: float,
    comm_penalty_weight: float = 0.15,
    energy_penalty_weight: float = 0.05,
) -> dict[str, float]:
    """Combine performance gain and communication/energy penalties."""

    reward = (
        float(f1_improvement)
        - float(comm_penalty_weight) * float(communication_cost)
        - float(energy_penalty_weight) * float(energy_cost)
    )
    return {
        "reward": float(reward),
        "f1_improvement": float(f1_improvement),
        "communication_penalty": float(comm_penalty_weight) * float(communication_cost),
        "energy_penalty": float(energy_penalty_weight) * float(energy_cost),
    }

