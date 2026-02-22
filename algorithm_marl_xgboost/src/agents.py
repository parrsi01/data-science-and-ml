"""MARL agent policies for decentralized communication decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MARLAgent:
    """Simple MARL agent using epsilon-greedy neighbor selection and trust updates."""

    agent_id: int
    epsilon: float
    epsilon_end: float
    epsilon_decay: float
    feature_names: list[str]
    trust_scores: dict[int, float] = field(default_factory=dict)
    feature_mask: np.ndarray | None = None
    last_reward: float = 0.0

    def initialize_neighbors(self, neighbors: list[int]) -> None:
        for n in neighbors:
            self.trust_scores.setdefault(int(n), 0.5)
        if self.feature_mask is None:
            self.feature_mask = np.ones(len(self.feature_names), dtype=float)

    def choose_neighbors(
        self,
        neighbor_ids: list[int],
        *,
        strategy: str,
        communication_budget: float,
        rng: np.random.Generator,
    ) -> list[int]:
        """Choose neighbors for communication under budget."""

        if not neighbor_ids:
            return []
        self.initialize_neighbors(neighbor_ids)
        neighbor_ids = sorted(int(n) for n in neighbor_ids)
        k = max(1, int(round(max(0.0, min(1.0, communication_budget)) * len(neighbor_ids))))
        k = min(k, len(neighbor_ids))

        explore = rng.random() < float(self.epsilon)
        if explore or strategy == "random":
            chosen = rng.choice(neighbor_ids, size=k, replace=False).tolist()
            return sorted(int(c) for c in chosen)

        if strategy in {"adaptive", "trust_weighted"}:
            scored = sorted(
                neighbor_ids,
                key=lambda n: (self.trust_scores.get(int(n), 0.5), -int(n)),
                reverse=True,
            )
            return sorted(scored[:k])

        raise ValueError(f"Unsupported neighbor selection strategy: {strategy}")

    def update_from_aggregate(self, aggregated_update: np.ndarray | None) -> None:
        """Update local feature mask using aggregated peer feature-importance signal."""

        if aggregated_update is None or self.feature_mask is None or len(aggregated_update) == 0:
            return
        vec = np.asarray(aggregated_update, dtype=float)
        if np.all(vec == 0):
            return
        vec = vec / (np.max(np.abs(vec)) + 1e-9)
        target = 1.0 + 0.15 * vec
        self.feature_mask = 0.9 * self.feature_mask + 0.1 * target
        self.feature_mask = np.clip(self.feature_mask, 0.7, 1.3)

    def update_after_round(
        self,
        *,
        reward: float,
        observed_neighbor_scores: dict[int, float],
    ) -> None:
        """Update trust scores and decay exploration rate."""

        self.last_reward = float(reward)
        for neighbor_id, score in observed_neighbor_scores.items():
            prev = self.trust_scores.get(int(neighbor_id), 0.5)
            normalized = max(0.0, min(1.0, float(score)))
            self.trust_scores[int(neighbor_id)] = 0.8 * prev + 0.2 * normalized
        self.epsilon = max(float(self.epsilon_end), float(self.epsilon) * float(self.epsilon_decay))

    def feature_weight_mapping(self) -> dict[str, float]:
        """Return feature weights for raw feature columns."""

        if self.feature_mask is None:
            return {name: 1.0 for name in self.feature_names}
        return {name: float(weight) for name, weight in zip(self.feature_names, self.feature_mask)}

