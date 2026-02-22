"""Topology builders for decentralized peer-to-peer communication."""

from __future__ import annotations

from typing import Any

import numpy as np


Adjacency = dict[int, list[int]]


def _normalize(adjacency: dict[int, set[int]], n_agents: int) -> Adjacency:
    return {i: sorted(int(j) for j in adjacency.get(i, set()) if j != i and 0 <= j < n_agents) for i in range(n_agents)}


def build_topology(
    name: str,
    n_agents: int,
    p: float | None = None,
    seed: int | None = None,
) -> Adjacency:
    """Build deterministic adjacency list for supported topologies."""

    if n_agents < 2:
        raise ValueError("n_agents must be >= 2")
    name = str(name).lower()
    adjacency: dict[int, set[int]] = {i: set() for i in range(n_agents)}

    if name == "ring":
        for i in range(n_agents):
            left = (i - 1) % n_agents
            right = (i + 1) % n_agents
            adjacency[i].update({left, right})
    elif name == "star":
        hub = 0
        for i in range(1, n_agents):
            adjacency[hub].add(i)
            adjacency[i].add(hub)
    elif name == "fully_connected":
        for i in range(n_agents):
            adjacency[i] = {j for j in range(n_agents) if j != i}
    elif name == "random":
        rng = np.random.default_rng(seed)
        prob = float(0.25 if p is None else p)
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if rng.random() < prob:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        # Ensure no isolated agents by adding deterministic ring fallback edges.
        for i in range(n_agents):
            if not adjacency[i]:
                j = (i + 1) % n_agents
                adjacency[i].add(j)
                adjacency[j].add(i)
    else:
        raise ValueError(f"Unsupported topology: {name}")

    return _normalize(adjacency, n_agents)

