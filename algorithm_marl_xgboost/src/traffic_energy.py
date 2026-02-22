"""Traffic and energy metric simulation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def simulate_bandwidth(bytes_sent: int, rng: np.random.Generator) -> dict[str, float]:
    """Simulate bandwidth usage from payload bytes."""

    jitter = float(rng.uniform(0.95, 1.05))
    return {
        "bytes_sent": float(bytes_sent),
        "kb_sent": float(bytes_sent) / 1024.0,
        "bandwidth_score": (float(bytes_sent) / 4096.0) * jitter,
    }


def simulate_latency(ms_base: float, congestion_factor: float, rng: np.random.Generator) -> float:
    """Simulate latency with congestion sensitivity."""

    noise = float(rng.normal(0.0, 1.5))
    return max(0.0, float(ms_base) * (1.0 + float(congestion_factor)) + noise)


def simulate_packet_loss(p_base: float, rng: np.random.Generator) -> float:
    """Simulate packet loss as a probability estimate."""

    return float(np.clip(float(p_base) + rng.normal(0.0, 0.005), 0.0, 1.0))


def simulate_energy(communication_cost: float, training_cost: float) -> float:
    """Simulate energy cost from communication and local training activity."""

    return float(0.8 * communication_cost + 0.02 * training_cost)


def summarize_round_traffic(events: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate traffic/energy metrics across communication events."""

    if not events:
        return {
            "bytes_sent_total": 0.0,
            "avg_latency_ms": 0.0,
            "packet_loss_mean": 0.0,
            "energy_total": 0.0,
            "n_events": 0.0,
        }
    return {
        "bytes_sent_total": float(sum(float(e.get("bytes_sent", 0.0)) for e in events)),
        "avg_latency_ms": float(np.mean([float(e.get("latency_ms", 0.0)) for e in events])),
        "packet_loss_mean": float(np.mean([float(e.get("packet_loss", 0.0)) for e in events])),
        "energy_total": float(sum(float(e.get("energy_j", 0.0)) for e in events)),
        "n_events": float(len(events)),
    }

