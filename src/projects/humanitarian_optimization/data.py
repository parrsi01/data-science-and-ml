"""Synthetic humanitarian demand generation for optimization experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_regional_demand(
    config: dict[str, Any],
    output_path: str | Path = "datasets/humanitarian_demand.csv",
) -> pd.DataFrame:
    """Generate reproducible regional humanitarian demand inputs."""

    regions = list(config["regions"])
    seed = int(config["demand_model"]["seed"])
    variance = float(config["demand_model"]["demand_variance"])
    rng = np.random.default_rng(seed)

    base_demand = rng.integers(7_500, 20_000, size=len(regions))
    demand_multiplier = rng.normal(loc=1.0, scale=max(variance, 1e-6), size=len(regions))
    demand_units = np.maximum(
        1_000, np.round(base_demand * np.clip(demand_multiplier, 0.4, 1.8))
    ).astype(int)

    priority_scores = rng.integers(2, 6, size=len(regions))
    if len(priority_scores) > 0:
        priority_scores[int(rng.integers(0, len(priority_scores)))] = 5

    cost_per_unit = np.round(
        rng.uniform(9.0, 35.0, size=len(regions)) * (1.0 + variance * 0.5), 2
    )
    risk_index = np.round(
        np.clip(
            0.25 + 0.12 * (priority_scores - 1) + rng.normal(0.0, 0.12, size=len(regions)),
            0.0,
            1.0,
        ),
        3,
    )

    df = pd.DataFrame(
        {
            "region": regions,
            "demand_units": demand_units,
            "priority_score": priority_scores.astype(int),
            "cost_per_unit": cost_per_unit.astype(float),
            "risk_index": risk_index.astype(float),
        }
    ).sort_values(
        ["priority_score", "region"], ascending=[False, True], kind="mergesort"
    )
    df = df.reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

