"""Data generation and loading for the institutional ML core pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "datasets" / "ml_core_synth.csv"


def make_synthetic_classification(n_rows: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic classification dataset with ~2% positives.

    Columns:
    - id
    - timestamp
    - metric_a
    - metric_b
    - category
    - label_rare_event
    """

    if n_rows < 100:
        raise ValueError("n_rows must be at least 100 to preserve class imbalance behavior")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq="min")
    categories = np.array(["aviation", "humanitarian", "scientific", "supply_chain"])
    category = rng.choice(categories, size=n_rows, p=[0.4, 0.25, 0.2, 0.15])

    metric_a = rng.normal(loc=100.0, scale=15.0, size=n_rows)
    metric_b = np.clip(rng.normal(loc=50.0, scale=8.0, size=n_rows), 0.1, None)

    # Create a learnable latent score correlated with category and metrics.
    category_weight_map = {
        "aviation": 0.1,
        "humanitarian": 0.0,
        "scientific": 0.45,
        "supply_chain": -0.1,
    }
    category_weights = np.vectorize(category_weight_map.get)(category)
    latent_score = (
        (metric_a - 100.0) / 15.0
        + 0.6 * ((metric_b - 50.0) / 8.0)
        + category_weights
        + rng.normal(0.0, 0.35, size=n_rows)
    )

    positive_count = max(1, int(round(n_rows * 0.02)))
    threshold = np.partition(latent_score, -positive_count)[-positive_count]
    labels = (latent_score >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1, dtype=np.int64),
            "timestamp": timestamps,
            "metric_a": metric_a.astype(float),
            "metric_b": metric_b.astype(float),
            "category": category.astype(str),
            "label_rare_event": labels.astype(int),
        }
    )

    DEFAULT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_DATASET_PATH, index=False)
    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from CSV into a DataFrame."""

    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def ensure_dataset_from_config(config: dict[str, Any]) -> pd.DataFrame:
    """Load or generate dataset based on config."""

    dataset_cfg = config["dataset"]
    source = dataset_cfg.get("source", "synthetic")
    if source == "synthetic":
        return make_synthetic_classification(
            n_rows=int(dataset_cfg["n_rows"]),
            seed=int(dataset_cfg["seed"]),
        )
    path = dataset_cfg.get("path")
    if not path:
        raise ValueError("Non-synthetic dataset source requires dataset.path")
    return load_dataset(path)
