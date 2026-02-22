"""Data loading/generation and non-IID partitioning for MARL-XGB experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from imblearn.over_sampling import SMOTE  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    SMOTE = None  # type: ignore[assignment]


RAW_NUMERIC_COLUMNS = [f"f{i:02d}" for i in range(10)]
RAW_CATEGORICAL_COLUMNS = ["proto", "service", "segment"]


@dataclass
class AgentSplit:
    """Per-agent dataset split container."""

    X: pd.DataFrame
    y: pd.Series


def load_or_generate_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Load UNSW_NB15 if available or generate deterministic synthetic anomaly data."""

    data_cfg = config["data"]
    source = str(data_cfg.get("source", "UNSW_NB15_or_synthetic"))
    if source != "UNSW_NB15_or_synthetic":
        raise ValueError("This institutional module currently supports synthetic generation only")

    rng = np.random.default_rng(int(data_cfg["seed"]))
    n_samples = int(data_cfg.get("n_samples", 2000))
    n_numeric = int(data_cfg.get("n_numeric_features", 10))
    anomaly_rate = float(data_cfg.get("anomaly_rate", 0.12))
    if n_numeric < 4:
        raise ValueError("n_numeric_features must be >= 4")

    numeric_cols = [f"f{i:02d}" for i in range(n_numeric)]
    proto = rng.choice(["tcp", "udp", "icmp"], size=n_samples, p=[0.6, 0.3, 0.1])
    service = rng.choice(["web", "dns", "ssh", "db", "iot"], size=n_samples, p=[0.35, 0.2, 0.15, 0.2, 0.1])
    segment = rng.choice(["edge", "core", "dmz"], size=n_samples, p=[0.5, 0.35, 0.15])

    X_num = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_numeric))
    # Embed structured signal and non-linear interactions.
    latent = (
        1.2 * X_num[:, 0]
        - 0.8 * X_num[:, 1]
        + 0.6 * X_num[:, 2] * X_num[:, 3]
        + 0.5 * (proto == "icmp").astype(float)
        + 0.35 * (service == "iot").astype(float)
        + 0.4 * (segment == "dmz").astype(float)
        + rng.normal(0.0, 0.35, size=n_samples)
    )
    positive_count = max(1, int(round(n_samples * anomaly_rate)))
    threshold = np.partition(latent, -positive_count)[-positive_count]
    y = (latent >= threshold).astype(int)

    X = pd.DataFrame(X_num, columns=numeric_cols)
    for col in RAW_NUMERIC_COLUMNS:
        if col not in X.columns:
            X[col] = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    X = X[RAW_NUMERIC_COLUMNS]
    X["proto"] = proto.astype(str)
    X["service"] = service.astype(str)
    X["segment"] = segment.astype(str)
    X["row_id"] = np.arange(1, n_samples + 1, dtype=np.int64)

    y_series = pd.Series(y, name="label", dtype="int64")
    return X.reset_index(drop=True), y_series.reset_index(drop=True)


def partition_dirichlet(
    X: pd.DataFrame,
    y: pd.Series,
    n_agents: int,
    alpha: float,
    seed: int,
) -> list[tuple[pd.DataFrame, pd.Series]]:
    """Partition data by label using Dirichlet-distributed proportions (non-IID)."""

    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    if n_agents < 2:
        raise ValueError("n_agents must be >= 2")

    rng = np.random.default_rng(seed)
    partitions_idx: list[list[int]] = [[] for _ in range(n_agents)]
    y_np = y.to_numpy()
    labels = np.unique(y_np)
    for label in labels:
        label_idx = np.where(y_np == label)[0]
        rng.shuffle(label_idx)
        props = rng.dirichlet(np.full(n_agents, float(alpha)))
        cuts = np.floor(np.cumsum(props) * len(label_idx)).astype(int)
        splits = np.split(label_idx, cuts[:-1])
        for agent_id, split in enumerate(splits):
            partitions_idx[agent_id].extend(int(i) for i in split.tolist())

    partitions: list[tuple[pd.DataFrame, pd.Series]] = []
    for idxs in partitions_idx:
        idxs_sorted = sorted(idxs)
        Xi = X.iloc[idxs_sorted].reset_index(drop=True)
        yi = y.iloc[idxs_sorted].reset_index(drop=True)
        partitions.append((Xi, yi))
    return partitions


def enforce_minority_presence(
    partitions: list[tuple[pd.DataFrame, pd.Series]],
    *,
    min_pos_per_agent: int,
    seed: int,
    max_attempts: int = 5,
) -> list[tuple[pd.DataFrame, pd.Series]]:
    """Ensure each agent has enough minority samples using retries and deterministic repair."""

    if min_pos_per_agent <= 0:
        return [(X.reset_index(drop=True), y.reset_index(drop=True)) for X, y in partitions]

    rng = np.random.default_rng(seed)
    repaired: list[tuple[pd.DataFrame, pd.Series]] = [(X.copy(), y.copy()) for X, y in partitions]

    for _attempt in range(max_attempts):
        pos_counts = [int((y == 1).sum()) for _, y in repaired]
        if all(count >= min_pos_per_agent for count in pos_counts):
            return [(X.reset_index(drop=True), y.reset_index(drop=True)) for X, y in repaired]

        donor_pool: list[pd.DataFrame] = []
        for X, y in repaired:
            donor_pool.append(X.loc[y == 1].copy())
        donor_df = pd.concat(donor_pool, ignore_index=True) if donor_pool else pd.DataFrame()
        if donor_df.empty:
            break

        for agent_id, (X, y) in enumerate(repaired):
            pos_count = int((y == 1).sum())
            deficit = max(0, min_pos_per_agent - pos_count)
            if deficit == 0:
                continue
            take_idx = rng.integers(0, len(donor_df), size=deficit)
            add_X = donor_df.iloc[take_idx].copy().reset_index(drop=True)
            # Keep deterministic uniqueness of row_id surrogate.
            add_X["row_id"] = add_X["row_id"].astype(int) * 10_000 + np.arange(1, deficit + 1)
            add_y = pd.Series(np.ones(deficit, dtype="int64"), name=y.name)
            repaired[agent_id] = (
                pd.concat([X, add_X], ignore_index=True).reset_index(drop=True),
                pd.concat([y, add_y], ignore_index=True).reset_index(drop=True),
            )

    return [(X.reset_index(drop=True), y.reset_index(drop=True)) for X, y in repaired]


def split_train_val(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Deterministic stratified train/validation split without sklearn dependency."""

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    y_np = y.to_numpy()
    for label in np.unique(y_np):
        idxs = np.where(y_np == label)[0]
        rng.shuffle(idxs)
        cut = max(1, min(len(idxs) - 1, int(round(len(idxs) * float(train_fraction))))) if len(idxs) > 1 else len(idxs)
        train_idx.extend(int(i) for i in idxs[:cut])
        val_idx.extend(int(i) for i in idxs[cut:])
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    if not val_idx:
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1] or train_idx
    return (
        X.iloc[train_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        X.iloc[val_idx].reset_index(drop=True),
        y.iloc[val_idx].reset_index(drop=True),
    )


def apply_smote_training_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    enabled: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Optionally apply SMOTE (or deterministic fallback oversampling) on training data only."""

    if not enabled:
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True)

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    if pos_count < 2 or pos_count >= neg_count:
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True)

    cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    if SMOTE is not None and not cat_cols:
        smote = SMOTE(random_state=seed)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)

    # Deterministic fallback random oversampling.
    rng = np.random.default_rng(seed)
    deficit = neg_count - pos_count
    pos_df = X_train.loc[y_train == 1].reset_index(drop=True)
    take_idx = rng.integers(0, len(pos_df), size=deficit)
    add_X = pos_df.iloc[take_idx].copy().reset_index(drop=True)
    add_y = pd.Series(np.ones(deficit, dtype="int64"), name=y_train.name)
    X_out = pd.concat([X_train.reset_index(drop=True), add_X], ignore_index=True).reset_index(drop=True)
    y_out = pd.concat([y_train.reset_index(drop=True), add_y], ignore_index=True).reset_index(drop=True)
    return X_out, y_out
