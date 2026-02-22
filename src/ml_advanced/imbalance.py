"""Imbalance handling decisions and SMOTE utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE


LOGGER = logging.getLogger("ml_advanced.imbalance")


def choose_imbalance_strategy(y_train, config: dict[str, Any]) -> dict[str, Any]:
    """Choose imbalance strategy for training based on data and config.

    Returns keys:
    - use_smote
    - use_class_weights
    - scale_pos_weight
    - reason
    """

    y = np.asarray(y_train)
    positives = int((y == 1).sum())
    total = int(len(y))
    negatives = total - positives
    positive_rate = (positives / total) if total else 0.0
    scale_pos_weight = float(negatives / positives) if positives > 0 else 1.0

    cfg = config.get("imbalance", {})
    requested = str(cfg.get("strategy", "auto")).lower()
    k_neighbors = int(cfg.get("smote_k_neighbors", 5))
    can_smote = positives > max(5, k_neighbors)

    use_smote = False
    use_class_weights = False
    reason = ""

    if requested == "none":
        reason = "imbalance.strategy=none"
    elif requested == "weights":
        use_class_weights = True
        reason = "imbalance.strategy=weights"
    elif requested == "smote":
        use_smote = can_smote
        use_class_weights = not can_smote
        reason = "imbalance.strategy=smote" if can_smote else "smote_requested_but_insufficient_positives_fallback_to_weights"
    elif requested == "auto":
        if positive_rate < 0.01:
            use_class_weights = True
            reason = "auto: positives <1%, prefer weights/scale_pos_weight"
        elif can_smote and total >= 1000:
            use_smote = True
            reason = "auto: moderate rare class and sufficient samples, using SMOTE on training split"
        else:
            use_class_weights = True
            reason = "auto: fallback to class weights"
    else:
        raise ValueError(f"Unknown imbalance strategy: {requested}")

    decision = {
        "use_smote": bool(use_smote),
        "use_class_weights": bool(use_class_weights),
        "scale_pos_weight": float(scale_pos_weight),
        "positive_rate": float(positive_rate),
        "positives": positives,
        "negatives": negatives,
        "reason": reason,
    }
    LOGGER.info("imbalance_decision %s", decision)
    return decision


def apply_smote(X_train, y_train, k_neighbors: int = 5):
    """Apply SMOTE to training data only (caller is responsible for split order)."""

    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res
