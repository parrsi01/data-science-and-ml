"""Cross-validation utilities for institutional evaluation workflows."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate


LOGGER = logging.getLogger("evaluation.cv")


def _summarize_cv_results(results: dict[str, np.ndarray]) -> dict[str, dict[str, float | list[float]]]:
    summary: dict[str, dict[str, float | list[float]]] = {}
    for key, values in results.items():
        if not key.startswith("test_"):
            continue
        metric = key.replace("test_", "")
        arr = np.asarray(values, dtype=float)
        summary[metric] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "fold_values": [float(v) for v in arr.tolist()],
        }
    return summary


def stratified_cv_scores(pipeline, X, y, folds: int, metrics: Iterable[str]) -> dict[str, object]:
    """Run deterministic stratified cross-validation and summarize scores."""

    LOGGER.info("Running stratified CV: folds=%s metrics=%s", folds, list(metrics))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    raw = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=list(metrics),
        n_jobs=1,
        return_train_score=False,
    )
    raw_arrays = {k: np.asarray(v, dtype=float) for k, v in raw.items() if k.startswith("test_")}
    return {
        "folds": int(folds),
        "seed": 42,
        "metrics": _summarize_cv_results(raw_arrays),
    }


def repeated_stratified_cv(pipeline, X, y, folds: int, repeats: int, metrics: Iterable[str]) -> dict[str, object]:
    """Run repeated stratified CV with deterministic seed and summarize scores."""

    LOGGER.info(
        "Running repeated stratified CV: folds=%s repeats=%s metrics=%s",
        folds,
        repeats,
        list(metrics),
    )
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    raw = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=list(metrics),
        n_jobs=1,
        return_train_score=False,
    )
    raw_arrays = {k: np.asarray(v, dtype=float) for k, v in raw.items() if k.startswith("test_")}
    return {
        "folds": int(folds),
        "repeats": int(repeats),
        "seed": 42,
        "metrics": _summarize_cv_results(raw_arrays),
    }
