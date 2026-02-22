from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluation.cv import stratified_cv_scores
from evaluation.drift import categorical_drift_report, numeric_drift_report
from evaluation.group_metrics import compute_group_metrics
from evaluation.stability import run_seed_sweep
from evaluation.thresholds import find_best_threshold
from ml_core.preprocess import build_preprocessor


def _tiny_df(n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "category": rng.choice(["a", "b", "c"], size=n),
            "metric_a": rng.normal(10, 2, size=n),
            "metric_b": np.clip(rng.normal(5, 1, size=n), 0.1, None),
        }
    )
    y = pd.Series((df["metric_a"] + 0.5 * df["metric_b"] + (df["category"] == "c") * 1.5 > 13).astype(int))
    return df, y


def test_threshold_finder_returns_threshold_in_range() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    result = find_best_threshold(y_true, y_proba, metric="f1")
    assert 0.0 <= result["best_threshold"] <= 1.0


def test_drift_report_keys_exist() -> None:
    train_df, _ = _tiny_df(120, 1)
    test_df, _ = _tiny_df(100, 2)
    num = numeric_drift_report(train_df, test_df, ["metric_a", "metric_b"])
    cat = categorical_drift_report(train_df, test_df, ["category"])
    assert "numeric" in num and "metric_a" in num["numeric"]
    assert "categorical" in cat and "category" in cat["categorical"]


def test_group_metrics_outputs_expected_columns() -> None:
    df, y_true = _tiny_df(100, 3)
    y_pred = y_true.copy()
    out = compute_group_metrics(df, y_true, y_pred, "category")
    assert {
        "support",
        "precision",
        "recall",
        "f1",
        "positive_rate",
        "false_positive_rate",
        "false_negative_rate",
    } <= set(out.columns)


def test_seed_sweep_returns_mean_std_keys() -> None:
    def train_fn(seed: int) -> dict[str, float]:
        return {"f1": 0.5 + (seed % 3) * 0.01, "roc_auc": 0.8 + (seed % 2) * 0.02}

    summary = run_seed_sweep(train_fn, [1, 2, 3])
    assert "summary" in summary
    assert "f1" in summary["summary"]
    assert "mean" in summary["summary"]["f1"]


def test_cv_utilities_run() -> None:
    X, y = _tiny_df(180, 4)
    pipeline = Pipeline(
        [
            ("pre", build_preprocessor(["category"], ["metric_a", "metric_b"])),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    out = stratified_cv_scores(pipeline, X, y, folds=3, metrics=["accuracy", "f1", "roc_auc"])
    assert "metrics" in out
    assert "f1" in out["metrics"]
