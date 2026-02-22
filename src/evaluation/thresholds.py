"""Threshold calibration utilities for imbalanced classification."""

from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _score_metric(metric: str, y_true, y_pred) -> float:
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    raise ValueError(f"Unsupported threshold metric: {metric}")


def find_best_threshold(y_true, y_proba, metric: str = "f1") -> dict[str, Any]:
    """Search thresholds [0.01..0.99] and return the best threshold for the metric."""

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    scores: list[dict[str, float]] = []
    for threshold in grid:
        y_pred = (y_proba >= threshold).astype(int)
        scores.append(
            {
                "threshold": float(threshold),
                "accuracy": _score_metric("accuracy", y_true, y_pred),
                "precision": _score_metric("precision", y_true, y_pred),
                "recall": _score_metric("recall", y_true, y_pred),
                "f1": _score_metric("f1", y_true, y_pred),
            }
        )

    best = max(scores, key=lambda row: (row[metric], -abs(row["threshold"] - 0.5)))
    tradeoff_thresholds = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    tradeoff_table = [row for row in scores if round(row["threshold"], 2) in tradeoff_thresholds]

    return {
        "optimized_metric": metric,
        "best_threshold": float(best["threshold"]),
        "best_score": float(best[metric]),
        "best_metrics_at_threshold": best,
        "threshold_grid_scores": scores,
        "tradeoff_table": tradeoff_table,
    }


def write_threshold_reports(result: dict[str, Any], report_dir: str | Path) -> dict[str, str]:
    """Write threshold analysis JSON and threshold tradeoff CSV."""

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "threshold_analysis.json"
    csv_path = report_dir / "threshold_tradeoff.csv"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["threshold", "accuracy", "precision", "recall", "f1"],
        )
        writer.writeheader()
        writer.writerows(result["tradeoff_table"])
    return {"json": str(json_path), "csv": str(csv_path)}
