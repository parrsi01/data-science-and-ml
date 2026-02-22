"""Basic group metric calculations for institutional bias/fairness checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def compute_group_metrics(df: pd.DataFrame, y_true, y_pred, group_col: str) -> pd.DataFrame:
    """Compute per-group classification metrics for a specified group column."""

    if group_col not in df.columns:
        raise ValueError(f"group_col not found: {group_col}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    work = pd.DataFrame(
        {
            group_col: df[group_col].values,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    rows: list[dict[str, Any]] = []
    for group_value, g in work.groupby(group_col, dropna=False):
        yt = g["y_true"].to_numpy()
        yp = g["y_pred"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        support = int(len(g))
        positive_rate = float((yp == 1).mean()) if support else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        rows.append(
            {
                group_col: str(group_value),
                "support": support,
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "positive_rate": positive_rate,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
            }
        )

    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def write_group_metrics_reports(group_df: pd.DataFrame, report_dir: str | Path) -> dict[str, str]:
    """Write group metrics CSV and Markdown reports."""

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "group_metrics.csv"
    md_path = report_dir / "group_metrics.md"
    group_df.to_csv(csv_path, index=False)

    lines = [
        "# Group Metrics",
        "",
        "| Group | Support | Precision | Recall | F1 | Positive Rate | FPR | FNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    group_col = group_df.columns[0]
    for _, row in group_df.iterrows():
        lines.append(
            f"| {row[group_col]} | {int(row['support'])} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['positive_rate']:.4f} | {row['false_positive_rate']:.4f} | {row['false_negative_rate']:.4f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path)}
