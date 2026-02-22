"""Train-vs-test drift snapshot reports for numeric and categorical features."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


def numeric_drift_report(train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    """Compute numeric drift summary using mean/std shifts and KS statistic."""

    report: dict[str, Any] = {"numeric": {}}
    for col in numeric_cols:
        train_vals = pd.to_numeric(train_df[col], errors="coerce").dropna()
        test_vals = pd.to_numeric(test_df[col], errors="coerce").dropna()
        ks_stat, ks_p = ks_2samp(train_vals, test_vals)
        report["numeric"][col] = {
            "train_mean": float(train_vals.mean()),
            "test_mean": float(test_vals.mean()),
            "mean_shift": float(test_vals.mean() - train_vals.mean()),
            "train_std": float(train_vals.std(ddof=1)),
            "test_std": float(test_vals.std(ddof=1)),
            "std_shift": float(test_vals.std(ddof=1) - train_vals.std(ddof=1)),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
        }
    return report


def categorical_drift_report(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list[str]) -> dict[str, Any]:
    """Compute categorical drift via total variation distance and chi-square test."""

    report: dict[str, Any] = {"categorical": {}}
    for col in cat_cols:
        train_counts = train_df[col].astype(str).value_counts(normalize=True)
        test_counts = test_df[col].astype(str).value_counts(normalize=True)
        categories = sorted(set(train_counts.index) | set(test_counts.index))

        train_dist = np.array([float(train_counts.get(cat, 0.0)) for cat in categories], dtype=float)
        test_dist = np.array([float(test_counts.get(cat, 0.0)) for cat in categories], dtype=float)
        tv_distance = 0.5 * float(np.abs(train_dist - test_dist).sum())

        contingency = np.vstack(
            [
                [int((train_df[col].astype(str) == cat).sum()) for cat in categories],
                [int((test_df[col].astype(str) == cat).sum()) for cat in categories],
            ]
        )
        chi2, pvalue, _, _ = chi2_contingency(contingency)

        report["categorical"][col] = {
            "categories": categories,
            "train_distribution": {cat: float(train_counts.get(cat, 0.0)) for cat in categories},
            "test_distribution": {cat: float(test_counts.get(cat, 0.0)) for cat in categories},
            "tv_distance": tv_distance,
            "chi_square_statistic": float(chi2),
            "chi_square_pvalue": float(pvalue),
        }
    return report


def write_drift_reports(report: dict[str, Any], report_dir: str | Path) -> dict[str, str]:
    """Write drift JSON and Markdown reports."""

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "drift_report.json"
    md_path = report_dir / "drift_report.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = ["# Drift Report", ""]
    lines.append("## Numeric Drift")
    lines.append("")
    lines.append("| Feature | Mean Shift | Std Shift | KS Statistic | KS p-value |")
    lines.append("|---|---:|---:|---:|---:|")
    for col, payload in report.get("numeric", {}).items():
        lines.append(
            f"| {col} | {payload['mean_shift']:.6f} | {payload['std_shift']:.6f} | "
            f"{payload['ks_statistic']:.6f} | {payload['ks_pvalue']:.6f} |"
        )
    lines.append("")
    lines.append("## Categorical Drift")
    lines.append("")
    lines.append("| Feature | TV Distance | Chi-square | p-value |")
    lines.append("|---|---:|---:|---:|")
    for col, payload in report.get("categorical", {}).items():
        lines.append(
            f"| {col} | {payload['tv_distance']:.6f} | {payload['chi_square_statistic']:.6f} | {payload['chi_square_pvalue']:.6f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}
