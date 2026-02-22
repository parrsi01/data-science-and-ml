"""Executive reporting for the air traffic delay project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _bottleneck_table(graph_metrics_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    df = graph_metrics_df.copy()
    df["degree_total"] = df["in_degree"] + df["out_degree"]
    return df.sort_values(
        ["betweenness_centrality", "pagerank", "degree_total"],
        ascending=[False, False, False],
        kind="mergesort",
    ).head(top_n)


def write_executive_summary(
    *,
    graph_metrics_df: pd.DataFrame,
    model_result: dict[str, Any],
    forecast_result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write an executive summary markdown with bottlenecks and recommendations."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bottlenecks = _bottleneck_table(graph_metrics_df, top_n=5)
    top_features = model_result.get("top_feature_importance", [])[:8]
    task = model_result.get("task", "classification")
    metrics = model_result.get("metrics", {})

    recs = []
    if len(bottlenecks) > 0:
        recs.append(
            f"Prioritize flow-management staffing or slot coordination at {bottlenecks.iloc[0]['airport']} and {bottlenecks.iloc[1]['airport']} due to high betweenness/PageRank bottleneck risk."
            if len(bottlenecks) > 1
            else f"Prioritize monitoring at {bottlenecks.iloc[0]['airport']} due to high centrality."
        )
    if task == "classification" and float(metrics.get("recall", 0.0)) < 0.75:
        recs.append("Increase focus on recall-oriented threshold tuning for delay alerts to reduce missed delay-risk flights.")
    recs.append("Use forecast trend and bottleneck airports together to pre-position gate, ramp, and turnaround resources during expected congestion windows.")

    lines = [
        "# Executive Summary: Air Traffic Flow & Delay Forecasting",
        "",
        "## Busiest / Bottleneck Airports (Degree + Centrality)",
        "",
        "| Airport | In Degree | Out Degree | Betweenness | PageRank | Clustering |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in bottlenecks.itertuples(index=False):
        lines.append(
            f"| {row.airport} | {row.in_degree:.0f} | {row.out_degree:.0f} | {row.betweenness_centrality:.4f} | {row.pagerank:.4f} | {row.clustering:.4f} |"
        )

    lines.extend(["", "## Delay Model Metrics", ""])
    if task == "classification":
        lines.extend(
            [
                f"- Accuracy: {float(metrics.get('accuracy', 0.0)):.4f}",
                f"- Precision: {float(metrics.get('precision', 0.0)):.4f}",
                f"- Recall: {float(metrics.get('recall', 0.0)):.4f}",
                f"- F1: {float(metrics.get('f1', 0.0)):.4f}",
                f"- ROC-AUC: {float(metrics.get('roc_auc', 0.0)):.4f}",
            ]
        )
    else:
        lines.extend(
            [
                f"- MAE: {float(metrics.get('mae', 0.0)):.4f}",
                f"- RMSE: {float(metrics.get('rmse', 0.0)):.4f}",
                f"- R2: {float(metrics.get('r2', 0.0)):.4f}",
            ]
        )

    lines.extend(["", "## Top Delay Predictors (XGBoost Feature Importance)", ""])
    for item in top_features:
        lines.append(f"- `{item['feature']}`: {float(item['importance']):.4f}")

    lines.extend(["", "## Forecast Trend Summary", ""])
    if forecast_result.get("enabled"):
        lines.extend(
            [
                f"- Forecast method: {forecast_result.get('method', 'unknown')}",
                f"- Horizon (days): {int(forecast_result.get('horizon_days', 0))}",
                f"- Trend label: {forecast_result.get('trend_label', 'unknown')}",
                f"- Trend delta (minutes): {float(forecast_result.get('trend_delta_minutes', 0.0)):.3f}",
                f"- Notes: {forecast_result.get('notes', '')}",
            ]
        )
    else:
        lines.append("- Forecasting disabled in config.")

    lines.extend(["", "## Operational Recommendations", ""])
    for rec in recs:
        lines.append(f"- {rec}")

    md_path = output_dir / "executive_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "executive_summary_md": str(md_path),
        "bottleneck_airports": bottlenecks["airport"].astype(str).tolist(),
        "top_delay_predictors": [str(item["feature"]) for item in top_features],
    }

