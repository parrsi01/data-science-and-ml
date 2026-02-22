"""Seed sensitivity / stability evaluation utilities."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Callable

import numpy as np


def run_seed_sweep(train_fn: Callable[[int], dict[str, float]], seeds: list[int]) -> dict[str, Any]:
    """Run a metric-producing train function across random seeds and summarize stability."""

    if not seeds:
        raise ValueError("seeds must not be empty")

    per_seed: dict[str, dict[str, float]] = {}
    metric_names: set[str] = set()
    for seed in seeds:
        metrics = train_fn(int(seed))
        per_seed[str(seed)] = {k: float(v) for k, v in metrics.items()}
        metric_names.update(metrics.keys())

    summary: dict[str, Any] = {"seeds": list(seeds), "per_seed_metrics": per_seed, "summary": {}}
    for metric in sorted(metric_names):
        values = np.asarray([per_seed[str(seed)][metric] for seed in seeds], dtype=float)
        summary["summary"][metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "worst_seed": int(seeds[int(np.nanargmin(values))]),
            "best_seed": int(seeds[int(np.nanargmax(values))]),
        }
    summary["stability_score_f1_std"] = float(summary["summary"].get("f1", {}).get("std", float("nan")))
    return summary


def write_stability_reports(summary: dict[str, Any], report_dir: str | Path) -> dict[str, str]:
    """Write JSON and Markdown stability reports."""

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "stability_seed_sweep.json"
    md_path = report_dir / "stability_seed_sweep.md"

    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Stability Seed Sweep",
        "",
        f"- Seeds: `{summary['seeds']}`",
        f"- Stability score (std of F1): `{summary['stability_score_f1_std']:.6f}`",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, stats in summary["summary"].items():
        lines.append(
            f"| {metric} | {stats['mean']:.6f} | {stats['std']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}
