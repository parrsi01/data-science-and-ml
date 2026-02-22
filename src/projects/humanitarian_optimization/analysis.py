"""Sensitivity and trade-off analysis for humanitarian optimization."""

from __future__ import annotations

from pathlib import Path
import json
import os
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from src.projects.humanitarian_optimization.model import solve_humanitarian_allocation  # type: ignore[import-not-found]
except Exception:
    from projects.humanitarian_optimization.model import solve_humanitarian_allocation


def _ensure_report_dir(report_dir: str | Path) -> Path:
    path = Path(report_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_budget_sensitivity(
    demand_df: pd.DataFrame,
    config: dict[str, Any],
    budget_values: Iterable[float],
    *,
    priority_weight: float = 1.0,
    report_dir: str | Path = "reports/projects/humanitarian_optimization",
) -> list[dict[str, Any]]:
    """Vary budget and collect unmet demand and fairness outputs."""

    rows: list[dict[str, Any]] = []
    for budget in budget_values:
        scenario_cfg = json.loads(json.dumps(config))
        scenario_cfg["resources"]["total_budget"] = float(budget)
        solved = solve_humanitarian_allocation(
            demand_df,
            scenario_cfg,
            priority_weight=priority_weight,
            report_dir=report_dir,
            save_csv=False,
        )
        summary = solved["summary"]
        rows.append(
            {
                "budget": float(budget),
                "total_unmet_demand": float(summary["total_unmet_demand"]),
                "fairness_std_allocation_ratio": float(
                    summary["fairness_std_allocation_ratio"]
                ),
                "priority_coverage_share": float(summary["priority_coverage_share"]),
                "budget_utilization_pct": float(summary["budget_utilization_pct"]),
            }
        )
    return rows


def run_priority_weight_sensitivity(
    demand_df: pd.DataFrame,
    config: dict[str, Any],
    weight_values: Iterable[float],
    *,
    report_dir: str | Path = "reports/projects/humanitarian_optimization",
) -> list[dict[str, Any]]:
    """Vary priority-weight objective coefficient and collect trade-off outputs."""

    rows: list[dict[str, Any]] = []
    for weight in weight_values:
        solved = solve_humanitarian_allocation(
            demand_df,
            config,
            priority_weight=float(weight),
            report_dir=report_dir,
            save_csv=False,
        )
        summary = solved["summary"]
        rows.append(
            {
                "priority_weight": float(weight),
                "total_unmet_demand": float(summary["total_unmet_demand"]),
                "fairness_std_allocation_ratio": float(
                    summary["fairness_std_allocation_ratio"]
                ),
                "priority_coverage_share": float(summary["priority_coverage_share"]),
                "objective_value": float(summary["objective_value"]),
            }
        )
    return rows


def save_sensitivity_outputs(
    budget_results: list[dict[str, Any]],
    weight_results: list[dict[str, Any]],
    *,
    report_dir: str | Path = "reports/projects/humanitarian_optimization",
) -> dict[str, str]:
    """Write JSON and visualization artifacts for sensitivity analysis."""

    report_path = _ensure_report_dir(report_dir)
    json_path = report_path / "sensitivity_analysis.json"
    payload = {
        "budget_sensitivity": budget_results,
        "priority_weight_sensitivity": weight_results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    budget_df = pd.DataFrame(budget_results).sort_values("budget", kind="mergesort")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        budget_df["budget"],
        budget_df["total_unmet_demand"],
        marker="o",
        color="#0B6E4F",
        linewidth=2,
    )
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Budget vs Unmet Demand")
    ax1.set_xlabel("Total Budget")
    ax1.set_ylabel("Total Unmet Demand (Units)")
    budget_plot = report_path / "budget_vs_unmet_demand.png"
    fig1.tight_layout()
    fig1.savefig(budget_plot, dpi=150)
    plt.close(fig1)

    weight_df = pd.DataFrame(weight_results).sort_values(
        "priority_weight", kind="mergesort"
    )
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(
        weight_df["priority_weight"],
        weight_df["priority_coverage_share"],
        marker="o",
        color="#C84C09",
        linewidth=2,
    )
    ax2.grid(True, alpha=0.25)
    ax2.set_title("Priority Weight vs Priority Coverage")
    ax2.set_xlabel("Priority Weight")
    ax2.set_ylabel("Priority Coverage Share")
    ax2.set_ylim(0.0, 1.05)
    weight_plot = report_path / "weight_vs_priority_coverage.png"
    fig2.tight_layout()
    fig2.savefig(weight_plot, dpi=150)
    plt.close(fig2)

    return {
        "json": str(json_path),
        "budget_plot_png": str(budget_plot),
        "weight_plot_png": str(weight_plot),
    }
