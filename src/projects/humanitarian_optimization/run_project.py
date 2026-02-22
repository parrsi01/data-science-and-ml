"""CLI runner for the humanitarian logistics optimization project."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any

import pandas as pd
import yaml

try:
    from src.projects.humanitarian_optimization.analysis import (  # type: ignore[import-not-found]
        run_budget_sensitivity,
        run_priority_weight_sensitivity,
        save_sensitivity_outputs,
    )
    from src.projects.humanitarian_optimization.data import generate_regional_demand  # type: ignore[import-not-found]
    from src.projects.humanitarian_optimization.model import solve_humanitarian_allocation  # type: ignore[import-not-found]
except Exception:
    from projects.humanitarian_optimization.analysis import (
        run_budget_sensitivity,
        run_priority_weight_sensitivity,
        save_sensitivity_outputs,
    )
    from projects.humanitarian_optimization.data import generate_regional_demand
    from projects.humanitarian_optimization.model import solve_humanitarian_allocation


DEFAULT_REPORT_DIR = Path("reports/projects/humanitarian_optimization")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML config for the project."""

    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def _scenario_ranges(config: dict[str, Any]) -> tuple[list[float], list[float]]:
    base_budget = float(config["resources"]["total_budget"])
    budget_values = [round(base_budget * factor, 2) for factor in [0.5, 0.75, 1.0, 1.25, 1.5]]
    weight_values = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    return budget_values, weight_values


def _write_executive_report(
    demand_df: pd.DataFrame,
    allocation_df: pd.DataFrame,
    summary: dict[str, Any],
    sensitivity_artifacts: dict[str, str],
    report_dir: Path,
) -> str:
    total_demand = float(demand_df["demand_units"].sum())
    total_allocated = float(allocation_df["allocated_units"].sum())
    underserved = allocation_df.sort_values(
        ["demand_fill_rate", "priority_score", "region"],
        ascending=[True, False, True],
        kind="mergesort",
    ).iloc[0]

    lines = [
        "# Executive Summary: Humanitarian Logistics Optimization",
        "",
        "## Top-Line Outcomes",
        "",
        f"- Total demand (units): {total_demand:,.0f}",
        f"- Total allocated (units): {total_allocated:,.0f}",
        f"- Unmet demand (%): {float(summary['unmet_demand_pct']) * 100:.2f}%",
        (
            "- Most under-served region: "
            f"{underserved['region']} (fill rate {float(underserved['demand_fill_rate']) * 100:.2f}%)"
        ),
        f"- Budget utilization (%): {float(summary['budget_utilization_pct']) * 100:.2f}%",
        "",
        "## Artifacts",
        "",
        f"- Allocation CSV: `{report_dir / 'allocation_results.csv'}`",
        f"- Sensitivity JSON: `{sensitivity_artifacts['json']}`",
        f"- Budget plot: `{sensitivity_artifacts['budget_plot_png']}`",
        f"- Priority plot: `{sensitivity_artifacts['weight_plot_png']}`",
    ]
    md_path = report_dir / "executive_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return str(md_path)


def run_project(config_path: str | Path) -> dict[str, Any]:
    """Run demand generation, optimization, sensitivity analysis, and reporting."""

    config = load_config(config_path)
    report_dir = DEFAULT_REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    demand_df = generate_regional_demand(config)
    solve_result = solve_humanitarian_allocation(demand_df, config, report_dir=report_dir)
    allocation_df = solve_result["allocation_table"]
    summary = solve_result["summary"]

    budget_values, weight_values = _scenario_ranges(config)
    budget_results = run_budget_sensitivity(demand_df, config, budget_values, report_dir=report_dir)
    weight_results = run_priority_weight_sensitivity(demand_df, config, weight_values, report_dir=report_dir)
    sensitivity_artifacts = save_sensitivity_outputs(
        budget_results, weight_results, report_dir=report_dir
    )

    exec_report_path = _write_executive_report(
        demand_df, allocation_df, summary, sensitivity_artifacts, report_dir
    )

    most_underserved = allocation_df.sort_values(
        "demand_fill_rate", ascending=True, kind="mergesort"
    ).iloc[0]
    payload: dict[str, Any] = {
        "config_path": str(config_path),
        "demand_dataset_path": "datasets/humanitarian_demand.csv",
        "allocation_csv_path": solve_result.get("allocation_csv_path"),
        "sensitivity_artifacts": sensitivity_artifacts,
        "executive_summary_md": exec_report_path,
        "executive_summary": {
            "total_demand": float(demand_df["demand_units"].sum()),
            "total_allocated": float(allocation_df["allocated_units"].sum()),
            "unmet_demand_pct": float(summary["unmet_demand_pct"]),
            "most_underserved_region": str(most_underserved["region"]),
            "most_underserved_fill_rate": float(most_underserved["demand_fill_rate"]),
            "budget_utilization_pct": float(summary["budget_utilization_pct"]),
            "priority_coverage_share": float(summary["priority_coverage_share"]),
        },
    }
    summary_json_path = report_dir / "run_summary.json"
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["run_summary_json"] = str(summary_json_path)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Humanitarian logistics optimization project runner"
    )
    parser.add_argument("--config", default="configs/projects/humanitarian_optimization.yaml")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = run_project(args.config)
    s = result["executive_summary"]
    print("Executive Summary")
    print(f"Total demand: {s['total_demand']:.0f}")
    print(f"Total allocated: {s['total_allocated']:.0f}")
    print(f"% unmet demand: {s['unmet_demand_pct'] * 100:.2f}%")
    print(f"Most under-served region: {s['most_underserved_region']}")
    print(f"Budget utilization %: {s['budget_utilization_pct'] * 100:.2f}%")
    print(f"Run summary JSON: {result['run_summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

