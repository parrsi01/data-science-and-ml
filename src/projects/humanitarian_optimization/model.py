"""Linear programming model for humanitarian aid allocation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pulp


def solve_humanitarian_allocation(
    demand_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    priority_weight: float = 1.0,
    report_dir: str | Path = "reports/projects/humanitarian_optimization",
    save_csv: bool = True,
) -> dict[str, Any]:
    """Solve the constrained allocation problem using linear programming."""

    df = demand_df.copy()
    required_cols = {
        "region",
        "demand_units",
        "priority_score",
        "cost_per_unit",
        "risk_index",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Demand dataframe missing required columns: {sorted(missing)}")

    total_budget = float(config["resources"]["total_budget"])
    total_units = float(config["resources"]["total_units"])
    max_ratio = float(config["constraints"]["max_allocation_per_region_ratio"])
    min_priority_share = float(config["constraints"]["min_priority_share"])

    problem = pulp.LpProblem("HumanitarianAllocation", pulp.LpMinimize)
    regions = df["region"].tolist()
    demand = {r: float(df.loc[df["region"] == r, "demand_units"].iloc[0]) for r in regions}
    priority = {
        r: float(df.loc[df["region"] == r, "priority_score"].iloc[0]) for r in regions
    }
    cost = {r: float(df.loc[df["region"] == r, "cost_per_unit"].iloc[0]) for r in regions}

    x = {r: pulp.LpVariable(f"x_{r}", lowBound=0, cat=pulp.LpContinuous) for r in regions}
    unmet = {
        r: pulp.LpVariable(f"unmet_{r}", lowBound=0, cat=pulp.LpContinuous) for r in regions
    }

    max_priority = max(priority.values()) if priority else 1.0
    priority_norm = {r: priority[r] / max_priority for r in regions}

    # Composite objective: minimize unmet demand while rewarding high-priority allocation.
    problem += (
        pulp.lpSum(unmet[r] for r in regions)
        - float(priority_weight) * pulp.lpSum(priority_norm[r] * x[r] for r in regions)
    ), "WeightedUnmetDemandMinusPriorityReward"

    for r in regions:
        problem += unmet[r] == demand[r] - x[r], f"unmet_definition_{r}"
        problem += x[r] <= demand[r], f"demand_cap_{r}"
        problem += x[r] <= max_ratio * total_units, f"region_max_ratio_{r}"

    problem += pulp.lpSum(cost[r] * x[r] for r in regions) <= total_budget, "budget_limit"
    problem += pulp.lpSum(x[r] for r in regions) <= total_units, "unit_limit"

    high_priority_regions = [r for r in regions if priority[r] >= 4]
    if high_priority_regions:
        problem += (
            pulp.lpSum(x[r] for r in high_priority_regions)
            >= min_priority_share * pulp.lpSum(x[r] for r in regions)
        ), "min_priority_share"

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status_name = pulp.LpStatus.get(status, str(status))
    if status_name != "Optimal":
        raise RuntimeError(f"Optimization did not find an optimal solution: {status_name}")

    region_order = df["region"].tolist()
    df["allocated_units"] = [float(x[r].value() or 0.0) for r in region_order]
    df["unmet_demand"] = (df["demand_units"].astype(float) - df["allocated_units"]).clip(
        lower=0.0
    )
    total_allocated = float(df["allocated_units"].sum())
    df["allocation_ratio_total"] = df["allocated_units"] / max(total_allocated, 1.0)
    df["demand_fill_rate"] = df["allocated_units"] / df["demand_units"].replace(0, 1)
    df["total_cost"] = df["allocated_units"] * df["cost_per_unit"]
    df = df.sort_values(
        ["priority_score", "risk_index", "region"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    total_cost = float(df["total_cost"].sum())
    total_demand = float(df["demand_units"].sum())
    total_unmet = float(df["unmet_demand"].sum())
    high_priority_alloc = float(df.loc[df["priority_score"] >= 4, "allocated_units"].sum())
    priority_coverage_share = high_priority_alloc / max(total_allocated, 1.0)

    result: dict[str, Any] = {
        "status": status_name,
        "allocation_table": df,
        "summary": {
            "total_demand": total_demand,
            "total_allocated": total_allocated,
            "total_unmet_demand": total_unmet,
            "unmet_demand_pct": total_unmet / max(total_demand, 1.0),
            "total_cost": total_cost,
            "budget_utilization_pct": total_cost / max(total_budget, 1.0),
            "total_units_utilization_pct": total_allocated / max(total_units, 1.0),
            "priority_coverage_share": priority_coverage_share,
            "objective_value": float(pulp.value(problem.objective)),
            "priority_weight": float(priority_weight),
            "fairness_std_allocation_ratio": float(df["allocation_ratio_total"].std(ddof=0)),
        },
    }

    if save_csv:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        csv_path = report_dir / "allocation_results.csv"
        df.to_csv(csv_path, index=False)
        result["allocation_csv_path"] = str(csv_path)
    return result

