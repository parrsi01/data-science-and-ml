from __future__ import annotations

from pathlib import Path
import json

import yaml

from projects.humanitarian_optimization.analysis import (
    run_budget_sensitivity,
    run_priority_weight_sensitivity,
    save_sensitivity_outputs,
)
from projects.humanitarian_optimization.data import generate_regional_demand
from projects.humanitarian_optimization.model import solve_humanitarian_allocation
from projects.humanitarian_optimization.run_project import run_project


def _small_config() -> dict:
    return {
        "regions": ["Africa", "MiddleEast", "Asia", "SouthAmerica"],
        "resources": {"total_budget": 200_000, "total_units": 8_000},
        "demand_model": {"seed": 7, "demand_variance": 0.15},
        "objectives": ["minimize_unmet_demand", "maximize_priority_coverage"],
        "constraints": {"max_allocation_per_region_ratio": 0.7, "min_priority_share": 0.25},
    }


def test_allocation_respects_demand_and_unit_limits(tmp_path: Path) -> None:
    cfg = _small_config()
    demand_df = generate_regional_demand(cfg, output_path=tmp_path / "demand.csv")
    solved = solve_humanitarian_allocation(demand_df, cfg, save_csv=False)
    alloc = solved["allocation_table"]
    assert (alloc["allocated_units"] <= alloc["demand_units"] + 1e-9).all()
    assert float(alloc["allocated_units"].sum()) <= float(cfg["resources"]["total_units"]) + 1e-9


def test_budget_constraint_respected(tmp_path: Path) -> None:
    cfg = _small_config()
    demand_df = generate_regional_demand(cfg, output_path=tmp_path / "demand.csv")
    solved = solve_humanitarian_allocation(demand_df, cfg, save_csv=False)
    total_cost = float(solved["allocation_table"]["total_cost"].sum())
    assert total_cost <= float(cfg["resources"]["total_budget"]) + 1e-6


def test_sensitivity_outputs_exist(tmp_path: Path) -> None:
    cfg = _small_config()
    demand_df = generate_regional_demand(cfg, output_path=tmp_path / "demand.csv")
    budget_results = run_budget_sensitivity(
        demand_df, cfg, [120_000, 180_000, 240_000], report_dir=tmp_path
    )
    weight_results = run_priority_weight_sensitivity(
        demand_df, cfg, [0.0, 1.0, 2.0], report_dir=tmp_path
    )
    paths = save_sensitivity_outputs(budget_results, weight_results, report_dir=tmp_path)
    assert Path(paths["json"]).exists()
    assert Path(paths["budget_plot_png"]).exists()
    assert Path(paths["weight_plot_png"]).exists()


def test_project_runner_writes_summary_artifacts(tmp_path: Path) -> None:
    cfg_path = tmp_path / "humanitarian.yaml"
    cfg_path.write_text(yaml.safe_dump(_small_config()), encoding="utf-8")
    result = run_project(
        cfg_path,
        report_dir=tmp_path / "reports",
        demand_output_path=tmp_path / "datasets" / "humanitarian_demand.csv",
    )
    assert Path(result["demand_dataset_path"]).exists()
    assert Path(result["allocation_csv_path"]).exists()
    assert Path(result["sensitivity_artifacts"]["json"]).exists()
    summary_json = Path(result["run_summary_json"])
    assert summary_json.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert "executive_summary" in payload
