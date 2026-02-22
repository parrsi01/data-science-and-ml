from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from algorithm_marl_xgboost.src.experiments.parameter_study import generate_param_grid
from algorithm_marl_xgboost.src.experiments.plotting import generate_parameter_study_plots
from algorithm_marl_xgboost.src.experiments.statistics import compare_methods


def test_generate_param_grid_cartesian_product(tmp_path: Path) -> None:
    cfg = {
        "vary": {
            "non_iid_alpha": [0.1, 0.5],
            "n_agents": [5, 10],
            "topology": ["ring"],
            "communication_budget": [0.5, 1.0],
        },
        "fixed": {"seed_base": 1000, "random_graph_p": 0.25, "per_agent_smote": True},
        "rounds": 2,
    }
    cfg_path = tmp_path / "study.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    grid = generate_param_grid(cfg_path)
    assert len(grid) == 2 * 2 * 1 * 2
    assert all("_study" in item for item in grid)


def test_compare_methods_returns_p_value() -> None:
    result = compare_methods([0.7, 0.8, 0.75, 0.78], [0.65, 0.7, 0.71, 0.69], test="wilcoxon")
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0


def test_plotting_writes_pngs(tmp_path: Path) -> None:
    repeats_df = pd.DataFrame(
        [
            {"topology": "ring", "non_iid_alpha": 0.1, "n_agents": 5, "communication_budget": 0.5, "marl_f1": 0.7, "marl_energy_total": 10.0, "marl_bytes_sent_total": 1000.0},
            {"topology": "ring", "non_iid_alpha": 0.1, "n_agents": 5, "communication_budget": 1.0, "marl_f1": 0.75, "marl_energy_total": 12.0, "marl_bytes_sent_total": 1500.0},
            {"topology": "star", "non_iid_alpha": 0.5, "n_agents": 10, "communication_budget": 0.5, "marl_f1": 0.68, "marl_energy_total": 14.0, "marl_bytes_sent_total": 900.0},
            {"topology": "star", "non_iid_alpha": 0.5, "n_agents": 10, "communication_budget": 1.0, "marl_f1": 0.72, "marl_energy_total": 16.0, "marl_bytes_sent_total": 1700.0},
        ]
    )
    summary_df = pd.DataFrame(
        [
            {"label": "a", "topology": "ring", "non_iid_alpha": 0.1, "n_agents": 5, "communication_budget": 0.5, "marl_f1_mean": 0.70, "marl_f1_sem": 0.01, "marl_bytes_sent_total_mean": 1000.0, "marl_bytes_sent_total_sem": 10.0},
            {"label": "b", "topology": "ring", "non_iid_alpha": 0.1, "n_agents": 5, "communication_budget": 1.0, "marl_f1_mean": 0.75, "marl_f1_sem": 0.01, "marl_bytes_sent_total_mean": 1500.0, "marl_bytes_sent_total_sem": 12.0},
            {"label": "c", "topology": "star", "non_iid_alpha": 0.5, "n_agents": 10, "communication_budget": 0.5, "marl_f1_mean": 0.68, "marl_f1_sem": 0.02, "marl_bytes_sent_total_mean": 900.0, "marl_bytes_sent_total_sem": 15.0},
            {"label": "d", "topology": "star", "non_iid_alpha": 0.5, "n_agents": 10, "communication_budget": 1.0, "marl_f1_mean": 0.72, "marl_f1_sem": 0.02, "marl_bytes_sent_total_mean": 1700.0, "marl_bytes_sent_total_sem": 20.0},
        ]
    )
    paths = generate_parameter_study_plots(repeats_df, summary_df, output_dir=tmp_path / "plots")
    assert len(paths) >= 6
    assert all(Path(p).exists() for p in paths.values())

