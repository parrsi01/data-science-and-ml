"""Parameter study orchestrator for MARL+XGBoost repeated experiments."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import yaml

from algorithm_marl_xgboost.src.experiments.plotting import generate_parameter_study_plots
from algorithm_marl_xgboost.src.experiments.reporting import generate_reports
from algorithm_marl_xgboost.src.experiments.runner import build_label, run_repeats
from algorithm_marl_xgboost.src.experiments.statistics import compare_methods


STUDY_REPORT_DIR = Path("algorithm_marl_xgboost/reports/parameter_study")


def load_parameter_study_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Parameter study config must be a YAML mapping")
    return payload


def generate_param_grid(config_or_path: dict[str, Any] | str | Path) -> list[dict[str, Any]]:
    """Generate full cartesian grid of parameter overrides from study config."""

    cfg = load_parameter_study_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    vary = cfg.get("vary", {})
    keys = ["non_iid_alpha", "n_agents", "topology", "communication_budget"]
    values = [list(vary.get(k, [])) for k in keys]
    if any(len(v) == 0 for v in values):
        raise ValueError("Parameter study vary grid must define all required keys")

    base_rounds = int(cfg.get("rounds", 5))
    fixed = cfg.get("fixed", {})
    grid: list[dict[str, Any]] = []
    for alpha, n_agents, topology, comm_budget in product(*values):
        study_vals = {
            "non_iid_alpha": float(alpha),
            "n_agents": int(n_agents),
            "topology": str(topology),
            "communication_budget": float(comm_budget),
        }
        override = {
            "_study": study_vals,
            "data": {
                "non_iid_alpha": float(alpha),
                "n_agents": int(n_agents),
                "per_agent_smote": bool(fixed.get("per_agent_smote", True)),
            },
            "marl": {"communication_budget": float(comm_budget)},
            "training": {
                "rounds": base_rounds,
                "topology": str(topology),
                "random_graph_p": float(fixed.get("random_graph_p", 0.25)),
            },
        }
        grid.append(override)
    return grid


def _aggregate_setting_results(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"repeat_index", "seed"}]
    row: dict[str, Any] = {}
    for key in ["label", "non_iid_alpha", "n_agents", "topology", "communication_budget"]:
        if key in df.columns:
            row[key] = df[key].iloc[0]
    row["repeats"] = int(len(df))
    for col in numeric_cols:
        row[f"{col}_mean"] = float(df[col].mean())
        row[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
        row[f"{col}_sem"] = float(df[col].std(ddof=1) / np.sqrt(len(df))) if len(df) > 1 else 0.0
    return row


def _write_slice_summaries(master_df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    mapping = {
        "non_iid_alpha": "parameter_non_iid_alpha.csv",
        "n_agents": "parameter_n_agents.csv",
        "topology": "parameter_topology.csv",
        "communication_budget": "parameter_comm_budget.csv",
    }
    paths: dict[str, str] = {}
    for col, filename in mapping.items():
        if col not in master_df.columns:
            continue
        grouped = (
            master_df.groupby(col, as_index=False)
            .agg(
                marl_f1_mean=("marl_f1_mean", "mean"),
                marl_f1_std=("marl_f1_mean", "std"),
                marl_bytes_sent_total_mean=("marl_bytes_sent_total_mean", "mean"),
                marl_energy_total_mean=("marl_energy_total_mean", "mean"),
            )
        )
        path = out_dir / filename
        grouped.to_csv(path, index=False)
        paths[col] = str(path)
    return paths


def _compute_significance(repeats_df: pd.DataFrame, study_cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    test_name = str(study_cfg.get("stats", {}).get("significance_test", "wilcoxon"))
    alpha = float(study_cfg.get("stats", {}).get("alpha", 0.05))

    payload = {
        "config": {"test": test_name, "alpha": alpha},
        "comparisons": {
            "f1_marl_vs_naive": compare_methods(
                repeats_df["marl_f1"].astype(float).tolist(),
                repeats_df["naive_f1"].astype(float).tolist(),
                test=test_name,
            ),
            "energy_marl_vs_naive": compare_methods(
                repeats_df["marl_energy_total"].astype(float).tolist(),
                repeats_df["naive_energy_total"].astype(float).tolist(),
                test=test_name,
            ),
            "bandwidth_marl_vs_naive": compare_methods(
                repeats_df["marl_bytes_sent_total"].astype(float).tolist(),
                repeats_df["naive_bytes_sent_total"].astype(float).tolist(),
                test=test_name,
            ),
        },
    }
    for key, comparison in payload["comparisons"].items():
        comparison["significant_at_alpha"] = bool(comparison["p_value"] < alpha)
        comparison["interpretation"] = (
            "statistically significant difference detected"
            if comparison["significant_at_alpha"]
            else "no statistically significant difference detected at configured alpha"
        )
    path = out_dir / "significance_tests.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["artifacts"] = {"significance_tests_json": str(path)}
    return payload


def run_parameter_study(
    config_path: str | Path,
    *,
    max_combinations: int | None = None,
    repeats_override: int | None = None,
    rounds_override: int | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run repeated parameter study, aggregate results, plot, and report."""

    study_cfg = load_parameter_study_config(config_path)
    grid = generate_param_grid(study_cfg)
    if max_combinations is not None:
        grid = grid[: int(max_combinations)]

    out_dir = STUDY_REPORT_DIR
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    repeats = int(repeats_override if repeats_override is not None else study_cfg.get("repeats", 10))
    seed_base = int(study_cfg.get("fixed", {}).get("seed_base", 1000))
    rounds = int(rounds_override if rounds_override is not None else study_cfg.get("rounds", 5))

    all_repeat_rows: list[pd.DataFrame] = []
    setting_summary_rows: list[dict[str, Any]] = []

    for idx, override in enumerate(grid):
        override = json.loads(json.dumps(override))  # deep copy via JSON-serializable structure
        override.setdefault("training", {})["rounds"] = rounds
        if quick:
            override.setdefault("data", {})
            override["data"].setdefault("n_samples", 700)
            override["data"]["min_pos_per_agent"] = min(override["data"].get("min_pos_per_agent", 20), 8)
            override.setdefault("xgboost", {})
            override["xgboost"].update({"n_estimators": 60, "max_depth": 4, "early_stopping_rounds": 10})
        label = build_label(override)
        print(f"[{idx+1}/{len(grid)}] running {label} repeats={repeats}")
        repeat_df = run_repeats(override, repeats=repeats, seed_base=seed_base, label=label)
        all_repeat_rows.append(repeat_df)
        setting_summary_rows.append(_aggregate_setting_results(repeat_df))

    repeats_df = pd.concat(all_repeat_rows, ignore_index=True) if all_repeat_rows else pd.DataFrame()
    repeats_csv = out_dir / "repeats_all.csv"
    repeats_df.to_csv(repeats_csv, index=False)

    master_df = pd.DataFrame(setting_summary_rows)
    if not master_df.empty:
        master_df = master_df.sort_values(["marl_f1_mean", "marl_f1_std"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    master_csv = out_dir / "parameter_study_results.csv"
    master_df.to_csv(master_csv, index=False)

    slice_paths = _write_slice_summaries(master_df, out_dir)
    sig_payload = _compute_significance(repeats_df, study_cfg, out_dir) if not repeats_df.empty else {"comparisons": {}}
    plot_paths = generate_parameter_study_plots(repeats_df, master_df, output_dir=plots_dir)
    report_paths = generate_reports(master_df, repeats_df, sig_payload, report_dir=out_dir)

    summary = {
        "config_path": str(config_path),
        "repeats": repeats,
        "rounds": rounds,
        "n_settings": int(len(grid)),
        "artifacts": {
            "parameter_study_results_csv": str(master_csv),
            "repeats_all_csv": str(repeats_csv),
            "plots_dir": str(plots_dir),
            **slice_paths,
            **plot_paths,
            **report_paths,
            **sig_payload.get("artifacts", {}),
        },
    }
    (out_dir / "parameter_study_run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run parameter study for MARL+XGBoost")
    parser.add_argument("--config", default="algorithm_marl_xgboost/configs/parameter_study.yaml")
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument("--repeats-override", type=int, default=None)
    parser.add_argument("--rounds-override", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Use smaller XGBoost/data settings for fast demo runs")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = run_parameter_study(
        args.config,
        max_combinations=args.max_combinations,
        repeats_override=args.repeats_override,
        rounds_override=args.rounds_override,
        quick=args.quick,
    )
    print(f"Parameter study complete: {summary['artifacts']['parameter_study_results_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

