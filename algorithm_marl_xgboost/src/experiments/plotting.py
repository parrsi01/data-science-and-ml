"""Plotting utilities for parameter studies and repeat analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOTS_DIR = Path("algorithm_marl_xgboost/reports/parameter_study/plots")


def _ensure_plots_dir(output_dir: str | Path = PLOTS_DIR) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _boxplot_by_category(
    df: pd.DataFrame,
    *,
    category_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> str:
    groups = []
    labels = []
    for label, subset in df.groupby(category_col, sort=True):
        vals = subset[value_col].dropna().astype(float).tolist()
        if vals:
            labels.append(str(label))
            groups.append(vals)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    if groups:
        ax.boxplot(groups, tick_labels=labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def _line_with_error(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    yerr_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> str:
    plot_df = df[[x_col, y_col, yerr_col]].copy().sort_values(x_col, kind="mergesort")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x_vals = plot_df[x_col].astype(str).tolist() if plot_df[x_col].dtype == "object" else plot_df[x_col].to_numpy()
    if plot_df[x_col].dtype == "object":
        ax.errorbar(range(len(plot_df)), plot_df[y_col], yerr=plot_df[yerr_col], fmt="-o", capsize=4)
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(x_vals, rotation=20)
    else:
        ax.errorbar(plot_df[x_col], plot_df[y_col], yerr=plot_df[yerr_col], fmt="-o", capsize=4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def generate_parameter_study_plots(
    repeats_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    output_dir: str | Path = PLOTS_DIR,
) -> dict[str, str]:
    """Generate required boxplots and parameter curves for the study."""

    out_dir = _ensure_plots_dir(output_dir)
    paths: dict[str, str] = {}

    if "topology" in repeats_df.columns:
        paths["f1_by_topology_boxplot"] = _boxplot_by_category(
            repeats_df,
            category_col="topology",
            value_col="marl_f1",
            title="F1 by Topology",
            ylabel="F1",
            output_path=out_dir / "f1_by_topology_boxplot.png",
        )
        paths["energy_by_topology_boxplot"] = _boxplot_by_category(
            repeats_df,
            category_col="topology",
            value_col="marl_energy_total",
            title="Energy by Topology",
            ylabel="Energy Total (J)",
            output_path=out_dir / "energy_by_topology_boxplot.png",
        )
        paths["bandwidth_by_topology_boxplot"] = _boxplot_by_category(
            repeats_df,
            category_col="topology",
            value_col="marl_bytes_sent_total",
            title="Bandwidth by Topology",
            ylabel="Bytes Sent Total",
            output_path=out_dir / "bandwidth_by_topology_boxplot.png",
        )

    def _curve_source(col: str) -> pd.DataFrame:
        if summary_df.empty:
            return pd.DataFrame(columns=[col, "marl_f1_mean", "marl_f1_sem"])
        return (
            summary_df.groupby(col, as_index=False)
            .agg(
                marl_f1_mean=("marl_f1_mean", "mean"),
                marl_f1_sem=("marl_f1_sem", "mean"),
                marl_bytes_sent_total_mean=("marl_bytes_sent_total_mean", "mean"),
                marl_bytes_sent_total_sem=("marl_bytes_sent_total_sem", "mean"),
            )
        )

    if "non_iid_alpha" in summary_df.columns:
        alpha_df = _curve_source("non_iid_alpha")
        paths["alpha_vs_f1"] = _line_with_error(
            alpha_df,
            x_col="non_iid_alpha",
            y_col="marl_f1_mean",
            yerr_col="marl_f1_sem",
            title="Non-IID Alpha vs Mean F1",
            ylabel="Mean F1",
            output_path=out_dir / "alpha_vs_mean_f1.png",
        )
    if "n_agents" in summary_df.columns:
        agents_df = _curve_source("n_agents")
        paths["n_agents_vs_f1"] = _line_with_error(
            agents_df,
            x_col="n_agents",
            y_col="marl_f1_mean",
            yerr_col="marl_f1_sem",
            title="Number of Agents vs Mean F1",
            ylabel="Mean F1",
            output_path=out_dir / "n_agents_vs_mean_f1.png",
        )
    if "communication_budget" in summary_df.columns:
        budget_df = _curve_source("communication_budget")
        paths["comm_budget_vs_f1"] = _line_with_error(
            budget_df,
            x_col="communication_budget",
            y_col="marl_f1_mean",
            yerr_col="marl_f1_sem",
            title="Communication Budget vs Mean F1",
            ylabel="Mean F1",
            output_path=out_dir / "comm_budget_vs_mean_f1.png",
        )
        paths["comm_budget_vs_bandwidth"] = _line_with_error(
            budget_df,
            x_col="communication_budget",
            y_col="marl_bytes_sent_total_mean",
            yerr_col="marl_bytes_sent_total_sem",
            title="Communication Budget vs Bandwidth",
            ylabel="Bytes Sent Total",
            output_path=out_dir / "comm_budget_vs_bandwidth.png",
        )
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate parameter study plots")
    parser.add_argument(
        "--repeats-csv",
        default="algorithm_marl_xgboost/reports/parameter_study/repeats_all.csv",
    )
    parser.add_argument(
        "--summary-csv",
        default="algorithm_marl_xgboost/reports/parameter_study/parameter_study_results.csv",
    )
    parser.add_argument("--output-dir", default=str(PLOTS_DIR))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repeats_df = pd.read_csv(args.repeats_csv)
    summary_df = pd.read_csv(args.summary_csv)
    paths = generate_parameter_study_plots(repeats_df, summary_df, output_dir=args.output_dir)
    print(f"Generated {len(paths)} plots in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
