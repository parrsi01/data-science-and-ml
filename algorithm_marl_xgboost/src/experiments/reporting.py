"""Executive and research-style reporting for parameter studies."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any

import pandas as pd


REPORT_DIR = Path("algorithm_marl_xgboost/reports/parameter_study")


def _format_config_row(row: pd.Series) -> str:
    return (
        f"alpha={row['non_iid_alpha']}, agents={int(row['n_agents'])}, "
        f"topology={row['topology']}, comm_budget={row['communication_budget']}"
    )


def generate_reports(
    summary_df: pd.DataFrame,
    repeats_df: pd.DataFrame,
    significance_payload: dict[str, Any],
    *,
    report_dir: str | Path = REPORT_DIR,
) -> dict[str, str]:
    """Generate executive and research summaries."""

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    ranked = summary_df.sort_values(["marl_f1_mean", "marl_f1_std"], ascending=[False, True], kind="mergesort")
    top3 = ranked.head(3)
    efficient = summary_df.copy()
    efficient = efficient[efficient["marl_f1_mean"] >= efficient["marl_f1_mean"].quantile(0.6)]
    efficient = efficient.sort_values(
        ["marl_bytes_sent_total_mean", "marl_energy_total_mean", "marl_f1_mean"],
        ascending=[True, True, False],
        kind="mergesort",
    ).head(3)

    exec_lines = [
        "# Executive Summary — Parameter Study (MARL + XGBoost)",
        "",
        "## Top 3 Configurations by Mean F1",
        "",
    ]
    for idx, row in enumerate(top3.itertuples(index=False), start=1):
        exec_lines.append(
            f"{idx}. alpha={row.non_iid_alpha}, agents={int(row.n_agents)}, topology={row.topology}, "
            f"comm_budget={row.communication_budget} -> mean F1={row.marl_f1_mean:.4f} (+/- {row.marl_f1_std:.4f})"
        )
    exec_lines.extend(["", "## Top 3 Most Efficient (Low Bandwidth/Energy, Acceptable F1)", ""])
    for idx, row in enumerate(efficient.itertuples(index=False), start=1):
        exec_lines.append(
            f"{idx}. alpha={row.non_iid_alpha}, agents={int(row.n_agents)}, topology={row.topology}, "
            f"comm_budget={row.communication_budget} -> F1={row.marl_f1_mean:.4f}, "
            f"bytes={row.marl_bytes_sent_total_mean:.1f}, energy={row.marl_energy_total_mean:.2f}"
        )
    exec_lines.extend(
        [
            "",
            "## Plain-Language Interpretation",
            "",
            "- Communication budget increases generally improve collaboration quality, but bandwidth/energy costs increase.",
            "- Topology choice changes both performance and operational cost; no single topology is universally best under all constraints.",
            "- Non-IID severity (Dirichlet alpha) can materially reduce consistency, so repeat-based reporting is required.",
            "",
            "## Deployment Recommendations",
            "",
            "- Use parameter settings with high mean F1 and low variance, not only peak F1.",
            "- Select communication budget based on infrastructure constraints (bandwidth/energy) and SLA priorities.",
            "- Retest significance and stability after any topology or agent-count changes in production-like environments.",
        ]
    )
    exec_path = report_dir / "EXECUTIVE_SUMMARY.md"
    exec_path.write_text("\n".join(exec_lines) + "\n", encoding="utf-8")

    research_lines = [
        "# Research Summary — Parameter Study (MARL + XGBoost)",
        "",
        "## Experimental Protocol",
        "",
        "- Repeated seeded runs were executed per configuration setting.",
        "- Metrics include performance (F1 primary) and traffic/energy costs.",
        "- Baselines (local-only, naive decentralized) were recorded alongside MARL trust-weighted results.",
        "",
        "## Repeats Details",
        "",
        f"- Total repeat rows: {len(repeats_df)}",
        f"- Total unique parameter settings: {summary_df['label'].nunique() if 'label' in summary_df.columns else len(summary_df)}",
        "",
        "## Significance Test Outcomes",
        "",
        "```json",
        json.dumps(significance_payload, indent=2),
        "```",
        "",
        "## Limitations",
        "",
        "- Synthetic data is a proxy and may not capture real institutional network traffic behavior.",
        "- The update abstraction uses feature-importance vectors, not full model parameters or secure aggregation.",
        "- Significance outcomes depend on repeat count and the selected parameter subset.",
        "",
        "## Reproducibility",
        "",
        "1. Keep study config YAML and experiment base config fixed",
        "2. Preserve repeat artifacts and aggregated CSV outputs",
        "3. Regenerate plots and summaries from saved CSV/JSON outputs",
        "",
        "## How to Reproduce",
        "",
        "```bash",
        "venv/bin/python -m algorithm_marl_xgboost.src.experiments.parameter_study \\",
        "  --config algorithm_marl_xgboost/configs/parameter_study.yaml",
        "```",
    ]
    research_path = report_dir / "RESEARCH_SUMMARY.md"
    research_path.write_text("\n".join(research_lines) + "\n", encoding="utf-8")
    return {
        "executive_summary_md": str(exec_path),
        "research_summary_md": str(research_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate parameter study reports")
    parser.add_argument(
        "--summary-csv",
        default="algorithm_marl_xgboost/reports/parameter_study/parameter_study_results.csv",
    )
    parser.add_argument(
        "--repeats-csv",
        default="algorithm_marl_xgboost/reports/parameter_study/repeats_all.csv",
    )
    parser.add_argument(
        "--significance-json",
        default="algorithm_marl_xgboost/reports/parameter_study/significance_tests.json",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary_df = pd.read_csv(args.summary_csv)
    repeats_df = pd.read_csv(args.repeats_csv)
    payload = json.loads(Path(args.significance_json).read_text(encoding="utf-8"))
    paths = generate_reports(summary_df, repeats_df, payload)
    print(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

