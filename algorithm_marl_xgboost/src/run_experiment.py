"""Experiment runner for decentralized federated anomaly detection (MARL + XGBoost)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import os
import random
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithm_marl_xgboost.src.agents import MARLAgent
from algorithm_marl_xgboost.src.baselines import run_local_only_baseline, run_naive_decentralized_baseline
from algorithm_marl_xgboost.src.config import load_yaml
from algorithm_marl_xgboost.src.data import (
    RAW_CATEGORICAL_COLUMNS,
    RAW_NUMERIC_COLUMNS,
    apply_smote_training_only,
    enforce_minority_presence,
    load_or_generate_data,
    partition_dirichlet,
    split_train_val,
)
from algorithm_marl_xgboost.src.decentralized_agg import build_update, trust_weighted_aggregate, uniform_aggregate
from algorithm_marl_xgboost.src.rewards import compute_reward
from algorithm_marl_xgboost.src.topologies import build_topology
from algorithm_marl_xgboost.src.traffic_energy import (
    simulate_bandwidth,
    simulate_energy,
    simulate_latency,
    simulate_packet_loss,
    summarize_round_traffic,
)
from algorithm_marl_xgboost.src.xgb_local import train_local_xgb

try:
    from src.data_quality.quality_metrics import compute_quality_metrics  # type: ignore[import-not-found]
    from src.evaluation.drift import categorical_drift_report, numeric_drift_report  # type: ignore[import-not-found]
except Exception:
    from data_quality.quality_metrics import compute_quality_metrics
    from evaluation.drift import categorical_drift_report, numeric_drift_report


@dataclass
class PreparedAgentData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series


def _ensure_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    out_dir = Path(cfg["artifacts"]["out_dir"])
    model_dir = Path(cfg["artifacts"]["model_dir"])
    log_path = Path(cfg["artifacts"]["log_path"])
    per_round_dir = out_dir / "per_round"
    for p in [out_dir, model_dir, log_path.parent, per_round_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return {"out_dir": out_dir, "model_dir": model_dir, "log_path": log_path, "per_round_dir": per_round_dir}


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _quality_snapshot(X: pd.DataFrame, y: pd.Series, out_dir: Path) -> str:
    df = X.copy()
    df["label"] = y.astype(int)
    rows = df.to_dict(orient="records")
    metrics = compute_quality_metrics(
        dataset_name="marl_xgb_synthetic",
        raw_df=rows,
        valid_df=rows,
        schema_invalid_df=[],
        domain_invalid_df=[],
        primary_id_col="row_id",
    )
    path = out_dir / "data_quality_snapshot.json"
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return str(path)


def _partition_drift_snapshot(X: pd.DataFrame, partitions: list[tuple[pd.DataFrame, pd.Series]], out_dir: Path) -> str:
    first_agent_X = partitions[0][0] if partitions else X.head(1)
    num_cols = [c for c in RAW_NUMERIC_COLUMNS if c in X.columns]
    cat_cols = [c for c in RAW_CATEGORICAL_COLUMNS if c in X.columns]
    report: dict[str, Any] = {}
    report.update(numeric_drift_report(X[num_cols], first_agent_X[num_cols], num_cols))
    report.update(categorical_drift_report(X[cat_cols], first_agent_X[cat_cols], cat_cols))
    path = out_dir / "partition_drift_snapshot.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return str(path)


def _prepare_agent_datasets(
    partitions: list[tuple[pd.DataFrame, pd.Series]],
    cfg: dict[str, Any],
) -> list[PreparedAgentData]:
    seed = int(cfg["data"]["seed"])
    train_fraction = float(cfg["data"].get("train_val_split", 0.8))
    prepared: list[PreparedAgentData] = []
    for agent_id, (Xi, yi) in enumerate(partitions):
        X_train, y_train, X_val, y_val = split_train_val(
            Xi, yi, train_fraction=train_fraction, seed=seed + 17 * (agent_id + 1)
        )
        X_train, y_train = apply_smote_training_only(
            X_train,
            y_train,
            enabled=bool(cfg["data"].get("per_agent_smote", True)),
            seed=seed + 1000 + agent_id,
        )
        prepared.append(PreparedAgentData(X_train, y_train, X_val, y_val))
    return prepared


def _plot_round_metrics(round_payload: dict[str, Any], output_path: Path) -> None:
    perf = round_payload["aggregate_metrics"]
    traffic = round_payload["traffic_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    perf_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    perf_values = [float(perf.get(name, 0.0)) for name in perf_names]
    axes[0].bar(perf_names, perf_values, color=["#0B6E4F", "#1A759F", "#F4A261", "#C84C09", "#5E548E"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title(f"Round {round_payload['round']:02d} Performance")
    axes[0].tick_params(axis="x", rotation=35)

    traffic_names = ["bytes_sent_total", "avg_latency_ms", "packet_loss_mean", "energy_total"]
    traffic_values = [float(traffic.get(name, 0.0)) for name in traffic_names]
    axes[1].bar(traffic_names, traffic_values, color=["#264653", "#2A9D8F", "#E9C46A", "#E76F51"])
    axes[1].set_title(f"Round {round_payload['round']:02d} Traffic/Energy")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_round_summary_txt(round_payload: dict[str, Any], output_path: Path) -> None:
    perf = round_payload["aggregate_metrics"]
    traffic = round_payload["traffic_metrics"]
    lines = [
        f"Round {round_payload['round']:02d} Summary",
        f"Accuracy: {perf['accuracy']:.4f}",
        f"Precision: {perf['precision']:.4f}",
        f"Recall: {perf['recall']:.4f}",
        f"F1: {perf['f1']:.4f}",
        f"ROC-AUC: {perf['roc_auc']:.4f}",
        f"Bytes Sent Total: {traffic['bytes_sent_total']:.1f}",
        f"Avg Latency (ms): {traffic['avg_latency_ms']:.2f}",
        f"Packet Loss Mean: {traffic['packet_loss_mean']:.4f}",
        f"Energy Total (J): {traffic['energy_total']:.3f}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_agent_metrics(local_metrics_by_agent: dict[int, dict[str, Any]]) -> dict[str, float]:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    payload: dict[str, float] = {}
    for key in keys:
        values = [float(m.get(key, 0.0)) for m in local_metrics_by_agent.values()]
        payload[key] = float(np.nanmean(values)) if values else 0.0
    return payload


def _save_traffic_csv(round_payloads: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "round",
        "bytes_sent_total",
        "avg_latency_ms",
        "packet_loss_mean",
        "energy_total",
        "n_events",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for p in round_payloads:
            row = {"round": p["round"], **p["traffic_metrics"], **p["aggregate_metrics"]}
            writer.writerow(row)


def _plot_average_traffic(round_payloads: list[dict[str, Any]], out_path: Path) -> None:
    if not round_payloads:
        return
    keys = ["bytes_sent_total", "avg_latency_ms", "packet_loss_mean", "energy_total"]
    avg_vals = [float(np.mean([float(p["traffic_metrics"].get(k, 0.0)) for p in round_payloads])) for k in keys]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(keys, avg_vals, color=["#355070", "#6D597A", "#B56576", "#E56B6F"])
    ax.set_title("Average Traffic/Energy Across Rounds")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_baseline_comparison(main_metrics: dict[str, float], baselines: list[dict[str, Any]], out_path: Path) -> None:
    labels = ["marl_trust_xgb"] + [str(b["name"]) for b in baselines]
    values = [float(main_metrics.get("f1", 0.0))] + [float(b["metrics"].get("f1", 0.0)) for b in baselines]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(labels, values, color=["#0B6E4F", "#457B9D", "#E76F51"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Performance Comparison (F1)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _final_summary_txt(
    *,
    cfg: dict[str, Any],
    round_payloads: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    artifact_paths: dict[str, str],
    out_path: Path,
) -> None:
    final_metrics = round_payloads[-1]["aggregate_metrics"] if round_payloads else {}
    lines = [
        "Decentralized Federated Anomaly Detection via MARL + XGBoost",
        f"Rounds: {int(cfg['training']['rounds'])}",
        f"Agents: {int(cfg['data']['n_agents'])}",
        f"Topology: {cfg['training']['topology']}",
        "",
        "Final MARL Metrics",
        f"Accuracy: {float(final_metrics.get('accuracy', 0.0)):.4f}",
        f"Precision: {float(final_metrics.get('precision', 0.0)):.4f}",
        f"Recall: {float(final_metrics.get('recall', 0.0)):.4f}",
        f"F1: {float(final_metrics.get('f1', 0.0)):.4f}",
        f"ROC-AUC: {float(final_metrics.get('roc_auc', 0.0)):.4f}",
        "",
        "Baselines",
    ]
    for b in baselines:
        m = b["metrics"]
        lines.append(
            f"- {b['name']}: F1={float(m.get('f1', 0.0)):.4f}, "
            f"Precision={float(m.get('precision', 0.0)):.4f}, Recall={float(m.get('recall', 0.0)):.4f}"
        )
    lines.extend(["", "Artifacts"])
    for key, value in artifact_paths.items():
        lines.append(f"- {key}: {value}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single_round_training(
    prepared_agents: list[PreparedAgentData],
    agents: list[MARLAgent],
    adjacency: dict[int, list[int]],
    cfg: dict[str, Any],
    *,
    round_idx: int,
    rng: np.random.Generator,
    previous_f1_by_agent: dict[int, float],
) -> dict[str, Any]:
    raw_feature_order = RAW_NUMERIC_COLUMNS + RAW_CATEGORICAL_COLUMNS
    local_models: dict[int, Any] = {}
    local_metrics_by_agent: dict[int, dict[str, Any]] = {}
    updates_by_agent: dict[int, dict[str, Any]] = {}

    for agent_id, prepared in enumerate(prepared_agents):
        model, metrics = train_local_xgb(
            prepared.X_train,
            prepared.y_train,
            prepared.X_val,
            prepared.y_val,
            cfg,
            seed=int(cfg["data"]["seed"]) + round_idx * 100 + agent_id,
            raw_feature_order=raw_feature_order,
            feature_weights=agents[agent_id].feature_weight_mapping(),
        )
        local_models[agent_id] = model
        local_metrics_by_agent[agent_id] = metrics
        updates_by_agent[agent_id] = build_update(
            agent_id=agent_id,
            feature_importance_vector=metrics["feature_importance_vector"],
            local_metrics=metrics,
        )

    communication_events: list[dict[str, Any]] = []
    observed_neighbor_scores_by_agent: dict[int, dict[int, float]] = {i: {} for i in range(len(agents))}
    strategy = str(cfg["marl"]["neighbor_selection"])
    trust_agg_enabled = bool(cfg["training"].get("trust_aggregation", True))
    comm_budget = float(cfg["marl"]["communication_budget"])

    for agent_id, agent in enumerate(agents):
        neighbors = adjacency.get(agent_id, [])
        selected = agent.choose_neighbors(
            neighbors,
            strategy=strategy,
            communication_budget=comm_budget,
            rng=rng,
        )
        received_updates = [updates_by_agent[n] for n in selected if n in updates_by_agent]
        if trust_agg_enabled:
            aggregated = trust_weighted_aggregate(received_updates, agent.trust_scores)
        else:
            aggregated = uniform_aggregate(received_updates)
        agent.update_from_aggregate(aggregated)

        for n in selected:
            observed_neighbor_scores_by_agent[agent_id][n] = float(local_metrics_by_agent[n]["f1"])

        bytes_sent = int(len(selected) * (len(raw_feature_order) * 8 + 128))
        bw = simulate_bandwidth(bytes_sent, rng)
        congestion_factor = min(1.0, len(selected) / max(1, len(neighbors)))
        latency_ms = simulate_latency(8.0, congestion_factor, rng)
        packet_loss = simulate_packet_loss(0.01 + 0.02 * congestion_factor, rng)
        training_cost = float(local_metrics_by_agent[agent_id]["rows_train"])
        energy_j = simulate_energy(bw["bandwidth_score"], training_cost)
        communication_events.append(
            {
                "agent_id": agent_id,
                "round": round_idx,
                "neighbors_selected": len(selected),
                "bytes_sent": bw["bytes_sent"],
                "latency_ms": latency_ms,
                "packet_loss": packet_loss,
                "energy_j": energy_j,
            }
        )

    traffic_summary = summarize_round_traffic(communication_events)
    per_agent_reward: dict[int, dict[str, float]] = {}
    for agent_id, agent in enumerate(agents):
        current_f1 = float(local_metrics_by_agent[agent_id]["f1"])
        prev_f1 = float(previous_f1_by_agent.get(agent_id, current_f1))
        f1_improvement = current_f1 - prev_f1
        agent_event = next(e for e in communication_events if int(e["agent_id"]) == agent_id)
        reward_payload = compute_reward(
            f1_improvement=f1_improvement,
            communication_cost=float(agent_event["bytes_sent"]) / 4096.0,
            energy_cost=float(agent_event["energy_j"]) / 10.0,
        )
        per_agent_reward[agent_id] = reward_payload
        agent.update_after_round(
            reward=reward_payload["reward"],
            observed_neighbor_scores=observed_neighbor_scores_by_agent[agent_id],
        )
        previous_f1_by_agent[agent_id] = current_f1

    aggregate_metrics = _aggregate_agent_metrics(local_metrics_by_agent)
    round_payload = {
        "round": int(round_idx),
        "aggregate_metrics": aggregate_metrics,
        "traffic_metrics": traffic_summary,
        "agent_metrics": {str(k): v for k, v in local_metrics_by_agent.items()},
        "agent_rewards": {str(k): v for k, v in per_agent_reward.items()},
        "agent_epsilons": {str(i): float(a.epsilon) for i, a in enumerate(agents)},
    }
    return round_payload


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    """Run the full decentralized MARL + XGBoost experiment and save artifacts."""

    cfg = load_yaml(config_path)
    paths = _ensure_dirs(cfg)
    out_dir = paths["out_dir"]
    per_round_dir = paths["per_round_dir"]
    log_path = paths["log_path"]
    if log_path.exists():
        log_path.unlink()

    seed = int(cfg["data"]["seed"])
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    _append_jsonl(log_path, {"event": "run_start", "seed": seed, "config_path": str(config_path)})

    X, y = load_or_generate_data(cfg)
    partitions = partition_dirichlet(
        X,
        y,
        n_agents=int(cfg["data"]["n_agents"]),
        alpha=float(cfg["data"]["non_iid_alpha"]),
        seed=seed,
    )
    partitions = enforce_minority_presence(
        partitions,
        min_pos_per_agent=int(cfg["data"]["min_pos_per_agent"]),
        seed=seed + 1,
        max_attempts=5,
    )
    adjacency = build_topology(
        cfg["training"]["topology"],
        n_agents=int(cfg["data"]["n_agents"]),
        p=float(cfg["training"].get("random_graph_p", 0.25)),
        seed=seed,
    )
    prepared_agents = _prepare_agent_datasets(partitions, cfg)
    raw_feature_order = RAW_NUMERIC_COLUMNS + RAW_CATEGORICAL_COLUMNS

    agents = [
        MARLAgent(
            agent_id=i,
            epsilon=float(cfg["marl"]["epsilon_start"]),
            epsilon_end=float(cfg["marl"]["epsilon_end"]),
            epsilon_decay=float(cfg["marl"]["epsilon_decay"]),
            feature_names=raw_feature_order,
        )
        for i in range(int(cfg["data"]["n_agents"]))
    ]
    for i, agent in enumerate(agents):
        agent.initialize_neighbors(adjacency.get(i, []))

    quality_snapshot_path = _quality_snapshot(X, y, out_dir)
    partition_drift_path = _partition_drift_snapshot(X, partitions, out_dir)

    round_payloads: list[dict[str, Any]] = []
    previous_f1_by_agent: dict[int, float] = {}
    rounds = int(cfg["training"]["rounds"])
    for round_idx in range(1, rounds + 1):
        round_payload = _run_single_round_training(
            prepared_agents,
            agents,
            adjacency,
            cfg,
            round_idx=round_idx,
            rng=rng,
            previous_f1_by_agent=previous_f1_by_agent,
        )
        round_payloads.append(round_payload)

        metrics_json = per_round_dir / f"round_{round_idx:02d}_metrics.json"
        metrics_json.write_text(json.dumps(round_payload, indent=2, default=str), encoding="utf-8")
        summary_txt = per_round_dir / f"round_{round_idx:02d}_summary.txt"
        _write_round_summary_txt(round_payload, summary_txt)
        plot_png = per_round_dir / f"round_{round_idx:02d}_plots.png"
        _plot_round_metrics(round_payload, plot_png)

        _append_jsonl(
            log_path,
            {
                "event": "round_complete",
                "round": round_idx,
                "aggregate_metrics": round_payload["aggregate_metrics"],
                "traffic_metrics": round_payload["traffic_metrics"],
            },
        )
        perf = round_payload["aggregate_metrics"]
        traffic = round_payload["traffic_metrics"]
        print(
            f"Round {round_idx:02d} | "
            f"Accuracy={perf['accuracy']:.4f} Precision={perf['precision']:.4f} "
            f"Recall={perf['recall']:.4f} F1={perf['f1']:.4f} | "
            f"Bytes={traffic['bytes_sent_total']:.0f} Latency={traffic['avg_latency_ms']:.2f}ms "
            f"Loss={traffic['packet_loss_mean']:.4f} Energy={traffic['energy_total']:.2f}J"
        )

    baselines = [
        run_local_only_baseline(partitions, cfg, raw_feature_order=raw_feature_order, seed=seed + 5000),
        run_naive_decentralized_baseline(
            partitions, adjacency, cfg, raw_feature_order=raw_feature_order, seed=seed + 6000
        ),
    ]
    baselines_path = out_dir / "baseline_results.json"
    baselines_path.write_text(json.dumps(baselines, indent=2, default=str), encoding="utf-8")

    traffic_csv = out_dir / "traffic_metrics_per_round.csv"
    _save_traffic_csv(round_payloads, traffic_csv)
    traffic_avg_png = out_dir / "traffic_metrics_average.png"
    _plot_average_traffic(round_payloads, traffic_avg_png)
    comparison_png = out_dir / "performance_comparison_boxplot.png"
    _plot_baseline_comparison(round_payloads[-1]["aggregate_metrics"], baselines, comparison_png)

    artifact_paths = {
        "jsonl_log": str(log_path),
        "quality_snapshot_json": quality_snapshot_path,
        "partition_drift_snapshot_json": partition_drift_path,
        "traffic_metrics_per_round_csv": str(traffic_csv),
        "traffic_metrics_average_png": str(traffic_avg_png),
        "performance_comparison_boxplot_png": str(comparison_png),
        "baseline_results_json": str(baselines_path),
    }
    final_summary_path = out_dir / "final_summary.txt"
    _final_summary_txt(
        cfg=cfg,
        round_payloads=round_payloads,
        baselines=baselines,
        artifact_paths=artifact_paths,
        out_path=final_summary_path,
    )
    _append_jsonl(
        log_path,
        {
            "event": "run_complete",
            "rounds": rounds,
            "final_metrics": round_payloads[-1]["aggregate_metrics"] if round_payloads else {},
            "artifacts": artifact_paths,
        },
    )

    result = {
        "config": cfg,
        "rounds": round_payloads,
        "baselines": baselines,
        "artifacts": {**artifact_paths, "final_summary_txt": str(final_summary_path)},
    }
    result_json = out_dir / "run_result.json"
    result_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["artifacts"]["run_result_json"] = str(result_json)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run decentralized MARL + XGBoost experiment")
    parser.add_argument(
        "--config",
        default="algorithm_marl_xgboost/configs/experiment.yaml",
        help="Path to experiment YAML config",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    run_experiment(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
