"""Repeated-run experiment harness wrapping the core MARL+XGBoost experiment."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha1
from pathlib import Path
import json
import time
from typing import Any

import pandas as pd
import yaml

from algorithm_marl_xgboost.src.config import load_yaml as load_experiment_config
from algorithm_marl_xgboost.src.run_experiment import run_experiment


BASE_EXPERIMENT_CONFIG_PATH = Path("algorithm_marl_xgboost/configs/experiment.yaml")
REPEATS_ROOT = Path("algorithm_marl_xgboost/reports/repeats")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _sanitize_value(value: Any) -> str:
    if isinstance(value, float):
        return str(value).replace(".", "p")
    return str(value).replace(" ", "_")


def build_label(config_overrides: dict[str, Any]) -> str:
    study = dict(config_overrides.get("_study", {}))
    ordered_items = sorted(study.items())
    human = "__".join(f"{k}-{_sanitize_value(v)}" for k, v in ordered_items) if ordered_items else "default"
    digest = sha1(json.dumps(ordered_items, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return f"{human}__{digest}"


def _extract_baseline(result: dict[str, Any], name: str) -> dict[str, Any]:
    for b in result.get("baselines", []):
        if b.get("name") == name:
            return b
    return {"metrics": {}, "traffic": {}}


def run_single(
    config_overrides: dict[str, Any],
    seed: int,
    *,
    repeat_index: int = 0,
    label: str | None = None,
    base_config_path: str | Path = BASE_EXPERIMENT_CONFIG_PATH,
) -> dict[str, Any]:
    base_cfg = load_experiment_config(base_config_path)
    cfg = _deep_merge(base_cfg, {k: v for k, v in config_overrides.items() if k != "_study"})
    cfg.setdefault("data", {})
    cfg["data"]["seed"] = int(seed)
    label = label or build_label(config_overrides)

    repeat_dir = REPEATS_ROOT / label / f"repeat_{repeat_index:02d}"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    cfg["artifacts"] = {
        "out_dir": str(repeat_dir / "reports"),
        "model_dir": str(repeat_dir / "models"),
        "log_path": str(repeat_dir / "logs" / "experiment.jsonl"),
    }
    effective_cfg_path = repeat_dir / "effective_experiment.yaml"
    effective_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    t0 = time.perf_counter()
    result = run_experiment(effective_cfg_path)
    runtime_seconds = time.perf_counter() - t0

    final_round = result["rounds"][-1]
    local_baseline = _extract_baseline(result, "local_only_xgboost")
    naive_baseline = _extract_baseline(result, "naive_decentralized_uniform")
    summary = {
        "label": label,
        "repeat_index": int(repeat_index),
        "seed": int(seed),
        "runtime_seconds": float(runtime_seconds),
        "config_overrides": config_overrides,
        "final_metrics": {k: float(v) for k, v in final_round["aggregate_metrics"].items()},
        "traffic_metrics": {k: float(v) for k, v in final_round["traffic_metrics"].items()},
        "baseline_metrics": {
            "local_only_xgboost": {k: float(v) for k, v in local_baseline.get("metrics", {}).items()},
            "naive_decentralized_uniform": {k: float(v) for k, v in naive_baseline.get("metrics", {}).items()},
        },
        "baseline_traffic": {
            "local_only_xgboost": {k: float(v) for k, v in local_baseline.get("traffic", {}).items()},
            "naive_decentralized_uniform": {k: float(v) for k, v in naive_baseline.get("traffic", {}).items()},
        },
        "per_round_metrics": result["rounds"],
        "artifacts": result.get("artifacts", {}),
    }
    out_json = repeat_dir / f"repeat_{repeat_index:02d}.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["repeat_json_path"] = str(out_json)
    return summary


def _summary_to_row(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": summary["label"],
        "repeat_index": int(summary["repeat_index"]),
        "seed": int(summary["seed"]),
        "runtime_seconds": float(summary["runtime_seconds"]),
    }
    row.update(summary.get("config_overrides", {}).get("_study", {}))
    for k, v in summary.get("final_metrics", {}).items():
        row[f"marl_{k}"] = float(v)
    for k, v in summary.get("traffic_metrics", {}).items():
        row[f"marl_{k}"] = float(v)
    for baseline_name, metrics in summary.get("baseline_metrics", {}).items():
        prefix = "local" if "local_only" in baseline_name else "naive"
        for k, v in metrics.items():
            row[f"{prefix}_{k}"] = float(v)
    for baseline_name, metrics in summary.get("baseline_traffic", {}).items():
        prefix = "local" if "local_only" in baseline_name else "naive"
        for k, v in metrics.items():
            row[f"{prefix}_{k}"] = float(v)
    return row


def run_repeats(
    config_overrides: dict[str, Any],
    repeats: int,
    seed_base: int,
    *,
    label: str | None = None,
    base_config_path: str | Path = BASE_EXPERIMENT_CONFIG_PATH,
) -> pd.DataFrame:
    label = label or build_label(config_overrides)
    rows: list[dict[str, Any]] = []
    root = REPEATS_ROOT / label
    root.mkdir(parents=True, exist_ok=True)
    for i in range(int(repeats)):
        summary = run_single(
            config_overrides,
            seed=int(seed_base) + i,
            repeat_index=i,
            label=label,
            base_config_path=base_config_path,
        )
        rows.append(_summary_to_row(summary))
    df = pd.DataFrame(rows).sort_values(["repeat_index"], kind="mergesort").reset_index(drop=True)
    df.to_csv(root / "aggregated_results.csv", index=False)
    return df

