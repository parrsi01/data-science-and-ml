"""Optuna tuning for XGBoost with stratified CV and reproducible artifacts."""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from .imbalance import apply_smote, choose_imbalance_strategy


def _metric_score(metric_name: str, y_true, y_pred, y_proba) -> float:
    if metric_name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric_name == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if metric_name == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, y_proba))
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    raise ValueError(f"Unsupported tuning metric: {metric_name}")


def _sample_for_tuning(X, y, seed: int, max_rows: int = 5000):
    """Subsample while preserving class ratio to keep tuning runtime bounded."""

    if len(y) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    y_arr = np.asarray(y)
    pos_idx = idx[y_arr == 1]
    neg_idx = idx[y_arr == 0]
    target_pos = max(1, int(round(max_rows * (len(pos_idx) / len(y_arr)))))
    target_pos = min(target_pos, len(pos_idx))
    target_neg = max_rows - target_pos
    target_neg = min(target_neg, len(neg_idx))
    chosen = np.concatenate(
        [
            rng.choice(pos_idx, size=target_pos, replace=False),
            rng.choice(neg_idx, size=target_neg, replace=False),
        ]
    )
    chosen.sort()
    if hasattr(X, "iloc"):
        X_sub = X.iloc[chosen]
    else:
        X_sub = X[chosen]
    if hasattr(y, "iloc"):
        y_sub = y.iloc[chosen]
    else:
        y_sub = y[chosen]
    return X_sub, y_sub


def _build_xgb_from_trial(trial: optuna.Trial, adv_config: dict[str, Any], base_config: dict[str, Any], scale_pos_weight: float):
    from xgboost import XGBClassifier  # type: ignore[import-not-found]

    ss = adv_config["xgboost_search_space"]
    seed = int(base_config["dataset"]["seed"])
    params = {
        "max_depth": trial.suggest_int("max_depth", int(ss["max_depth"][0]), int(ss["max_depth"][1])),
        "learning_rate": trial.suggest_float("learning_rate", float(ss["learning_rate"][0]), float(ss["learning_rate"][1]), log=True),
        "n_estimators": trial.suggest_int("n_estimators", int(ss["n_estimators"][0]), int(ss["n_estimators"][1])),
        "subsample": trial.suggest_float("subsample", float(ss["subsample"][0]), float(ss["subsample"][1])),
        "colsample_bytree": trial.suggest_float("colsample_bytree", float(ss["colsample_bytree"][0]), float(ss["colsample_bytree"][1])),
        "reg_lambda": trial.suggest_float("reg_lambda", float(ss["reg_lambda"][0]), float(ss["reg_lambda"][1])),
        "min_child_weight": trial.suggest_int("min_child_weight", int(ss["min_child_weight"][0]), int(ss["min_child_weight"][1])),
        "gamma": trial.suggest_float("gamma", float(ss["gamma"][0]), float(ss["gamma"][1])),
    }
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric=str(base_config.get("xgboost", {}).get("eval_metric", "logloss")),
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        **params,
    )
    return model, params


def objective(trial: optuna.Trial, X, y, preprocessor, config: dict[str, Any]) -> float:
    """Optuna objective: stratified CV for XGBoost pipeline score."""

    base_config = config["base"]
    adv_config = config["advanced"]
    tuning_cfg = adv_config["tuning"]
    seed = int(base_config["dataset"]["seed"])
    metric_name = str(tuning_cfg.get("metric", "f1"))
    cv_folds = int(tuning_cfg.get("cv_folds", 5))

    X_tune, y_tune = _sample_for_tuning(X, y, seed=seed, max_rows=5000)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_scores: list[float] = []

    for train_idx, valid_idx in skf.split(X_tune, y_tune):
        X_train_fold = X_tune.iloc[train_idx]
        X_valid_fold = X_tune.iloc[valid_idx]
        y_train_fold = y_tune.iloc[train_idx]
        y_valid_fold = y_tune.iloc[valid_idx]

        X_train_trans = preprocessor.fit_transform(X_train_fold)
        X_valid_trans = preprocessor.transform(X_valid_fold)

        imbalance_decision = choose_imbalance_strategy(y_train_fold, adv_config)
        if imbalance_decision["use_smote"]:
            X_fit, y_fit = apply_smote(
                X_train_trans, y_train_fold, k_neighbors=int(adv_config["imbalance"]["smote_k_neighbors"])
            )
            scale_pos_weight = 1.0
        else:
            X_fit, y_fit = X_train_trans, y_train_fold
            scale_pos_weight = (
                imbalance_decision["scale_pos_weight"] if imbalance_decision["use_class_weights"] else 1.0
            )

        model, _ = _build_xgb_from_trial(trial, adv_config, base_config, scale_pos_weight)
        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_valid_trans)
        y_proba = model.predict_proba(X_valid_trans)[:, 1]
        score = _metric_score(metric_name, y_valid_fold, y_pred, y_proba)
        fold_scores.append(score)

    return float(np.mean(fold_scores))


def run_optuna_tuning(X, y, preprocessor, config: dict[str, Any], report_dir: str | Path) -> dict[str, Any]:
    """Run Optuna tuning and write best params / study summary artifacts."""

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    tuning_cfg = config["advanced"]["tuning"]
    seed = int(config["base"]["dataset"]["seed"])
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    start = time.perf_counter()
    study.optimize(
        lambda trial: objective(trial, X, y, preprocessor, config),
        n_trials=int(tuning_cfg.get("n_trials", 30)),
        timeout=int(tuning_cfg.get("timeout_seconds", 600)),
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - start

    best_params_path = report_dir / "optuna_best_params.json"
    study_summary_path = report_dir / "optuna_study_summary.json"

    best_payload = {
        "best_score": float(study.best_value),
        "best_params": study.best_params,
        "metric": str(tuning_cfg.get("metric", "f1")),
    }
    summary_payload = {
        "n_trials_requested": int(tuning_cfg.get("n_trials", 30)),
        "n_trials_completed": len(study.trials),
        "timeout_seconds": int(tuning_cfg.get("timeout_seconds", 600)),
        "best_value": float(study.best_value),
        "elapsed_seconds": elapsed,
        "best_trial_number": int(study.best_trial.number),
        "trials": [
            {
                "number": int(t.number),
                "value": None if t.value is None else float(t.value),
                "state": str(t.state),
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    best_params_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    study_summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Optuna best score: {best_payload['best_score']:.6f}")
    print(f"Optuna best params: {best_payload['best_params']}")

    return {
        "best_score": best_payload["best_score"],
        "best_params": dict(study.best_params),
        "best_params_path": str(best_params_path),
        "study_summary_path": str(study_summary_path),
        "elapsed_seconds": elapsed,
    }
