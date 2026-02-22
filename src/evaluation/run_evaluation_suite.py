"""Institutional evaluation suite runner: stability, thresholding, group checks, drift."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import json
import os
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

try:  # `python -m src.evaluation.run_evaluation_suite`
    from src.ml_core.data import ensure_dataset_from_config  # type: ignore[import-not-found]
    from src.ml_core.evaluate import compute_metrics  # type: ignore[import-not-found]
    from src.ml_core.preprocess import add_feature_engineering, build_preprocessor  # type: ignore[import-not-found]
    from src.ml_advanced.features import add_advanced_features  # type: ignore[import-not-found]
    from src.ml_advanced.train_advanced import load_advanced_config  # type: ignore[import-not-found]
    from src.evaluation.cv import repeated_stratified_cv, stratified_cv_scores  # type: ignore[import-not-found]
    from src.evaluation.drift import categorical_drift_report, numeric_drift_report, write_drift_reports  # type: ignore[import-not-found]
    from src.evaluation.group_metrics import compute_group_metrics, write_group_metrics_reports  # type: ignore[import-not-found]
    from src.evaluation.stability import run_seed_sweep, write_stability_reports  # type: ignore[import-not-found]
    from src.evaluation.thresholds import find_best_threshold, write_threshold_reports  # type: ignore[import-not-found]
except Exception:  # `PYTHONPATH=src ... -m evaluation.run_evaluation_suite`
    from ml_core.data import ensure_dataset_from_config
    from ml_core.evaluate import compute_metrics
    from ml_core.preprocess import add_feature_engineering, build_preprocessor
    from ml_advanced.features import add_advanced_features
    from ml_advanced.train_advanced import load_advanced_config
    from evaluation.cv import repeated_stratified_cv, stratified_cv_scores
    from evaluation.drift import categorical_drift_report, numeric_drift_report, write_drift_reports
    from evaluation.group_metrics import compute_group_metrics, write_group_metrics_reports
    from evaluation.stability import run_seed_sweep, write_stability_reports
    from evaluation.thresholds import find_best_threshold, write_threshold_reports


REPORT_DIR = Path("reports/evaluation")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _prepare_data(config_bundle: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str], str]:
    base_cfg = config_bundle["base"]
    merged_cfg = config_bundle["merged"]
    df = ensure_dataset_from_config(base_cfg)
    df = add_feature_engineering(df)
    df = add_advanced_features(df)
    categorical = list(merged_cfg["features"]["categorical"])
    numerical = list(merged_cfg["features"]["numerical"])
    for col in ["lag_metric_a", "rolling_std_a", "interaction_a_b"]:
        if col not in numerical:
            numerical.append(col)
    target = merged_cfg["target"]["name"]
    return df, categorical, numerical, target


def _load_tuned_params_or_fallback(report_dir: Path, base_cfg: dict[str, Any]) -> dict[str, Any]:
    optuna_path = report_dir / "optuna_best_params.json"
    if optuna_path.exists():
        payload = json.loads(optuna_path.read_text(encoding="utf-8"))
        return dict(payload["best_params"])
    xgb = base_cfg["xgboost"]
    return {
        "n_estimators": int(xgb["n_estimators"]),
        "max_depth": int(xgb["max_depth"]),
        "learning_rate": float(xgb["learning_rate"]),
        "subsample": float(xgb["subsample"]),
        "colsample_bytree": float(xgb["colsample_bytree"]),
        "reg_lambda": float(xgb["reg_lambda"]),
        "min_child_weight": 1,
        "gamma": 0.0,
    }


def _build_xgb_model(params: dict[str, Any], seed: int, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        scale_pos_weight=float(scale_pos_weight),
        **params,
    )


def _class_imbalance_scale(y_train: pd.Series) -> float:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    return float(negatives / positives) if positives > 0 else 1.0


def _fit_preprocessor_model(X_train, y_train, categorical, numerical, params, seed: int):
    pre = build_preprocessor(categorical, numerical)
    X_train_t = pre.fit_transform(X_train)
    model = _build_xgb_model(params, seed=seed, scale_pos_weight=_class_imbalance_scale(y_train))
    model.fit(X_train_t, y_train)
    return pre, model


def _predict(pre, model, X):
    Xt = pre.transform(X)
    y_pred = model.predict(Xt)
    y_proba = model.predict_proba(Xt)[:, 1]
    return Xt, np.asarray(y_pred), np.asarray(y_proba)


def _seed_train_fn_factory(X_train, y_train, X_test, y_test, categorical, numerical, params):
    def _train_fn(seed: int) -> dict[str, float]:
        pre, model = _fit_preprocessor_model(X_train, y_train, categorical, numerical, params, seed=seed)
        _, y_pred, y_proba = _predict(pre, model, X_test)
        return compute_metrics(y_test, y_pred, y_proba)

    return _train_fn


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def run_evaluation_suite(config_path: str | Path) -> dict[str, Any]:
    """Run institutional evaluation suite and write all evaluation reports."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_bundle = load_advanced_config(config_path)
    base_cfg = cfg_bundle["base"]
    adv_cfg = cfg_bundle["advanced"]
    merged_cfg = cfg_bundle["merged"]

    df, categorical, numerical, target = _prepare_data(cfg_bundle)
    X = df[categorical + numerical].copy()
    y = df[target].astype(int).copy()
    seed = int(merged_cfg["dataset"]["seed"])
    test_size = float(merged_cfg["dataset"]["test_size"])

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Validation split only from training set for threshold calibration.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=seed + 1,
        stratify=y_train_full,
    )

    params = _load_tuned_params_or_fallback(Path(adv_cfg["artifacts"]["report_dir"]), base_cfg)

    # Fit main model on train split, use validation for threshold calibration, test for final evaluation.
    pre, model = _fit_preprocessor_model(X_train, y_train, categorical, numerical, params, seed=seed)
    _, y_valid_pred, y_valid_proba = _predict(pre, model, X_valid)
    X_test_t, y_test_pred_default, y_test_proba = _predict(pre, model, X_test)

    threshold_result = find_best_threshold(y_valid, y_valid_proba, metric="f1")
    threshold_paths = write_threshold_reports(threshold_result, REPORT_DIR)
    best_threshold = float(threshold_result["best_threshold"])
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

    # CV on a bounded sample to keep suite runtime reasonable.
    cv_sample_n = min(6000, len(X_train_full))
    cv_idx = np.arange(len(X_train_full))[:cv_sample_n]
    X_cv = X_train_full.iloc[cv_idx]
    y_cv = y_train_full.iloc[cv_idx]
    cv_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(categorical, numerical)),
            (
                "model",
                _build_xgb_model(
                    params=params,
                    seed=seed,
                    scale_pos_weight=_class_imbalance_scale(y_train_full),
                ),
            ),
        ]
    )
    metric_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_result = stratified_cv_scores(cv_pipeline, X_cv, y_cv, folds=3, metrics=metric_list)
    repeated_cv_result = repeated_stratified_cv(
        cv_pipeline, X_cv, y_cv, folds=3, repeats=2, metrics=metric_list
    )
    cv_json_path = REPORT_DIR / "cv_report.json"
    _write_json(cv_json_path, {"stratified_cv": cv_result, "repeated_stratified_cv": repeated_cv_result})

    # Group metrics on test set using category as synthetic group.
    group_df = compute_group_metrics(X_test.reset_index(drop=True), y_test.to_numpy(), y_test_pred, "category")
    group_paths = write_group_metrics_reports(group_df, REPORT_DIR)

    # Drift snapshot: train vs test (full train split vs test split).
    numeric_drift = numeric_drift_report(X_train_full, X_test, numeric_cols=numerical)
    categorical_drift = categorical_drift_report(X_train_full, X_test, cat_cols=categorical)
    drift_payload = _deep_merge(numeric_drift, categorical_drift)
    drift_paths = write_drift_reports(drift_payload, REPORT_DIR)

    # Stability seed sweep using same train/test split and tuned params.
    seeds = [7, 11, 21, 42, 99]
    seed_train_fn = _seed_train_fn_factory(
        X_train_full,
        y_train_full,
        X_test,
        y_test,
        categorical,
        numerical,
        params,
    )
    stability_summary = run_seed_sweep(seed_train_fn, seeds)
    stability_paths = write_stability_reports(stability_summary, REPORT_DIR)

    # Find largest drift feature and worst group recall for summary.
    largest_drift_feature = None
    largest_drift_value = -1.0
    for col, payload in drift_payload.get("numeric", {}).items():
        score = abs(float(payload["mean_shift"])) + abs(float(payload["std_shift"]))
        if score > largest_drift_value:
            largest_drift_value = score
            largest_drift_feature = col
    for col, payload in drift_payload.get("categorical", {}).items():
        score = float(payload["tv_distance"])
        if score > largest_drift_value:
            largest_drift_value = score
            largest_drift_feature = col

    worst_group_row = group_df.sort_values("recall", ascending=True).iloc[0]
    summary = {
        "test_metrics": test_metrics,
        "best_threshold": best_threshold,
        "stability_score_f1_std": float(stability_summary["stability_score_f1_std"]),
        "largest_drift_feature": largest_drift_feature,
        "largest_drift_score": float(largest_drift_value),
        "worst_group_recall": {
            "group": str(worst_group_row.iloc[0]),
            "recall": float(worst_group_row["recall"]),
        },
        "artifacts": {
            "cv_report_json": str(cv_json_path),
            **threshold_paths,
            **group_paths,
            **drift_paths,
            **stability_paths,
        },
    }
    summary_path = REPORT_DIR / "evaluation_summary.json"
    _write_json(summary_path, summary)
    summary["artifacts"]["evaluation_summary_json"] = str(summary_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run institutional evaluation suite")
    parser.add_argument("--config", default="configs/ml_advanced/config.yaml")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = run_evaluation_suite(args.config)
    print("Institutional Evaluation Summary")
    print(
        f"test_f1={summary['test_metrics']['f1']:.4f} "
        f"test_roc_auc={summary['test_metrics']['roc_auc']:.4f}"
    )
    print(f"best_threshold={summary['best_threshold']:.2f}")
    print(f"stability_std_f1={summary['stability_score_f1_std']:.6f}")
    print(f"largest_drift_feature={summary['largest_drift_feature']}")
    print(
        f"worst_group_recall={summary['worst_group_recall']['group']}:{summary['worst_group_recall']['recall']:.4f}"
    )
    print(f"reports_dir={REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
