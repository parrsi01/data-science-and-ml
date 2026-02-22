"""Advanced ML orchestrator: imbalance handling, Optuna tuning, SHAP explainability."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import json
import os
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

try:  # Supports `python -m src.ml_advanced.train_advanced`
    from src.ml_core.data import ensure_dataset_from_config  # type: ignore[import-not-found]
    from src.ml_core.evaluate import compute_metrics  # type: ignore[import-not-found]
    from src.ml_core.plots import save_confusion_matrix_plot, save_roc_curve_plot  # type: ignore[import-not-found]
    from src.ml_core.preprocess import add_feature_engineering, build_preprocessor  # type: ignore[import-not-found]
    from src.ml_advanced.explain_shap import generate_shap_reports  # type: ignore[import-not-found]
    from src.ml_advanced.features import add_advanced_features  # type: ignore[import-not-found]
    from src.ml_advanced.imbalance import apply_smote, choose_imbalance_strategy  # type: ignore[import-not-found]
    from src.ml_advanced.tune_xgb import run_optuna_tuning  # type: ignore[import-not-found]
except Exception:  # Supports `PYTHONPATH=src ... -m ml_advanced.train_advanced`
    from ml_core.data import ensure_dataset_from_config
    from ml_core.evaluate import compute_metrics
    from ml_core.plots import save_confusion_matrix_plot, save_roc_curve_plot
    from ml_core.preprocess import add_feature_engineering, build_preprocessor
    from ml_advanced.explain_shap import generate_shap_reports
    from ml_advanced.features import add_advanced_features
    from ml_advanced.imbalance import apply_smote, choose_imbalance_strategy
    from ml_advanced.tune_xgb import run_optuna_tuning


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_advanced_config(config_path: str | Path) -> dict[str, Any]:
    """Load and merge base + advanced YAML configs."""

    adv_path = Path(config_path)
    adv_cfg = yaml.safe_load(adv_path.read_text(encoding="utf-8"))
    base_path = Path(adv_cfg["base_config"])
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    merged = _deep_merge(base_cfg, adv_cfg)
    return {"base": base_cfg, "advanced": adv_cfg, "merged": merged, "paths": {"advanced": str(adv_path), "base": str(base_path)}}


def _prepare_dataframe(config_bundle: dict[str, Any]) -> pd.DataFrame:
    base_cfg = config_bundle["base"]
    df = ensure_dataset_from_config(base_cfg)
    df = add_feature_engineering(df)
    df = add_advanced_features(df)
    return df


def _advanced_feature_lists(merged_config: dict[str, Any]) -> tuple[list[str], list[str]]:
    categorical = list(merged_config["features"]["categorical"])
    numerical = list(merged_config["features"]["numerical"])
    for col in ["lag_metric_a", "rolling_std_a", "interaction_a_b"]:
        if col not in numerical:
            numerical.append(col)
    return categorical, numerical


def _build_xgb_estimator(params: dict[str, Any], seed: int, scale_pos_weight: float):
    from xgboost import XGBClassifier  # type: ignore[import-not-found]

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        **params,
    )


def _transform_and_resample(preprocessor, X_train, y_train, adv_cfg):
    X_train_trans = preprocessor.fit_transform(X_train)
    decision = choose_imbalance_strategy(y_train, adv_cfg)
    if decision["use_smote"]:
        X_fit, y_fit = apply_smote(
            X_train_trans,
            y_train,
            k_neighbors=int(adv_cfg["imbalance"]["smote_k_neighbors"]),
        )
        scale_pos_weight = 1.0
    else:
        X_fit, y_fit = X_train_trans, y_train
        scale_pos_weight = decision["scale_pos_weight"] if decision["use_class_weights"] else 1.0
    return X_fit, y_fit, X_train_trans, decision, scale_pos_weight


def run_advanced_training(config_path: str | Path) -> dict[str, Any]:
    """Run advanced XGBoost training with tuning and SHAP explainability."""

    cfg_bundle = load_advanced_config(config_path)
    base_cfg = cfg_bundle["base"]
    adv_cfg = cfg_bundle["advanced"]
    merged_cfg = cfg_bundle["merged"]

    report_dir = Path(adv_cfg["artifacts"]["report_dir"])
    model_dir = Path(adv_cfg["artifacts"]["model_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_dataframe(cfg_bundle)
    categorical_cols, numerical_cols = _advanced_feature_lists(merged_cfg)

    target_name = merged_cfg["target"]["name"]
    X = df[categorical_cols + numerical_cols].copy()
    y = df[target_name].astype(int).copy()

    seed = int(merged_cfg["dataset"]["seed"])
    test_size = float(merged_cfg["dataset"]["test_size"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Tuning uses a clean preprocessor instance separate from final fit.
    tune_preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    tuning_result: dict[str, Any] | None = None
    best_params: dict[str, Any]
    if bool(adv_cfg["tuning"]["enabled"]):
        tuning_result = run_optuna_tuning(
            X_train,
            y_train,
            tune_preprocessor,
            {"base": base_cfg, "advanced": adv_cfg},
            report_dir=report_dir,
        )
        best_params = dict(tuning_result["best_params"])
    else:
        best_params = {
            "n_estimators": int(base_cfg["xgboost"]["n_estimators"]),
            "max_depth": int(base_cfg["xgboost"]["max_depth"]),
            "learning_rate": float(base_cfg["xgboost"]["learning_rate"]),
            "subsample": float(base_cfg["xgboost"]["subsample"]),
            "colsample_bytree": float(base_cfg["xgboost"]["colsample_bytree"]),
            "reg_lambda": float(base_cfg["xgboost"]["reg_lambda"]),
            "min_child_weight": 1,
            "gamma": 0.0,
        }

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    X_fit, y_fit, X_train_trans, imbalance_decision, scale_pos_weight = _transform_and_resample(
        preprocessor, X_train, y_train, adv_cfg
    )
    X_test_trans = preprocessor.transform(X_test)

    model = _build_xgb_estimator(best_params, seed=seed, scale_pos_weight=scale_pos_weight)
    model.fit(X_fit, y_fit)

    y_pred = model.predict(X_test_trans)
    y_proba = model.predict_proba(X_test_trans)[:, 1]
    test_metrics = compute_metrics(y_test, y_pred, y_proba)

    model_bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "best_params": best_params,
        "imbalance_decision": imbalance_decision,
    }
    model_path = model_dir / "xgboost_tuned.joblib"
    joblib.dump(model_bundle, model_path)

    cm_path = report_dir / "confusion_matrix_xgboost_tuned.png"
    roc_path = report_dir / "roc_xgboost_tuned.png"
    save_confusion_matrix_plot(y_test, y_pred, cm_path, model_name="xgboost_tuned")
    save_roc_curve_plot(y_test, y_proba, roc_path, model_name="xgboost_tuned")

    metrics_payload = {
        "test_metrics": test_metrics,
        "best_params": best_params,
        "imbalance_decision": imbalance_decision,
        "tuning": tuning_result,
        "dataset": {
            "rows_total": int(len(df)),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
        },
    }
    metrics_path = report_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8")

    shap_artifacts = None
    if bool(adv_cfg["explainability"]["enabled"]):
        sample_size = int(adv_cfg["explainability"]["sample_size"])
        rng = np.random.default_rng(seed)
        n = X_test_trans.shape[0]
        sample_n = min(sample_size, n)
        sample_idx = np.sort(rng.choice(np.arange(n), size=sample_n, replace=False))
        X_shap = X_test_trans[sample_idx]
        feature_names = preprocessor.get_feature_names_out()
        shap_artifacts = generate_shap_reports(
            model,
            X_shap,
            feature_names=feature_names,
            output_dir=adv_cfg["explainability"]["output_dir"],
        )

    summary = {
        "config_paths": cfg_bundle["paths"],
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "confusion_matrix_png": str(cm_path),
        "roc_png": str(roc_path),
        "shap_artifacts": shap_artifacts,
        "test_metrics": test_metrics,
        "best_params": best_params,
        "tuning_result": tuning_result,
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Advanced XGBoost training pipeline")
    parser.add_argument("--config", default="configs/ml_advanced/config.yaml")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = run_advanced_training(args.config)
    print(f"Optuna best score: {summary['tuning_result']['best_score']:.6f}" if summary["tuning_result"] else "Optuna disabled")
    print(f"Best params: {summary['best_params']}")
    print(f"Test metrics: {summary['test_metrics']}")
    print(f"Model saved: {summary['model_path']}")
    print(f"Metrics saved: {summary['metrics_path']}")
    if summary["shap_artifacts"]:
        print(f"SHAP artifacts: {summary['shap_artifacts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
