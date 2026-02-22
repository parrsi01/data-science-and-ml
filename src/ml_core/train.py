"""Config-driven institutional ML core training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .data import ensure_dataset_from_config
from .evaluate import compute_confusion_matrix, compute_metrics
from .models import get_model
from .plots import save_confusion_matrix_plot, save_roc_curve_plot
from .preprocess import add_feature_engineering, build_preprocessor


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config from disk."""

    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _class_weight_info(y_train: pd.Series) -> dict[str, float]:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = float(negatives / positives) if positives > 0 else 1.0
    return {
        "positives": float(positives),
        "negatives": float(negatives),
        "scale_pos_weight": scale_pos_weight,
    }


def _ensure_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    artifacts_cfg = config["artifacts"]
    model_dir = Path(artifacts_cfg["model_dir"])
    report_dir = Path(artifacts_cfg["report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, report_dir


def _evaluate_model(pipeline, X_test, y_test) -> tuple[dict[str, float], dict[str, Any]]:
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(X_test)
        scores = np.asarray(scores, dtype=float)
        # Map scores to [0,1] for ROC/AUC compatibility.
        y_proba = 1.0 / (1.0 + np.exp(-scores))
    else:
        y_proba = np.asarray(y_pred, dtype=float)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    aux = {
        "y_pred": np.asarray(y_pred),
        "y_proba": np.asarray(y_proba),
        "confusion_matrix": compute_confusion_matrix(y_test, y_pred),
    }
    return metrics, aux


def _cross_validate_model(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    eval_cfg = config["evaluation"]
    cv_folds = int(eval_cfg.get("cv_folds", 5))
    seed = int(config["dataset"]["seed"])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )
    summary: dict[str, Any] = {}
    for key, values in results.items():
        if key.startswith("test_"):
            metric_name = key.replace("test_", "")
            values = np.asarray(values, dtype=float)
            summary[metric_name] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "fold_values": [float(v) for v in values.tolist()],
            }
    return summary


def run_training(config_path: str | Path) -> dict[str, Any]:
    """Run the end-to-end training pipeline and return summary payload."""

    config = load_config(config_path)
    model_dir, report_dir = _ensure_dirs(config)

    df = ensure_dataset_from_config(config)
    df = add_feature_engineering(df)

    target_name = config["target"]["name"]
    categorical_cols = list(config["features"]["categorical"])
    numerical_cols = list(config["features"]["numerical"])

    X = df[categorical_cols + numerical_cols].copy()
    y = df[target_name].astype(int).copy()

    test_size = float(config["dataset"]["test_size"])
    seed = int(config["dataset"]["seed"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    class_info = _class_weight_info(y_train)

    metrics_report: dict[str, Any] = {}
    cv_results_report: dict[str, Any] = {}
    artifact_registry: dict[str, Any] = {"models": {}, "plots": {}}

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        preprocessor = build_preprocessor(categorical_cols, numerical_cols)
        estimator = get_model(model_name, config, class_info)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

        cv_summary = _cross_validate_model(pipeline, X_train, y_train, config)
        pipeline.fit(X_train, y_train)
        test_metrics, aux = _evaluate_model(pipeline, X_test, y_test)

        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)

        cm_path = report_dir / f"confusion_matrix_{model_name}.png"
        roc_path = report_dir / f"roc_{model_name}.png"
        save_confusion_matrix_plot(y_test, aux["y_pred"], cm_path, model_name=model_name)
        save_roc_curve_plot(y_test, aux["y_proba"], roc_path, model_name=model_name)

        metrics_report[model_name] = {
            "test_metrics": test_metrics,
            "confusion_matrix": aux["confusion_matrix"],
            "class_weight_info": class_info,
        }
        cv_results_report[model_name] = cv_summary
        artifact_registry["models"][model_name] = str(model_path)
        artifact_registry["plots"][model_name] = {
            "confusion_matrix": str(cm_path),
            "roc": str(roc_path),
        }

    metrics_path = report_dir / "metrics.json"
    cv_results_path = report_dir / "cv_results.json"
    metrics_path.write_text(json.dumps(metrics_report, indent=2, default=str), encoding="utf-8")
    cv_results_path.write_text(
        json.dumps(cv_results_report, indent=2, default=str), encoding="utf-8"
    )

    summary = {
        "config_path": str(config_path),
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics_path": str(metrics_path),
        "cv_results_path": str(cv_results_path),
        "artifacts": artifact_registry,
        "metrics": metrics_report,
    }
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run institutional ML core training pipeline")
    parser.add_argument(
        "--config",
        default="configs/ml_core/config.yaml",
        help="Path to YAML config file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = _build_arg_parser().parse_args(argv)
    summary = run_training(args.config)

    for model_name, payload in summary["metrics"].items():
        m = payload["test_metrics"]
        print(
            f"{model_name}: "
            f"accuracy={m['accuracy']:.4f} "
            f"precision={m['precision']:.4f} "
            f"recall={m['recall']:.4f} "
            f"f1={m['f1']:.4f} "
            f"roc_auc={m['roc_auc']:.4f}"
        )
    print(f"metrics.json: {summary['metrics_path']}")
    print(f"cv_results.json: {summary['cv_results_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
