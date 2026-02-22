from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from ml_advanced.explain_shap import generate_shap_reports
from ml_advanced.features import add_advanced_features
from ml_advanced.imbalance import choose_imbalance_strategy
from ml_advanced.tune_xgb import run_optuna_tuning
from ml_advanced.train_advanced import run_advanced_training
from ml_core.data import make_synthetic_classification
from ml_core.preprocess import add_feature_engineering, build_preprocessor


def _tiny_dataset(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    df = make_synthetic_classification(n_rows=n_rows, seed=seed)
    df = add_feature_engineering(df)
    df = add_advanced_features(df)
    return df


def test_advanced_features_add_expected_columns() -> None:
    df = _tiny_dataset(500, 101)
    assert {"lag_metric_a", "rolling_std_a", "interaction_a_b"} <= set(df.columns)


def test_imbalance_strategy_chooser_returns_keys() -> None:
    df = _tiny_dataset(1000, 102)
    y = df["label_rare_event"]
    decision = choose_imbalance_strategy(y, {"imbalance": {"strategy": "auto", "smote_k_neighbors": 3}})
    assert {"use_smote", "use_class_weights", "scale_pos_weight", "reason"} <= set(decision.keys())


def test_optuna_tuning_runs_fast_two_trials(tmp_path: Path) -> None:
    df = _tiny_dataset(800, 103)
    feature_cols = ["category", "metric_a", "metric_b", "ratio_ab", "rolling_mean_a", "lag_metric_a", "rolling_std_a", "interaction_a_b"]
    X = df[feature_cols]
    y = df["label_rare_event"].astype(int)

    preprocessor = build_preprocessor(
        categorical_cols=["category"],
        numerical_cols=["metric_a", "metric_b", "ratio_ab", "rolling_mean_a", "lag_metric_a", "rolling_std_a", "interaction_a_b"],
    )
    config = {
        "base": {"dataset": {"seed": 42}, "xgboost": {"eval_metric": "logloss"}},
        "advanced": {
            "imbalance": {"strategy": "auto", "smote_k_neighbors": 3},
            "tuning": {"n_trials": 2, "timeout_seconds": 120, "metric": "f1", "cv_folds": 2},
            "xgboost_search_space": {
                "max_depth": [3, 4],
                "learning_rate": [0.05, 0.1],
                "n_estimators": [20, 40],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [0.0, 2.0],
                "min_child_weight": [1, 3],
                "gamma": [0.0, 1.0],
            },
        },
    }
    result = run_optuna_tuning(X, y, preprocessor, config, report_dir=tmp_path)
    assert "best_score" in result
    assert Path(result["best_params_path"]).exists()
    assert Path(result["study_summary_path"]).exists()


def test_shap_report_generation_writes_expected_files(tmp_path: Path) -> None:
    from xgboost import XGBClassifier

    df = _tiny_dataset(500, 104)
    X = df[["category", "metric_a", "metric_b", "ratio_ab", "rolling_mean_a", "lag_metric_a", "rolling_std_a", "interaction_a_b"]]
    y = df["label_rare_event"].astype(int)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pre = build_preprocessor(["category"], ["metric_a", "metric_b", "ratio_ab", "rolling_mean_a", "lag_metric_a", "rolling_std_a", "interaction_a_b"])
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    model = XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(X_train_t, y_train)

    if hasattr(X_test_t, "shape") and X_test_t.shape[0] > 50:
        X_sample = X_test_t[:50]
    else:
        X_sample = X_test_t
    paths = generate_shap_reports(model, X_sample, pre.get_feature_names_out(), tmp_path)
    assert Path(paths["shap_summary_png"]).exists()
    assert Path(paths["shap_bar_png"]).exists()
    assert Path(paths["shap_values_npz"]).exists()
    assert Path(paths["explainability_notes_md"]).exists()


def test_training_advanced_runs_small_and_writes_metrics(tmp_path: Path) -> None:
    base_cfg = {
        "dataset": {"source": "synthetic", "n_rows": 2000, "seed": 42, "test_size": 0.2},
        "target": {"name": "label_rare_event", "positive_label": 1},
        "features": {
            "categorical": ["category"],
            "numerical": ["metric_a", "metric_b", "ratio_ab", "rolling_mean_a"],
        },
        "xgboost": {
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
        },
        "evaluation": {"metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"], "cv_folds": 3},
        "artifacts": {"model_dir": str(tmp_path / "unused_models"), "report_dir": str(tmp_path / "unused_reports")},
    }
    adv_cfg = {
        "base_config": str(tmp_path / "base.yaml"),
        "imbalance": {"strategy": "weights", "smote_k_neighbors": 3},
        "tuning": {"enabled": True, "n_trials": 2, "timeout_seconds": 120, "metric": "f1", "cv_folds": 2},
        "xgboost_search_space": {
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [20, 40],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [0.0, 2.0],
            "min_child_weight": [1, 3],
            "gamma": [0.0, 1.0],
        },
        "explainability": {"enabled": False, "sample_size": 200, "output_dir": str(tmp_path / "shap")},
        "artifacts": {"model_dir": str(tmp_path / "models"), "report_dir": str(tmp_path / "reports")},
    }
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base_cfg), encoding="utf-8")
    adv_path = tmp_path / "adv.yaml"
    adv_path.write_text(yaml.safe_dump(adv_cfg), encoding="utf-8")

    summary = run_advanced_training(adv_path)
    assert Path(summary["metrics_path"]).exists()
    payload = json.loads(Path(summary["metrics_path"]).read_text(encoding="utf-8"))
    assert "test_metrics" in payload
