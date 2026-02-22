from __future__ import annotations

from pathlib import Path
import json

import yaml

from ml_core.data import make_synthetic_classification
from ml_core.models import get_model
from ml_core.preprocess import add_feature_engineering, build_preprocessor
from ml_core.train import run_training


def test_data_generation_shape_and_imbalance_approx() -> None:
    df = make_synthetic_classification(n_rows=2000, seed=123)
    assert df.shape[0] == 2000
    positive_rate = float(df["label_rare_event"].mean())
    assert 0.015 <= positive_rate <= 0.03


def test_preprocess_transformer_builds() -> None:
    df = make_synthetic_classification(n_rows=500, seed=124)
    df = add_feature_engineering(df)
    assert {"ratio_ab", "rolling_mean_a"} <= set(df.columns)
    transformer = build_preprocessor(["category"], ["metric_a", "metric_b", "ratio_ab", "rolling_mean_a"])
    assert transformer is not None


def test_model_factory_returns_correct_estimator() -> None:
    config = {
        "dataset": {"seed": 42},
        "xgboost": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
        },
    }
    class_info = {"scale_pos_weight": 10.0}
    assert get_model("logreg", config, class_info).__class__.__name__ == "LogisticRegression"
    assert get_model("random_forest", config, class_info).__class__.__name__ == "RandomForestClassifier"
    assert get_model("xgboost", config, class_info).__class__.__name__ == "XGBClassifier"


def test_training_runs_small_and_writes_metrics(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    config = {
        "dataset": {"source": "synthetic", "n_rows": 2000, "seed": 42, "test_size": 0.2},
        "target": {"name": "label_rare_event", "positive_label": 1},
        "features": {
            "categorical": ["category"],
            "numerical": ["metric_a", "metric_b", "ratio_ab", "rolling_mean_a"],
        },
        "models": [{"name": "logreg"}],
        "xgboost": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
        },
        "evaluation": {"metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"], "cv_folds": 3},
        "artifacts": {"model_dir": str(model_dir), "report_dir": str(report_dir)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = run_training(config_path)
    metrics_path = Path(summary["metrics_path"])
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "logreg" in metrics
    assert "test_metrics" in metrics["logreg"]
