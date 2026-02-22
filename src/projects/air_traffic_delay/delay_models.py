"""Delay prediction models for aviation operations analytics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


def _build_preprocessor(categorical_cols: list[str], numerical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numerical", StandardScaler(), numerical_cols),
        ]
    )


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _save_classification_plots(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    output_dir: Path,
) -> dict[str, str]:
    cm_path = output_dir / "confusion_matrix.png"
    roc_path = output_dir / "roc_curve.png"

    cm = confusion_matrix(y_true, y_pred)
    fig1, ax1 = plt.subplots(figsize=(4.8, 4.0))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax1, colorbar=False)
    ax1.set_title("Delay Classification Confusion Matrix")
    fig1.tight_layout()
    fig1.savefig(cm_path, dpi=150)
    plt.close(fig1)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig2, ax2 = plt.subplots(figsize=(5.0, 4.0))
    ax2.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}", color="#C84C09")
    ax2.plot([0, 1], [0, 1], "--", linewidth=1, color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Delay Classification ROC Curve")
    ax2.legend(loc="lower right")
    fig2.tight_layout()
    fig2.savefig(roc_path, dpi=150)
    plt.close(fig2)

    return {"confusion_matrix_png": str(cm_path), "roc_curve_png": str(roc_path)}


def _feature_importance_payload(preprocessor: ColumnTransformer, model: Any, top_n: int = 15) -> list[dict[str, float | str]]:
    feature_names = preprocessor.get_feature_names_out().tolist()
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    pairs = list(zip(feature_names, np.asarray(importances, dtype=float).tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [{"feature": str(name), "importance": float(val)} for name, val in pairs[:top_n]]


def train_delay_model(
    flights_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    model_dir: str | Path = "models/air_traffic_delay",
) -> dict[str, Any]:
    """Train classification or regression delay model and save artifacts."""

    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    task = str(config["modeling"]["task"]).lower()
    if task not in {"classification", "regression"}:
        raise ValueError("modeling.task must be 'classification' or 'regression'")

    categorical_cols = ["dep", "arr"]
    numerical_cols = [
        "weather_index",
        "congestion_index",
        "distance_km",
        "hour_of_day",
        "day_of_week",
        "dep_in_degree",
        "dep_out_degree",
        "dep_betweenness_centrality",
        "dep_pagerank",
        "dep_clustering",
        "arr_in_degree",
        "arr_out_degree",
        "arr_betweenness_centrality",
        "arr_pagerank",
        "arr_clustering",
    ]
    use_cols = categorical_cols + numerical_cols
    df = flights_df.sort_values(["scheduled_time", "flight_id"], kind="mergesort").reset_index(drop=True)
    X = df[use_cols].copy()

    seed = int(config["simulation"]["seed"])
    test_size = float(config["modeling"]["test_size"])

    if task == "classification":
        y = df["delayed"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        pos = max(int(y_train.sum()), 1)
        neg = max(int((1 - y_train).sum()), 1)
        scale_pos_weight = neg / pos
        estimator = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
        )
    else:
        y = df["delay_minutes"].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        estimator = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )

    preprocessor = _build_preprocessor(categorical_cols, numerical_cols)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if task == "classification":
        y_pred = np.asarray(y_pred).astype(int)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = _classification_metrics(y_test, y_pred, y_proba)
        plot_artifacts = _save_classification_plots(y_test, y_pred, y_proba, output_dir)
    else:
        y_pred = np.asarray(y_pred, dtype=float)
        metrics = _regression_metrics(y_test, y_pred)
        plot_artifacts = {}

    fitted_pre = pipeline.named_steps["preprocessor"]
    fitted_model = pipeline.named_steps["model"]
    top_features = _feature_importance_payload(fitted_pre, fitted_model, top_n=15)

    model_path = model_dir / "xgb.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "task": task,
            "categorical_cols": categorical_cols,
            "numerical_cols": numerical_cols,
            "top_features": top_features,
        },
        model_path,
    )

    metrics_payload: dict[str, Any] = {
        "task": task,
        "model": str(config["modeling"]["model"]),
        "metrics": metrics,
        "rows": {
            "total": int(len(df)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
        },
        "top_feature_importance": top_features,
        "artifacts": {"model_joblib": str(model_path), **plot_artifacts},
    }
    metrics_path = output_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    metrics_payload["metrics_json"] = str(metrics_path)
    return metrics_payload

