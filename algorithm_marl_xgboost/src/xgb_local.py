"""Local XGBoost training wrapper with deterministic feature processing and metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


def _apply_feature_weights(
    X: pd.DataFrame,
    feature_weights: dict[str, float] | None,
) -> pd.DataFrame:
    out = X.copy()
    if not feature_weights:
        return out
    for col, weight in feature_weights.items():
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(float) * float(weight)
    return out


def _onehot_align(X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([X_train.assign(_split=0), X_val.assign(_split=1)], ignore_index=True)
    combined_encoded = pd.get_dummies(combined.drop(columns=["_split"]), drop_first=False)
    train_rows = combined["_split"].to_numpy() == 0
    X_train_enc = combined_encoded.loc[train_rows].reset_index(drop=True)
    X_val_enc = combined_encoded.loc[~train_rows].reset_index(drop=True)
    return X_train_enc, X_val_enc


def _aggregate_feature_importance_by_raw(
    encoded_columns: list[str],
    importances: np.ndarray,
    raw_feature_order: list[str],
) -> list[float]:
    scores = {feature: 0.0 for feature in raw_feature_order}
    for col, value in zip(encoded_columns, importances.tolist()):
        matched = None
        for raw in raw_feature_order:
            if col == raw or col.startswith(f"{raw}_"):
                matched = raw
                break
        if matched is None:
            # get_dummies often encodes category as "proto_tcp"
            prefix = col.split("_", 1)[0]
            if prefix in scores:
                matched = prefix
        if matched is not None:
            scores[matched] += float(value)
    return [float(scores[f]) for f in raw_feature_order]


def train_local_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict[str, Any],
    *,
    seed: int,
    raw_feature_order: list[str],
    feature_weights: dict[str, float] | None = None,
) -> tuple[XGBClassifier, dict[str, Any]]:
    """Train a local XGBoost classifier and return model + metrics."""

    xgb_cfg = config["xgboost"]
    X_train_w = _apply_feature_weights(X_train, feature_weights)
    X_val_w = _apply_feature_weights(X_val, feature_weights)
    X_train_enc, X_val_enc = _onehot_align(X_train_w, X_val_w)

    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        subsample=float(xgb_cfg["subsample"]),
        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
        reg_lambda=float(xgb_cfg["reg_lambda"]),
        eval_metric=str(xgb_cfg.get("eval_metric", "logloss")),
        objective="binary:logistic",
        random_state=int(seed),
        n_jobs=1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=int(xgb_cfg.get("early_stopping_rounds", 30)),
    )
    model.fit(
        X_train_enc,
        y_train,
        eval_set=[(X_val_enc, y_val)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_val_enc)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    if len(np.unique(np.asarray(y_val))) < 2:
        roc_auc = float("nan")
    else:
        try:
            roc_auc = float(roc_auc_score(y_val, y_proba))
        except Exception:
            roc_auc = float("nan")
    importances = np.asarray(model.feature_importances_, dtype=float)
    raw_importance_vector = _aggregate_feature_importance_by_raw(
        encoded_columns=list(X_train_enc.columns),
        importances=importances,
        raw_feature_order=raw_feature_order,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "positive_rate_val": float(y_val.mean()) if len(y_val) else 0.0,
        "scale_pos_weight": float(scale_pos_weight),
        "feature_importance_vector": raw_importance_vector,
        "feature_importance_feature_order": list(raw_feature_order),
        "best_iteration": int(getattr(model, "best_iteration", -1)) if getattr(model, "best_iteration", None) is not None else -1,
        "rows_train": int(len(X_train)),
        "rows_val": int(len(X_val)),
    }
    return model, metrics
