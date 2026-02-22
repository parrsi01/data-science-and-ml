"""Model factory for the institutional ML core pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_model(
    name: str,
    config: dict[str, Any],
    class_weight_info: dict[str, float],
):
    """Return a configured estimator by name."""

    seed = int(config["dataset"]["seed"])

    if name == "logreg":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=1,
        )

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("xgboost is required for model 'xgboost'") from exc

        xgb_cfg = config.get("xgboost", {})
        return XGBClassifier(
            n_estimators=int(xgb_cfg.get("n_estimators", 300)),
            max_depth=int(xgb_cfg.get("max_depth", 6)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
            subsample=float(xgb_cfg.get("subsample", 0.9)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.9)),
            reg_lambda=float(xgb_cfg.get("reg_lambda", 1.0)),
            eval_metric=str(xgb_cfg.get("eval_metric", "logloss")),
            objective="binary:logistic",
            random_state=seed,
            n_jobs=1,
            scale_pos_weight=float(class_weight_info.get("scale_pos_weight", 1.0)),
        )

    raise ValueError(f"Unsupported model name: {name}")
