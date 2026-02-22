# Advanced ML (Imbalance, Tuning, Explainability)

- Author: Simon Parris
- Date: 2026-02-22

## What SMOTE is (simple + risks)

SMOTE creates synthetic minority-class examples by interpolating between nearby minority samples. It can help models learn rare classes, but it may also create unrealistic samples if features are noisy or if time-order leakage is ignored.

## What hyperparameter tuning is (simple)

Hyperparameter tuning means systematically trying different model settings (like tree depth or learning rate) to improve performance on validation data.

## Why Optuna helps

Optuna automates the search for good hyperparameters using efficient trial selection instead of manual guessing. It also records trials and best settings for reproducibility.

## What SHAP means (simple)

SHAP estimates how much each feature contributes to a model prediction. It helps explain which variables push risk scores up or down.

## Institutional explainability requirements

- Explanations must be reproducible and saved as artifacts
- Feature importance should be interpreted with domain context, not used as proof of causality
- Rare-event models require careful review of false positives/false negatives alongside explainability
- Audit-ready reports should include metrics, plots, and model/tuning settings

## How to rebuild without AI

1. Install dependencies: `imbalanced-learn`, `optuna`, `shap`, `pyyaml`, `joblib`
2. Create `src/ml_advanced`, `tests/ml_advanced`, `docs/ml_advanced`, `reports/ml_advanced`, `configs/ml_advanced`
3. Add advanced features (lags, rolling std, interactions)
4. Implement imbalance strategy chooser + SMOTE utility (training split only)
5. Add Optuna tuning for XGBoost with stratified CV
6. Add SHAP explainability artifact generation
7. Build advanced training orchestrator and save model/metrics/plots
8. Add tests for features, imbalance, tuning, and SHAP outputs
9. Run `venv/bin/python -m src.ml_advanced.train_advanced --config configs/ml_advanced/config.yaml`

## JIRA-Style Ticket Examples

### MLADV-101: Add Imbalance Strategy Controls and SMOTE Safeguards

- Type: Story
- Goal: Implement reproducible imbalance decisions (weights vs SMOTE) with explicit anti-leakage guardrails.
- Acceptance Criteria:
  - Strategy chooser logs decision and reason
  - SMOTE applied only after train/test split
  - Returned decision includes `scale_pos_weight`

### MLADV-102: Tune XGBoost and Publish SHAP Explainability Artifacts

- Type: Task
- Goal: Tune XGBoost with Optuna and produce institutional explainability artifacts for review.
- Acceptance Criteria:
  - Optuna best params and study summary JSON artifacts are written
  - Advanced metrics and plots are saved
  - SHAP summary/bar PNGs and values archive are saved
