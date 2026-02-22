# Machine Learning Core Pipeline (Institutional Standard)

- Author: Simon Parris
- Date: 2026-02-22

## What a pipeline is (simple)

A pipeline is a repeatable sequence of steps that turns raw data into model outputs and evaluation results. It makes training easier to reproduce, review, and improve.

## Why configs matter

Configs keep model settings, dataset sizes, and artifact paths outside the code. This makes experiments easier to compare and rebuild without editing source files.

## Why stratified splits matter

Stratified train/test splits preserve the class ratio (for example, rare events) in both training and test sets. This reduces misleading evaluations on imbalanced datasets.

## Why class imbalance matters

If only a small percentage of records are positive, a model can look accurate while still missing most important events. Institutional systems need metrics like recall, precision, and ROC-AUC, not just accuracy.

## How to run training + interpret outputs

1. Run `make ml-train`
2. Inspect `reports/ml_core/metrics.json` and `reports/ml_core/cv_results.json`
3. Review plots:
   - `confusion_matrix_<model>.png`
   - `roc_<model>.png`
4. Load saved models from `models/ml_core/*.joblib` for downstream testing/inference

## How to rebuild without AI

1. Create `src/ml_core`, `tests/ml_core`, `docs/ml_core`, `reports/ml_core`, `configs/ml_core`
2. Add a YAML config defining data, features, models, and artifact paths
3. Build synthetic data loader/generator with class imbalance
4. Implement deterministic feature engineering and sklearn preprocessing
5. Add model factory for Logistic Regression, Random Forest, and XGBoost
6. Build training orchestrator with stratified split, CV, evaluation, and artifact saving
7. Add plots for confusion matrix and ROC curves
8. Add tests for data generation, preprocessing, model factory, and small training run
9. Run `venv/bin/python -m src.ml_core.train --config configs/ml_core/config.yaml`

## JIRA-Style Ticket Examples

### MLCORE-101: Build Config-Driven Baseline Training Pipeline

- Type: Story
- Goal: Deliver a reproducible binary-classification training workflow with preprocessing, CV, and saved artifacts.
- Acceptance Criteria:
  - YAML config controls dataset size, models, and artifact paths
  - Training saves `metrics.json` and `cv_results.json`
  - Models saved to `models/ml_core/*.joblib`
  - Test suite includes a small end-to-end training test

### MLCORE-102: Add Institutional Evaluation Artifacts and Plot Outputs

- Type: Task
- Goal: Produce audit-friendly evaluation plots and top-line metrics for baseline + XGBoost models.
- Acceptance Criteria:
  - Confusion matrix and ROC PNGs saved per model
  - Metrics include precision/recall/F1/ROC-AUC
  - `make ml-report` prints concise top-line model metrics
