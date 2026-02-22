# Institutional Evaluation (Stability, Bias, Drift)

- Author: Simon Parris
- Date: 2026-02-22

## Why accuracy is misleading under imbalance

When positive cases are rare, a model can achieve high accuracy by predicting mostly negatives. Institutional systems should also review precision, recall, F1, and ROC-AUC to understand rare-event performance.

## Why stability across seeds matters

If model performance changes too much across random seeds, results may not be reliable. Stability checks help detect fragile pipelines and over-tuned settings before operational use.

## What “threshold calibration” means (simple)

Threshold calibration means choosing the decision cutoff (for example, 0.30 instead of 0.50) that best matches institutional priorities such as recall vs false positives. This should be done on validation data, not the test set.

## What drift means and why it breaks models

Drift is a change in data patterns between training and future data. If distributions shift enough, model assumptions degrade and performance may fall even if code does not change.

## What group metrics mean in institutional settings

Group metrics compare performance across categories (for example, operational domains or regions). They help identify uneven error patterns that could create fairness, safety, or governance risks.

## How to rebuild without AI

1. Create `src/evaluation`, `tests/evaluation`, `docs/evaluation`, `reports/evaluation`
2. Implement stratified and repeated CV helpers
3. Add seed sweep stability runner and report writers
4. Add threshold search and tradeoff report generation
5. Add group metric calculations and Markdown/CSV exports
6. Add numeric/categorical drift reports (KS + TV distance / chi-square)
7. Build evaluation suite runner that prints a concise summary and writes artifacts
8. Add tests for thresholds, drift, group metrics, stability, and CV helpers
9. Run `venv/bin/python -m src.evaluation.run_evaluation_suite --config configs/ml_advanced/config.yaml`

## JIRA-Style Ticket Examples

### EVAL-101: Add Institutional Threshold Calibration and Group Metrics Reports

- Type: Story
- Goal: Provide validation-only threshold calibration and group-level performance reporting for imbalanced classifier decisions.
- Acceptance Criteria:
  - Threshold analysis JSON + tradeoff CSV saved
  - Group metrics CSV + Markdown saved
  - Evaluation summary prints best threshold and worst group recall

### EVAL-102: Add Stability and Drift Monitoring Snapshot Suite

- Type: Task
- Goal: Quantify seed sensitivity and train-vs-test drift for institutional review before deployment.
- Acceptance Criteria:
  - Seed sweep JSON + Markdown saved with F1 stability score
  - Drift JSON + Markdown saved with numeric and categorical drift measures
  - Evaluation runner prints largest drift feature
