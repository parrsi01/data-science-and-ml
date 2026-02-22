# Evaluation Cheatsheet (Institutional Data/AI Lab)

## Metric Selection Guide

- Imbalanced classification baseline: `precision`, `recall`, `f1`, `roc_auc`
- Operational triage use cases: emphasize `recall` + false positive review
- Resource-limited interventions: emphasize `precision`
- Always keep a confusion matrix for error-type visibility

## Thresholding Guide

- Tune threshold on validation data only
- Save tradeoff tables for policy discussions
- Compare thresholds against operational cost of false positives/negatives
- Revisit thresholds when drift is detected

## Drift Detection Basics

- Numeric drift: compare mean/std shifts + KS statistic
- Categorical drift: compare distributions (TV distance / chi-square)
- Large drift does not always mean failure, but it should trigger review
- Track drift snapshots per model release

## Group Metrics Pitfalls

- Small groups can produce unstable metrics
- Equal overall accuracy can hide unequal recalls
- Group labels may be proxies for sensitive or operational factors
- Review domain context before concluding fairness issues
