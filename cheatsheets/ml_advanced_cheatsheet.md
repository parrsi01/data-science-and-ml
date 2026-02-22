# Advanced ML Cheatsheet (Institutional Data/AI Lab)

## SMOTE Dos / Don'ts

- Do apply SMOTE only on the training split
- Do compare SMOTE vs class-weight baselines
- Do inspect whether synthetic samples make domain sense
- Don't apply SMOTE before train/test split (data leakage)
- Don't assume SMOTE always improves rare-event recall

## Tuning Workflow

1. Define search space
2. Pick optimization metric (for rare events often `f1` or `roc_auc`)
3. Use stratified CV
4. Track best params + trial history
5. Retrain final model with best params
6. Save metrics and artifacts

## SHAP Interpretation

- SHAP value magnitude = impact strength
- Sign (+/-) = pushes prediction higher/lower
- Summary plot = distribution of impacts across samples
- Bar plot = average absolute importance (global view)
- Importance is not causality

## Common Pitfalls

- Tuning on test set instead of validation/CV
- Ignoring class imbalance and reporting accuracy only
- Applying SHAP on mismatched feature names after one-hot encoding
- Forgetting to save tuning/config artifacts for reproducibility
