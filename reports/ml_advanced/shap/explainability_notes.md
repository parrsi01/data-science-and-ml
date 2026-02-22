# Explainability Notes (SHAP)

- SHAP values estimate how each feature pushes predictions toward or away from the rare-event class.
- `shap_summary.png` shows per-sample effect spread for each feature.
- `shap_bar.png` shows average absolute impact (global importance).
- Interpret high-impact features with domain context; importance alone does not prove causality.
- Review one-hot encoded categorical features together when summarizing institutional decisions.