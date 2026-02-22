# Statistics Cheatsheet (Institutional Data/AI Lab)

## Short Simplified Definitions

- Mean/Median: Measures of central tendency (average and middle value).
- Variance/Standard deviation: Measures of spread in data.
- Correlation: Degree to which two variables move together.
- Hypothesis test: Procedure to assess whether an observed effect is likely due to chance.

## Core Commands

```python
import pandas as pd
df.describe()
df.corr(numeric_only=True)
df["target"].value_counts(normalize=True)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
```

## Common Pitfalls

- Confusing correlation with causation
- Ignoring class imbalance in evaluation
- Using data leakage features during training
- Reporting only one metric for complex institutional decisions

## Institutional Best Practices

- Use validation plans defined before model fitting
- Report uncertainty, assumptions, and data limitations
- Track metric definitions consistently across teams
- Prefer interpretable baselines before complex models
