# Machine Learning Advanced

---

> **Field** — Machine Learning and Feature Engineering
> **Scope** — Advanced techniques for handling
> imbalanced data, tuning models, engineering
> features, and interpreting predictions

---

## Overview

Once you understand the fundamentals of machine
learning, the next level involves techniques that
separate good models from great ones. This
reference covers advanced strategies for handling
imbalanced datasets, systematic hyperparameter
tuning, feature engineering patterns, and model
interpretability methods that are essential for
production-quality machine learning.

---

## Definitions

---

### `SMOTE`

**Definition.**
SMOTE (Synthetic Minority Over-sampling
Technique) is a method that creates new,
synthetic examples of the minority class
by interpolating between existing minority
samples. Instead of simply duplicating
minority records, it generates new data
points that lie between existing ones.

**Context.**
SMOTE is the most popular technique for
dealing with class imbalance when you want
to balance the training data without simply
copying rows. It helps the model see more
diverse examples of the minority class.
Important: SMOTE should only be applied to
the training set, never to the test set.
Applying SMOTE to the full dataset before
splitting causes data leakage.

**Example.**
```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Split FIRST, then apply SMOTE to train only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y,
    random_state=42
)

print(f"Before SMOTE: {y_train.value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train, y_train
)

print(f"After SMOTE: "
      f"{y_train_resampled.value_counts().to_dict()}")
# Classes are now balanced
```

How SMOTE works:

1. Pick a minority sample
2. Find its K nearest minority neighbors
3. Pick one neighbor at random
4. Create a new point on the line segment
   between the original and the neighbor
5. Repeat until desired balance is achieved

Variants:

- **SMOTE:** Standard version
- **BorderlineSMOTE:** Focuses on samples
  near the decision boundary
- **ADASYN:** Generates more samples in
  harder-to-learn regions

---

### `Hyperparameter Tuning`

**Definition.**
Hyperparameter tuning is the systematic
process of finding the best combination of
hyperparameters for a model. Instead of
guessing or manually testing values, you
use automated methods to search the
parameter space.

**Context.**
Hyperparameter tuning can significantly
improve model performance, but it must be
done carefully to avoid overfitting to the
validation set. The search should use
cross-validation, not a single train/test
split. Common approaches range from simple
(grid search) to sophisticated (Bayesian
optimization).

**Example.**
**Grid Search** (exhaustive):

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [1, 5, 10],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best F1: {grid.best_score_:.3f}")
```

**Random Search** (samples randomly):

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 20),
    'min_samples_leaf': randint(1, 20),
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
```

Comparison of methods:

- **Grid search:** Tests every combination.
  Thorough but slow.
- **Random search:** Samples randomly.
  Faster, often finds good results.
- **Bayesian optimization (Optuna):** Learns
  from previous trials. Most efficient.

---

### `Optuna`

**Definition.**
Optuna is a Python library for automatic
hyperparameter optimization using Bayesian
methods. It learns from previous trials to
intelligently suggest the next set of
hyperparameters to try, focusing on the
most promising regions of the search space.

**Context.**
Optuna is more efficient than grid search
or random search because it does not waste
time on parameter combinations that are
unlikely to be good. It supports pruning
(stopping unpromising trials early), handles
complex search spaces (conditional
parameters), and provides visualization
tools. It has become the standard tool for
hyperparameter tuning in the Python data
science ecosystem.

**Example.**
```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int(
            'n_estimators', 50, 500
        ),
        'max_depth': trial.suggest_int(
            'max_depth', 3, 20
        ),
        'min_samples_leaf': trial.suggest_int(
            'min_samples_leaf', 1, 20
        ),
    }

    model = RandomForestClassifier(
        **params, random_state=42
    )

    scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='f1'
    )
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best F1: {study.best_value:.3f}")
print(f"Best params: {study.best_params}")
```

Key Optuna features:

- `suggest_int()`: Integer parameters
- `suggest_float()`: Float parameters
- `suggest_categorical()`: Choice parameters
- Automatic pruning of bad trials
- Built-in visualization
- Parallel execution support

---

### `SHAP (SHapley Additive exPlanations)`

**Definition.**
SHAP is a method for explaining individual
predictions by calculating the contribution
of each feature to the prediction. It is
based on Shapley values from game theory,
which fairly distribute a "payout" (the
prediction) among the "players" (the
features).

**Context.**
SHAP is the gold standard for model
interpretability. Unlike simple feature
importance (which tells you which features
matter globally), SHAP explains why a
specific prediction was made for a specific
data point. This is critical for regulated
industries (finance, healthcare) where you
must explain individual decisions, and for
debugging models to understand unexpected
predictions.

**Example.**
```python
import shap

# Train a model
model.fit(X_train, y_train)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot: global feature importance
# with direction of effect
shap.summary_plot(shap_values, X_test)

# Force plot: explain a single prediction
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0]
)

# Waterfall plot: another single-prediction view
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0]
    )
)
```

How to read SHAP values:

- **Positive SHAP value:** Feature pushes
  prediction higher (toward positive class)
- **Negative SHAP value:** Feature pushes
  prediction lower (toward negative class)
- **Large absolute value:** Feature has a
  big impact on this prediction
- **Small absolute value:** Feature has
  little impact on this prediction

---

### `Feature Engineering`

**Definition.**
Feature engineering is the process of
creating new input variables from raw data
to improve model performance. It transforms,
combines, or extracts information from
existing features to make patterns more
visible to the model.

**Context.**
Feature engineering is often the most
impactful part of a machine learning
project. A good feature can improve model
performance far more than tuning
hyperparameters or switching algorithms.
It requires domain knowledge and creativity.
The best features capture meaningful
relationships that the raw data does not
directly express.

**Example.**
Common feature engineering techniques:

```python
import pandas as pd
import numpy as np

df = pd.read_csv('transactions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract date parts
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# Mathematical combinations
df['price_per_unit'] = df['total'] / df['quantity']
df['log_amount'] = np.log1p(df['amount'])

# Aggregation features
customer_stats = df.groupby('customer_id').agg(
    avg_amount=('amount', 'mean'),
    total_transactions=('amount', 'count'),
    days_since_first=('timestamp',
        lambda x: (x.max() - x.min()).days)
)
df = df.merge(customer_stats, on='customer_id')

# Binning
df['amount_bin'] = pd.cut(
    df['amount'],
    bins=[0, 10, 50, 100, float('inf')],
    labels=['low', 'medium', 'high', 'very_high']
)
```

Categories of features:

- **Temporal:** Hour, day, month, is_weekend
- **Mathematical:** Ratios, logs, differences
- **Aggregation:** Per-group statistics
- **Interaction:** Feature A times Feature B
- **Domain-specific:** Business logic rules

---

### `Stratified KFold`

**Definition.**
Stratified KFold is a variation of K-fold
cross validation that ensures each fold has
approximately the same class distribution as
the full dataset. It combines stratification
with cross validation.

**Context.**
Standard KFold can produce folds where the
minority class is under-represented or even
absent, giving unreliable performance
estimates. Stratified KFold prevents this
and should always be used for classification
tasks, especially with imbalanced classes.
It is the default in scikit-learn's
`cross_val_score` for classifiers.

**Example.**
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = []
for fold, (train_idx, val_idx) in enumerate(
    skf.split(X, y)
):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    scores.append(f1)

    print(f"Fold {fold}: F1={f1:.3f}, "
          f"positive rate="
          f"{y_val.mean():.3f}")

print(f"\nMean F1: {np.mean(scores):.3f}")
print(f"Std F1:  {np.std(scores):.3f}")
```

Each fold preserves the original class ratio,
so a dataset with 5% positive rate will have
approximately 5% in every fold.

---

### `Imbalance Strategy`

**Definition.**
An imbalance strategy is a deliberate
approach for handling class imbalance in
a machine learning problem. It combines
one or more techniques at the data level,
algorithm level, or evaluation level to
ensure the model learns effectively from
skewed class distributions.

**Context.**
There is no single best strategy for
handling imbalance. The right approach
depends on how severe the imbalance is,
how much data you have, and what the
consequences of different types of errors
are. Often the best results come from
combining multiple techniques (e.g.,
SMOTE with class weights and threshold
tuning).

**Example.**
A typical imbalance strategy might combine:

1. **Data level:** SMOTE on the training set
2. **Algorithm level:** Class weights
3. **Evaluation level:** F1 and ROC-AUC
   instead of accuracy
4. **Decision level:** Tune the classification
   threshold

```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np

# Data level: SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(
    X_train, y_train
)

# Algorithm level: class weights
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
model.fit(X_resampled, y_resampled)

# Decision level: threshold tuning
y_prob = model.predict_proba(X_test)[:, 1]
best_f1 = 0
best_threshold = 0.5
for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f}")
print(f"Best F1: {best_f1:.3f}")
```

---

### `Feature Selection`

**Definition.**
Feature selection is the process of
identifying and keeping only the most
relevant features for your model, removing
those that are redundant, irrelevant, or
noisy. It reduces dimensionality without
creating new features.

**Context.**
Too many features can cause overfitting,
increase training time, and make models
harder to interpret. Feature selection helps
by removing features that add noise without
adding signal. It is different from feature
engineering (which creates new features).
The two processes are complementary: first
engineer features, then select the best ones.

**Example.**
Three main approaches:

**Filter methods** (statistical tests):

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif
)

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)

# Which features were selected?
selected = X_train.columns[selector.get_support()]
print(f"Selected features: {list(selected)}")
```

**Wrapper methods** (train and evaluate):

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

selector = RFECV(
    RandomForestClassifier(random_state=42),
    cv=5,
    scoring='f1'
)
selector.fit(X_train, y_train)
print(f"Optimal features: {selector.n_features_}")
```

**Embedded methods** (built into model):

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Drop features with low importance
threshold = np.percentile(
    model.feature_importances_, 25
)
important = X_train.columns[
    model.feature_importances_ > threshold
]
print(f"Important features: {list(important)}")
```

---

### `One-hot Encoding`

**Definition.**
One-hot encoding converts a categorical
variable into multiple binary (0/1) columns,
one for each category. Each row has a 1 in
the column corresponding to its category and
0 in all other columns.

**Context.**
Most machine learning algorithms require
numerical input. One-hot encoding is the
standard way to convert categorical variables
(like "color" or "country") into a format
that models can use. It is preferred over
label encoding for nominal categories (those
without a natural order) because it does not
impose an artificial ranking.

**Example.**
Original data:

```
color
-----
red
blue
green
red
```

After one-hot encoding:

```
color_red  color_blue  color_green
---------  ----------  -----------
1          0           0
0          1           0
0          0           1
1          0           0
```

```python
import pandas as pd

df = pd.DataFrame({'color': ['red', 'blue',
                              'green', 'red']})

# Pandas get_dummies
encoded = pd.get_dummies(df, columns=['color'])
print(encoded)

# Scikit-learn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)
result = encoder.fit_transform(df[['color']])
```

When to use one-hot encoding:

- Nominal categories (no order): color,
  country, product type
- Categories with fewer than ~20 values
- Most tree-based and linear models

When NOT to use:

- High-cardinality features (1000+ categories)
  use target encoding or embeddings instead
- Ordinal categories (use label encoding)

---

### `Label Encoding`

**Definition.**
Label encoding converts each category into
a single integer. For example, "low" becomes
0, "medium" becomes 1, and "high" becomes 2.

**Context.**
Label encoding is appropriate for ordinal
variables where there is a natural order
(like "low < medium < high"). For nominal
variables (no natural order), label encoding
is misleading because it implies that
"green" (2) is somehow greater than
"red" (0). Tree-based models can handle
label encoding for any category type, but
linear models cannot.

**Example.**
```python
from sklearn.preprocessing import LabelEncoder

# Ordinal: has natural order
sizes = ['small', 'medium', 'large', 'medium']
le = LabelEncoder()
encoded = le.fit_transform(sizes)
print(encoded)  # [2, 1, 0, 1]

# Manual ordinal encoding (explicit order)
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(
    categories=[['small', 'medium', 'large']]
)
result = oe.fit_transform(
    [['small'], ['medium'], ['large'], ['medium']]
)
print(result)  # [[0.], [1.], [2.], [1.]]
```

Label encoding vs one-hot encoding:

- **Label encoding:**
  One column, integer values.
  Good for: ordinal data, tree models.
  Bad for: nominal data with linear models.

- **One-hot encoding:**
  Multiple binary columns.
  Good for: nominal data, any model.
  Bad for: high-cardinality features.

---

### `Rolling Statistics`

**Definition.**
Rolling statistics (also called moving
statistics or window statistics) calculate
a statistic (mean, sum, std, etc.) over a
sliding window of consecutive data points.
Each value is computed from the N most
recent observations.

**Context.**
Rolling statistics are fundamental for
time-series feature engineering. A 7-day
rolling average smooths out daily noise to
reveal weekly trends. A 30-day rolling
standard deviation shows how volatile a
series has been recently. These features
give models information about recent
behavior patterns that a single point-in-
time value cannot provide.

**Example.**
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'date': pd.date_range('2024-01-01',
                           periods=30),
    'sales': np.random.randint(50, 200, 30)
})

# 7-day rolling mean
df['rolling_mean_7'] = (
    df['sales'].rolling(window=7).mean()
)

# 7-day rolling standard deviation
df['rolling_std_7'] = (
    df['sales'].rolling(window=7).std()
)

# 7-day rolling min and max
df['rolling_min_7'] = (
    df['sales'].rolling(window=7).min()
)
df['rolling_max_7'] = (
    df['sales'].rolling(window=7).max()
)

print(df.head(10))
```

Important notes:

- The first (window - 1) values will be NaN
  because there are not enough prior points
- Always use `.shift(1)` when creating
  features for prediction to avoid data
  leakage (do not include the current value
  in the window when predicting it)

```python
# CORRECT: shift to avoid leakage
df['rolling_mean_7'] = (
    df['sales'].shift(1).rolling(7).mean()
)
```

---

### `Lag Features`

**Definition.**
A lag feature is a value from a previous
time step used as an input feature. For
example, "yesterday's sales" is a lag-1
feature, and "sales from 7 days ago" is
a lag-7 feature.

**Context.**
Lag features capture temporal patterns and
are among the most powerful features in
time-series modeling. If today's sales are
often similar to yesterday's, then
yesterday's value is a strong predictor.
Different lag periods capture different
patterns: lag-1 for day-to-day momentum,
lag-7 for weekly patterns, lag-365 for
annual seasonality.

**Example.**
```python
import pandas as pd

df = pd.DataFrame({
    'date': pd.date_range('2024-01-01',
                           periods=30),
    'sales': [100, 110, 105, 115, 120,
              108, 112, 125, 130, 128,
              135, 140, 138, 142, 145,
              132, 136, 148, 150, 147,
              155, 160, 158, 162, 165,
              152, 156, 168, 170, 167]
})

# Lag features
df['sales_lag_1'] = df['sales'].shift(1)
df['sales_lag_7'] = df['sales'].shift(7)

# Change from previous day
df['sales_change'] = df['sales'].diff(1)

# Percentage change from previous day
df['sales_pct_change'] = (
    df['sales'].pct_change(1)
)

print(df.head(10))
```

Important: lag features introduce NaN values
at the beginning of the dataset (lag-7 means
the first 7 rows have NaN). Handle these by
either dropping those rows or filling them
appropriately.

---

### `Interaction Features`

**Definition.**
An interaction feature is created by
combining two or more existing features
(typically by multiplication or division)
to capture relationships that neither
feature captures alone.

**Context.**
Some patterns only emerge when you consider
features together. For example, "income"
and "number of dependents" individually
might not predict loan risk well, but
"income per dependent" might be a strong
predictor. Linear models especially benefit
from interaction features because they
cannot learn these relationships on their
own. Tree-based models can discover
interactions automatically but still often
benefit from explicit interaction features.

**Example.**
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'income': [50000, 80000, 35000, 120000],
    'dependents': [2, 1, 3, 0],
    'age': [30, 45, 28, 55],
    'years_employed': [5, 20, 3, 30]
})

# Ratio interaction
df['income_per_dependent'] = (
    df['income'] / (df['dependents'] + 1)
)

# Product interaction
df['age_x_income'] = df['age'] * df['income']

# Ratio of two features
df['employment_ratio'] = (
    df['years_employed'] / df['age']
)

# Polynomial interaction
from sklearn.preprocessing import (
    PolynomialFeatures
)
poly = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)
interactions = poly.fit_transform(
    df[['income', 'age']]
)
```

Tips for interaction features:

- Use domain knowledge to guide which
  features to combine
- Ratios are often more meaningful than
  raw products
- Add 1 to denominators to avoid division
  by zero
- Tree-based models find interactions
  automatically but explicit ones help

---

## See Also

- [Machine Learning Fundamentals](./05_machine_learning_fundamentals.md)
- [Model Evaluation and Monitoring](./07_model_evaluation_and_monitoring.md)
- [Statistical Foundations](./01_statistical_foundations.md)
- [Data Quality and Validation](./04_data_quality_and_validation.md)

---

> **Author** — Simon Parris | Data Science Reference Library
