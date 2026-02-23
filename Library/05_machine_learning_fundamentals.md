# Machine Learning Fundamentals

---

> **Field** — Machine Learning and Predictive Modeling
> **Scope** — Core machine learning concepts, algorithms,
> evaluation strategies, and best practices for
> building reliable predictive models

---

## Overview

Machine learning is the practice of building
systems that learn patterns from data and use
those patterns to make predictions or decisions.
This reference covers the foundational concepts
you need before training your first model: how
to split data, evaluate performance, avoid common
pitfalls, and choose the right algorithm for
your problem.

---

## Definitions

---

### `Pipeline (ML)`

**Definition.**
An ML pipeline is a sequence of data processing
and modeling steps chained together so that the
output of one step feeds into the next. It
bundles preprocessing, feature engineering, and
model training into a single reproducible object.

**Context.**
Pipelines prevent a common and dangerous mistake
called data leakage. When you fit a scaler on
the full dataset and then split into train/test,
the test set has already influenced the scaler.
A pipeline ensures that each step is fit only
on training data and applied consistently to new
data. Pipelines also make deployment easier
because you ship one object, not a series of
manual steps.

**Example.**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Fit the entire pipeline on training data
pipe.fit(X_train, y_train)

# Predict on new data (scaling is automatic)
predictions = pipe.predict(X_test)

# The scaler was fit ONLY on X_train
# and applied to X_test without leakage
```

Benefits of pipelines:

- Prevent data leakage
- Reproducible preprocessing
- Easy to deploy as a single artifact
- Compatible with cross-validation

---

### `Train/Test Split`

**Definition.**
A train/test split divides your dataset into
two parts: a training set used to build the
model and a test set used to evaluate how well
the model performs on unseen data.

**Context.**
If you evaluate a model on the same data it
was trained on, it will appear to perform
much better than it actually does. The test
set simulates real-world data that the model
has never seen. A typical split is 80% train
and 20% test. The test set should be used
only once, at the very end, to get a final
performance estimate.

**Example.**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # preserve class proportions
)

print(f"Train: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples")
```

Common mistakes:

- Using the test set to tune hyperparameters
  (use a validation set or cross-validation)
- Not stratifying when classes are imbalanced
- Splitting time-series data randomly instead
  of chronologically

---

### `Stratification`

**Definition.**
Stratification ensures that each subset of
your data (train, test, validation) has the
same proportion of each class as the original
dataset. It preserves the class distribution
across splits.

**Context.**
Stratification is critical when you have
imbalanced classes. If your dataset is 95%
negative and 5% positive, a random split
could give you a test set with 0% positive
samples by bad luck. Stratification prevents
this by guaranteeing each split has
approximately 95/5 proportions. Always use
stratification for classification tasks.

**Example.**
```python
from sklearn.model_selection import train_test_split
import numpy as np

# Imbalanced dataset: 95% class 0, 5% class 1
y = np.array([0]*950 + [1]*50)

# Without stratification (risky)
_, _, _, y_test_bad = train_test_split(
    np.zeros_like(y), y,
    test_size=0.2, random_state=42
)
print(f"Without stratify: "
      f"{y_test_bad.mean():.3f} positive rate")

# With stratification (safe)
_, _, _, y_test_good = train_test_split(
    np.zeros_like(y), y,
    test_size=0.2, random_state=42,
    stratify=y
)
print(f"With stratify: "
      f"{y_test_good.mean():.3f} positive rate")
# Should be close to 0.05 (5%)
```

---

### `Cross Validation`

**Definition.**
Cross validation is a technique that splits
the data into K equal parts (folds), trains
the model K times (each time using K-1 folds
for training and 1 fold for testing), and
averages the results. This gives a more
reliable performance estimate than a single
train/test split.

**Context.**
A single train/test split can give misleading
results depending on which data points end up
in which set. Cross validation solves this by
testing on every part of the data. The most
common version is 5-fold or 10-fold cross
validation. It is the standard way to evaluate
models and compare algorithms during
development.

**Example.**
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

# 5-fold cross validation
scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring='f1'
)

print(f"F1 scores: {scores}")
print(f"Mean F1: {scores.mean():.3f}")
print(f"Std F1:  {scores.std():.3f}")
```

How 5-fold CV works:

```
Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]
```

Each data point appears in the test set
exactly once.

---

### `Overfitting`

**Definition.**
Overfitting occurs when a model learns the
training data too well, including its noise
and random quirks, and performs poorly on new,
unseen data. The model has memorized the
training set instead of learning general
patterns.

**Context.**
Overfitting is the most common problem in
machine learning. Signs include: high accuracy
on training data but much lower accuracy on
test data, and a model that is overly complex
for the amount of data available. Detecting
and preventing overfitting is a core skill.

**Example.**
Signs of overfitting:

```
Training accuracy:  98%
Test accuracy:      65%
Gap:                33%  (too large)
```

Remedies:

- **More data:** Harder to memorize
- **Simpler model:** Fewer parameters
- **Regularization:** Penalize complexity
  (L1, L2, dropout)
- **Cross validation:** Detect it early
- **Early stopping:** Stop training before
  the model overfits
- **Feature selection:** Remove noisy features

```python
from sklearn.ensemble import RandomForestClassifier

# Overfitting: too many features, deep trees
overfit_model = RandomForestClassifier(
    max_depth=None,  # Unlimited depth
    min_samples_leaf=1,  # Can memorize
    random_state=42
)

# Better: constrained model
good_model = RandomForestClassifier(
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
```

---

### `Underfitting`

**Definition.**
Underfitting occurs when a model is too
simple to capture the patterns in the data.
It performs poorly on both training and test
data because it has not learned enough.

**Context.**
Underfitting is the opposite of overfitting.
It happens when you use a model that lacks
the capacity to learn the underlying
relationships, or when you do not provide
enough features. While less common than
overfitting, underfitting results in a model
that is essentially useless.

**Example.**
Signs of underfitting:

```
Training accuracy:  60%
Test accuracy:      58%
Both are low — the model is too simple.
```

Remedies:

- **More complex model:** Use a model with
  more capacity (e.g., random forest instead
  of logistic regression)
- **More features:** Provide more information
- **Less regularization:** If you constrained
  the model too aggressively
- **Feature engineering:** Create better
  representations of the data
- **Train longer:** More epochs (for neural
  networks)

---

### `Class Imbalance`

**Definition.**
Class imbalance occurs when one class in
your dataset is far more common than another.
For example, 98% of transactions are
legitimate and 2% are fraudulent.

**Context.**
Class imbalance is extremely common in
real-world problems: fraud detection, disease
diagnosis, equipment failure prediction, and
rare event detection. Standard accuracy is
misleading with imbalanced data because a
model that predicts "not fraud" every time
achieves 98% accuracy while catching zero
fraud. You need specialized metrics (F1,
precision, recall, ROC-AUC) and strategies
(oversampling, undersampling, class weights).

**Example.**
```python
import numpy as np

y = np.array([0]*9800 + [1]*200)
print(f"Class 0: {(y==0).sum()} "
      f"({(y==0).mean():.1%})")
print(f"Class 1: {(y==1).sum()} "
      f"({(y==1).mean():.1%})")
# Class 0: 9800 (98.0%)
# Class 1: 200 (2.0%)
```

Strategies for handling imbalance:

- **Class weights:** Tell the model that
  minority class errors cost more
- **Oversampling (SMOTE):** Create synthetic
  minority samples
- **Undersampling:** Reduce majority class
- **Threshold tuning:** Adjust the decision
  boundary
- **Use proper metrics:** F1, precision,
  recall, ROC-AUC (not accuracy)

---

### `Confusion Matrix`

**Definition.**
A confusion matrix is a table that shows
the counts of correct and incorrect
predictions broken down by class. For binary
classification, it has four cells: true
positives, true negatives, false positives,
and false negatives.

**Context.**
The confusion matrix is the single most
informative evaluation tool for
classification. From it, you can calculate
accuracy, precision, recall, F1 score, and
many other metrics. It shows you exactly
where the model makes mistakes and helps
you decide whether those mistakes are
acceptable for your use case.

**Example.**
```
                  Predicted
                  Neg    Pos
Actual  Neg       TN     FP
        Pos       FN     TP
```

- **TP (True Positive):** Correctly predicted
  positive
- **TN (True Negative):** Correctly predicted
  negative
- **FP (False Positive):** Predicted positive,
  actually negative (Type I error)
- **FN (False Negative):** Predicted negative,
  actually positive (Type II error)

```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[TN, FP],
#  [FN, TP]]

# Visual display
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred
)
```

---

### `ROC-AUC`

**Definition.**
ROC-AUC (Receiver Operating Characteristic -
Area Under the Curve) measures how well a
classifier distinguishes between classes
across all possible decision thresholds.
A score of 1.0 is perfect; 0.5 is random
guessing.

**Context.**
ROC-AUC is a widely used metric for binary
classification, especially with imbalanced
data. Unlike accuracy, it is not affected
by class proportions and evaluates the
model's ranking ability (how well it
separates positive from negative cases)
rather than its predictions at a single
threshold. Use ROC-AUC when comparing
models or when the decision threshold has
not been chosen yet.

**Example.**
```python
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {auc:.3f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(
    y_test, y_prob
)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--',
         label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png', dpi=150)
```

Interpretation guide:

- 0.90 - 1.00: Excellent
- 0.80 - 0.90: Good
- 0.70 - 0.80: Fair
- 0.60 - 0.70: Poor
- 0.50 - 0.60: Nearly random

---

### `Feature Importance`

**Definition.**
Feature importance measures how much each
input variable contributes to the model's
predictions. Features with high importance
have a strong influence on the output;
features with low importance contribute
little.

**Context.**
Feature importance helps you understand
your model, identify the most valuable data
sources, and simplify models by removing
unimportant features. It is essential for
model explainability. Stakeholders want to
know not just what the model predicts but
why it makes those predictions.

**Example.**
```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("Top 5 features:")
print(importances.head())

# Plot
importances.head(10).plot(kind='barh')
```

Methods for feature importance:

- **Tree-based:** Built-in importance
  (Random Forest, XGBoost)
- **Permutation importance:** Shuffle each
  feature and measure performance drop
- **SHAP values:** Game-theory-based
  explanations (most rigorous)
- **Coefficient magnitude:** For linear
  models (after scaling)

---

### `Hyperparameter`

**Definition.**
A hyperparameter is a setting that you
choose before training a model. Unlike
model parameters (which are learned from
data), hyperparameters are set by the
practitioner and control how the model
learns.

**Context.**
Choosing good hyperparameters can be the
difference between a mediocre model and an
excellent one. Examples include the number
of trees in a random forest, the learning
rate in gradient boosting, and the
regularization strength in logistic
regression. Hyperparameter tuning is the
process of finding the best values.

**Example.**
Common hyperparameters:

- **Random Forest:**
  `n_estimators`, `max_depth`,
  `min_samples_leaf`
- **XGBoost:**
  `learning_rate`, `max_depth`,
  `n_estimators`, `subsample`
- **Logistic Regression:**
  `C` (regularization), `penalty`

```python
from sklearn.ensemble import RandomForestClassifier

# These are all hyperparameters
model = RandomForestClassifier(
    n_estimators=200,    # Number of trees
    max_depth=10,        # Tree depth limit
    min_samples_leaf=5,  # Min samples per leaf
    random_state=42
)

model.fit(X_train, y_train)
```

Hyperparameters vs parameters:

- **Hyperparameters:** Set before training.
  You choose them.
- **Parameters:** Learned during training.
  The model discovers them.
  (e.g., tree split points, regression
  coefficients)

---

### `Artifact`

**Definition.**
An artifact is any file produced during
a machine learning workflow: trained models,
preprocessor objects, evaluation reports,
plots, configuration files, or datasets.
Artifacts are the tangible outputs of your
work.

**Context.**
Tracking artifacts is essential for
reproducibility and deployment. When you
train a model, you need to save not just
the model file but also the scaler, the
feature list, the configuration, and the
evaluation metrics. Without proper artifact
management, you cannot reproduce results or
deploy models reliably.

**Example.**
Common ML artifacts:

- `model.pkl` — trained model
- `scaler.pkl` — fitted preprocessor
- `config.yaml` — training configuration
- `metrics.json` — evaluation results
- `feature_importance.png` — plot
- `confusion_matrix.png` — plot

```python
import joblib
import json

# Save model artifact
joblib.dump(model, 'artifacts/model.pkl')
joblib.dump(scaler, 'artifacts/scaler.pkl')

# Save metrics artifact
metrics = {
    'accuracy': 0.85,
    'f1': 0.72,
    'roc_auc': 0.91
}
with open('artifacts/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Load artifacts later
loaded_model = joblib.load('artifacts/model.pkl')
```

---

### `Logistic Regression`

**Definition.**
Logistic regression is a linear model for
binary classification. Despite the name
"regression," it predicts probabilities of
class membership (between 0 and 1) by
applying a sigmoid function to a linear
combination of features.

**Context.**
Logistic regression is often the first model
you should try for classification. It is
fast, interpretable, and works well as a
baseline. Its coefficients tell you the
direction and strength of each feature's
influence. Even if you end up using a more
complex model, logistic regression is
valuable as a comparison point and for
understanding your data.

**Example.**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print(classification_report(y_test, y_pred))

# Interpret coefficients
import pandas as pd
coefs = pd.Series(
    model.coef_[0],
    index=X_train.columns
).sort_values()
print(coefs)
```

When to use logistic regression:

- Binary classification problems
- When interpretability matters
- As a baseline before trying complex models
- When you have limited data
- When features have linear relationships
  with the log-odds of the outcome

---

### `Random Forest`

**Definition.**
A random forest is an ensemble model that
builds many decision trees on random subsets
of the data and features, then combines
their predictions by majority vote
(classification) or averaging (regression).

**Context.**
Random forests are among the most popular
and reliable machine learning algorithms.
They handle non-linear relationships,
are resistant to overfitting (compared to
individual trees), require minimal
preprocessing, and provide built-in feature
importance. They are an excellent default
choice for tabular data problems.

**Example.**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
```

How it works:

1. Create many bootstrap samples (random
   subsets with replacement)
2. Build a decision tree on each sample,
   using a random subset of features at
   each split
3. Combine all trees: majority vote for
   classification, average for regression

Key hyperparameters:

- `n_estimators`: Number of trees (more is
  usually better, but slower)
- `max_depth`: Maximum tree depth
- `min_samples_leaf`: Minimum samples per
  leaf node

---

### `Decision Tree`

**Definition.**
A decision tree is a model that makes
predictions by learning a series of
if/then/else rules from the data. It splits
the data at each internal node based on a
feature value, creating branches that lead
to predictions at the leaf nodes.

**Context.**
Decision trees are the building blocks of
random forests and gradient boosting. On
their own, they are highly interpretable
(you can draw and explain them) but prone
to overfitting. They are useful for
understanding data structure and as a
teaching tool for machine learning concepts.

**Example.**
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# Print the tree rules
tree_rules = export_text(
    model,
    feature_names=list(X_train.columns)
)
print(tree_rules)
```

Sample output:

```
|--- income <= 50000.00
|   |--- age <= 25.00
|   |   |--- class: denied
|   |--- age > 25.00
|   |   |--- class: approved
|--- income > 50000.00
|   |--- class: approved
```

The tree asks questions about features
and follows branches based on the answers
until it reaches a final prediction.

---

### `XGBoost`

**Definition.**
XGBoost (Extreme Gradient Boosting) is an
optimized gradient boosting library that
builds an ensemble of decision trees
sequentially, where each new tree corrects
the errors of the previous ones.

**Context.**
XGBoost is one of the most successful ML
algorithms for tabular data. It dominates
Kaggle competitions and is widely used in
industry for problems like fraud detection,
credit scoring, and recommendation systems.
It is faster than traditional gradient
boosting, handles missing values natively,
and includes built-in regularization.

**Example.**
```python
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

Key hyperparameters:

- `n_estimators`: Number of boosting rounds
- `learning_rate`: Step size (lower = more
  robust, needs more trees)
- `max_depth`: Tree depth (lower = less
  overfitting)
- `subsample`: Fraction of data per tree
- `colsample_bytree`: Fraction of features
  per tree

---

### `Gradient Boosting`

**Definition.**
Gradient boosting is an ensemble technique
that builds models sequentially. Each new
model is trained to correct the residual
errors of the combined previous models.
The "gradient" refers to using gradient
descent to minimize a loss function.

**Context.**
Gradient boosting is the general framework
behind XGBoost, LightGBM, and CatBoost.
It typically produces the best performance
on structured (tabular) data. The tradeoff
is that it is slower to train than random
forests (because trees are built
sequentially, not in parallel) and has
more hyperparameters to tune.

**Example.**
How gradient boosting works conceptually:

```
Step 1: Train tree on original target
Step 2: Calculate errors (residuals)
Step 3: Train new tree on the residuals
Step 4: Add new tree to the ensemble
        (with a small learning rate)
Step 5: Repeat steps 2-4
```

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Accuracy: {score:.3f}")
```

Gradient boosting vs random forest:

- **Gradient boosting:** Sequential trees,
  often higher accuracy, slower training,
  more hyperparameters
- **Random forest:** Parallel trees, more
  robust to overfitting, fewer
  hyperparameters, faster training

---

### `Data Leakage`

**Definition.**
Data leakage occurs when information from
outside the training set is used to build
the model. This makes the model appear to
perform much better during development than
it actually will in production.

**Context.**
Data leakage is one of the most dangerous
and common mistakes in machine learning.
It can be subtle and hard to detect. The
model gets "cheating" access to information
it would not have in the real world, leading
to overly optimistic metrics and a model
that fails when deployed.

**Example.**
Common sources of leakage:

- **Target leakage:** Using a feature that
  is derived from the target variable.
  Example: using "loan_status" as a feature
  when predicting "will_default."

- **Train-test contamination:** Fitting
  preprocessing (scaling, encoding) on the
  full dataset before splitting.

- **Temporal leakage:** Using future data
  to predict the past. Example: using
  tomorrow's stock price as a feature.

```python
# BAD: leakage from fitting scaler on all data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # All data
X_train, X_test = X_scaled[:800], X_scaled[800:]

# GOOD: fit scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Note: transform only, not fit_transform
```

---

### `Config-driven Training`

**Definition.**
Config-driven training is a practice where
all model training parameters (hyperparameters,
file paths, feature lists, thresholds) are
stored in external configuration files (YAML,
JSON, TOML) rather than hardcoded in scripts.

**Context.**
Config-driven training makes experiments
reproducible and manageable. Instead of
editing code to change hyperparameters, you
edit a configuration file and rerun the same
script. This also makes it easy to track what
changed between experiments, version control
your settings, and automate parameter sweeps.

**Example.**
Configuration file (`config.yaml`):

```yaml
data:
  train_path: data/train.csv
  test_path: data/test.csv
  target_column: is_fraud

features:
  - amount
  - hour_of_day
  - merchant_category
  - distance_from_home

model:
  type: xgboost
  params:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1
    random_state: 42

evaluation:
  metrics:
    - f1
    - roc_auc
    - precision
  threshold: 0.5
```

Training script reads the config:

```python
import yaml
from xgboost import XGBClassifier

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = XGBClassifier(**config['model']['params'])
```

---

### `Bias-Variance Tradeoff`

**Definition.**
The bias-variance tradeoff is the fundamental
tension in machine learning between a model
that is too simple (high bias, underfitting)
and a model that is too complex (high variance,
overfitting). The goal is to find the sweet
spot that minimizes total error.

**Context.**
Every model sits somewhere on the
bias-variance spectrum. Linear regression has
high bias (assumes a linear relationship) but
low variance (predictions are stable). A deep
decision tree has low bias (can fit any
pattern) but high variance (predictions change
dramatically with different training data).
Understanding this tradeoff guides model
selection and regularization decisions.

**Example.**
```
Total Error = Bias^2 + Variance + Noise

High bias:
  Model is too simple.
  Training error: HIGH
  Test error: HIGH
  Example: linear model on non-linear data

High variance:
  Model is too complex.
  Training error: LOW
  Test error: HIGH
  Example: deep tree on small dataset

Sweet spot:
  Moderate complexity.
  Training error: MODERATE
  Test error: MODERATE (and close to training)
```

How to adjust:

- **Reduce bias:** More complex model,
  more features, less regularization
- **Reduce variance:** Simpler model,
  more data, more regularization,
  ensemble methods (random forest)

---

### `Regularization`

**Definition.**
Regularization is a technique that adds a
penalty for model complexity during training.
It discourages the model from learning overly
complex patterns (noise) by shrinking or
eliminating less important parameters.

**Context.**
Regularization is one of the primary tools
for fighting overfitting. It works by adding
a term to the loss function that penalizes
large parameter values. Common forms are
L1 (Lasso), which can zero out features
entirely, and L2 (Ridge), which shrinks
all parameters toward zero without
eliminating them.

**Example.**
```python
from sklearn.linear_model import (
    LogisticRegression, Lasso, Ridge
)

# L1 regularization (Lasso)
# Can set some coefficients to exactly 0
# Good for feature selection
l1_model = LogisticRegression(
    penalty='l1',
    C=0.1,  # Lower C = stronger regularization
    solver='saga',
    max_iter=1000
)

# L2 regularization (Ridge)
# Shrinks all coefficients toward 0
# Good general-purpose regularization
l2_model = LogisticRegression(
    penalty='l2',
    C=0.1,
    max_iter=1000
)
```

The regularization parameter:

- **C** in scikit-learn:
  Inverse of regularization strength.
  Smaller C = stronger regularization.
- **alpha** in some libraries:
  Direct regularization strength.
  Larger alpha = stronger regularization.

---

### `Class Weights`

**Definition.**
Class weights tell the model to treat errors
on different classes with different importance.
By assigning higher weight to the minority
class, you make the model pay more attention
to getting those predictions right.

**Context.**
Class weights are one of the simplest and
most effective strategies for dealing with
class imbalance. Instead of modifying the
data (oversampling/undersampling), you modify
the loss function. Most scikit-learn
classifiers support a `class_weight` parameter.
Setting it to "balanced" automatically
calculates weights inversely proportional to
class frequencies.

**Example.**
```python
from sklearn.ensemble import RandomForestClassifier

# Automatic balanced weights
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Manual weights
# If class 0 is 95% and class 1 is 5%
model = RandomForestClassifier(
    class_weight={0: 1, 1: 19},
    random_state=42
)
model.fit(X_train, y_train)
```

How "balanced" works:

```
weight = n_samples / (n_classes * n_class_i)

For 1000 samples, 950 class 0, 50 class 1:
  weight_0 = 1000 / (2 * 950) = 0.526
  weight_1 = 1000 / (2 * 50)  = 10.0
```

Class 1 errors cost ~19 times more,
forcing the model to prioritize getting
minority class predictions correct.

---

## See Also

- [Statistical Foundations](./01_statistical_foundations.md)
- [Machine Learning Advanced](./06_machine_learning_advanced.md)
- [Model Evaluation and Monitoring](./07_model_evaluation_and_monitoring.md)
- [Data Quality and Validation](./04_data_quality_and_validation.md)

---

> **Author** — Simon Parris | Data Science Reference Library
