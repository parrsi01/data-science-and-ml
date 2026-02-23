# Model Evaluation and Monitoring

---

> **Field** — MLOps and Model Management
> **Scope** — Metrics for evaluating model performance,
> techniques for monitoring deployed models, and
> strategies for maintaining model reliability
> in production

---

## Overview

Building a model is only half the job. You also
need to rigorously evaluate how well it performs,
understand its failure modes, and continuously
monitor it after deployment. This reference
covers the metrics, tests, and monitoring
strategies that ensure your model works reliably
not just on test data, but in the real world
over time.

---

## Definitions

---

### `Precision`

**Definition.**
Precision is the proportion of positive
predictions that are actually correct. Of
all the items the model said were positive,
how many really were positive?

**Context.**
Precision matters most when false positives
are costly. In spam detection, low precision
means legitimate emails end up in the spam
folder, which is very annoying for users.
In fraud detection, low precision means
flagging too many legitimate transactions,
causing customer friction. High precision
means "when the model says yes, you can
trust it."

**Example.**
```
Precision = TP / (TP + FP)

If the model predicts 100 transactions
as fraudulent:
  - 80 are actually fraud (TP = 80)
  - 20 are legitimate (FP = 20)
  - Precision = 80 / 100 = 0.80 (80%)
```

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.3f}")
```

Precision vs Recall tradeoff:

- Raising the prediction threshold increases
  precision but decreases recall
- Lowering the threshold increases recall
  but decreases precision
- You cannot maximize both simultaneously

---

### `Recall`

**Definition.**
Recall (also called sensitivity or true
positive rate) is the proportion of actual
positive cases that the model correctly
identified. Of all the items that truly
were positive, how many did the model
catch?

**Context.**
Recall matters most when false negatives
are costly. In cancer screening, low recall
means missing actual cancer cases, which
could be fatal. In security threat
detection, low recall means missing real
threats. High recall means "the model
catches most of the real positives."

**Example.**
```
Recall = TP / (TP + FN)

If there are 100 actual fraud cases:
  - Model catches 70 of them (TP = 70)
  - Model misses 30 of them (FN = 30)
  - Recall = 70 / 100 = 0.70 (70%)
```

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.3f}")
```

When to prioritize recall over precision:

- Medical diagnosis (missing disease is worse
  than a false alarm)
- Security (missing a threat is worse than
  investigating a false alarm)
- Safety systems (missing a failure is worse
  than a false shutdown)

---

### `F1 Score`

**Definition.**
The F1 score is the harmonic mean of
precision and recall. It provides a single
number that balances both metrics. It ranges
from 0 (worst) to 1 (best).

**Context.**
F1 is the go-to metric when you need to
balance precision and recall and do not have
a strong reason to favor one over the other.
It is especially useful for imbalanced
datasets where accuracy is misleading. The
harmonic mean ensures that F1 is low if
either precision or recall is low, even if
the other is high.

**Example.**
```
F1 = 2 * (Precision * Recall) /
         (Precision + Recall)

Precision = 0.80, Recall = 0.70
F1 = 2 * (0.80 * 0.70) / (0.80 + 0.70)
F1 = 2 * 0.56 / 1.50
F1 = 0.747
```

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print(f"F1: {f1:.3f}")

# Full classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

F1 variants:

- **F1 (binary):** For binary classification
- **F1 macro:** Average F1 across classes
  (treats all classes equally)
- **F1 weighted:** Average F1 weighted by
  class frequency
- **F-beta:** Generalized form where
  beta > 1 favors recall,
  beta < 1 favors precision

---

### `Calibration Threshold`

**Definition.**
A calibration threshold (or classification
threshold or decision threshold) is the
probability cutoff above which a model's
output is classified as positive. The
default is 0.5, but the optimal threshold
depends on the relative costs of false
positives and false negatives.

**Context.**
Most classification models output
probabilities, not binary labels. The
threshold converts probabilities into
decisions. Tuning this threshold is a
powerful way to control the precision/recall
tradeoff without retraining the model.
For imbalanced problems, 0.5 is almost
never the best threshold.

**Example.**
```python
import numpy as np
from sklearn.metrics import f1_score

# Model outputs probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Find best threshold for F1
best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.05, 0.95, 0.05):
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f}")
print(f"Best F1: {best_f1:.3f}")
```

Common threshold strategies:

- **Default (0.5):** Rarely optimal for
  imbalanced data
- **Maximize F1:** Best balance of precision
  and recall
- **Maximize recall at minimum precision:**
  "Catch at least 90% of positives while
  maintaining at least 50% precision"
- **Business-driven:** Based on the dollar
  cost of false positives vs false negatives

---

### `Stability (Seed Sensitivity)`

**Definition.**
Stability (or seed sensitivity) measures
how much a model's performance varies when
you change only the random seed. A stable
model produces similar results across
different seeds; an unstable model's
performance fluctuates significantly.

**Context.**
If your model's F1 score is 0.85 with
seed 42 but 0.68 with seed 123, the
model is unstable and the reported
performance is unreliable. Stability
testing reveals whether your results are
robust or just lucky. This is especially
important for small datasets and complex
models. Always test multiple seeds before
reporting final results.

**Example.**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

seeds = [42, 123, 456, 789, 0, 17, 99, 314]
results = []

for seed in seeds:
    model = RandomForestClassifier(
        random_state=seed,
        n_estimators=100
    )
    scores = cross_val_score(
        model, X, y, cv=5, scoring='f1'
    )
    results.append(scores.mean())

results = np.array(results)
print(f"Mean F1:   {results.mean():.3f}")
print(f"Std F1:    {results.std():.3f}")
print(f"Min F1:    {results.min():.3f}")
print(f"Max F1:    {results.max():.3f}")
print(f"Range:     {results.max() - results.min():.3f}")
```

Interpretation:

- Std < 0.01: Very stable
- Std 0.01-0.03: Acceptable stability
- Std > 0.03: Investigate further
  (more data, simpler model, or
  regularization may help)

---

### `Bias (Group Metrics)`

**Definition.**
Bias in model evaluation refers to
systematic differences in model performance
across demographic or subgroup populations.
A model is biased if it performs significantly
better or worse for one group compared to
another.

**Context.**
Bias testing is critical for fairness and
regulatory compliance. A loan approval model
that works well overall but denies qualified
applicants from a particular demographic
group is biased and potentially illegal.
Always evaluate model performance across
relevant subgroups (gender, age group,
geography, etc.) to detect and address
disparities.

**Example.**
```python
from sklearn.metrics import f1_score
import pandas as pd

# Evaluate F1 by subgroup
groups = df_test['gender'].unique()

for group in groups:
    mask = df_test['gender'] == group
    group_f1 = f1_score(
        y_test[mask], y_pred[mask]
    )
    group_count = mask.sum()
    print(f"{group}: F1={group_f1:.3f} "
          f"(n={group_count})")
```

Sample output that reveals bias:

```
Male:   F1=0.88 (n=4500)
Female: F1=0.71 (n=4500)
```

The 0.17 gap in F1 score indicates potential
bias that needs investigation.

Metrics to check across groups:

- F1 score
- False positive rate
- False negative rate
- Precision
- Recall
- Selection rate (percentage predicted positive)

---

### `Drift Detection`

**Definition.**
Drift detection is the process of
continuously monitoring whether the
statistical properties of incoming data
or model predictions have changed compared
to a reference baseline (typically the
training data distribution).

**Context.**
Models degrade over time because the world
changes. Customer behavior shifts, fraud
tactics evolve, and economic conditions
fluctuate. Drift detection is the early
warning system that tells you when your
model's assumptions no longer hold. Without
it, a model can silently produce
increasingly wrong predictions for weeks
or months before anyone notices.

**Example.**
```python
from scipy import stats
import numpy as np

def detect_drift(
    reference, current,
    threshold=0.05
):
    """
    Detect drift using the KS test.
    Returns True if drift is detected.
    """
    results = {}
    for column in reference.columns:
        ks_stat, p_value = stats.ks_2samp(
            reference[column].dropna(),
            current[column].dropna()
        )
        drifted = p_value < threshold
        results[column] = {
            'ks_stat': round(ks_stat, 4),
            'p_value': round(p_value, 4),
            'drifted': drifted
        }
    return results

# Compare training data to this week's data
drift_report = detect_drift(
    df_training[features],
    df_this_week[features]
)

for feature, result in drift_report.items():
    if result['drifted']:
        print(f"DRIFT: {feature} "
              f"(KS={result['ks_stat']}, "
              f"p={result['p_value']})")
```

What to monitor:

- **Input drift:** Feature distributions
- **Prediction drift:** Output distribution
- **Performance drift:** Metrics on labeled
  data (when available)
- **Concept drift:** Relationship between
  features and target changes

---

### `KS Test (Kolmogorov-Smirnov)`

**Definition.**
The Kolmogorov-Smirnov test is a
non-parametric statistical test that
compares two distributions to determine
whether they are significantly different.
The KS statistic measures the maximum
distance between the two cumulative
distribution functions.

**Context.**
The KS test is the most commonly used
statistical test for drift detection in
MLOps. It compares each feature's
distribution in the current data to its
distribution in the training data. A
significant KS test result (low p-value)
means the distributions have changed,
which may indicate drift. It works for any
continuous distribution and makes no
assumptions about the data's shape.

**Example.**
```python
from scipy import stats
import numpy as np

# Training data distribution
train_ages = np.random.normal(35, 10, 10000)

# Production data distribution
prod_ages = np.random.normal(40, 12, 5000)

ks_stat, p_value = stats.ks_2samp(
    train_ages, prod_ages
)

print(f"KS statistic: {ks_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("Drift detected: distributions "
          "are significantly different")
else:
    print("No significant drift detected")
```

Interpretation:

- **KS statistic:** Ranges from 0 to 1.
  Higher = more different.
- **p-value < 0.05:** Distributions are
  significantly different (drift detected)
- **p-value >= 0.05:** No significant
  difference detected

KS test limitations:

- Sensitive to sample size (large samples
  detect tiny, irrelevant differences)
- Only works for continuous variables
- For categorical variables, use chi-squared
  test instead

---

### `Total Variation Distance`

**Definition.**
Total Variation Distance (TVD) measures the
maximum difference in probability between
two distributions across all possible events.
For discrete distributions, it is half the
sum of absolute differences in probabilities.

**Context.**
TVD is commonly used to compare categorical
distributions, where the KS test does not
apply. It is useful for monitoring drift in
categorical features (e.g., has the
distribution of product categories changed?).
TVD ranges from 0 (identical distributions)
to 1 (completely different distributions).

**Example.**
```python
import numpy as np
from collections import Counter

def total_variation_distance(dist_a, dist_b):
    """
    Calculate TVD between two categorical
    distributions represented as arrays.
    """
    # Get frequency distributions
    all_categories = set(dist_a) | set(dist_b)
    n_a = len(dist_a)
    n_b = len(dist_b)

    count_a = Counter(dist_a)
    count_b = Counter(dist_b)

    tvd = 0
    for cat in all_categories:
        p_a = count_a.get(cat, 0) / n_a
        p_b = count_b.get(cat, 0) / n_b
        tvd += abs(p_a - p_b)

    return tvd / 2

# Training distribution
train_cats = ['A']*500 + ['B']*300 + ['C']*200

# Production distribution (shifted)
prod_cats = ['A']*300 + ['B']*400 + ['C']*300

tvd = total_variation_distance(
    train_cats, prod_cats
)
print(f"TVD: {tvd:.3f}")
# TVD > 0.1 often indicates meaningful drift
```

Interpretation:

- TVD = 0: Distributions are identical
- TVD < 0.05: Very similar
- TVD 0.05-0.15: Some difference
- TVD > 0.15: Significant difference
- TVD = 1: Completely different

---

### `Validation Split`

**Definition.**
A validation split is a portion of the
training data held out to evaluate the model
during development and tune hyperparameters.
It sits between the training set and the
test set: used for model selection but not
for final evaluation.

**Context.**
If you use the test set to tune
hyperparameters, you are effectively training
on the test set, which makes your final
metric estimate optimistic. The validation
set solves this: use it for all tuning
decisions, and only touch the test set
once at the very end. In practice,
cross-validation often replaces a single
validation split for more reliable estimates.

**Example.**
Three-way split:

```python
from sklearn.model_selection import train_test_split

# First split: separate out test set (final)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15,
    stratify=y, random_state=42
)

# Second split: separate train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.18,
    stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train)} samples")
print(f"Val:   {len(X_val)} samples")
print(f"Test:  {len(X_test)} samples")
```

How each split is used:

- **Training set (70%):**
  Fit the model parameters
- **Validation set (15%):**
  Tune hyperparameters, select the best model
- **Test set (15%):**
  Final, unbiased performance estimate.
  Use ONCE at the very end.

---

### `Threshold Tradeoff`

**Definition.**
The threshold tradeoff is the relationship
between the classification threshold and
the resulting precision and recall. Moving
the threshold in one direction improves one
metric at the expense of the other.

**Context.**
Understanding the threshold tradeoff is
essential for deploying models in production.
The optimal threshold depends entirely on
your use case. A fraud system might accept
lower precision (more false alarms) to
achieve higher recall (catch more fraud).
A medical screening might need very high
recall even if precision drops significantly.

**Example.**
```python
from sklearn.metrics import (
    precision_recall_curve
)
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = (
    precision_recall_curve(y_test, y_prob)
)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1],
         label='Precision')
plt.plot(thresholds, recalls[:-1],
         label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.grid(True)
plt.savefig('threshold_tradeoff.png', dpi=150)
```

Typical behavior:

```
Threshold  Precision  Recall
---------  ---------  ------
0.1        0.15       0.98
0.3        0.45       0.85
0.5        0.72       0.65
0.7        0.88       0.40
0.9        0.95       0.15
```

As threshold increases:

- Precision goes UP (fewer false positives)
- Recall goes DOWN (more false negatives)

---

### `Seed Sweep`

**Definition.**
A seed sweep is the practice of training a
model multiple times with different random
seeds and evaluating the distribution of
results. It quantifies how sensitive the
model's performance is to random
initialization and data shuffling.

**Context.**
A seed sweep is the practical implementation
of stability testing. By running 10-50
experiments with different seeds, you get a
distribution of performance metrics instead
of a single point estimate. This tells you
both the expected performance and how much
it varies. Report results as "mean plus or
minus standard deviation" rather than a
single number.

**Example.**
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold
)

def seed_sweep(X, y, n_seeds=20):
    """Run a seed sweep and report statistics."""
    f1_scores = []
    auc_scores = []

    for seed in range(n_seeds):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=seed
        )

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed
        )

        f1 = cross_val_score(
            model, X, y,
            cv=cv, scoring='f1'
        ).mean()

        auc = cross_val_score(
            model, X, y,
            cv=cv, scoring='roc_auc'
        ).mean()

        f1_scores.append(f1)
        auc_scores.append(auc)

    f1_arr = np.array(f1_scores)
    auc_arr = np.array(auc_scores)

    print(f"F1:  {f1_arr.mean():.3f} "
          f"+/- {f1_arr.std():.3f}")
    print(f"AUC: {auc_arr.mean():.3f} "
          f"+/- {auc_arr.std():.3f}")
    print(f"F1 range: [{f1_arr.min():.3f}, "
          f"{f1_arr.max():.3f}]")

    return f1_scores, auc_scores

f1_results, auc_results = seed_sweep(X, y)
```

---

### `False Positive Rate`

**Definition.**
False positive rate (FPR) is the proportion
of actual negative cases that the model
incorrectly classified as positive. It
measures how often the model triggers
false alarms.

**Context.**
FPR is critical in applications where false
alarms have real costs: security systems
that alert on non-threats, medical tests
that cause unnecessary procedures, or fraud
systems that block legitimate transactions.
FPR is also the x-axis of the ROC curve.
A low FPR means the model rarely cries wolf.

**Example.**
```
FPR = FP / (FP + TN)

If there are 1000 legitimate transactions:
  - Model flags 50 as fraud (FP = 50)
  - Model correctly passes 950 (TN = 950)
  - FPR = 50 / 1000 = 0.05 (5%)
```

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.3f}")
```

Relationship to other metrics:

- FPR = 1 - Specificity
- Specificity = TN / (TN + FP)
- Low FPR = High Specificity

---

### `False Negative Rate`

**Definition.**
False negative rate (FNR) is the proportion
of actual positive cases that the model
incorrectly classified as negative. It
measures how often the model misses real
positives.

**Context.**
FNR is the complement of recall (FNR = 1 -
Recall). It is critical when missing a
positive case is dangerous or expensive:
missing a disease diagnosis, failing to
detect a security breach, or overlooking
a defective product. A low FNR means the
model rarely misses real positives.

**Example.**
```
FNR = FN / (FN + TP)

If there are 100 actual fraud cases:
  - Model misses 15 of them (FN = 15)
  - Model catches 85 of them (TP = 85)
  - FNR = 15 / 100 = 0.15 (15%)
  - Recall = 1 - 0.15 = 0.85 (85%)
```

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

fnr = fn / (fn + tp)
print(f"False Negative Rate: {fnr:.3f}")
print(f"Recall: {1 - fnr:.3f}")
```

---

### `Monitoring`

**Definition.**
Model monitoring is the ongoing process of
tracking a deployed model's inputs,
predictions, and performance metrics to
detect problems before they cause harm. It
is the production equivalent of testing.

**Context.**
A model that worked well during development
can degrade in production for many reasons:
data drift, upstream system changes, schema
changes, or infrastructure failures.
Monitoring provides visibility into model
health and triggers alerts when something
goes wrong. Without monitoring, failures are
silent and can persist for weeks or months.

**Example.**
What to monitor:

1. **Input data quality:**
   - Missing value rates
   - Feature distributions
   - Schema conformance

2. **Prediction behavior:**
   - Prediction distribution
   - Prediction volume
   - Latency

3. **Performance metrics:**
   - F1, precision, recall (when labels
     arrive)
   - ROC-AUC

4. **System health:**
   - Memory usage
   - CPU usage
   - Error rates

```python
import json
from datetime import datetime

def log_prediction(
    features, prediction, probability
):
    """Log each prediction for monitoring."""
    record = {
        'timestamp': datetime.utcnow().isoformat(),
        'features': features,
        'prediction': int(prediction),
        'probability': float(probability),
    }
    with open('prediction_log.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')
```

Alerting rules example:

- Alert if null rate for any feature
  exceeds 5%
- Alert if prediction positive rate
  changes by more than 20%
- Alert if inference latency exceeds
  500ms for the p95
- Alert if error rate exceeds 1%

---

### `Inference`

**Definition.**
Inference is the process of using a trained
model to make predictions on new, unseen
data. It is the deployment-time counterpart
of training: training learns the patterns,
inference applies them.

**Context.**
Inference is when your model creates real
value. Everything in the ML lifecycle
(data collection, feature engineering,
training, evaluation) leads to this moment.
Inference performance matters: it needs to
be fast enough for the use case (real-time
for web APIs, batch for reports) and
reliable enough for production use.

**Example.**
**Batch inference:**

```python
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Load new data
new_data = pd.read_csv('today_transactions.csv')

# Preprocess
X = scaler.transform(new_data[features])

# Predict
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

# Save results
new_data['prediction'] = predictions
new_data['fraud_probability'] = probabilities
new_data.to_csv('predictions_today.csv',
                index=False)
```

**Real-time inference (API):**

```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.post('/predict')
def predict(data: dict):
    X = scaler.transform([data['features']])
    prob = model.predict_proba(X)[0, 1]
    return {
        'probability': float(prob),
        'prediction': int(prob >= 0.5)
    }
```

Inference types:

- **Batch:** Process many records at once.
  Used for reports, nightly pipelines.
- **Real-time:** Process one record at a
  time via API. Used for web apps, live
  systems.
- **Near real-time:** Micro-batches every
  few minutes. Balance of speed and
  efficiency.

---

### `Health Check`

**Definition.**
A health check is an automated test that
verifies a deployed model and its
dependencies are functioning correctly.
It confirms the model can load, preprocess
data, and return predictions without errors.

**Context.**
Health checks are the first thing to look
at when a model-powered system has problems.
They run continuously (every few minutes)
and immediately surface issues like: the
model file is corrupted, a dependency is
missing, the database is unreachable, or
inference is timing out. They are standard
practice in production ML systems.

**Example.**
```python
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.get('/health')
def health_check():
    """
    Verify the model pipeline works end-to-end
    with a synthetic input.
    """
    try:
        # Create dummy input
        dummy = np.zeros((1, len(feature_names)))

        # Test preprocessing
        scaled = scaler.transform(dummy)

        # Test inference
        prediction = model.predict(scaled)
        probability = model.predict_proba(scaled)

        return {
            'status': 'healthy',
            'model_loaded': True,
            'scaler_loaded': True,
            'can_predict': True,
            'sample_prediction': int(prediction[0])
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }
```

What a health check should verify:

- Model file loads successfully
- Preprocessor (scaler, encoder) loads
- A dummy prediction completes without error
- Prediction latency is within bounds
- Required external services are reachable
  (database, feature store)

---

### `Model Serving`

**Definition.**
Model serving is the infrastructure and
software that makes a trained model
available for inference requests. It handles
loading the model into memory, accepting
input data, running predictions, and
returning results.

**Context.**
Model serving bridges the gap between
a model that works in a notebook and one
that works in production. Serving systems
handle concerns that do not exist during
development: concurrency (many simultaneous
requests), latency requirements, version
management, A/B testing, and graceful
failover. Choosing the right serving
approach depends on your scale and latency
requirements.

**Example.**
Common model serving approaches:

**Simple: Flask/FastAPI**

```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

@app.post('/predict')
def predict(data: dict):
    prediction = model.predict([data['features']])
    return {'prediction': int(prediction[0])}
```

**Production: containerized**

```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model.pkl scaler.pkl app.py ./
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

Serving architecture levels:

- **Level 1:** Script that loads model and
  runs batch predictions
- **Level 2:** Simple API (Flask/FastAPI)
- **Level 3:** Containerized API with
  health checks and logging
- **Level 4:** Dedicated serving platform
  (MLflow, Seldon, BentoML, TFServing)

Key considerations:

- Latency: How fast must predictions be?
- Throughput: How many predictions per second?
- Availability: What happens if it goes down?
- Versioning: Can you roll back to a
  previous model?

---

### `Drift Flag`

**Definition.**
A drift flag is a boolean indicator that is
raised when a monitoring system detects
that data or model behavior has shifted
beyond a predefined threshold. It signals
that investigation or retraining may be
needed.

**Context.**
Drift flags are the actionable output of
drift detection systems. When a flag is
raised, it triggers a workflow: an alert
is sent, the drift is investigated, and
a decision is made about whether to
retrain the model. Not every drift flag
requires immediate action; some drift is
transient (seasonal fluctuations), while
other drift is permanent and requires
model updates.

**Example.**
```python
from scipy import stats
import json
from datetime import datetime

def check_and_flag_drift(
    reference_data,
    current_data,
    features,
    ks_threshold=0.1,
    p_threshold=0.05
):
    """
    Check each feature for drift and
    return a drift report with flags.
    """
    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'features': {},
        'any_drift': False
    }

    for feature in features:
        ks_stat, p_value = stats.ks_2samp(
            reference_data[feature].dropna(),
            current_data[feature].dropna()
        )

        drifted = (
            ks_stat > ks_threshold and
            p_value < p_threshold
        )

        report['features'][feature] = {
            'ks_statistic': round(ks_stat, 4),
            'p_value': round(p_value, 4),
            'drift_flag': drifted
        }

        if drifted:
            report['any_drift'] = True

    return report

# Run daily drift check
drift_report = check_and_flag_drift(
    training_data, todays_data, feature_list
)

# Save report
with open('drift_report.json', 'w') as f:
    json.dump(drift_report, f, indent=2)

# Alert if drift detected
if drift_report['any_drift']:
    flagged = [
        f for f, v in drift_report['features'].items()
        if v['drift_flag']
    ]
    print(f"DRIFT ALERT: {flagged}")
```

Drift response workflow:

1. **Flag raised:** Automated detection
2. **Alert sent:** Team notified
3. **Investigation:** Is it real? Is it
   impactful?
4. **Decision:** Retrain, adjust threshold,
   or dismiss
5. **Action:** Retrain model or update
   monitoring thresholds

---

### `MAE`

**Definition.**
MAE (Mean Absolute Error) is the average
of the absolute differences between
predicted values and actual values. It
measures how far off predictions are, on
average, in the original units of the
target variable.

**Context.**
MAE is the most intuitive regression metric.
If MAE is 5.2 and you are predicting
temperature in degrees, your predictions
are off by an average of 5.2 degrees.
Unlike MSE/RMSE, MAE treats all errors
equally (no squaring), so it is less
sensitive to outliers. Use MAE when every
error matters equally, regardless of size.

**Example.**
```
MAE = mean(|actual - predicted|)

Actual:    [10, 20, 30, 40, 50]
Predicted: [12, 18, 35, 38, 52]
Errors:    [ 2,  2,  5,  2,  2]
MAE = (2 + 2 + 5 + 2 + 2) / 5 = 2.6
```

```python
from sklearn.metrics import mean_absolute_error
import numpy as np

y_actual = np.array([10, 20, 30, 40, 50])
y_predicted = np.array([12, 18, 35, 38, 52])

mae = mean_absolute_error(y_actual, y_predicted)
print(f"MAE: {mae:.1f}")
# MAE: 2.6
```

MAE vs RMSE:

- **MAE:** Treats all errors equally.
  More robust to outliers.
- **RMSE:** Penalizes large errors more.
  More sensitive to outliers.
- If MAE and RMSE are very different,
  you have some predictions with large errors.

---

### `RMSE`

**Definition.**
RMSE (Root Mean Squared Error) is the square
root of the average of squared differences
between predicted and actual values. It
penalizes large errors more heavily than
small ones.

**Context.**
RMSE is one of the most commonly used
regression metrics. Because it squares errors
before averaging, a single large error has a
disproportionate effect on RMSE. This makes
RMSE useful when large errors are
particularly undesirable (e.g., predicting
delivery times where a 2-hour error is
much worse than four 30-minute errors).
RMSE is in the same units as the target
variable, making it interpretable.

**Example.**
```
RMSE = sqrt(mean((actual - predicted)^2))

Actual:    [10, 20, 30, 40, 50]
Predicted: [12, 18, 35, 38, 52]
Sq Errors: [ 4,  4, 25,  4,  4]
MSE = (4 + 4 + 25 + 4 + 4) / 5 = 8.2
RMSE = sqrt(8.2) = 2.86
```

```python
from sklearn.metrics import root_mean_squared_error
import numpy as np

y_actual = np.array([10, 20, 30, 40, 50])
y_predicted = np.array([12, 18, 35, 38, 52])

rmse = root_mean_squared_error(
    y_actual, y_predicted
)
print(f"RMSE: {rmse:.2f}")
# RMSE: 2.86
```

Notice RMSE (2.86) > MAE (2.60) for the
same data. This gap tells you some errors
are larger than others. If all errors
were exactly the same size, MAE and RMSE
would be equal.

Comparison:

```
                MAE    RMSE
All errors = 3: 3.00   3.00
Mixed errors:   2.60   2.86
One big error:  2.60   4.47
```

The bigger the gap between RMSE and MAE,
the more your errors vary in size.

---

## See Also

- [Machine Learning Fundamentals](./05_machine_learning_fundamentals.md)
- [Machine Learning Advanced](./06_machine_learning_advanced.md)
- [Statistical Foundations](./01_statistical_foundations.md)
- [Data Quality and Validation](./04_data_quality_and_validation.md)

---

> **Author** — Simon Parris | Data Science Reference Library
