# Data Quality and Validation

---

> **Field** — Data Engineering and Data Governance
> **Scope** — Techniques and tools for ensuring data
> is correct, complete, consistent, and fit for use
> in analysis and machine learning

---

## Overview

Data quality is the foundation of trustworthy
analysis. If your data contains errors, missing
values, duplicates, or structural problems,
every model and report built on it will be
unreliable. This reference covers the core
concepts and tools for validating, measuring,
and maintaining data quality throughout the
data science lifecycle.

---

## Definitions

---

### `Data Validation`

**Definition.**
Data validation is the process of checking
whether data meets a defined set of rules
before accepting it for use. It answers the
question: "Does this data conform to what
we expect?"

**Context.**
Data validation is the first line of defense
against bad data. In data science, you
validate data at multiple points: when it
enters a pipeline, before training a model,
and before generating a report. Catching
problems early prevents corrupted analyses
and wasted compute time. Validation should
be automated, not manual.

**Example.**
Common validation checks:

- **Type check:** Is the age column numeric?
- **Range check:** Is age between 0 and 150?
- **Null check:** Are required fields filled?
- **Format check:** Do emails contain "@"?
- **Uniqueness:** Are IDs unique?

```python
import pandas as pd

df = pd.read_csv('data.csv')

# Type validation
assert df['age'].dtype in ['int64', 'float64'], \
    "age must be numeric"

# Range validation
assert df['age'].between(0, 150).all(), \
    "age values out of range"

# Null validation
assert df['customer_id'].notna().all(), \
    "customer_id must not be null"

# Uniqueness validation
assert df['customer_id'].is_unique, \
    "customer_id must be unique"

print("All validations passed.")
```

---

### `Data Quality`

**Definition.**
Data quality is a measure of how well data
serves its intended purpose. High-quality
data is accurate, complete, consistent,
timely, and valid. Low-quality data leads
to wrong conclusions, failed models, and
poor decisions.

**Context.**
Data quality is not binary (good or bad);
it is a spectrum. The same dataset might
be "good enough" for exploratory analysis
but insufficient for training a medical
diagnostic model. Data scientists must
assess quality relative to the task at
hand and document known issues. Poor data
quality is the most common reason data
science projects fail.

**Example.**
The six dimensions of data quality:

- **Accuracy:**
  Do values reflect the real world?
  (Is this really the customer's age?)

- **Completeness:**
  Are all required values present?
  (What percentage of rows have nulls?)

- **Consistency:**
  Do related fields agree?
  (Does zip code match the state?)

- **Timeliness:**
  Is the data recent enough?
  (Last updated 3 years ago?)

- **Validity:**
  Does data conform to expected formats?
  (Are dates in YYYY-MM-DD format?)

- **Uniqueness:**
  Are there duplicate records?
  (Same customer listed twice?)

```python
import pandas as pd

df = pd.read_csv('data.csv')

# Quick data quality report
quality = {
    'total_rows': len(df),
    'null_pct': df.isnull().mean().to_dict(),
    'duplicate_rows': df.duplicated().sum(),
    'unique_ids': df['id'].nunique(),
}
print(quality)
```

---

### `Schema Validation`

**Definition.**
Schema validation checks whether data
conforms to a predefined structure: the
correct columns exist, they have the right
data types, and they follow naming
conventions. It validates the shape of the
data, not the values.

**Context.**
Schema validation catches structural
problems before they cause cryptic errors
downstream. If an upstream system renames
a column, changes a type from integer to
string, or adds unexpected columns, schema
validation catches it immediately. It is
especially important in automated pipelines
where data arrives from sources you do not
control.

**Example.**
```python
# Simple schema validation with assertions
expected_columns = [
    'customer_id', 'name', 'email',
    'age', 'signup_date'
]

expected_types = {
    'customer_id': 'int64',
    'name': 'object',
    'email': 'object',
    'age': 'int64',
    'signup_date': 'object',
}

import pandas as pd

df = pd.read_csv('data.csv')

# Check columns exist
missing = set(expected_columns) - set(df.columns)
assert not missing, f"Missing columns: {missing}"

# Check data types
for col, dtype in expected_types.items():
    assert str(df[col].dtype) == dtype, \
        f"{col} should be {dtype}, " \
        f"got {df[col].dtype}"

print("Schema validation passed.")
```

For production pipelines, use dedicated
schema validation libraries:

- **Pydantic** for Python objects and APIs
- **pandera** for Pandas DataFrames
- **Great Expectations** for full data
  quality suites

---

### `Domain Rule`

**Definition.**
A domain rule is a business-specific
constraint that defines what valid data
looks like in a particular context. It
goes beyond basic type and format checks
to encode real-world knowledge about what
the data should contain.

**Context.**
Domain rules catch errors that generic
validation misses. A "temperature" column
might be numeric and non-null (passing
basic validation) but contain the value
500 degrees Celsius, which is physically
impossible for a weather station. Domain
rules encode this kind of expert knowledge.
They are essential for catching subtle data
quality issues.

**Example.**
Domain rules for a medical dataset:

- Heart rate must be between 30 and 250 bpm
- Systolic blood pressure must exceed
  diastolic blood pressure
- Patient age must be >= 0 and <= 120
- Admission date must be before or equal
  to discharge date

```python
import pandas as pd

df = pd.read_csv('patients.csv')

# Domain rule: heart rate range
invalid_hr = df[
    ~df['heart_rate'].between(30, 250)
]
if len(invalid_hr) > 0:
    print(f"WARNING: {len(invalid_hr)} rows "
          f"have invalid heart rate")

# Domain rule: blood pressure relationship
invalid_bp = df[
    df['systolic'] <= df['diastolic']
]
if len(invalid_bp) > 0:
    print(f"WARNING: {len(invalid_bp)} rows "
          f"have systolic <= diastolic")

# Domain rule: date ordering
df['admit'] = pd.to_datetime(df['admit_date'])
df['discharge'] = pd.to_datetime(df['discharge_date'])
invalid_dates = df[df['admit'] > df['discharge']]
if len(invalid_dates) > 0:
    print(f"WARNING: {len(invalid_dates)} rows "
          f"have admit after discharge")
```

---

### `Outlier`

**Definition.**
An outlier is a data point that is
significantly different from other
observations in the dataset. It lies
far outside the normal range of values.

**Context.**
Outliers can be genuine extreme values
(a CEO's salary in an employee dataset)
or errors (a typo that turned 25 into
2500). How you handle outliers depends on
context. In fraud detection, outliers ARE
the signal. In regression modeling, outliers
can distort results. Always investigate
outliers before removing them. Blindly
deleting outliers can introduce bias.

**Example.**
Common outlier detection methods:

**IQR Method:**
```python
import numpy as np

data = [10, 12, 14, 13, 15, 11, 100]

q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = [x for x in data
            if x < lower or x > upper]
print(f"Outliers: {outliers}")
# [100]
```

**Z-Score Method:**
```python
from scipy import stats
import numpy as np

data = np.array([10, 12, 14, 13, 15, 11, 100])
z_scores = np.abs(stats.zscore(data))
outliers = data[z_scores > 3]
print(f"Outliers: {outliers}")
```

What to do with outliers:

- **Investigate:** Is it real or an error?
- **Cap/Clip:** Replace with boundary value
- **Transform:** Use log scale
- **Remove:** Only if clearly erroneous
- **Keep:** If they are valid data points

---

### `Drift (data)`

**Definition.**
Data drift (also called dataset drift or
covariate shift) occurs when the statistical
properties of your data change over time.
The distribution of input features or
target values shifts from what your model
was trained on.

**Context.**
Drift is one of the main reasons ML models
degrade in production. A fraud detection
model trained on 2023 data may perform
poorly in 2025 because fraud patterns have
changed. Monitoring for drift is essential
for maintaining model accuracy. When drift
is detected, you may need to retrain the
model with recent data.

**Example.**
Types of drift:

- **Feature drift (covariate shift):**
  Input distribution changes.
  Example: average customer age increases.

- **Target drift (concept drift):**
  The relationship between features and
  target changes.
  Example: what makes a good credit risk
  changes due to economic conditions.

- **Prediction drift:**
  Model predictions shift over time.

Detecting feature drift:

```python
from scipy import stats

# Compare training and production data
train_ages = [25, 30, 35, 28, 32, 27]
prod_ages  = [45, 50, 42, 48, 55, 47]

# KS test: are these from the same
# distribution?
ks_stat, p_value = stats.ks_2samp(
    train_ages, prod_ages
)
print(f"KS statistic: {ks_stat:.3f}")
print(f"p-value: {p_value:.4f}")
# Low p-value = drift detected
```

---

### `Quality Gate`

**Definition.**
A quality gate is an automated checkpoint
in a data pipeline that halts processing if
data quality falls below a defined threshold.
It acts as a go/no-go decision point that
prevents bad data from reaching downstream
systems.

**Context.**
Quality gates are how teams enforce data
quality in production. Instead of hoping
data is clean, you build automated checks
that block bad data from entering your
models or reports. This protects against
silent failures where a model quietly
produces wrong predictions because it
received corrupted input data.

**Example.**
```python
import pandas as pd
import sys

def quality_gate(df, config):
    """
    Check data quality.
    Exit with error if thresholds breached.
    """
    issues = []

    # Check completeness
    null_pct = df.isnull().mean()
    for col, threshold in config['max_null_pct'].items():
        if null_pct[col] > threshold:
            issues.append(
                f"{col}: {null_pct[col]:.1%} null "
                f"(max {threshold:.1%})"
            )

    # Check row count
    if len(df) < config['min_rows']:
        issues.append(
            f"Only {len(df)} rows "
            f"(min {config['min_rows']})"
        )

    # Check duplicates
    dup_pct = df.duplicated().mean()
    if dup_pct > config['max_duplicate_pct']:
        issues.append(
            f"{dup_pct:.1%} duplicates "
            f"(max {config['max_duplicate_pct']:.1%})"
        )

    if issues:
        print("QUALITY GATE FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("Quality gate passed.")

# Usage
config = {
    'max_null_pct': {
        'customer_id': 0.0,
        'email': 0.05,
        'age': 0.10,
    },
    'min_rows': 1000,
    'max_duplicate_pct': 0.01,
}

df = pd.read_csv('data.csv')
quality_gate(df, config)
```

---

### `JSON Lines (JSONL)`

**Definition.**
JSON Lines (JSONL) is a file format where
each line is a valid JSON object, separated
by newline characters. Unlike regular JSON,
which wraps everything in a single array,
JSONL stores one record per line.

**Context.**
JSONL is popular in data science because it
is easy to process incrementally. You can
read one line at a time without loading the
entire file into memory, append new records
without rewriting the file, and process it
with simple Unix tools. It is commonly used
for log files, ML training data, and data
pipeline outputs.

**Example.**
A JSONL file (`data.jsonl`):

```
{"id": 1, "name": "Alice", "score": 88}
{"id": 2, "name": "Bob", "score": 92}
{"id": 3, "name": "Carol", "score": 79}
```

Reading JSONL in Python:

```python
import json

# Read line by line (memory efficient)
records = []
with open('data.jsonl', 'r') as f:
    for line in f:
        records.append(json.loads(line))

# Or with Pandas (one line)
import pandas as pd
df = pd.read_json('data.jsonl', lines=True)
```

Writing JSONL:

```python
import json

records = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
]

with open('output.jsonl', 'w') as f:
    for record in records:
        f.write(json.dumps(record) + '\n')
```

JSONL vs CSV:

- **JSONL:** handles nested data, no header
  row needed, self-describing
- **CSV:** smaller file size, faster to
  parse, universally supported

---

### `Pydantic`

**Definition.**
Pydantic is a Python library for data
validation using type annotations. You
define a data model as a Python class, and
Pydantic automatically validates that
incoming data matches the expected types,
formats, and constraints.

**Context.**
Pydantic is the standard tool for data
validation in modern Python applications.
It is used in FastAPI, data pipelines, and
configuration management. For data science,
Pydantic is useful for validating
configuration files, API inputs, and
individual records before they enter your
pipeline. It gives you clear error messages
when validation fails.

**Example.**
```python
from pydantic import BaseModel, Field
from pydantic import field_validator
from datetime import date

class PatientRecord(BaseModel):
    patient_id: int
    name: str
    age: int = Field(ge=0, le=120)
    heart_rate: float = Field(ge=30, le=250)
    admit_date: date

    @field_validator('name')
    @classmethod
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError('name cannot be empty')
        return v.strip()

# Valid record
patient = PatientRecord(
    patient_id=1,
    name="Alice Smith",
    age=35,
    heart_rate=72.0,
    admit_date="2024-01-15"
)
print(patient)

# Invalid record (age out of range)
try:
    bad = PatientRecord(
        patient_id=2,
        name="Bob",
        age=200,  # Too high
        heart_rate=72.0,
        admit_date="2024-01-15"
    )
except Exception as e:
    print(f"Validation error: {e}")
```

Key features:

- Automatic type coercion (string "42" to int 42)
- Clear error messages
- JSON serialization built in
- Widely used in the Python ecosystem

---

### `Out-of-core Processing`

**Definition.**
Out-of-core processing refers to techniques
for working with datasets that are too large
to fit in memory (RAM). Instead of loading
everything at once, you process the data in
smaller chunks.

**Context.**
When your CSV is 50 GB and your laptop has
16 GB of RAM, you need out-of-core methods.
This is increasingly common as datasets
grow. Techniques include chunked reading,
memory-mapped files, and tools designed for
large data (Dask, Polars, Vaex). Every data
scientist eventually encounters a dataset
that does not fit in memory.

**Example.**
Chunked processing with Pandas:

```python
import pandas as pd

# Process a large CSV in 10,000-row chunks
total_rows = 0
total_amount = 0

for chunk in pd.read_csv('huge_file.csv',
                          chunksize=10_000):
    total_rows += len(chunk)
    total_amount += chunk['amount'].sum()

average = total_amount / total_rows
print(f"Average amount: {average:.2f}")
```

Using Dask for out-of-core DataFrames:

```python
import dask.dataframe as dd

# Reads lazily (no memory spike)
df = dd.read_csv('huge_file.csv')

# Computations are lazy until .compute()
result = df.groupby('category')['amount'].mean()
print(result.compute())
```

Other out-of-core tools:

- **Dask:** parallel, out-of-core Pandas
- **Polars:** fast DataFrame library in Rust
- **Vaex:** memory-mapped DataFrames
- **SQLite:** query large data with SQL
- **Apache Arrow:** columnar in-memory format

---

### `Completeness`

**Definition.**
Completeness measures the proportion of
required data that is actually present. It
answers the question: "How much data is
missing?"

**Context.**
Completeness is one of the most commonly
measured data quality dimensions. Missing
data (nulls, blanks, "N/A" strings) can
break calculations, bias models, and lead
to wrong conclusions. Before any analysis,
you should assess completeness and decide
how to handle missing values: drop them,
fill them, or flag them.

**Example.**
```python
import pandas as pd

df = pd.read_csv('data.csv')

# Completeness per column
completeness = 1 - df.isnull().mean()
print("Completeness by column:")
print(completeness.round(3))

# Overall completeness
total_cells = df.size
filled_cells = df.notna().sum().sum()
overall = filled_cells / total_cells
print(f"\nOverall completeness: {overall:.1%}")
```

Sample output:

```
Completeness by column:
customer_id    1.000
name           0.998
email          0.950
phone          0.720
age            0.880

Overall completeness: 90.9%
```

Strategies for handling missing data:

- **Drop rows:** when few rows are affected
- **Drop columns:** when a column is mostly null
- **Fill with mean/median:** for numerical data
- **Fill with mode:** for categorical data
- **Flag as missing:** add a boolean column
- **Model-based imputation:** predict missing values

---

### `Consistency`

**Definition.**
Consistency means that related data values
do not contradict each other, both within
a single dataset and across multiple
datasets. Inconsistent data contains
conflicting information.

**Context.**
Inconsistency is a subtle data quality
problem. Each value might look valid in
isolation, but together they do not make
sense. For example, a customer's city says
"London" but their country says "Japan."
Consistency checks require understanding
the relationships between fields and are
often the hardest data quality issues to
detect automatically.

**Example.**
Types of inconsistency:

- **Internal:** A person's age is 5 but
  their job title is "Senior Manager"
- **Cross-field:** Ship date is before
  order date
- **Cross-table:** Customer address differs
  between the customers and orders tables
- **Temporal:** Sales were $1M on Monday
  but $0 on Tuesday with no explanation

```python
import pandas as pd

df = pd.read_csv('orders.csv')

# Cross-field consistency check
df['order_date'] = pd.to_datetime(df['order_date'])
df['ship_date'] = pd.to_datetime(df['ship_date'])

inconsistent = df[df['ship_date'] < df['order_date']]
print(f"Inconsistent records: {len(inconsistent)}")

# Cross-field: total should equal
# quantity * unit_price
df['expected_total'] = df['quantity'] * df['unit_price']
mismatch = df[
    abs(df['total'] - df['expected_total']) > 0.01
]
print(f"Total mismatches: {len(mismatch)}")
```

---

### `Accuracy (data)`

**Definition.**
Accuracy measures how closely data values
represent the real-world entities they
describe. Accurate data correctly reflects
reality.

**Context.**
Accuracy is the hardest data quality
dimension to measure because it requires
knowing the true values. You can check
that an age is numeric and in range (0-120),
but you cannot easily verify that the age
is actually correct without an external
reference. Accuracy issues come from data
entry errors, sensor malfunctions, outdated
records, and conversion mistakes.

**Example.**
Accuracy assessment approaches:

- **Spot checking:** Manually verify a
  random sample against source documents
- **Cross-referencing:** Compare against
  a trusted external source
- **Reasonableness checks:** Flag values
  that are technically valid but suspicious

```python
import pandas as pd

df = pd.read_csv('employees.csv')

# Reasonableness check for salary
median_salary = df['salary'].median()
suspicious = df[
    (df['salary'] < 10_000) |
    (df['salary'] > 500_000)
]
print(f"Suspicious salaries: {len(suspicious)}")

# Cross-reference with known data
known_ceos = {'Alice': 150_000, 'Bob': 160_000}
for name, expected in known_ceos.items():
    actual = df.loc[
        df['name'] == name, 'salary'
    ].values
    if len(actual) > 0 and actual[0] != expected:
        print(f"Accuracy issue: {name} "
              f"salary is {actual[0]}, "
              f"expected {expected}")
```

Key principle: accuracy checks should be
proportional to the cost of errors. Medical
data needs more accuracy verification than
a marketing email list.

---

## See Also

- [Databases and Data Engineering](./03_databases_and_data_engineering.md)
- [Python and Numerical Computing](./02_python_and_numerical_computing.md)
- [Machine Learning Fundamentals](./05_machine_learning_fundamentals.md)
- [Model Evaluation and Monitoring](./07_model_evaluation_and_monitoring.md)

---

> **Author** — Simon Parris | Data Science Reference Library
