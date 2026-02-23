# Python and Numerical Computing

---

> **Field** — Software Engineering and Scientific Computing
> **Scope** — Python programming patterns, performance
> optimization, and numerical libraries essential
> for data science workflows

---

## Overview

Python is the dominant language in data science,
but writing fast, efficient Python code requires
understanding how tools like NumPy and Pandas
work under the hood. This reference covers the
programming concepts, data structures, and
performance techniques you need to write code
that scales from prototypes to production
data pipelines.

---

## Definitions

---

### `Vectorization`

**Definition.**
Vectorization is the practice of replacing
explicit Python loops with operations that
act on entire arrays or columns at once.
Instead of processing one element at a time,
you process thousands or millions in a
single operation.

**Context.**
Vectorization is the single most important
performance technique in numerical Python.
A vectorized NumPy operation can be 10 to
100 times faster than the equivalent Python
for-loop because the actual computation
happens in optimized C code. Whenever you
find yourself writing a for-loop over a
NumPy array or Pandas DataFrame, ask
whether you can vectorize it instead.

**Example.**
Slow (loop):

```python
import numpy as np

data = np.random.rand(1_000_000)
result = np.empty_like(data)
for i in range(len(data)):
    result[i] = data[i] * 2 + 1
```

Fast (vectorized):

```python
result = data * 2 + 1
```

Both produce identical results, but the
vectorized version runs dramatically faster
because NumPy handles the loop internally
in compiled C code.

---

### `Broadcasting`

**Definition.**
Broadcasting is a set of rules that NumPy
uses to perform arithmetic between arrays
of different shapes. Instead of requiring
arrays to have identical dimensions, NumPy
"stretches" the smaller array to match the
larger one automatically.

**Context.**
Broadcasting eliminates the need to
manually reshape or tile arrays before
doing math. It is the reason you can
subtract a single mean value from an entire
column, or add a row vector to every row
of a matrix. Understanding broadcasting
rules helps you write concise, efficient
code and avoid confusing shape errors.

**Example.**
Subtracting the column mean from a matrix:

```python
import numpy as np

# 3 rows, 4 columns
matrix = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])

# Column means: shape (4,)
col_means = matrix.mean(axis=0)

# Broadcasting: (3,4) - (4,) works
centered = matrix - col_means
print(centered)
```

NumPy automatically "broadcasts" the
1D mean array across all 3 rows.

Broadcasting rules:

- Dimensions are compared from right to left
- Dimensions must be equal or one of them
  must be 1
- Missing dimensions are treated as 1

---

### `Generator`

**Definition.**
A generator is a Python function that yields
values one at a time instead of returning
them all at once. It uses the `yield`
keyword instead of `return` and produces
values lazily (only when asked for).

**Context.**
Generators are essential when working with
large datasets that do not fit in memory.
Instead of loading a million records into a
list, a generator processes them one at a
time, keeping memory usage constant. They
are used in data pipelines, batch processing,
and streaming applications throughout data
science.

**Example.**
```python
def read_large_file(filepath):
    """Yield lines one at a time."""
    with open(filepath, 'r') as f:
        for line in f:
            yield line.strip()

# This never loads the whole file into memory
for line in read_large_file('big_data.csv'):
    process(line)
```

Compare with the non-generator approach:

```python
# BAD: loads entire file into memory
lines = open('big_data.csv').readlines()
```

You can also use generator expressions:

```python
# Generator expression (lazy, low memory)
squares = (x**2 for x in range(1_000_000))

# List comprehension (eager, high memory)
squares = [x**2 for x in range(1_000_000)]
```

---

### `Memory Profiling`

**Definition.**
Memory profiling is the process of measuring
how much RAM your Python program uses and
identifying which parts of the code consume
the most memory. It helps you find memory
leaks and optimize data processing.

**Context.**
Data science code often processes large
datasets, and running out of memory is a
common failure mode. Memory profiling helps
you understand where memory is being used
so you can optimize (e.g., switching dtypes,
using generators, or processing in chunks).
It is especially important before deploying
code to production where memory is limited.

**Example.**
Using the `memory_profiler` package:

```python
# Install: pip install memory-profiler

from memory_profiler import profile

@profile
def process_data():
    import pandas as pd
    df = pd.read_csv('large_file.csv')
    result = df.groupby('category').sum()
    return result

process_data()
```

Running this prints a line-by-line memory
report showing how much memory each line
uses. You can also run it from the command
line:

```bash
python -m memory_profiler my_script.py
```

Quick memory check without a library:

```python
import sys

data = [1, 2, 3, 4, 5]
print(sys.getsizeof(data))  # bytes
```

---

### `dtype Optimization`

**Definition.**
dtype optimization means choosing the
smallest appropriate data type for each
column in your dataset. For example, using
`int8` instead of `int64` when values only
range from 0 to 100.

**Context.**
By default, Pandas uses 64-bit types for
numbers, which wastes memory when smaller
types would suffice. A column of ages (0-120)
stored as int64 uses 8 bytes per value, but
int8 uses only 1 byte. For a dataset with
millions of rows, this difference adds up
to gigabytes. dtype optimization is a quick
win for reducing memory usage.

**Example.**
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': np.random.randint(0, 100, 1_000_000),
    'score': np.random.random(1_000_000),
    'category': np.random.choice(
        ['A', 'B', 'C'], 1_000_000
    )
})

print(f"Before: {df.memory_usage().sum():,} bytes")

# Optimize integer column
df['age'] = df['age'].astype('int8')

# Optimize float column
df['score'] = df['score'].astype('float32')

# Optimize string column to category
df['category'] = df['category'].astype('category')

print(f"After: {df.memory_usage().sum():,} bytes")
# Typically 60-80% memory reduction
```

Common dtype choices:

- `int8`: -128 to 127
- `int16`: -32,768 to 32,767
- `int32`: up to ~2 billion
- `float32`: sufficient for most ML tasks
- `category`: best for low-cardinality strings

---

### `Time Complexity`

**Definition.**
Time complexity describes how the runtime
of an algorithm grows as the input size
increases. It is expressed using Big O
notation, such as O(n), O(n log n), or
O(n squared).

**Context.**
Understanding time complexity helps you
predict whether your code will be fast
enough for production data. A function
that works fine on 1,000 rows might be
impossibly slow on 10 million rows if it
has O(n squared) complexity. Data scientists
need this awareness to choose efficient
algorithms and data structures, especially
when building pipelines that must scale.

**Example.**
Common complexities (n = number of items):

- **O(1):** Constant time.
  Dictionary lookup. Always fast.
- **O(n):** Linear.
  Single pass through a list.
- **O(n log n):** Log-linear.
  Efficient sorting (merge sort).
- **O(n squared):** Quadratic.
  Nested loops. Becomes very slow.

```python
# O(n) - linear search
def find_item(lst, target):
    for item in lst:
        if item == target:
            return True
    return False

# O(1) - dictionary lookup
def find_item_fast(dct, target):
    return target in dct
```

For 1 million items:

- O(n) does ~1,000,000 operations
- O(n squared) does ~1,000,000,000,000

---

### `Multiprocessing`

**Definition.**
Multiprocessing is a technique that runs
multiple Python processes in parallel, each
with its own memory space. This bypasses
Python's Global Interpreter Lock (GIL),
allowing true parallel execution on
multi-core CPUs.

**Context.**
Python's GIL prevents threads from running
Python code simultaneously. Multiprocessing
solves this by using separate processes
instead of threads. It is useful for
CPU-bound data science tasks like feature
engineering on large datasets, running
multiple model trainings, or processing
files in parallel. For I/O-bound tasks
(network requests, file reads), threading
or asyncio may be more appropriate.

**Example.**
```python
from multiprocessing import Pool
import numpy as np

def process_chunk(chunk):
    """CPU-intensive processing."""
    return np.mean(chunk ** 2)

# Split data into chunks
data = np.random.rand(1_000_000)
chunks = np.array_split(data, 4)

# Process 4 chunks in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_chunk, chunks)

print(results)
```

Key points:

- Each process has its own memory
- Data must be serialized between processes
- Start with `Pool` for simple parallelism
- Use `joblib` for scikit-learn workflows

---

### `Virtual Environment`

**Definition.**
A virtual environment is an isolated Python
installation that has its own set of
packages, separate from the system Python
and other projects. Changes to packages in
one environment do not affect others.

**Context.**
Virtual environments prevent dependency
conflicts. Project A might need pandas 1.5
while Project B needs pandas 2.0. Without
virtual environments, installing one would
break the other. Every data science project
should use a virtual environment. It is the
first step in making your work reproducible.

**Example.**
Creating and using a virtual environment:

```bash
# Create a new virtual environment
python -m venv myproject_env

# Activate it (Linux/Mac)
source myproject_env/bin/activate

# Activate it (Windows)
myproject_env\Scripts\activate

# Install packages (isolated to this env)
pip install pandas numpy scikit-learn

# See what is installed
pip list

# Save dependencies
pip freeze > requirements.txt

# Deactivate when done
deactivate
```

Common alternatives:

- `venv`: built into Python (recommended)
- `conda`: includes non-Python packages
- `virtualenv`: third-party, more features

---

### `Dependency Pinning`

**Definition.**
Dependency pinning means specifying exact
version numbers for every package your
project uses. Instead of saying "install
pandas," you say "install pandas==2.1.4."

**Context.**
Without pinning, someone installing your
project tomorrow might get a newer version
of a library that behaves differently or
has breaking changes. Pinning ensures that
everyone working on the project (and
production servers) use the exact same
versions. This is critical for
reproducibility in data science, where even
minor version changes can alter model
outputs.

**Example.**
Unpinned (risky):

```
# requirements.txt
pandas
scikit-learn
numpy
```

Pinned (safe):

```
# requirements.txt
pandas==2.1.4
scikit-learn==1.3.2
numpy==1.26.2
```

Workflow for pinning:

```bash
# Install the packages you need
pip install pandas scikit-learn numpy

# Generate pinned requirements
pip freeze > requirements.txt

# Reproduce the exact environment
pip install -r requirements.txt
```

Modern alternatives:

- `pip-compile` (from pip-tools)
- `poetry lock`
- `uv lock`

---

### `NumPy Array`

**Definition.**
A NumPy array (ndarray) is a fixed-size,
homogeneous (all elements same type),
multi-dimensional container for numerical
data. It is the fundamental data structure
for numerical computing in Python.

**Context.**
NumPy arrays are the foundation that Pandas,
scikit-learn, TensorFlow, and nearly every
Python data science library is built on.
They are dramatically faster than Python
lists for numerical operations because they
store data in contiguous memory blocks and
perform operations in compiled C code.
Understanding arrays is essential for
efficient data science work.

**Example.**
```python
import numpy as np

# Create arrays
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3, 4))       # 3x4 of zeros
c = np.arange(0, 10, 0.5)  # 0 to 9.5

# Vectorized math
result = a * 2 + 1  # [3, 5, 7, 9, 11]

# Slicing
first_three = a[:3]  # [1, 2, 3]

# Shape and dtype
print(a.shape)  # (5,)
print(a.dtype)  # int64

# Reshape
matrix = np.arange(12).reshape(3, 4)
```

Key properties:

- Fixed size after creation
- All elements must be the same type
- Supports multi-dimensional indexing
- Much faster than Python lists for math

---

### `Pandas DataFrame`

**Definition.**
A Pandas DataFrame is a two-dimensional,
labeled data structure with columns of
potentially different types. Think of it
as a spreadsheet or SQL table in Python.

**Context.**
The DataFrame is the most-used data
structure in data science. Almost every
data science workflow starts with loading
data into a DataFrame, exploring it,
cleaning it, and transforming it. Pandas
provides hundreds of methods for filtering,
grouping, merging, and reshaping data. It
is built on top of NumPy arrays.

**Example.**
```python
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Carol'],
    'age': [30, 25, 35],
    'score': [88.5, 92.0, 79.3]
})

# Load from CSV
df = pd.read_csv('data.csv')

# Explore
print(df.shape)        # (rows, columns)
print(df.dtypes)       # column types
print(df.describe())   # summary stats

# Filter
adults = df[df['age'] >= 18]

# Group and aggregate
avg_by_group = df.groupby('name')['score'].mean()

# Add new column
df['passed'] = df['score'] >= 80
```

Common operations:

- `df.head()` — first 5 rows
- `df.info()` — column types and nulls
- `df.value_counts()` — frequency counts
- `df.merge()` — join two DataFrames
- `df.to_csv()` — export to CSV

---

### `List Comprehension`

**Definition.**
A list comprehension is a concise Python
syntax for creating a new list by applying
an expression to each item in an iterable,
optionally filtering items with a condition.
It replaces multi-line for-loops with a
single readable line.

**Context.**
List comprehensions are a Pythonic way to
transform and filter data. They are faster
than equivalent for-loops because Python
optimizes them internally. In data science,
you use them for quick data transformations,
building feature lists, filtering file names,
and many other small tasks. However, for
numerical work on large arrays, vectorized
NumPy/Pandas operations are preferred.

**Example.**
```python
# Basic: square each number
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With filter: only even squares
even_sq = [x**2 for x in range(10) if x % 2 == 0]
# [0, 4, 16, 36, 64]

# Transform strings
names = ['alice', 'bob', 'carol']
upper = [n.upper() for n in names]
# ['ALICE', 'BOB', 'CAROL']

# Nested: flatten a list of lists
nested = [[1, 2], [3, 4], [5, 6]]
flat = [x for sublist in nested for x in sublist]
# [1, 2, 3, 4, 5, 6]

# Dictionary comprehension
scores = {'alice': 90, 'bob': 75, 'carol': 88}
passed = {k: v for k, v in scores.items()
          if v >= 80}
# {'alice': 90, 'carol': 88}
```

---

### `Lambda Function`

**Definition.**
A lambda function is a small, anonymous
(unnamed) function defined in a single line
using the `lambda` keyword. It can take any
number of arguments but contains only one
expression.

**Context.**
Lambda functions are frequently used in
Pandas operations like `apply()`, `map()`,
and `sort()` where you need a quick
throwaway function. They keep your code
concise when the function logic is simple.
For anything complex, use a regular `def`
function instead.

**Example.**
```python
# Basic lambda
double = lambda x: x * 2
print(double(5))  # 10

# Sorting with lambda
students = [
    ('Alice', 90),
    ('Bob', 75),
    ('Carol', 88)
]
students.sort(key=lambda s: s[1])
# Sorted by score

# Pandas apply with lambda
import pandas as pd

df = pd.DataFrame({'price': [10, 20, 30]})
df['with_tax'] = df['price'].apply(
    lambda x: x * 1.08
)

# Filter with lambda
high_prices = df[
    df['price'].apply(lambda x: x > 15)
]
```

When to use lambda vs def:

- **Lambda:** one-line, used once, simple
- **Def:** multi-line, reusable, complex
- **Rule:** if a lambda is hard to read,
  use a regular function instead

---

## See Also

- [Statistical Foundations](./01_statistical_foundations.md)
- [Databases and Data Engineering](./03_databases_and_data_engineering.md)
- [Data Quality and Validation](./04_data_quality_and_validation.md)

---

> **Author** — Simon Parris | Data Science Reference Library
