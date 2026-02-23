# Scaling and Distributed Processing

---

> **Field** — High-Performance Computing, Data Engineering
> **Scope** — Processing large datasets efficiently using
> Dask, Parquet, chunking, profiling, and parallel
> execution strategies

---

## Overview

When datasets grow too large to fit in memory or take
too long to process on a single core, you need scaling
strategies. This topic covers tools and techniques for
processing data that exceeds the capacity of a single
machine or a single pandas DataFrame. The core idea is
to break work into smaller pieces and process them
efficiently, either in parallel or sequentially with
minimal memory usage.

---

## Definitions

### `Dask`

**Definition.**
Dask is a Python library for parallel and distributed
computing. It provides familiar pandas-like and
NumPy-like interfaces but breaks work into smaller
tasks that can run on multiple CPU cores or across
multiple machines. It processes data that does not fit
in memory by working on it in chunks.

**Context.**
Dask is the most common "next step" when pandas runs
out of memory or takes too long. You can often replace
`import pandas as pd` with `import dask.dataframe as dd`
and keep most of your code the same. Dask builds a task
graph of operations and only computes results when you
explicitly ask for them (lazy evaluation).

**Example.**
```python
import dask.dataframe as dd

# Read a large CSV that won't fit in memory
df = dd.read_csv("huge_file_*.csv")

# Operations look like pandas
result = (
    df.groupby("category")["value"]
    .mean()
)

# Nothing has computed yet (lazy)
# Trigger computation:
output = result.compute()
print(output)
```

Install with:

```bash
pip install dask[complete]
```

---

### `Parquet`

**Definition.**
Parquet is a columnar file format designed for
efficient storage and retrieval of large datasets.
Unlike CSV, which stores data row by row, Parquet
stores data column by column. This makes it much
faster for analytical queries that only need a few
columns from a wide table.

**Context.**
Parquet is the standard file format for data science
at scale. It compresses data typically 5-10x smaller
than CSV, reads 10-100x faster for column-selective
queries, and preserves data types (no more guessing
whether a column is a string or integer). Use Parquet
whenever you are saving data for later analysis.

**Example.**
```python
import pandas as pd

# Save as Parquet
df = pd.DataFrame({
    "id": range(1000000),
    "value": range(1000000),
    "category": ["A", "B"] * 500000
})
df.to_parquet("data.parquet")

# Read back (much faster than CSV)
df2 = pd.read_parquet("data.parquet")

# Read only specific columns
df3 = pd.read_parquet(
    "data.parquet",
    columns=["id", "value"]
)
```

With Dask:

```python
import dask.dataframe as dd

# Dask + Parquet = fast large-scale reads
df = dd.read_parquet(
    "data_partitioned/",
    columns=["value"]
)
```

---

### `Profiling`

**Definition.**
Profiling means measuring where your code spends its
time and memory. A profiler shows you exactly which
functions are slow, which lines allocate the most
memory, and where the bottlenecks are. It replaces
guessing with measurement.

**Context.**
Before optimizing, you must profile. Developers often
guess wrong about what is slow. A profiler might
reveal that 90% of your runtime is in one unexpected
function call. Profiling saves you from wasting time
optimizing code that does not matter.

**Example.**
Time profiling with cProfile:

```python
import cProfile

def process_data():
    data = list(range(1000000))
    total = sum(x ** 2 for x in data)
    return total

cProfile.run("process_data()")
```

Memory profiling with memory_profiler:

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def load_data():
    import pandas as pd
    df = pd.read_csv("big_file.csv")
    return df.describe()

load_data()
```

Line-by-line timing with line_profiler:

```bash
pip install line_profiler
kernprof -l -v script.py
```

---

### `Chunking`

**Definition.**
Chunking means processing a large dataset in small,
manageable pieces instead of loading it all at once.
You read a chunk, process it, save or accumulate the
result, then move to the next chunk. This keeps
memory usage constant regardless of data size.

**Context.**
Chunking is the simplest scaling strategy. It requires
no special libraries, just a loop. It works with any
file format and any processing logic. The trade-off
is that it is harder to do operations that need the
full dataset at once (like sorting or joins).

**Example.**
```python
import pandas as pd

# Process a large CSV in chunks of 10,000
totals = []
for chunk in pd.read_csv(
    "huge_file.csv", chunksize=10_000
):
    chunk_total = chunk["value"].sum()
    totals.append(chunk_total)

grand_total = sum(totals)
print(f"Total: {grand_total}")
```

Chunking for aggregation:

```python
import pandas as pd

counts = pd.Series(dtype=int)
for chunk in pd.read_csv(
    "huge.csv", chunksize=50_000
):
    c = chunk["category"].value_counts()
    counts = counts.add(c, fill_value=0)

print(counts.sort_values(ascending=False))
```

---

### `Deterministic Output`

**Definition.**
Deterministic output means that running the same code
on the same data always produces exactly the same
result, byte for byte. There is no randomness or
order-dependent variation in the output.

**Context.**
Determinism is critical for reproducible data science.
When processing data in parallel, the order of results
can vary between runs. You must sort or otherwise
normalize the output to ensure consistency. This
matters for testing, auditing, and comparing results
across experiments.

**Example.**
Non-deterministic (parallel order varies):

```python
import dask.dataframe as dd

df = dd.read_parquet("data/")
result = df.groupby("key").sum().compute()
# Row order may differ between runs
```

Made deterministic:

```python
import dask.dataframe as dd

df = dd.read_parquet("data/")
result = (
    df.groupby("key").sum()
    .compute()
    .sort_index()
    .reset_index()
)
# Row order is now consistent
```

Always sort output DataFrames by a stable key
before saving or comparing results.

---

### `Worker Pool`

**Definition.**
A worker pool is a group of processes or threads that
wait for tasks and execute them in parallel. Instead
of processing items one at a time, you distribute them
across multiple workers who run simultaneously.

**Context.**
Worker pools are the building block of parallel
processing. Python's Global Interpreter Lock (GIL)
means threads cannot run CPU-bound code in parallel,
so multiprocessing (separate processes) is used for
computation. Dask manages worker pools automatically,
but you can also create them manually.

**Example.**
Using Python's multiprocessing:

```python
from multiprocessing import Pool

def square(x):
    return x ** 2

with Pool(processes=4) as pool:
    results = pool.map(square, range(100))

print(results[:10])
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

Using concurrent.futures:

```python
from concurrent.futures import (
    ProcessPoolExecutor
)

def process_chunk(filename):
    import pandas as pd
    df = pd.read_csv(filename)
    return df["value"].sum()

files = ["chunk_1.csv", "chunk_2.csv",
         "chunk_3.csv", "chunk_4.csv"]

with ProcessPoolExecutor(max_workers=4) as ex:
    totals = list(ex.map(process_chunk, files))

print(f"Grand total: {sum(totals)}")
```

---

### `Benchmark`

**Definition.**
A benchmark is a standardized test that measures the
performance of code, hardware, or a system. It
produces quantitative metrics like execution time,
memory usage, or throughput that you can compare
across different approaches.

**Context.**
Benchmarks let you make data-driven decisions about
which approach to use. "Is Dask faster than pandas
for this workload?" is answered by running benchmarks,
not by guessing. Always benchmark on realistic data
sizes and representative workloads, not toy examples.

**Example.**
Simple timing benchmark:

```python
import time
import pandas as pd

def benchmark(func, *args, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg = sum(times) / len(times)
    print(f"  Mean: {avg:.3f}s "
          f"(over {repeats} runs)")
    return avg

# Compare two approaches
print("pandas read_csv:")
benchmark(pd.read_csv, "data.csv")

print("pandas read_parquet:")
benchmark(pd.read_parquet, "data.parquet")
```

Using pytest-benchmark for automated benchmarking:

```bash
pip install pytest-benchmark
pytest test_perf.py --benchmark-only
```

---

### `Out-of-core Processing`

**Definition.**
Out-of-core processing means working with data that
is too large to fit in RAM. The data stays on disk,
and only small portions are loaded into memory at a
time. The program streams through the data without
ever holding it all at once.

**Context.**
When your dataset is 50 GB and you have 16 GB of
RAM, you cannot use standard pandas. Out-of-core
tools like Dask, Vaex, or chunked pandas handle this
automatically. The key insight is that most operations
(filtering, aggregation, column transforms) do not
need all data in memory simultaneously.

**Example.**
Out-of-core with Dask:

```python
import dask.dataframe as dd

# This does NOT load the data into memory
df = dd.read_parquet("huge_dataset/")

# Filter, transform, aggregate
result = (
    df[df["status"] == "active"]
    .groupby("region")["revenue"]
    .sum()
    .compute()  # only now data is processed
)
```

Out-of-core with chunked pandas:

```python
import pandas as pd

# Process 50 GB file with 500 MB chunks
reader = pd.read_csv(
    "massive.csv", chunksize=100_000
)
results = []
for chunk in reader:
    filtered = chunk[chunk["score"] > 80]
    results.append(
        filtered.groupby("dept")["score"].mean()
    )

final = pd.concat(results).groupby(level=0).mean()
```

---

### `Lazy Evaluation`

**Definition.**
Lazy evaluation means that operations are recorded
but not executed until you explicitly ask for the
result. Instead of computing immediately, the system
builds a plan (task graph) of what needs to happen.
Execution only occurs when you call `.compute()` or
a similar trigger method.

**Context.**
Lazy evaluation is how Dask achieves its efficiency.
By recording all operations before executing them,
Dask can optimize the execution plan: eliminating
redundant work, combining steps, and minimizing memory
usage. It also means mistakes are caught at execution
time, not definition time.

**Example.**
```python
import dask.dataframe as dd

df = dd.read_parquet("data/")

# These lines build the task graph
# but do NOT execute anything
filtered = df[df["value"] > 100]
grouped = filtered.groupby("key").mean()
top_10 = grouped.nlargest(10, "value")

# Nothing has happened yet!
print(type(top_10))
# <class 'dask.dataframe.core.DataFrame'>

# NOW it executes the entire pipeline
result = top_10.compute()
print(type(result))
# <class 'pandas.core.frame.DataFrame'>
```

The benefit: if you add more operations before
calling `.compute()`, Dask optimizes them all
together instead of running each step separately.

---

### `Task Graph`

**Definition.**
A task graph is a directed acyclic graph (DAG) that
represents the computational steps needed to produce
a result. Each node is a task (like "read partition 3"
or "compute mean"), and edges show dependencies (which
tasks must finish before others can start).

**Context.**
Dask builds a task graph from your code, then executes
it efficiently. Visualizing the task graph helps you
understand what Dask is actually doing and identify
bottlenecks. A wide task graph means lots of
parallelism; a tall, narrow graph means mostly
sequential work.

**Example.**
```python
import dask.dataframe as dd

df = dd.read_parquet("data/")
result = df.groupby("key")["value"].mean()

# Visualize the task graph
result.visualize(filename="task_graph.png")
```

You can also inspect the graph structure:

```python
import dask

# See the raw task graph dictionary
graph = dict(result.__dask_graph__())
print(f"Number of tasks: {len(graph)}")

# Each key is a task, each value is
# (function, *arguments)
```

Install graphviz for visualization:

```bash
pip install graphviz
sudo apt install graphviz
```

---

### `Partitioning`

**Definition.**
Partitioning means splitting a large dataset into
smaller, independent pieces (partitions) that can be
processed separately. Each partition is a subset of
the data, small enough to fit in memory. The way you
partition (by row count, by a key column, by date)
affects performance significantly.

**Context.**
Good partitioning is the key to efficient parallel
processing. Dask partitions DataFrames automatically
when reading Parquet or CSV files. Each partition
becomes a separate task in the task graph. Too few
partitions means not enough parallelism; too many
means excessive overhead.

**Example.**
```python
import dask.dataframe as dd

# Dask automatically partitions on read
df = dd.read_parquet("data/")
print(f"Partitions: {df.npartitions}")

# Repartition to a specific number
df2 = df.repartition(npartitions=20)

# Partition by a column (for grouped ops)
# Save with partitioning for fast queries
df.to_parquet(
    "output/",
    partition_on=["year", "month"]
)
# Creates directory structure:
# output/year=2024/month=01/part.parquet
# output/year=2024/month=02/part.parquet
# ...
```

Rule of thumb: aim for partitions of 50-200 MB
each. A 10 GB dataset should have roughly 50-200
partitions.

---

### `Memory-mapped Files`

**Definition.**
A memory-mapped file is a file on disk that the
operating system treats as if it were in memory. The
OS loads portions of the file into RAM on demand, as
you access them. Your code reads from the file as if
it were an array in memory, but the OS manages what
is actually loaded.

**Context.**
Memory mapping is useful when you need random access
to a large file (not just sequential reading). NumPy
supports memory-mapped arrays natively. It is faster
than reading chunks manually because the OS optimizes
which portions to keep in memory based on your access
patterns.

**Example.**
NumPy memory-mapped array:

```python
import numpy as np

# Create a large array on disk
data = np.random.rand(10_000_000)
np.save("big_array.npy", data)

# Memory-map it (does NOT load into RAM)
mmap = np.load(
    "big_array.npy", mmap_mode="r"
)

# Access elements as if in memory
print(mmap[0:5])      # loads just these 5
print(mmap[999999])   # loads just this one
print(mmap.mean())    # streams through data

# Memory usage stays low regardless
# of array size
```

Modes:
- `"r"` = read-only
- `"r+"` = read-write (modifies file)
- `"c"` = copy-on-write (changes not saved)

---

### `Horizontal Scaling`

**Definition.**
Horizontal scaling means adding more machines to
handle a larger workload. Instead of making one
computer more powerful, you distribute the work across
many computers. Each machine handles a portion of the
data or computation.

**Context.**
Horizontal scaling is how cloud platforms handle
massive datasets. Services like Dask distributed,
Spark, and Kubernetes let you add worker machines on
demand. The advantage is near-unlimited capacity; the
disadvantage is the complexity of coordinating work
across machines and handling network communication.

**Example.**
Dask distributed with multiple workers:

```python
from dask.distributed import Client

# Connect to a cluster of machines
client = Client(
    "scheduler-address:8786"
)
print(client)

# Now Dask operations run across the cluster
import dask.dataframe as dd
df = dd.read_parquet("s3://bucket/data/")
result = df.groupby("key").mean().compute()
```

Setting up a local cluster for development:

```python
from dask.distributed import (
    Client, LocalCluster
)

cluster = LocalCluster(
    n_workers=4,
    threads_per_worker=2,
    memory_limit="4GB"
)
client = Client(cluster)
print(client.dashboard_link)
```

---

### `Vertical Scaling`

**Definition.**
Vertical scaling means making a single machine more
powerful by adding more RAM, faster CPUs, or faster
storage. Instead of adding more machines, you upgrade
the one you have. All your code runs on one machine,
but with more resources.

**Context.**
Vertical scaling is the simplest approach: no code
changes needed, just bigger hardware. Cloud providers
offer machines with up to several terabytes of RAM.
However, there are physical limits to how large a
single machine can be, and costs increase rapidly.
Vertical scaling is often the right first step before
investing in distributed computing complexity.

**Example.**
When to choose vertical vs horizontal:

```
Vertical scaling (scale up):
- Dataset: 20 GB, machine has 8 GB RAM
- Solution: upgrade to 64 GB RAM machine
- Pro: no code changes, simple
- Con: expensive, has upper limits

Horizontal scaling (scale out):
- Dataset: 2 TB, no single machine is enough
- Solution: distribute across 20 machines
- Pro: virtually unlimited capacity
- Con: code changes needed, more complex
```

In practice, try this progression:
1. Optimize code (profiling, vectorization)
2. Use efficient formats (Parquet)
3. Vertical scaling (more RAM)
4. Horizontal scaling (Dask distributed)

---

## See Also

- [Python and Numerical Computing](./02_python_and_numerical_computing.md)
- [Time Series and Forecasting](./10_time_series_and_forecasting.md)
- [Anomaly Detection and Operational ML](./12_anomaly_detection_and_operational_ml.md)
- [Reproducibility and Governance](./14_reproducibility_and_governance.md)

---

> **Author** — Simon Parris | Data Science Reference Library
