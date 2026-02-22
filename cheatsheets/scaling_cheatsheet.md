# Scaling Cheatsheet (Institutional Data/AI Lab)

## Multiprocessing Basics

- Use `multiprocessing.Pool` for CPU-bound independent tasks
- Keep worker functions top-level (picklable)
- Preserve chunk order if deterministic output is required
- Avoid tiny chunks (overhead can exceed compute time)

## Dask Basics

- `dask.dataframe` supports parallel dataframe-style operations
- Good for larger-than-memory or partitioned processing workflows
- Use `.compute()` to materialize results
- Prefer explicit output formats (Parquet) for repeatable analytics

## Parquet vs CSV

- CSV: human-readable, portable, simple
- Parquet: columnar, compressed, faster analytics reads, typed columns
- Prefer Parquet for repeated query/report pipelines
- Prefer CSV for manual inspection and quick interchange

## Profiling Commands

```bash
PYTHONPATH=src python3 -m scaling.benchmark
python3 -m pytest -q tests/scaling/test_scaling.py
/usr/bin/time -v python3 -m scaling.benchmark   # if available
```
