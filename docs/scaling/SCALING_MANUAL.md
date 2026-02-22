# Distributed Processing & Scaling for Institutional Data

- Author: Simon Parris
- Date: 2026-02-22

## What “out-of-core” means (simple)

Out-of-core processing means working on data in smaller chunks instead of loading the entire dataset into memory at once. This allows larger workloads to run on limited hardware.

## When multiprocessing helps vs hurts

- Helps when work can be split into independent CPU-heavy tasks
- Helps when single-thread processing becomes the bottleneck
- Hurts when chunks are tiny (overhead dominates)
- Hurts when work is mostly I/O-bound or requires heavy object serialization

## What Dask is (simple)

Dask is a Python library for parallel and larger-than-memory data processing. It can scale familiar dataframe-style workflows from a laptop to larger systems.

## Determinism and reproducibility in parallel systems

Parallel systems can produce inconsistent results if task order, seeds, or reductions are not controlled. This module uses stable chunk ordering and explicit seeds so outputs remain reproducible across runs.

## How to rebuild and rerun without AI

1. Create `src/scaling`, `tests/scaling`, `docs/scaling`, and `reports/scaling`
2. Implement chunked generation and chunk transforms in `chunking.py`
3. Implement deterministic multiprocessing pipeline in `multiprocessing_jobs.py`
4. Implement Dask processing path plus fallback in `dask_jobs.py`
5. Add profiling utilities for time and memory snapshots
6. Build a benchmark runner that writes JSON + Markdown reports
7. Add Makefile targets (`scale-generate`, `scale-mp`, `scale-dask`, `scale-bench`)
8. Run `python3 -m pytest -q tests/scaling/test_scaling.py`
9. Run `make scale-bench` and review `reports/scaling/benchmark_results.*`

## JIRA-Style Ticket Examples

### SCALING-101: Implement Deterministic Chunk Processing Baseline

- Type: Story
- Goal: Generate and process a large synthetic institutional dataset in fixed chunks with reproducible outputs.
- Acceptance Criteria:
  - Writes `datasets/large_synthetic.csv`
  - Writes `datasets/large_synthetic_processed.csv`
  - Row count preserved after chunk processing
  - Feature columns include rolling mean, ratio, anomaly flag

### SCALING-102: Add Parallel Processing Benchmark (Multiprocessing + Dask/Fallback)

- Type: Task
- Goal: Compare baseline chunking vs multiprocessing vs Dask/fallback and publish offline-readable results.
- Acceptance Criteria:
  - Benchmark writes `reports/scaling/benchmark_results.json`
  - Benchmark writes `reports/scaling/benchmark_results.md`
  - Output includes wall time, memory snapshots, output row counts
  - Dask path documents parquet usage and fallback behavior
