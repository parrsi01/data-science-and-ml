# CHANGELOG

## 2026-02-22 — Initial institutional module refactor and packaging

### 1) What changed

- Added full package layout under `algorithm_marl_xgboost/src/` with reusable modules for config, data partitioning, topologies, MARL agents, local XGBoost training, decentralized aggregation, rewards, traffic/energy simulation, baselines, and experiment runner.
- Added deterministic experiment config (`configs/experiment.yaml`), tests, docs, cheatsheets, JSONL logging, and artifact/report generation.
- Added integration hooks to existing lab layers:
  - structured JSONL logging style
  - drift utilities from `src/evaluation`
  - optional data-quality metric snapshots from `src/data_quality`

### 2) What was reused

- Existing repository dependencies and ML stack (XGBoost, matplotlib, pandas/numpy).
- Existing lab conventions for artifact directories, deterministic seeds, and test patterns.
- Existing drift utility logic (`src/evaluation/drift.py`) and quality metric helpers (`src/data_quality/quality_metrics.py`) via imports/wrappers.

### 3) What was intentionally left untouched

- Existing root-level project modules and prior prompt artifacts outside `algorithm_marl_xgboost/`.
- Existing local modifications in `reports/scaling/benchmark_results.json` and `reports/scaling/benchmark_results.md`.
- Existing `algorithm_marl_xgboost/.gitkeep` file (preserved).

### 4) Risks introduced

- XGBoost behavior can vary slightly across versions/architectures despite fixed seeds.
- MARL reward dynamics are simplified (feature-importance update abstraction rather than parameter-sharing/gradient exchange).
- Synthetic data is a controllable proxy and not a substitute for validated institutional datasets (e.g., UNSW-NB15 preprocessing pipeline).

## 2026-02-22 — Parameter study harness, repeats, plotting, and significance testing

### 1) What changed

- Added `src/experiments/` harness modules for repeated runs, parameter studies, significance testing, plotting, and summary report generation.
- Added `configs/parameter_study.yaml` for institutional-style parameter sweeps and repeat settings.
- Added experiment docs and cheatsheet for running/rebuilding studies offline.
- Added lightweight tests for parameter grid generation, significance testing, and plotting artifact creation.

### 2) What reused

- Reused `algorithm_marl_xgboost/src/run_experiment.py` as the core execution engine for each repeat.
- Reused existing artifact/report formats from Prompt 14 runs and baseline outputs.
- Reused SciPy/Matplotlib already present in the lab stack.

### 3) What was intentionally left untouched

- Core MARL+XGBoost training/aggregation logic in `algorithm_marl_xgboost/src/` (only wrapped, not replaced).
- Existing Prompt 14 reports and logs (parameter-study outputs are written to separate directories).
- Unrelated repository modules and existing scaling report local changes.

### 4) Risks introduced

- Full grid (`144` settings x `10` repeats) is computationally expensive; accidental full runs may take a long time.
- Statistical comparisons can be misread if repeat counts are too small or study slices are selectively reported.
- Quick/demo mode changes runtime settings and should not be treated as final research evidence.
