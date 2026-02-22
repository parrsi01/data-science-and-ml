# Reproducibility and Audit

## Seeds and Deterministic Controls

- Primary experiment seed is defined in `configs/experiment.yaml`
- NumPy and Python `random` seeds are set in the runner
- Topology generation uses seeded RNG
- Dirichlet partitioning uses seeded RNG
- Agent decisions use seeded RNG

Note: XGBoost can still show minor variation across versions/platforms.

## Logging Format (JSONL)

- Path: `algorithm_marl_xgboost/logs/experiment.jsonl`
- One JSON object per line
- Stable key ordering used for consistent diffs (`sort_keys=True`)

Typical events:
- `run_start`
- `round_complete`
- `run_complete`

## Artifact Naming Conventions

- Per-round metrics JSON:
  - `algorithm_marl_xgboost/reports/per_round/round_01_metrics.json`
- Per-round text summary:
  - `algorithm_marl_xgboost/reports/per_round/round_01_summary.txt`
- Per-round plots:
  - `algorithm_marl_xgboost/reports/per_round/round_01_plots.png`
- Final summary:
  - `algorithm_marl_xgboost/reports/final_summary.txt`
- Traffic CSV:
  - `algorithm_marl_xgboost/reports/traffic_metrics_per_round.csv`

## Audit Checklist

1. Record exact commit SHA
2. Archive `configs/experiment.yaml`
3. Preserve `experiment.jsonl`
4. Preserve per-round and final artifacts
5. Confirm seed and topology settings
6. Compare baseline results and MARL results from same run
7. Note library versions from `requirements.txt`

