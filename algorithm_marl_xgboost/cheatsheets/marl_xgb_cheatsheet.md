# MARL + XGBoost Cheatsheet

## Commands

```bash
venv/bin/python -m algorithm_marl_xgboost.src.run_experiment \
  --config algorithm_marl_xgboost/configs/experiment.yaml

venv/bin/python -m pytest -q algorithm_marl_xgboost/tests/test_algorithm.py
```

## Quick Concepts

- MARL agent: chooses communication neighbors each round
- Update abstraction: compact feature-importance vector + local metrics
- Trust-weighted aggregation: higher-trust peers influence more
- Reward: F1 gain minus communication/energy penalties

## Typical Errors + Fixes

- `ModuleNotFoundError`: run from repo root and use the module command exactly
- XGBoost errors on tiny partitions: reduce `min_pos_per_agent` or increase `n_samples`
- Very slow runs: reduce `rounds`, `n_estimators`, or `n_agents` for development

## Rebuild Without AI (Quick)

1. Recreate package structure under `algorithm_marl_xgboost/`
2. Add config loader, topology builder, synthetic data + partitioning
3. Add local XGBoost training wrapper
4. Add MARL agent policy + aggregation + rewards
5. Add runner + logging + plots + tests

