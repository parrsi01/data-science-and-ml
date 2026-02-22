# Experiment Runner Cheatsheet (MARL + XGBoost Parameter Study)

## Commands

Quick demo run (recommended for local validation):

```bash
venv/bin/python -m algorithm_marl_xgboost.src.experiments.parameter_study \
  --config algorithm_marl_xgboost/configs/parameter_study.yaml \
  --max-combinations 6 \
  --repeats-override 2 \
  --rounds-override 2 \
  --quick
```

Full study (can be very long):

```bash
venv/bin/python -m algorithm_marl_xgboost.src.experiments.parameter_study \
  --config algorithm_marl_xgboost/configs/parameter_study.yaml
```

Plots only:

```bash
venv/bin/python -m algorithm_marl_xgboost.src.experiments.plotting
```

Reports only:

```bash
venv/bin/python -m algorithm_marl_xgboost.src.experiments.reporting
```

## How to read plots

- Boxplots by topology: compare spread/stability across repeats
- Parameter curves: compare trend and uncertainty (error bars)
- F1 vs bandwidth plots: inspect performance-cost tradeoffs

## Interpreting p-values safely

- A small p-value does not prove practical importance
- Check effect size and cost metrics too
- Use paired tests when repeats share seeds/configs

## Common pitfalls

- p-hacking (trying many slices without correction/context)
- data leakage between train/validation partitions
- too few repeats for stable conclusions
- comparing configurations with different rounds or budgets unfairly

