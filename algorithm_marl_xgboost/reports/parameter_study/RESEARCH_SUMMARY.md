# Research Summary — Parameter Study (MARL + XGBoost)

## Experimental Protocol

- Repeated seeded runs were executed per configuration setting.
- Metrics include performance (F1 primary) and traffic/energy costs.
- Baselines (local-only, naive decentralized) were recorded alongside MARL trust-weighted results.

## Repeats Details

- Total repeat rows: 12
- Total unique parameter settings: 6

## Significance Test Outcomes

```json
{
  "config": {
    "test": "wilcoxon",
    "alpha": 0.05
  },
  "comparisons": {
    "f1_marl_vs_naive": {
      "test_requested": "wilcoxon",
      "test_used": "wilcoxon",
      "n_a": 12,
      "n_b": 12,
      "mean_a": 0.8751393534002229,
      "mean_b": 0.9385093167701862,
      "statistic": 0.0,
      "p_value": 0.00048828125,
      "effect_size": -1.0,
      "paired": true,
      "significant_at_alpha": true,
      "interpretation": "statistically significant difference detected"
    },
    "energy_marl_vs_naive": {
      "test_requested": "wilcoxon",
      "test_used": "wilcoxon",
      "n_a": 12,
      "n_b": 12,
      "mean_a": 21.34138030284116,
      "mean_b": 14.622946030941634,
      "statistic": 0.0,
      "p_value": 0.00048828125,
      "effect_size": 1.0,
      "paired": true,
      "significant_at_alpha": true,
      "interpretation": "statistically significant difference detected"
    },
    "bandwidth_marl_vs_naive": {
      "test_requested": "wilcoxon",
      "test_used": "wilcoxon",
      "n_a": 12,
      "n_b": 12,
      "mean_a": 1508.0,
      "mean_b": 936.0,
      "statistic": 0.0,
      "p_value": 0.00048828125,
      "effect_size": 1.0,
      "paired": true,
      "significant_at_alpha": true,
      "interpretation": "statistically significant difference detected"
    }
  },
  "artifacts": {
    "significance_tests_json": "algorithm_marl_xgboost/reports/parameter_study/significance_tests.json"
  }
}
```

## Limitations

- Synthetic data is a proxy and may not capture real institutional network traffic behavior.
- The update abstraction uses feature-importance vectors, not full model parameters or secure aggregation.
- Significance outcomes depend on repeat count and the selected parameter subset.

## Reproducibility

1. Keep study config YAML and experiment base config fixed
2. Preserve repeat artifacts and aggregated CSV outputs
3. Regenerate plots and summaries from saved CSV/JSON outputs

## How to Reproduce

```bash
venv/bin/python -m algorithm_marl_xgboost.src.experiments.parameter_study \
  --config algorithm_marl_xgboost/configs/parameter_study.yaml
```
