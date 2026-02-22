# Experiment Protocol

## How To Run Repeats

Run the experiment multiple times while changing the seed in `algorithm_marl_xgboost/configs/experiment.yaml` (or generate config copies).

Recommended repeat protocol:
- 5 to 10 seeds
- same topology and communication budget
- same data size and anomaly rate

## Metrics To Report

Primary:
- F1
- Precision
- Recall

Secondary:
- Accuracy
- ROC-AUC
- Total bytes sent
- Average latency
- Packet loss mean
- Simulated energy total

## How To Interpret Improvements

- An improvement is stronger if it appears across multiple seeds, not just one run.
- Small F1 gains may not be worth much if communication/energy costs increase heavily.
- Gains should be compared against both baselines:
  - local-only XGBoost
  - naive decentralized uniform aggregation

## How To Compare To Baselines

1. Use the same seed and same partitions
2. Use the same local XGBoost hyperparameters
3. Report identical metrics
4. Keep communication budget explicit
5. Include traffic/energy outputs in the comparison table

