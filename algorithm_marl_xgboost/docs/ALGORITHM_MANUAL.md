# Decentralized Federated Anomaly Detection using MARL + XGBoost

- Author: Simon Parris
- Date: 2026-02-22

## System Overview

This module implements a decentralized federated anomaly detection workflow where each agent trains a local XGBoost classifier, exchanges compact updates with peers, and uses a MARL-inspired policy to decide communication partners over rounds.

It exists to support institutional settings that need:
- anomaly detection under distributed ownership
- auditable communication/energy tradeoffs
- reproducible experiments with logs and per-round artifacts

## Components

- MARL Agent (`agents.py`): chooses neighbors using epsilon-greedy exploration + trust scores
- Local XGBoost (`xgb_local.py`): trains per-agent anomaly classifier and returns metrics + feature-importance update vector
- Topology (`topologies.py`): defines who can communicate (ring/star/fully connected/random)
- Aggregation (`decentralized_agg.py`): trust-weighted or uniform peer update aggregation (no central server)
- Reward (`rewards.py`): balances F1 improvement against communication and energy costs
- Traffic/Energy (`traffic_energy.py`): simulates operational communication metrics per round
- Baselines (`baselines.py`): local-only and naive decentralized comparison runs

## Data Flow (ASCII)

```text
Synthetic/UNSW-like Data
        |
        v
Dirichlet Non-IID Partitioning ---> Agent 1 split/train/val
        |                           Agent 2 split/train/val
        |                           ...
        v
Topology Graph (peer links)
        |
        v
For each round:
  Local XGBoost train -> local metrics + feature-importance update
  MARL agent chooses neighbors (epsilon + trust)
  Peer-to-peer exchange (no server)
  Trust-weighted aggregation of received updates
  Reward = F1 gain - comm penalty - energy penalty
  Update trust + epsilon
        |
        v
Per-round artifacts + JSONL log + final summary + baseline comparison
```

## How To Run

```bash
venv/bin/python -m algorithm_marl_xgboost.src.run_experiment \
  --config algorithm_marl_xgboost/configs/experiment.yaml
```

## How To Reproduce Results

1. Use the same `venv` and `requirements.txt`
2. Keep `algorithm_marl_xgboost/configs/experiment.yaml` unchanged (seeded run)
3. Delete prior artifacts if needed:
   - `algorithm_marl_xgboost/reports/*`
   - `algorithm_marl_xgboost/logs/experiment.jsonl`
4. Re-run the command above
5. Compare:
   - per-round JSON/TXT/PNG files
   - `traffic_metrics_per_round.csv`
   - `final_summary.txt`
   - JSONL logs

