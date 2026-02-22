# MARL + XGBoost Architecture

Author: Simon Parris  
Date: 2026-02-22

## Purpose

This diagram documents the decentralized federated anomaly detection workflow used by the `algorithm_marl_xgboost` module.

## ASCII Architecture (Peer-to-Peer)

```text
              Round t (No Central Server)

      +----------------+        +----------------+
      |    Agent i      | <----> |    Agent j      |
      | local data Xi   |        | local data Xj   |
      | local XGBoost   |        | local XGBoost   |
      +-------+--------+        +--------+-------+
              ^                          ^
              |                          |
              v                          v
      +----------------+        +----------------+
      |  Agent k       | <----> |  Agent m       |
      | local data Xk  |        | local data Xm  |
      | local XGBoost  |        | local XGBoost  |
      +----------------+        +----------------+

Per agent loop:
  1) Train local XGBoost on local partition
  2) Build update summary (feature-importance / calibration signal)
  3) MARL policy chooses neighbors (epsilon-greedy + trust)
  4) Peer-to-peer exchange under communication budget
  5) Trust-weighted aggregation of received updates
  6) Compute reward = performance gain - comm/energy cost penalty
  7) Update trust + epsilon -> next round
```

## Control/Data Flow

```text
Local Data -> Local Train -> Local Metrics ----+
                                                |
Topology + Trust + Epsilon -> Neighbor Choice --+--> P2P Exchange
                                                |
Traffic/Energy Simulation ----------------------+--> Reward
                                                |
Aggregated Update ------------------------------+--> Next Round State
```

## Auditability Hooks

- Per-round JSON metrics and text summaries
- JSONL run logs (`algorithm_marl_xgboost/logs/experiment.jsonl`)
- Stable artifact naming under `algorithm_marl_xgboost/reports/per_round/`
- Parameter-study repeat artifacts under `algorithm_marl_xgboost/reports/repeats/`
