# Overall Architecture

Author: Simon Parris  
Date: 2026-02-22

## Purpose

This diagram summarizes the institutional data and AI workflow implemented in this repository, from ingestion through monitoring and reporting.

## ASCII Architecture

```text
                +---------------------------+
                |   Data Sources / Feeds    |
                | Postgres | CSV | Synthetic |
                +------------+--------------+
                             |
                             v
+----------------+    +----------------------+    +-------------------+
| Data Pipeline   | -> | Data Validation &    | -> | Feature Engineering|
| SQL / ETL / ELT |    | Quality Gates (JSONL)|    | + Preprocessing    |
+----------------+    +----------------------+    +-------------------+
                             |                           |
                             v                           v
                    +------------------+         +----------------------+
                    | ML Training       | <-----> | Evaluation Suite     |
                    | Core + Advanced   |         | CV / Threshold / Bias|
                    +---------+---------+         +----------+-----------+
                              |                              |
                              v                              v
                    +------------------+           +---------------------+
                    | Artifacts         |           | Drift & Stability   |
                    | Models / Metrics  |           | Snapshots / Alerts  |
                    +---------+---------+           +----------+----------+
                              |                               |
                              +---------------+---------------+
                                              |
                                              v
                                  +-------------------------+
                                  | Operational ML Layer     |
                                  | Anomaly System + FastAPI |
                                  +------------+------------+
                                               |
                                               v
                                  +-------------------------+
                                  | Reports / Docs / Audit   |
                                  | Executive + Research     |
                                  +-------------------------+
```

## Key Design Notes

- Quality checks run before production-style inference.
- Evaluation and drift analysis are separate from training to preserve auditability.
- The MARL + XGBoost algorithm is isolated in `algorithm_marl_xgboost/` but reuses lab concepts (quality, drift, evaluation rigor).
- All major modules emit artifacts to `reports/` for offline review.
