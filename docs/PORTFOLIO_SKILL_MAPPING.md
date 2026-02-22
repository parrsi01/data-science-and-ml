# Portfolio Skill Mapping

Author: Simon Parris  
Date: 2026-02-22

This document maps repository components to institution-relevant technical skills and provides CV-ready statements grounded in implemented work.

## Data Engineering

Mapped modules:
- `src/data_engineering/` (schema creation, ingestion, querying, validation)
- `scripts/sql/001_create_schema.sql`
- `src/data_quality/` (quality metrics, validation gates, JSONL logs)

Skills demonstrated:
- PostgreSQL schema design, constraints, indexing, query patterns
- ETL/ELT pipeline structuring with reproducible scripts and Make targets
- Data validation and quality gating before downstream ML consumption
- Structured operational logging for audit-friendly pipelines

## Machine Learning

Mapped modules:
- `src/ml_core/`
- `src/ml_advanced/`
- `src/evaluation/`

Skills demonstrated:
- XGBoost + baseline model training pipelines
- Hyperparameter tuning with Optuna
- Imbalance handling (class weights / SMOTE decision logic)
- Explainability workflows with SHAP artifacts
- Threshold calibration, stability checks, and drift reporting

## Distributed Systems

Mapped modules:
- `src/scaling/`
- `algorithm_marl_xgboost/src/`

Skills demonstrated:
- Chunked/out-of-core processing mindset for larger-than-memory workflows
- Multiprocessing and Dask benchmarking with reproducible performance reports
- Decentralized topology-aware communication patterns (ring/star/random/fully connected)
- Trust-weighted peer aggregation under simulated bandwidth/latency/energy constraints

## Scientific Rigor

Mapped modules:
- `src/statistics/`
- `src/evaluation/`
- `algorithm_marl_xgboost/src/experiments/`
- `algorithm_marl_xgboost/docs/REPRODUCIBILITY_AND_AUDIT.md`

Skills demonstrated:
- Hypothesis testing and confidence intervals with reproducible reporting
- Parameter studies with repeated runs and seed control
- Statistical significance testing (Wilcoxon / t-test pathways)
- Experiment protocols, changelog discipline, and audit-oriented artifacts

## Operational ML

Mapped modules:
- `src/projects/ops_anomaly_system/`
- `src/data_quality/`
- `src/evaluation/drift.py`

Skills demonstrated:
- Operational inference workflow design (quality -> inference -> drift -> dashboard)
- FastAPI metrics/dashboard endpoints for monitoring and triage
- Drift flagging and recent-batch analysis for anomaly systems
- Artifact-based operational reporting for incident review and governance

## CV-Ready Bullet Statements

- Built a reproducible institutional data/AI lab spanning data engineering, ML pipelines, evaluation, optimization, and operational monitoring with offline-readable documentation and test coverage.
- Implemented PostgreSQL-style schema/ingestion/query workflows with validation gates, structured JSONL logging, and audit-oriented quality reports.
- Developed config-driven ML pipelines (XGBoost + baselines) with class imbalance handling, Optuna tuning, SHAP explainability, and artifact persistence.
- Designed an evaluation framework for imbalanced classification including threshold calibration, seed-stability analysis, group metrics, and drift snapshots.
- Engineered distributed/scaling demos covering chunked processing, multiprocessing, Dask-style workflows, and reproducible benchmark reporting.
- Built an operational anomaly detection system with quality checks, inference, drift monitoring, and a FastAPI dashboard interface for live metrics.
- Implemented a decentralized MARL + XGBoost federated anomaly detection research module with topology-aware peer communication and trust-weighted aggregation.
- Added a repeatable experiment harness with parameter studies, statistical significance testing, and executive/research summaries for institutional review.
