# Institutional Data & AI Engineering Lab

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/parrsi01/data-science-and-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/parrsi01/data-science-and-ml/actions/workflows/ci.yml)

Author: Simon Parris  
Date: 2026-02-22

## 1. Overview

This repository is a reproducible, institutional-grade data science and AI engineering lab designed for aviation, humanitarian, and scientific analytics contexts (UN / IATA / CERN-style expectations).

It combines data engineering, statistical rigor, ML pipelines, explainability, evaluation, operational monitoring, optimization, and decentralized anomaly detection research into one offline-readable portfolio package.

## 2. Architecture Diagram

```text
Data Sources -> Data Pipelines -> Quality Gates -> Feature Engineering -> ML Training
      -> Evaluation / Thresholding / Bias Checks -> Drift Monitoring -> Reporting / Dashboards
      -> Operational ML + Research Experimentation (MARL + XGBoost)
```

Detailed diagrams:
- `docs/architecture/overall_architecture.md`
- `docs/architecture/marl_architecture.md`

## 3. Core Modules

- Data Engineering: PostgreSQL-ready schema, ingestion, validation, and query examples (`src/data_engineering/`)
- ML Core: Config-driven baseline + XGBoost training pipeline (`src/ml_core/`)
- Advanced ML: SMOTE/weights logic, Optuna tuning, SHAP explainability (`src/ml_advanced/`)
- Evaluation Suite: Stability, threshold calibration, bias/group metrics, drift snapshots (`src/evaluation/`)
- Humanitarian Optimization: UN-style linear programming allocation and sensitivity analysis (`src/projects/humanitarian_optimization/`)
- Air Traffic Analytics: IATA-style graph flow metrics, delay modeling, forecasting (`src/projects/air_traffic_delay/`)
- Rare Event Detection: Imbalanced classification pipelines and anomaly-oriented evaluation artifacts (`src/ml_core/`, `src/ml_advanced/`, `src/evaluation/`)
- Operational Anomaly System: Quality + inference + drift + FastAPI dashboard (`src/projects/ops_anomaly_system/`)
- Decentralized MARL + XGBoost Algorithm: Simon Parris research module for decentralized federated anomaly detection (`algorithm_marl_xgboost/`)

## 4. Experiment Framework

The MARL + XGBoost module includes a dedicated experiment harness for repeated runs, parameter studies, plots, and significance testing.

Key paths:
- `algorithm_marl_xgboost/src/experiments/`
- `algorithm_marl_xgboost/configs/parameter_study.yaml`
- `algorithm_marl_xgboost/reports/parameter_study/`
- `algorithm_marl_xgboost/reports/repeats/`

## 5. Reproducibility & Auditability

- Config-driven execution (`configs/`, `algorithm_marl_xgboost/configs/`)
- JSON/CSV/PNG report artifacts stored in versioned directories
- JSONL structured logging for operational quality workflows and experiments
- Seeded synthetic data generation for deterministic reference runs where feasible
- Offline-readable manuals, cheat sheets, and experiment protocol documentation

Start with:
- `docs/PROJECT_MANUAL.md`
- `docs/CORE_CONCEPTS.md`
- `docs/OFFLINE_INDEX.md`
- `algorithm_marl_xgboost/docs/REPRODUCIBILITY_AND_AUDIT.md`

## 6. Tech Stack

- Python (local `venv`; CI targets Python 3.11)
- Data & ML: NumPy, Pandas, SciPy, scikit-learn, XGBoost, Optuna, SHAP
- Data Engineering: SQLAlchemy, PostgreSQL (local dev target), validation utilities
- APIs / Ops: FastAPI, Uvicorn, JSONL logging
- Analytics / Stats: statsmodels, Monte Carlo tools, hypothesis testing utilities
- Scaling: multiprocessing, Dask-style workflows, benchmarking utilities
- Research tooling: repeated experiment harness, significance testing, plotting

## 7. How to Run (Step-by-step)

1. Create environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run tests (root + algorithm module)
```bash
pytest -q tests algorithm_marl_xgboost/tests
```

3. Run core ML pipeline
```bash
make ml-train
```

4. Run advanced ML (tuning + explainability)
```bash
make ml-adv-train
```

5. Run evaluation suite
```bash
make eval-suite
```

6. Run portfolio projects
```bash
make project-humanitarian
make project-air-traffic
make project-ops-system
```

7. Run Simon Parris algorithm (MARL + XGBoost)
```bash
make algo-run
make algo-study
```

Note: PostgreSQL live runs require local PostgreSQL installation and service access. The repo also contains offline-safe fallbacks for reference execution in restricted environments.

## 8. Institutional Skill Mapping

This repository demonstrates institution-relevant capabilities across data engineering, ML, distributed systems, scientific rigor, and operational ML.

- Full mapping: `docs/PORTFOLIO_SKILL_MAPPING.md`
- CV-ready snippets: `docs/CV_READY_SUMMARY.md`

## 9. Directory Structure

```text
configs/                     Runtime and experiment configurations
cheatsheets/                 Offline quick references (lab-wide)
docs/                        Project manuals, architecture docs, portfolio docs
src/                         Core lab source modules (DE/ML/eval/projects)
tests/                       Core lab pytest suite
models/                      Saved model artifacts
datasets/                    Synthetic/reference datasets
reports/                     Generated reports and plots
algorithm_marl_xgboost/      Research module, experiments, docs, reports, tests
.github/workflows/           CI pipeline definitions
```

## 10. License (MIT)

This repository is licensed under the MIT License. See `LICENSE`.
