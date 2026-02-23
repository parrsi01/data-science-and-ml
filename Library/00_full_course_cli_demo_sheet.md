# Data Science and AI Engineering Full CLI Demo Sheet (Master)

Author: Simon Parris + additive Codex library extension
Date: 2026-02-23
Mode: Full CLI execution + GitHub mobile reading support

This is a single, ordered demo sheet for running the Data Science and AI Engineering course from the CLI.

It focuses on executable commands plus explanation of what each command proves.

## Before You Start (CLI Session Setup)

```bash
cd /home/sp/cyber-course/projects/datascience
pwd
git status --short
python3 --version
```

What you are doing: confirming repo context, local changes, and Python availability before starting.

## Recommended Two-Terminal Workflow

1. Terminal A: run commands.
2. Terminal B: read manuals (`docs/`) or `Library/` notes.
3. After each lesson, write a short evidence note (command -> output -> meaning).

## Core Setup Demo (Run First)

```bash
cd /home/sp/cyber-course/projects/datascience
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pytest -q tests algorithm_marl_xgboost/tests
```

What this proves: the local environment can run the core and algorithm test suites.

## Lesson Demo Order (1-19)

### Lesson 1 Demo - Project Manual

```bash
sed -n '1,220p' docs/PROJECT_MANUAL.md
make venv
make install
make test
```

What this proves: you can follow the repository operating manual and run the baseline workflow.

### Lesson 2 Demo - Core Concepts

```bash
sed -n '1,240p' docs/CORE_CONCEPTS.md
rg -n 'drift|threshold|feature|pipeline|bias|validation' docs/CORE_CONCEPTS.md src tests | head -40
```

What this proves: concepts in the manual map to real code and tests.

### Lesson 3 Demo - Foundations Guide

```bash
sed -n '1,240p' docs/foundations/FOUNDATIONS_GUIDE.md
python3 - <<'PY'
print('Foundations demo: CLI, Python, and reproducible execution path are available.')
PY
```

What this proves: the foundations reading is tied to actual executable context in the repo.

### Lesson 4 Demo - Statistics Manual

```bash
sed -n '1,240p' docs/statistics/STATISTICS_MANUAL.md
sed -n '1,160p' Library/01_statistical_foundations.md
```

What this proves: you can pair canonical statistics guidance with deeper library definitions.

### Lesson 5 Demo - Data Pipelines Manual

```bash
sed -n '1,240p' docs/data_engineering/DATA_PIPELINES_MANUAL.md
make db-init
make ingest
make queries
```

What this proves: schema initialization, ingestion, and query examples run as a pipeline sequence.

### Lesson 6 Demo - Data Quality Manual

```bash
sed -n '1,220p' docs/data_quality/DATA_QUALITY_MANUAL.md
make quality
```

What this proves: quality gates can be run as a distinct stage before ML training.

### Lesson 7 Demo - ML Core Manual

```bash
sed -n '1,240p' docs/ml_core/ML_CORE_MANUAL.md
make ml-train
ls -la reports/ml_core 2>/dev/null || true
ls -la models/ml_core 2>/dev/null || true
```

What this proves: the baseline training pipeline executes and generates artifacts.

### Lesson 8 Demo - ML Advanced Manual

```bash
sed -n '1,240p' docs/ml_advanced/ML_ADVANCED_MANUAL.md
make ml-adv-train
ls -la reports/ml_advanced 2>/dev/null || true
```

What this proves: advanced training/tuning workflow produces outputs separate from baseline.

### Lesson 9 Demo - Evaluation Manual

```bash
sed -n '1,240p' docs/evaluation/EVALUATION_MANUAL.md
make eval-suite
ls -la reports/evaluation 2>/dev/null || true
```

What this proves: evaluation runs as a dedicated stage with report outputs.

### Lesson 10 Demo - Scaling Manual

```bash
sed -n '1,240p' docs/scaling/SCALING_MANUAL.md
make scale-generate
make scale-mp
make scale-bench
```

What this proves: scaling workflows and benchmark generation can be executed from the CLI.

### Lesson 11 Demo - Humanitarian Optimization Project

```bash
sed -n '1,240p' docs/projects/HUMANITARIAN_OPTIMIZATION_MANUAL.md
make project-humanitarian
ls -la reports/projects/humanitarian_optimization 2>/dev/null || true
```

What this proves: optimization project execution produces domain-specific outputs and reports.

### Lesson 12 Demo - Air Traffic Delay Project

```bash
sed -n '1,240p' docs/projects/AIR_TRAFFIC_DELAY_MANUAL.md
make project-air-traffic
ls -la reports/projects/air_traffic_delay 2>/dev/null || true
```

What this proves: aviation analytics/forecasting workflows can be run and inspected via generated artifacts.

### Lesson 13 Demo - Ops Anomaly System Project

```bash
sed -n '1,240p' docs/projects/OPS_ANOMALY_SYSTEM_MANUAL.md
make project-ops-system
ls -la reports/projects/ops_anomaly_system 2>/dev/null || true
```

What this proves: operational anomaly pipeline execution and outputs are wired end-to-end.

Optional dashboard run:

```bash
make project-ops-dashboard
```

What this proves: the operational interface layer can be launched locally.

### Lesson 14 Demo - Architecture Docs

```bash
sed -n '1,220p' docs/architecture/overall_architecture.md
sed -n '1,220p' docs/architecture/marl_architecture.md
```

What this proves: you can explain system flow and experiment architecture from CLI-readable docs.

### Lesson 15 Demo - Algorithm Manual (MARL + XGBoost)

```bash
sed -n '1,240p' algorithm_marl_xgboost/docs/ALGORITHM_MANUAL.md
make algo-run
```

What this proves: the research algorithm execution path is runnable from a controlled config.

### Lesson 16 Demo - Parameter Study Manual

```bash
sed -n '1,240p' algorithm_marl_xgboost/docs/experiments/PARAMETER_STUDY_MANUAL.md
make algo-study
ls -la algorithm_marl_xgboost/reports/parameter_study 2>/dev/null || true
```

What this proves: parameter studies can be executed and inspected as reproducible experiments.

### Lesson 17 Demo - Simplified Concepts + Math Intuition

```bash
sed -n '1,220p' algorithm_marl_xgboost/docs/CONCEPTS_SIMPLIFIED.md
sed -n '1,220p' algorithm_marl_xgboost/docs/MATH_INTUITION.md
```

What this proves: advanced concepts can be reviewed in plain-language and intuition-first formats.

### Lesson 18 Demo - Reproducibility / Threat Model / Protocol

```bash
sed -n '1,220p' algorithm_marl_xgboost/docs/REPRODUCIBILITY_AND_AUDIT.md
sed -n '1,220p' algorithm_marl_xgboost/docs/THREAT_MODEL_AND_LIMITS.md
sed -n '1,220p' algorithm_marl_xgboost/docs/EXPERIMENT_PROTOCOL.md
```

What this proves: experiment governance and validity constraints are documented and reviewable.

### Lesson 19 Demo - Portfolio Mapping / Status Docs

```bash
sed -n '1,220p' docs/PORTFOLIO_SKILL_MAPPING.md
sed -n '1,220p' docs/CV_READY_SUMMARY.md
sed -n '1,220p' docs/REPOSITORY_STATUS_REPORT.md
```

What this proves: repository outputs can be translated into capability evidence for portfolio/review use.

## Full-Course CLI Run Loop (Condensed)

If you want one pass through the major executable stages:

```bash
cd /home/sp/cyber-course/projects/datascience
source venv/bin/activate
pytest -q tests algorithm_marl_xgboost/tests
make db-init ingest quality
make ml-train
make ml-adv-train
make eval-suite
make scale-generate scale-mp scale-bench
make project-humanitarian project-air-traffic project-ops-system
make algo-run
make algo-study
```

What this proves: the repo works as an integrated engineering course, not only isolated documents.

## Interpreting Outputs (What To Record)

For each lesson/demo, record:

- command run
- key output or artifact path
- meaning of the result
- failure mode to watch for
- next command you would run if it failed

## Notes About "Tickets" in This Repo

This repo does not have a dedicated ticket drill library like the DevOps repo.

Use issue-pattern drills instead:

- setup/environment failure (`venv`, dependency install, tests)
- data pipeline failure (`db-init`, `ingest`, `queries`, `quality`)
- training failure (`ml-train`, `ml-adv-train`)
- evaluation/report failure (`eval-suite`, report dirs)
- scaling/benchmark failure (`scale-*`)
- project pipeline failure (`project-*`)
- experiment/research failure (`algo-*`)

## Cross-References

- `Library/00_full_course_q_and_a_sheet.md`
- `docs/LESSON_EXECUTION_COMPANION.md`
- `docs/LESSON_RESEARCH_ANALYSIS_COMPANION.md`
- `README.md`
