# Offline Documentation Index

Author: Simon Parris  
Date: 2026-02-22

This index is the offline reading map for the entire repository. It is designed for cold-start onboarding, audit review, and rebuild workflows without AI assistance.

## Recommended Reading Order

1. `README.md` — repository overview, architecture summary, run commands, and module index.
2. `docs/PROJECT_MANUAL.md` — institutional lab setup and reproducibility rules.
3. `docs/CORE_CONCEPTS.md` — simplified shared vocabulary used across modules.
4. `docs/architecture/overall_architecture.md` — end-to-end data/ML/monitoring flow.
5. `docs/architecture/marl_architecture.md` — decentralized MARL + XGBoost module flow.
6. Domain/module manuals (choose by focus area; list below).
7. `docs/PORTFOLIO_SKILL_MAPPING.md` — capability mapping and CV-ready wording.
8. `docs/REPOSITORY_STATUS_REPORT.md` — current status, coverage, and known limitations.

## Manual and Guide Catalog (Root `docs/`)

- `docs/PROJECT_MANUAL.md`
- `docs/foundations/FOUNDATIONS_GUIDE.md`
- `docs/statistics/STATISTICS_MANUAL.md`
- `docs/data_engineering/DATA_PIPELINES_MANUAL.md`
- `docs/data_quality/DATA_QUALITY_MANUAL.md`
- `docs/scaling/SCALING_MANUAL.md`
- `docs/ml_core/ML_CORE_MANUAL.md`
- `docs/ml_advanced/ML_ADVANCED_MANUAL.md`
- `docs/evaluation/EVALUATION_MANUAL.md`
- `docs/projects/HUMANITARIAN_OPTIMIZATION_MANUAL.md`
- `docs/projects/AIR_TRAFFIC_DELAY_MANUAL.md`
- `docs/projects/OPS_ANOMALY_SYSTEM_MANUAL.md`

## Manual and Guide Catalog (`algorithm_marl_xgboost/docs/`)

- `algorithm_marl_xgboost/docs/ALGORITHM_MANUAL.md`
- `algorithm_marl_xgboost/docs/experiments/PARAMETER_STUDY_MANUAL.md`
- `algorithm_marl_xgboost/docs/CONCEPTS_SIMPLIFIED.md`
- `algorithm_marl_xgboost/docs/MATH_INTUITION.md`
- `algorithm_marl_xgboost/docs/REPRODUCIBILITY_AND_AUDIT.md`
- `algorithm_marl_xgboost/docs/THREAT_MODEL_AND_LIMITS.md`
- `algorithm_marl_xgboost/docs/EXPERIMENT_PROTOCOL.md`

## Rebuild the Entire Lab Without AI (High-Level)

1. Create and activate a virtual environment (`python3 -m venv venv && source venv/bin/activate`).
2. Install dependencies from `requirements.txt` (or `requirements.in` if regenerating).
3. Review `configs/` and `algorithm_marl_xgboost/configs/` for reproducible settings.
4. Run `make test` and targeted module commands (ML, evaluation, projects, algorithm).
5. Generate reports/artifacts using documented Make targets and module CLIs.
6. Record outputs in version control with changelog updates for reproducibility.

## Cold Start Checklist

- [ ] Read `README.md`
- [ ] Read `docs/PROJECT_MANUAL.md`
- [ ] Read `docs/CORE_CONCEPTS.md`
- [ ] Confirm local Python + `venv` setup
- [ ] Install dependencies from `requirements.txt`
- [ ] Run `pytest` (root + algorithm tests)
- [ ] Run one pipeline (`make ml-train`) and one project (`make project-air-traffic`)
- [ ] Run `make algo-run` or `make algo-study` for the MARL + XGBoost section
- [ ] Review generated artifacts in `reports/` and `algorithm_marl_xgboost/reports/`
- [ ] Review `docs/REPOSITORY_STATUS_REPORT.md` for limitations and next steps

## Companion Notes (Slow-Learning Add-on)

- `docs/LESSON_EXECUTION_COMPANION.md` - lesson-by-lesson what/why/do/evidence/stop guide
- `docs/LESSON_RESEARCH_ANALYSIS_COMPANION.md` - beginner definitions + analyst-style reading prompts
