# Institutional Data & AI Engineering Lab

"A reproducible institutional-grade data science and AI engineering laboratory environment designed for aviation, humanitarian, and scientific data systems."

This repository provides a professional, offline-readable, GitHub-ready project scaffold for data science and machine learning work aligned with institutional expectations (UN/IATA/CERN-style reproducibility, governance, and auditability).

## Tech Stack

- Python (`venv`-based local environment; current workspace uses `python3`)
- Scientific/ML libraries: NumPy, Pandas, SciPy, scikit-learn, XGBoost/LightGBM/CatBoost
- Visualization: Matplotlib, Seaborn, Plotly
- MLOps/API: MLflow, FastAPI, Uvicorn
- Data systems: SQLAlchemy, Redis, Dask
- Quality: Black, Flake8, MyPy, Pytest, pre-commit

## Folder Structure

- `docs/`: Project manual and core concepts for institutional operations
- `cheatsheets/`: Offline quick references (Linux, Git, Python, Statistics)
- `src/`: Source code package(s)
- `tests/`: Automated tests
- `models/`: Trained model artifacts (tracked selectively)
- `datasets/`: Data placeholders / metadata (no sensitive data committed)
- `notebooks/`: Exploratory and reporting notebooks
- `reports/`: Analysis outputs and stakeholder reports
- `configs/`: Runtime and experiment configuration files
- `docker/`: Container assets
- `mlops/`: MLOps workflows / CI-CD assets
- `algorithm_marl_xgboost/`: Dedicated research/algorithm work area

## Environment Setup (Current Folder)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.in
pip freeze > requirements.txt
```

Note: The original setup prompt specified `python3.11`. In this workspace, `python3.11` is not available, so the local scaffold uses the installed `python3`.

## How To Run Tests

```bash
source venv/bin/activate
pytest -q
```

or

```bash
make test
```

## Data Pipeline Make Targets

The repository includes reproducible PostgreSQL pipeline targets for Prompt 4:

- `make venv`: create local virtual environment
- `make install`: install dependencies from `requirements.in`
- `make db-init`: run `scripts/sql/001_create_schema.sql` against `DATABASE_URL`
- `make ingest`: generate and ingest synthetic records into PostgreSQL
- `make queries`: run institution-style SQL query examples and print outputs
- `make quality`: run institutional data quality gates, JSONL logging, and quality reports
- `make all`: run `db-init`, `ingest`, `quality`, and `test` in sequence
- `make scale-generate`: generate and chunk-process a large synthetic dataset
- `make scale-mp`: run deterministic multiprocessing CSV processing
- `make scale-dask`: run Dask processing (or offline-safe fallback) and write parquet output
- `make scale-bench`: run scaling benchmark and write JSON/Markdown reports
- `make ml-train`: run the config-driven ML training pipeline and save models/metrics/plots
- `make ml-report`: print top-line ML metrics from `reports/ml_core/metrics.json`
- `make ml-clean`: remove ML artifacts (`models/ml_core`, `reports/ml_core`)
- `make ml-adv-train`: run advanced XGBoost training with imbalance handling, Optuna tuning, and SHAP reports
- `make ml-adv-explain`: rerun advanced training/explainability pipeline (same command path)
- `make ml-adv-clean`: remove advanced ML artifacts (`models/ml_advanced`, `reports/ml_advanced`)
- `make ml-adv-all`: run advanced training + explainability + advanced ML tests
- `make test`: run the pytest suite
- `make clean`: remove common local test caches

For database usage, create `configs/db.env` from `configs/db.env.example` (do not commit credentials).

## How To Contribute

- Create a feature branch from `main`
- Keep changes reproducible and documented
- Add/update tests for behavioral changes
- Run formatting/linting/tests before opening a pull request
- Avoid committing secrets, credentials, or sensitive datasets
