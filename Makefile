PYTHON ?= python3
VENV ?= venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: venv install freeze db-init ingest queries quality all scale-generate scale-mp scale-dask scale-bench ml-train ml-report ml-clean ml-adv-train ml-adv-explain ml-adv-clean ml-adv-all eval-suite eval-clean project-humanitarian project-humanitarian-clean project-air-traffic project-air-traffic-clean project-ops-system project-ops-dashboard project-ops-clean test lint format tree clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install:
	$(PIP) install -r requirements.in

freeze:
	$(PIP) freeze > requirements.txt

test:
	$(PY) -m pytest -q

db-init:
	PYTHONPATH=src $(PYTHON) -m data_engineering.schema_init

ingest:
	PYTHONPATH=src $(PYTHON) -m data_engineering.ingest

queries:
	PYTHONPATH=src $(PYTHON) -m data_engineering.query_examples

quality:
	PYTHONPATH=src $(PYTHON) -m data_quality.run_quality_gate

all: db-init ingest quality test

scale-generate:
	PYTHONPATH=src $(PYTHON) -m scaling.chunking

scale-mp:
	PYTHONPATH=src $(PYTHON) -m scaling.multiprocessing_jobs

scale-dask:
	PYTHONPATH=src $(PYTHON) -m scaling.dask_jobs

scale-bench:
	PYTHONPATH=src $(PYTHON) -m scaling.benchmark

ml-train:
	$(PY) -m src.ml_core.train --config configs/ml_core/config.yaml

ml-report:
	PYTHONPATH=src $(PY) -m ml_core.report

ml-clean:
	rm -rf models/ml_core reports/ml_core

ml-adv-train:
	$(PY) -m src.ml_advanced.train_advanced --config configs/ml_advanced/config.yaml

ml-adv-explain:
	$(PY) -m src.ml_advanced.train_advanced --config configs/ml_advanced/config.yaml

ml-adv-clean:
	rm -rf models/ml_advanced reports/ml_advanced

ml-adv-all: ml-adv-train
	$(PY) -m pytest -q tests/ml_advanced/test_ml_advanced.py

eval-suite:
	$(PY) -m src.evaluation.run_evaluation_suite --config configs/ml_advanced/config.yaml

eval-clean:
	rm -rf reports/evaluation

project-humanitarian:
	$(PY) -m src.projects.humanitarian_optimization.run_project --config configs/projects/humanitarian_optimization.yaml

project-humanitarian-clean:
	rm -rf reports/projects/humanitarian_optimization datasets/humanitarian_demand.csv

project-air-traffic:
	$(PY) -m src.projects.air_traffic_delay.run_project --config configs/projects/air_traffic_delay.yaml

project-air-traffic-clean:
	rm -rf reports/projects/air_traffic_delay models/air_traffic_delay datasets/air_traffic_flights.csv datasets/air_traffic_routes.csv

project-ops-system:
	$(PY) -m src.projects.ops_anomaly_system.run_system --config configs/projects/ops_anomaly_system.yaml

project-ops-dashboard:
	$(PY) -m src.projects.ops_anomaly_system.dashboard --config configs/projects/ops_anomaly_system.yaml

project-ops-clean:
	rm -rf reports/projects/ops_anomaly_system models/ops_anomaly_system

lint:
	$(VENV)/bin/flake8 src tests
	$(VENV)/bin/mypy src

format:
	$(VENV)/bin/black src tests

tree:
	find . -maxdepth 3 -type d | sort

clean:
	rm -rf .pytest_cache .mypy_cache htmlcov
