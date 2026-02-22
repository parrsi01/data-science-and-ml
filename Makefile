PYTHON ?= python3
VENV ?= venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: venv install freeze db-init ingest queries quality all scale-generate scale-mp scale-dask scale-bench ml-train ml-report ml-clean test lint format tree clean

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

lint:
	$(VENV)/bin/flake8 src tests
	$(VENV)/bin/mypy src

format:
	$(VENV)/bin/black src tests

tree:
	find . -maxdepth 3 -type d | sort

clean:
	rm -rf .pytest_cache .mypy_cache htmlcov
