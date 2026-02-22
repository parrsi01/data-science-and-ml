PYTHON ?= python3
VENV ?= venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: venv install freeze db-init ingest queries test lint format tree clean

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

lint:
	$(VENV)/bin/flake8 src tests
	$(VENV)/bin/mypy src

format:
	$(VENV)/bin/black src tests

tree:
	find . -maxdepth 3 -type d | sort

clean:
	rm -rf .pytest_cache .mypy_cache htmlcov
