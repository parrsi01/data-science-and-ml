# Institutional Data Quality & Validation

- Author: Simon Parris
- Date: 2026-02-22

## What “data quality” means (simple)

Data quality means the data is complete enough, valid enough, and consistent enough for the decision or analysis being performed. High quality data reduces the risk of incorrect conclusions and operational failures.

## Why quality gates matter in institutions

Quality gates are explicit pass/fail checks that stop bad data from moving further into reporting, analytics, or model pipelines. Institutions use them to support accountability, auditability, and reliable operations.

## What JSONL logs are and why used

JSON Lines (`.jsonl`) stores one JSON object per line. It is easy to stream, append, parse by machines, and inspect with command-line tools, making it useful for pipeline logs and audit trails.

## How to run and interpret reports

1. Run `make quality` (or `PYTHONPATH=src python3 -m data_quality.run_quality_gate`)
2. Review CLI PASS/FAIL summary
3. Open `reports/data_quality/summary_quality_report.json`
4. Review per-dataset reports:
   - `<dataset>_quality_report.json`
   - `<dataset>_invalid_rows.csv`
5. Check `logs/data_quality.jsonl` for structured event history

## How to rebuild without AI

1. Create `src/data_quality`, `tests/data_quality`, `docs/data_quality`, and `reports/data_quality`
2. Add JSONL logging config (`get_logger`)
3. Define schema models (Pydantic or equivalent validators)
4. Implement row-level schema validation and domain rule validation
5. Compute missing, duplicate, outlier, and drift metrics
6. Write report artifacts (JSON + CSV)
7. Build a CLI quality gate runner with thresholds and non-zero exit on failure
8. Add tests for invalid-row detection, metrics keys, and gate failure
9. Run `python3 -m pytest -q tests/data_quality/test_quality_gate.py`
