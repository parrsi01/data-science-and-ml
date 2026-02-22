# Repository Status Report

Author: Simon Parris  
Date: 2026-02-22

## Snapshot

- Total Python modules (`src/` + `algorithm_marl_xgboost/src/`): 86
- Total test files (`tests/` + `algorithm_marl_xgboost/tests/`): 14
- Total Python LOC (approx., source + tests): 11,339 lines
- Primary documentation surfaces: README, module manuals, architecture docs, offline index, algorithm research docs

## Supported Use Cases

- Institutional data engineering workflows (schema design, ingestion, validation, SQL querying)
- Statistical training and scientific-thinking exercises (distributions, hypothesis tests, Monte Carlo)
- ML classification pipelines with imbalance handling, tuning, explainability, and evaluation
- Operational ML monitoring (quality gates, drift checks, anomaly scoring, FastAPI dashboard endpoints)
- Domain projects: humanitarian logistics optimization and air traffic flow / delay forecasting
- Distributed/scaling demonstrations (chunking, multiprocessing, Dask-style processing, benchmarks)
- Decentralized MARL + XGBoost anomaly detection experiments with repeatable parameter studies

## Repository Hygiene Checks

- Large CSV files > 10 MB committed: none identified in project datasets/reports (large binaries found only under local `venv/`, which is ignored)
- CI workflow present: `.github/workflows/ci.yml`
- License present: `LICENSE` (MIT)
- Offline documentation entry point: `docs/OFFLINE_INDEX.md`

## Known Limitations

- Live PostgreSQL pipeline execution requires local PostgreSQL install/service access; sandboxed environments may block `sudo` and `systemctl`.
- Some project workflows rely on optional dependencies (for example, Prophet) and use fallbacks when unavailable.
- Full MARL parameter-study grid (`10 repeats x full grid`) is implemented but runtime-intensive; quick subsets are preferable for routine checks.
- Root `pytest` config targets `tests/`; algorithm module tests must be included explicitly (handled in CI workflow and documented commands).
- Existing local benchmark report edits under `reports/scaling/` may be unrelated to the latest changes and should be reviewed before release tagging.

## Future Work Roadmap

1. Add containerized local services (PostgreSQL + Redis) for one-command reproducible ops demos.
2. Add dataset versioning / artifact metadata (hashes, schema versions, provenance manifests).
3. Expand CI into staged jobs (lint, tests, docs checks, smoke project runs).
4. Add security/compliance checks (dependency scanning, secret scanning, SBOM generation).
5. Extend MARL module with malicious-node simulations, privacy-preserving variants, and repeatable benchmark presets.
6. Publish release bundles (docs + key reports + frozen configs) for portfolio snapshots.
