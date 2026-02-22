# Operational Anomaly Detection System

- Author: Simon Parris
- Date: 2026-02-22

## What operational ML means

Operational ML is using models inside a live workflow to support day-to-day decisions, monitoring, and triage. It focuses on reliability, observability, and safe failure behavior.

## Why quality + drift must precede inference

If incoming data is malformed or significantly shifted from training conditions, predictions can become unreliable. Quality and drift checks reduce the risk of acting on bad model outputs.

## Why dashboards must show uncertainty

Operators need to know whether the system is healthy and whether outputs are trustworthy. A dashboard should show status flags and context, not only a single score.

## Institutional mapping

- UN: logistics monitoring for shipments, routing disruptions, and priority anomalies
- IATA: operations monitoring for flight delays, congestion, and schedule risk
- CERN: detector monitoring for unusual event patterns and data quality shifts

## How to rebuild without AI

1. Define a YAML config for data source, thresholds, model path, and dashboard host/port
2. Implement a Postgres loader with deterministic ordering and fallback handling
3. Reuse schema/domain validation and quality metrics modules for incoming batches
4. Load a trained model bundle and run inference with a reproducible threshold
5. Compare recent features vs training reference with drift checks
6. Save snapshots (quality, inference, drift, system state) as JSON/CSV artifacts
7. Create a minimal FastAPI dashboard for `/health`, `/metrics`, and `/top-anomalies`
8. Add tests with mock DataFrames and FastAPI TestClient
9. Run the system CLI and confirm dashboard endpoints respond

## JIRA-Style Ticket Examples

### OPSML-101: Build Postgres-to-Inference Operational Monitoring Pipeline

- Type: Story
- Goal: Create a production-style operational pipeline that loads recent records, validates quality, runs anomaly inference, and saves auditable artifacts.
- Acceptance Criteria:
  - `quality_snapshot.json`, `anomaly_results.csv`, `inference_metrics.json`, and `drift_snapshot.json` are generated
  - System summary prints rows processed, anomaly rate, drift flag, and quality flag
  - Postgres failures trigger deterministic local fallback without crashing

### OPSML-102: Add Minimal FastAPI Monitoring Dashboard

- Type: Task
- Goal: Expose core health and anomaly metrics for operations triage via simple HTTP endpoints and HTML page.
- Acceptance Criteria:
  - `/health`, `/metrics`, `/top-anomalies`, and `/` return 200
  - Dashboard reads latest system artifacts from output directory
  - Metrics include anomaly rate, drift flag, and quality flag

