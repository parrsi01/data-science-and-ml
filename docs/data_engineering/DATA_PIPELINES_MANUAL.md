# Data Engineering Foundations (PostgreSQL + Pipelines)

- Author: Simon Parris
- Date: 2026-02-22

## What is a schema (simple definition)

A schema is the organized structure of a database: tables, columns, types, and rules. It defines what data is allowed and how systems interpret it consistently.

## Why constraints matter (auditability)

Constraints enforce data quality at the database level (for example, no negative passenger counts). This supports auditability because invalid records are rejected before they can corrupt downstream analysis.

## Why indexing matters (performance)

Indexes make frequent lookups and filters much faster by helping the database find rows without scanning the whole table. In institutional workflows, this improves response time for dashboards, operations, and investigations.

## How this maps to UN/IATA/CERN needs

- UN/humanitarian workflows: shipment status and priority tracking across regions
- IATA/aviation workflows: flight delays, routes, and fuel efficiency analysis
- CERN/scientific workflows: event logs, rare event rates, and time-based analysis
- Shared requirement: reproducible ingestion, validation, query logic, and documented controls

## How to rebuild and run the pipeline without AI

1. Install PostgreSQL locally and start the service
2. Create local role/database (`ds_user` / `institutional_lab`)
3. Copy `configs/db.env.example` to `configs/db.env` and adjust credentials if needed
4. Install Python dependencies (`sqlalchemy`, `psycopg2-binary`, `pytest`)
5. Run `make db-init` to create schema and indexes
6. Run `make ingest` to load synthetic datasets
7. Run `make queries` to review example outputs
8. Run `python3 -m pytest -q tests/data_engineering/test_pipeline.py`

## JIRA-Style Ticket Examples

### DE-101: Build Institutional PostgreSQL Schema and Validation Baseline

- Type: Story
- Goal: Establish audited schema + constraints for flights, humanitarian shipments, and CERN-style events.
- Acceptance Criteria:
  - `scripts/sql/001_create_schema.sql` creates all tables and indexes
  - Validation module enforces required columns, types, and ranges
  - Tests skip gracefully when DB is unavailable

### DE-102: Add Reproducible Ingestion and Query Examples

- Type: Task
- Goal: Generate synthetic data, ingest in batches, and provide institution-style query examples (DB + offline reference fallback).
- Acceptance Criteria:
  - Ingestion outputs row counts by table
  - Query examples include delay, backlog, rare event, and quality checks
  - Offline query fallback is runnable for reference work
