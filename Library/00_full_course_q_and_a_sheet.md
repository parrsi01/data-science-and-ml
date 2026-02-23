# Data Science and AI Engineering Full Course Q and A Sheet (Master)

Author: Simon Parris + additive Codex library extension
Date: 2026-02-23
Mode: GitHub mobile / CLI study companion

This is a single, full question-and-answer sheet for the Data Science and AI Engineering repository.

It is designed to sit alongside the topic-specific `Library/*.md` files Claude generated, without replacing them.

## How To Use This Sheet

1. Read the course-level Q and A first.
2. Use the lesson Q and A for the module you are running.
3. Use the CLI demo master sheet for exact commands.
4. Return to the topic-specific `Library/*.md` files when you need deeper definitions.

## Course-Level Q and A

### Q1. What type of software engineering is this repository training?

This repo is data science plus AI engineering plus MLOps-style operational discipline.

It is not only statistics and notebooks. It includes:

- data engineering (schema, ingestion, queries, quality gates)
- ML engineering (training pipelines, configs, reproducibility)
- evaluation engineering (thresholds, bias, drift, stability)
- operational ML (dashboards, logging, anomaly monitoring)
- applied optimization and graph analytics projects
- research engineering (MARL + XGBoost experiments, parameter studies)

### Q2. Why is reproducibility emphasized so strongly?

Because results that cannot be reproduced cannot be trusted in institutional settings.

Reproducibility reduces risk in:

- audits
- peer review
- production incidents
- model regressions
- experiment comparisons

This repo treats reproducibility as an engineering requirement, not an optional academic preference.

### Q3. What is the difference between data science work and AI engineering work in this repo?

- Data science focuses on statistical reasoning, data understanding, and model evaluation.
- AI/ML engineering focuses on building repeatable pipelines, configs, artifacts, and operational workflows.

This repo intentionally combines both so you can move from analysis to production-quality execution.

### Q4. What is the core learning pattern across the repository?

1. Read the concept/manual.
2. Run a small command or workflow.
3. Inspect artifacts and outputs.
4. Explain what the result means.
5. Compare alternatives or detect tradeoffs.
6. Record evidence and interpretation.

This pattern is used from statistics to MARL experiments.

### Q5. Why are there both `docs/` manuals and `Library/` notes?

They serve different roles.

- `docs/` contains canonical manuals, runbooks, and repo standards.
- `Library/` contains extended concept notes / reference-style explanations.

Your new master sheets in `Library/` are the bridge between CLI execution and conceptual understanding.

### Q6. What counts as evidence in this repo?

Evidence is not a claim like "the model is better." Evidence is:

- test results
- metrics reports
- config files used
- generated artifacts in `reports/`
- model outputs in `models/`
- benchmark outputs
- experiment result directories
- logged parameters / seeds / versions

### Q7. Why are evaluation and drift separate from training?

Because good training metrics do not guarantee safe deployment behavior.

Evaluation protects against:

- overfitting
- threshold mistakes
- group performance disparities
- drift-related performance decay
- unstable deployment decisions

### Q8. What is the value of the project modules (humanitarian, air traffic, ops anomaly system)?

They turn abstract methods into domain-specific engineering decisions.

This shows:

- problem formulation
- constraints
- tradeoffs
- operational outputs
- communication value for a portfolio

### Q9. Why is there a research module (`algorithm_marl_xgboost/`) in the same repo?

It demonstrates advanced experimentation discipline and research-to-engineering translation.

It also proves you can manage:

- experiment harnesses
- repeated runs
- parameter studies
- statistical comparisons
- reproducibility and threat-model documentation

### Q10. What is the fastest way to use this repo fully in the CLI?

Use one terminal for execution and one for reading:

```bash
# terminal A
cd /home/sp/cyber-course/projects/datascience

# terminal B
less docs/LESSON_EXECUTION_COMPANION.md
# or open the docs / Library file on GitHub mobile
```

Run one lesson at a time and record outputs in plain language.

## Lesson-by-Lesson Q and A (Lessons 1-19)

### Lesson 1 - Project Manual

Q: What is the main skill?
A: Rebuilding the environment and repo workflow reproducibly.

Q: What proves progress?
A: You can create the environment, install deps, and run baseline tests.

### Lesson 2 - Core Concepts

Q: What is the main skill?
A: Vocabulary fluency across data engineering, ML, evaluation, and operations.

Q: What proves progress?
A: You can map terms to real files/commands in the repo.

### Lesson 3 - Foundations Guide

Q: What is the main skill?
A: Core Python/data/engineering foundations used by later modules.

Q: What proves progress?
A: You can identify which later workflows depend on each foundation.

### Lesson 4 - Statistics Manual

Q: What is the main skill?
A: Statistical reasoning about uncertainty, tests, and interpretation.

Q: What proves progress?
A: You can explain what a statistical result means and does not mean.

### Lesson 5 - Data Pipelines Manual

Q: What is the main skill?
A: Source -> transform -> validation -> output lineage reasoning.

Q: What proves progress?
A: You can explain where data came from and what changed.

### Lesson 6 - Data Quality Manual

Q: What is the main skill?
A: Detecting data defects before they become model defects.

Q: What proves progress?
A: You can name multiple quality failure modes and their checks.

### Lesson 7 - ML Core Manual

Q: What is the main skill?
A: Running and understanding the baseline training pipeline.

Q: What proves progress?
A: You can rerun baseline training and interpret key metrics/artifacts.

### Lesson 8 - ML Advanced Manual

Q: What is the main skill?
A: Applying advanced methods with explicit tradeoff reasoning.

Q: What proves progress?
A: You can compare advanced outputs to baseline and justify gains/risks.

### Lesson 9 - Evaluation Manual

Q: What is the main skill?
A: Translating metrics into decision quality, fairness, and reliability checks.

Q: What proves progress?
A: You can explain why evaluation is more than accuracy.

### Lesson 10 - Scaling Manual

Q: What is the main skill?
A: Understanding scale patterns (chunking, multiprocessing, distributed-style workflows, benchmarking).

Q: What proves progress?
A: You can explain which scaling method fits which workload shape.

### Lesson 11 - Humanitarian Optimization Project

Q: What is the main skill?
A: Formulating resource allocation problems as constrained optimization.

Q: What proves progress?
A: You can explain objective, constraints, and why the solution is operationally useful.

### Lesson 12 - Air Traffic Delay Project

Q: What is the main skill?
A: Applied analytics and forecasting on aviation-style network/traffic data.

Q: What proves progress?
A: You can explain the modeling outputs in domain terms (delay, routes, flow, forecast utility).

### Lesson 13 - Ops Anomaly System Project

Q: What is the main skill?
A: Operational anomaly detection workflow with quality + inference + drift + reporting/dashboard behavior.

Q: What proves progress?
A: You can trace data -> model -> monitoring -> operational output.

### Lesson 14 - Architecture Docs

Q: What is the main skill?
A: Reading system architecture as engineering decisions and module relationships.

Q: What proves progress?
A: You can explain data flow and experimental flow using the diagrams/docs.

### Lesson 15 - Algorithm Manual (MARL + XGBoost)

Q: What is the main skill?
A: Understanding the research algorithm design, inputs, outputs, and experiment controls.

Q: What proves progress?
A: You can explain what is algorithmic contribution vs baseline engineering scaffolding.

### Lesson 16 - Parameter Study Manual

Q: What is the main skill?
A: Systematic experiment comparison using controlled parameter variation.

Q: What proves progress?
A: You can explain what changed, what stayed constant, and what conclusion is justified.

### Lesson 17 - Simplified Concepts + Math Intuition

Q: What is the main skill?
A: Translating technical and mathematical ideas into plain language.

Q: What proves progress?
A: You can explain the idea without losing the logic.

### Lesson 18 - Reproducibility / Threat Model / Protocol

Q: What is the main skill?
A: Formal experiment governance: what can be trusted, what can fail, and how runs should be documented.

Q: What proves progress?
A: You can describe the experiment protocol and the main threats to validity.

### Lesson 19 - Portfolio Mapping / Status Docs

Q: What is the main skill?
A: Communicating capability, scope, and evidence to institutions/employers/reviewers.

Q: What proves progress?
A: You can map repository outputs to real-world engineering skills.

## Operational Q and A (CLI + GitHub)

### Q: How should I study this in the CLI without getting lost?

A: Always pair execution with interpretation.

For each module, record:

- command run
- output/artifact produced
- what it means
- what could go wrong
- what would prove a regression

### Q: How do I know when to run code vs just read docs?

A: Run code when the module is executable (training, evaluation, projects, experiments).
Read docs when the lesson is conceptual/architecture/protocol.
Do both when possible.

### Q: What if a dependency is missing?

A: Do not force progress by skipping understanding. Read the matching manual and inspect configs/artifacts while you resolve environment setup.

## Where The Existing `Library/` Topic Files Fit

The topic-specific `Library/*.md` files are deep references for concepts.

Use them when the lesson companion or CLI output raises a concept question such as:

- probability/statistics (`01_*`)
- Python/numerical performance (`02_*`)
- databases/data engineering (`03_*`)
- data quality (`04_*`)
- optimization (`08_*`)
- graph analytics (`09_*`)
- time series (`10_*`)
- scaling (`11_*`)

## Cross-References

- `Library/00_full_course_cli_demo_sheet.md`
- `docs/LESSON_EXECUTION_COMPANION.md`
- `docs/LESSON_RESEARCH_ANALYSIS_COMPANION.md`
- `docs/OFFLINE_INDEX.md`
- `docs/PROJECT_MANUAL.md`
