# Core Concepts for Institutional Data Science & AI

## Data Pipeline

- Short definition: A sequence of steps that collects, cleans, transforms, and delivers data for analysis or model use.
- Why it matters in UN/IATA/CERN context: Institutional programs rely on consistent, traceable data flows across teams, systems, and reporting cycles.

## Reproducibility

- Short definition: The ability to rerun the same workflow and obtain the same result using the same data, code, and settings.
- Why it matters in UN/IATA/CERN context: High-stakes operations and scientific work require repeatable evidence for audits, peer review, and policy decisions.

## Cross Validation

- Short definition: A method for testing model performance by training and validating on multiple data splits.
- Why it matters in UN/IATA/CERN context: It reduces overconfidence and gives stronger evidence that results generalize across regions, routes, or experiments.

## Overfitting

- Short definition: When a model learns noise or quirks in training data and performs poorly on new data.
- Why it matters in UN/IATA/CERN context: Overfit models can create unsafe or misleading recommendations in operational or scientific settings.

## Class Imbalance

- Short definition: A dataset condition where one outcome class is much more common than another.
- Why it matters in UN/IATA/CERN context: Rare but important events (incidents, anomalies, failures) may be missed unless imbalance is handled carefully.

## Anomaly Detection

- Short definition: Techniques used to identify unusual patterns that differ from normal behavior.
- Why it matters in UN/IATA/CERN context: Early detection of irregular events can support safety monitoring, humanitarian logistics, and instrument health checks.

## Distributed Computing

- Short definition: Running computation across multiple machines or processes to handle larger workloads faster.
- Why it matters in UN/IATA/CERN context: Large-scale datasets and simulations often exceed single-machine capacity and require coordinated processing.

## Explainability

- Short definition: The ability to describe why a model produced a specific output in understandable terms.
- Why it matters in UN/IATA/CERN context: Analysts, auditors, and decision-makers need interpretable evidence before trusting automated outputs.

## Model Drift

- Short definition: Performance decline caused by changes in data patterns, behavior, or operating conditions over time.
- Why it matters in UN/IATA/CERN context: Environments change; monitoring drift prevents silent degradation in mission-critical systems.

## Governance

- Short definition: The policies, roles, and controls that guide how data and AI systems are developed and used.
- Why it matters in UN/IATA/CERN context: Clear governance supports accountability, compliance, and cross-organizational coordination.

## Auditability

- Short definition: The ability to inspect and verify data, code, decisions, and model outputs after the fact.
- Why it matters in UN/IATA/CERN context: External review, compliance checks, and scientific validation depend on complete traceable records.

## Vectorization

- Short definition: Doing the same calculation on many values at once instead of looping through each item in Python.
- Why it matters in UN/IATA/CERN context: Faster processing supports timely analysis on large operational and scientific datasets.

## Broadcasting

- Short definition: A way to combine arrays of different shapes by automatically matching dimensions when possible.
- Why it matters in UN/IATA/CERN context: It simplifies large-scale transformations while reducing manual code errors in data pipelines.

## Generator

- Short definition: A Python object that produces values one at a time instead of storing them all in memory.
- Why it matters in UN/IATA/CERN context: Useful for memory-safe processing of logs, telemetry, and large streaming datasets.

## Time Complexity

- Short definition: A simple way to describe how runtime grows as the amount of data grows.
- Why it matters in UN/IATA/CERN context: Helps teams choose scalable methods before workloads become too slow or costly.

## Memory Profiling

- Short definition: Measuring how much memory a program or dataset uses during execution.
- Why it matters in UN/IATA/CERN context: Prevents failures in batch and research jobs that must run reliably on shared infrastructure.

## Probability Distribution

- Short definition: A rule or pattern that describes how likely different values or events are.
- Why it matters in UN/IATA/CERN context: Helps teams model uncertainty in operations, logistics, and scientific measurements.

## Standard Deviation

- Short definition: A measure of how spread out values are around the average.
- Why it matters in UN/IATA/CERN context: Quantifies variability in delays, demand, and instrument readings for planning and risk controls.

## p-value

- Short definition: A number showing how surprising the observed result would be if there were no real effect.
- Why it matters in UN/IATA/CERN context: Supports evidence-based decisions, but must be interpreted carefully with domain context.

## Statistical Significance

- Short definition: A threshold-based label used when a p-value is below a chosen cutoff (for example, 0.05).
- Why it matters in UN/IATA/CERN context: Can guide decisions, but should not replace operational significance, safety review, or scientific judgment.

## Confidence Interval

- Short definition: A range of plausible values for an unknown quantity, based on sample data and assumptions.
- Why it matters in UN/IATA/CERN context: Communicates uncertainty more clearly than a single estimate in reports and audits.

## Monte Carlo Simulation

- Short definition: Repeated random simulation used to estimate uncertainty, risk, or outcome ranges.
- Why it matters in UN/IATA/CERN context: Useful for planning under uncertainty, stress testing, and scientific scenario analysis.

## Sampling Bias

- Short definition: A distortion that happens when collected data does not fairly represent the real population.
- Why it matters in UN/IATA/CERN context: Biased samples can produce unsafe, unfair, or scientifically invalid conclusions.

## Schema

- Short definition: The defined structure of a database, including tables, columns, and rules.
- Why it matters in UN/IATA/CERN context: Shared schema definitions keep operational and scientific data consistent across teams and systems.

## Primary key

- Short definition: A column (or set of columns) that uniquely identifies each row in a table.
- Why it matters in UN/IATA/CERN context: Unique identifiers are essential for traceability, deduplication, and audit-ready data linking.

## Foreign key

- Short definition: A column that references a primary key in another table to enforce valid relationships.
- Why it matters in UN/IATA/CERN context: Prevents broken data relationships in multi-table operational and analytical pipelines.

## Index

- Short definition: A database structure that helps queries find rows faster.
- Why it matters in UN/IATA/CERN context: Improves performance for time-critical dashboards, investigations, and reporting workloads.

## ETL vs ELT

- Short definition: ETL transforms data before loading into storage; ELT loads first, then transforms inside the data platform.
- Why it matters in UN/IATA/CERN context: Choosing the right pattern affects reproducibility, performance, and governance controls.

## Data validation

- Short definition: Checks that data has the right columns, types, and value ranges before use.
- Why it matters in UN/IATA/CERN context: Prevents bad records from contaminating mission-critical analysis and decisions.

## Data Quality

- Short definition: A measure of whether data is accurate enough, complete enough, and consistent enough for its intended use.
- Why it matters in UN/IATA/CERN context: Low-quality data can produce unsafe operational decisions, weak humanitarian planning, or invalid scientific conclusions.

## Schema Validation

- Short definition: Checking that each row matches the required columns, types, and basic constraints.
- Why it matters in UN/IATA/CERN context: Enforces consistent structure across teams and systems before analysis or reporting begins.

## Domain Rule

- Short definition: A business or scientific rule that reflects real-world logic (for example, departure and arrival airports should differ).
- Why it matters in UN/IATA/CERN context: Domain rules catch errors that generic type checks cannot detect.

## Outlier

- Short definition: A value that is unusually far from most other values in the dataset.
- Why it matters in UN/IATA/CERN context: Outliers may indicate important rare events, sensor issues, or data entry problems.

## Drift (basic)

- Short definition: A change in data patterns over time, such as shifts in averages or variability.
- Why it matters in UN/IATA/CERN context: Drift can reduce reliability of analytics and models unless monitored and reviewed.

## Quality Gate

- Short definition: A pass/fail checkpoint that blocks data from progressing when quality thresholds are not met.
- Why it matters in UN/IATA/CERN context: Prevents bad data from entering mission-critical reporting and decision pipelines.

## JSON Lines (JSONL)

- Short definition: A file format where each line is a separate JSON object.
- Why it matters in UN/IATA/CERN context: Supports machine-readable logging, easy streaming, and audit-friendly pipeline records.

## Out-of-core processing

- Short definition: Processing data in smaller chunks instead of loading the full dataset into memory at once.
- Why it matters in UN/IATA/CERN context: Enables large analytical workloads to run reliably on limited or shared infrastructure.

## Determinism

- Short definition: Getting the same output every time when the same input, code, and settings are used.
- Why it matters in UN/IATA/CERN context: Deterministic pipelines improve reproducibility, audits, and incident debugging.

## Multiprocessing

- Short definition: Using multiple CPU processes in parallel to speed up independent tasks.
- Why it matters in UN/IATA/CERN context: Helps scale data transformations and simulations when timelines are tight.

## Dask

- Short definition: A Python library for parallel and larger-than-memory data processing.
- Why it matters in UN/IATA/CERN context: Supports scalable data workflows across research and operations without rewriting everything from scratch.

## Parquet

- Short definition: A compressed columnar data file format optimized for analytics.
- Why it matters in UN/IATA/CERN context: Reduces storage costs and speeds up repeated analytical queries on large datasets.

## Profiling

- Short definition: Measuring runtime and memory usage to find slow or inefficient code.
- Why it matters in UN/IATA/CERN context: Helps teams tune performance and avoid failures in production and scientific workloads.

## Pipeline

- Short definition: A repeatable sequence of steps that moves data from input through processing to outputs.
- Why it matters in UN/IATA/CERN context: Pipelines make institutional analytics and AI workflows reproducible, reviewable, and easier to audit.

## Train/Test Split

- Short definition: Dividing data into a training set for learning and a test set for final evaluation.
- Why it matters in UN/IATA/CERN context: Prevents over-optimistic performance claims in operational and scientific models.

## Stratification

- Short definition: Splitting data while preserving class proportions (for example, rare events) in each subset.
- Why it matters in UN/IATA/CERN context: Produces more reliable evaluation when important events are uncommon.

## Class Imbalance

- Short definition: A dataset condition where one class is much rarer than another.
- Why it matters in UN/IATA/CERN context: Rare but critical events can be missed unless models and metrics account for imbalance.

## ROC-AUC

- Short definition: A metric summarizing how well a classifier separates positives from negatives across thresholds.
- Why it matters in UN/IATA/CERN context: Useful for comparing models in imbalanced risk-detection settings.

## Confusion Matrix

- Short definition: A table showing counts of true/false positives and true/false negatives.
- Why it matters in UN/IATA/CERN context: Helps teams understand error types, not just overall accuracy.

## Hyperparameter

- Short definition: A model setting chosen before training (for example, tree depth or learning rate).
- Why it matters in UN/IATA/CERN context: Hyperparameters affect performance, runtime, and stability and should be tracked for reproducibility.

## Artifact

- Short definition: A saved output from a workflow, such as a model file, metrics report, or plot.
- Why it matters in UN/IATA/CERN context: Artifacts provide traceable evidence for review, deployment, and audits.

## SMOTE

- Short definition: A technique that creates synthetic minority-class samples to reduce class imbalance during training.
- Why it matters in UN/IATA/CERN context: Can improve rare-event detection, but must be applied carefully to avoid unrealistic samples or leakage.

## Hyperparameter Tuning

- Short definition: Systematically searching for better model settings using validation data.
- Why it matters in UN/IATA/CERN context: Improves model quality while keeping configuration choices traceable and reproducible.

## Optuna

- Short definition: A Python library that automates hyperparameter tuning by selecting and evaluating trial configurations.
- Why it matters in UN/IATA/CERN context: Helps teams tune models efficiently while recording trial histories for audit and review.

## SHAP

- Short definition: A method for explaining model predictions by estimating each feature's contribution.
- Why it matters in UN/IATA/CERN context: Supports interpretability requirements for high-stakes institutional models.

## Data Leakage

- Short definition: When information from outside the training process improperly influences model training or evaluation.
- Why it matters in UN/IATA/CERN context: Leakage can make models look stronger than they really are, creating false confidence in decisions.

## Calibration Threshold

- Short definition: The probability cutoff used to convert model scores into final yes/no decisions.
- Why it matters in UN/IATA/CERN context: Threshold choices affect false positives, false negatives, and operational policy outcomes.

## Stability (Seed Sensitivity)

- Short definition: How much model performance changes when training randomness (seed) changes.
- Why it matters in UN/IATA/CERN context: Stable models are easier to trust, reproduce, and operate in regulated workflows.

## Bias (Group Metrics)

- Short definition: Uneven performance or error rates across groups when comparing model outcomes.
- Why it matters in UN/IATA/CERN context: Group-level disparities can create fairness, safety, or governance concerns.

## Drift

- Short definition: A change in data distributions or behavior between training and later evaluation/production data.
- Why it matters in UN/IATA/CERN context: Drift can silently degrade model reliability in operational and scientific systems.

## KS Test

- Short definition: A statistical test that compares two distributions and measures how different they are.
- Why it matters in UN/IATA/CERN context: Useful for detecting numeric feature drift between training and evaluation periods.

## Total Variation Distance

- Short definition: A number that measures how different two probability distributions are.
- Why it matters in UN/IATA/CERN context: Helps quantify categorical drift in a simple, interpretable way.

## Validation Split

- Short definition: A subset of training-era data reserved for model selection and threshold tuning before final test evaluation.
- Why it matters in UN/IATA/CERN context: Prevents test-set leakage and supports audit-ready model selection decisions.

## Linear Programming

- Short definition: A mathematical optimization method that finds the best values for decision variables while obeying linear rules.
- Why it matters in UN/IATA/CERN context: Useful for transparent planning when budgets, capacity, and policy constraints must be enforced.

## Objective Function

- Short definition: The formula the optimization model tries to minimize or maximize.
- Why it matters in UN/IATA/CERN context: Makes trade-offs explicit so allocation decisions can be reviewed and justified.

## Constraint

- Short definition: A hard condition that every valid solution must satisfy.
- Why it matters in UN/IATA/CERN context: Encodes policy limits, resource caps, and operational realities directly into planning models.

## Feasible Region

- Short definition: The set of all possible solutions that satisfy every constraint.
- Why it matters in UN/IATA/CERN context: Helps teams understand what is possible before choosing the best plan.

## Sensitivity Analysis

- Short definition: Re-running a model with changed assumptions to see how results move.
- Why it matters in UN/IATA/CERN context: Shows whether plans remain robust when budgets, priorities, or risks change.

## Fairness Metric

- Short definition: A measure used to compare how outcomes are distributed across groups or regions.
- Why it matters in UN/IATA/CERN context: Supports review of equity and under-service risks in institutional allocation decisions.
