# Data Science Lesson Execution Companion (Slow-Learning Edition)

Author: Simon Parris + Codex companion notes  
Date: 2026-02-23

This guide is for learning while coding in another terminal. It slows the pace and makes each module readable like a study manual.

## How to Use This Companion

For every module/manual:

1. Read `What this lesson is`
2. Read `Why this matters`
3. Use the `Companion prompt`
4. Run one small command/task only
5. Record evidence before moving on

## Reading Rule (When Course Pace Feels Too Fast)

- First pass: run, do not optimize.
- Second pass: explain outputs in plain language.
- Third pass: write a short note: what was done, why, what proves it worked.

## Suggested Study Order (Execution Track)

1. `docs/PROJECT_MANUAL.md`
2. `docs/CORE_CONCEPTS.md`
3. `docs/foundations/FOUNDATIONS_GUIDE.md`
4. `docs/statistics/STATISTICS_MANUAL.md`
5. `docs/data_engineering/DATA_PIPELINES_MANUAL.md`
6. `docs/data_quality/DATA_QUALITY_MANUAL.md`
7. `docs/ml_core/ML_CORE_MANUAL.md`
8. `docs/ml_advanced/ML_ADVANCED_MANUAL.md`
9. `docs/evaluation/EVALUATION_MANUAL.md`
10. `docs/scaling/SCALING_MANUAL.md`
11. Project manuals (`docs/projects/*.md`)
12. Architecture docs
13. MARL + XGBoost docs (`algorithm_marl_xgboost/docs/*`)

## Lesson 1: Project Manual (`docs/PROJECT_MANUAL.md`)

### What this lesson is

The operating rules for rebuilding the repository, managing environments, and maintaining reproducibility and audit discipline.

### Why this matters

If your environment and process are unstable, later ML results are hard to trust and hard to repeat.

### Companion prompt

`Explain what setup step I am doing, what dependency or system requirement it enables, and what problem it prevents later in the ML pipeline. Define each environment term simply.`

### Do this now

- Create `venv`
- Activate it
- Install dependencies
- Run baseline tests

### Evidence to collect

- Python version
- `pip` install success
- Test summary

### Stop condition

You can rebuild the repo on a fresh machine using only the manual.

## Lesson 2: Core Concepts (`docs/CORE_CONCEPTS.md`)

### What this lesson is

Shared vocabulary across data engineering, ML, evaluation, and operational ML.

### Why this matters

Most confusion in fast courses is vocabulary confusion, not coding difficulty.

### Companion prompt

`Define each term in beginner language, then give one example from this repository where it appears in practice.`

### Do this now

- Read one concept section at a time
- Match each term to a real file or command in the repo

### Stop condition

You can explain the difference between data quality, model quality, and operational quality.

## Lesson 3: Foundations Guide (`docs/foundations/FOUNDATIONS_GUIDE.md`)

### What this lesson is

Core Python/data/engineering foundations used by later modules.

### Why this matters

This is the “load-bearing” layer for the rest of the course.

### Companion prompt

`Explain which foundational skill this lesson is training and how it will be reused in later modules (data pipelines, ML training, evaluation, or ops).`

### Do this now

- Run one small foundational example or command
- Confirm expected output shape/type/behavior

### Stop condition

You can identify which future modules depend on this foundation.

## Lesson 4: Statistics Manual (`docs/statistics/STATISTICS_MANUAL.md`)

### What this lesson is

Statistical reasoning for experiments, distributions, tests, and uncertainty.

### Why this matters

Without statistics, ML results can look impressive but be misleading.

### Companion prompt

`Explain what statistical question is being answered, what assumptions are required, and what decision would be wrong if I skipped this analysis.`

### Do this now

- Run one statistics example
- Write the hypothesis being tested in plain language
- Record the interpretation, not only the numeric result

### Stop condition

You can state what a result means and what it does not mean.

## Lesson 5: Data Pipelines Manual (`docs/data_engineering/DATA_PIPELINES_MANUAL.md`)

### What this lesson is

Ingestion, transformation, schema discipline, and pipeline execution patterns.

### Why this matters

Model quality depends on data pipeline quality. Bad inputs create believable but incorrect outputs.

### Companion prompt

`Explain this pipeline step as a chain: source -> transform -> validation -> output. What can fail here, and what evidence proves the failure location?`

### Do this now

- Run one pipeline step or example query/process
- Inspect input/output artifacts
- Check schema/shape assumptions

### Stop condition

You can explain lineage: where the data came from and what changed.

## Lesson 6: Data Quality Manual (`docs/data_quality/DATA_QUALITY_MANUAL.md`)

### What this lesson is

Validation rules and checks that ensure data is fit for downstream use.

### Why this matters

Data quality failures often look like model failures unless caught early.

### Companion prompt

`Explain which quality dimension is being checked (missingness, validity, consistency, drift, etc.), why it matters, and what action is taken on failure.`

### Do this now

- Run or review one quality check
- Identify pass/fail criteria
- Note what downstream module would break if this were skipped

### Stop condition

You can name at least three data quality failure modes and how to detect them.

## Lesson 7: ML Core Manual (`docs/ml_core/ML_CORE_MANUAL.md`)

### What this lesson is

Baseline training pipeline, feature preparation, model training, and core evaluation outputs.

### Why this matters

The baseline is your reference system; advanced work is only meaningful when compared to a stable baseline.

### Companion prompt

`Explain this ML pipeline as a sequence of transformations and decisions. What is the baseline, what metrics are produced, and why is reproducibility important here?`

### Do this now

- Run `make ml-train`
- Identify input dataset, config, and output artifacts
- Record key metrics

### Stop condition

You can explain what the baseline model does and how to rerun it reproducibly.

## Lesson 8: ML Advanced Manual (`docs/ml_advanced/ML_ADVANCED_MANUAL.md`)

### What this lesson is

Advanced modeling techniques: imbalance handling, tuning, explainability, and performance refinement.

### Why this matters

Advanced methods improve performance, but also add complexity and risk if not evaluated carefully.

### Companion prompt

`Explain what advanced technique is being introduced, what problem it solves, what tradeoff it adds, and how I should verify the gain is real.`

### Do this now

- Run `make ml-adv-train`
- Compare results to baseline, not only absolute values
- Inspect tuning/explainability outputs

### Stop condition

You can state one benefit and one risk for each advanced technique you used.

## Lesson 9: Evaluation Manual (`docs/evaluation/EVALUATION_MANUAL.md`)

### What this lesson is

Structured evaluation of model behavior: thresholds, bias/group metrics, drift, stability, and reliability.

### Why this matters

A model can score well overall and still fail operationally or ethically in important cases.

### Companion prompt

`Explain what this evaluation metric or check is protecting against, and what real-world decision error could happen if it is ignored.`

### Do this now

- Run `make eval-suite`
- Read outputs as decisions, not just numbers
- Note threshold and tradeoff implications

### Stop condition

You can explain why evaluation is more than “accuracy”.

## Lesson 10: Scaling Manual (`docs/scaling/SCALING_MANUAL.md`)

### What this lesson is

Performance and throughput techniques for larger workloads (parallelism/chunking/benchmarking patterns).

### Why this matters

A correct pipeline that cannot run on realistic workloads is not production-ready.

### Companion prompt

`Explain what scaling strategy is used, what bottleneck it targets (CPU, memory, IO), and how I should measure whether performance actually improved.`

### Do this now

- Run one benchmark/scaling example
- Record runtime/resource changes
- Check correctness remains unchanged

### Stop condition

You can explain a performance improvement with measurements, not impressions.

## Lesson 11: Humanitarian Optimization Project (`docs/projects/HUMANITARIAN_OPTIMIZATION_MANUAL.md`)

### What this lesson is

Optimization modeling for resource allocation under constraints.

### Why this matters

This introduces decision science thinking: best choice under limited resources, not only prediction.

### Companion prompt

`Explain the optimization problem in plain language: objective, constraints, decision variables, and what a feasible vs infeasible solution means.`

### Do this now

- Run the project target (`make project-humanitarian`)
- Read outputs as decisions and constraints

### Stop condition

You can explain why the chosen allocation is valid and what constraints shaped it.

## Lesson 12: Air Traffic Delay Project (`docs/projects/AIR_TRAFFIC_DELAY_MANUAL.md`)

### What this lesson is

Domain analytics and forecasting around air traffic flow and delay behavior.

### Why this matters

It connects data/ML methods to an operational domain with real timing and reliability consequences.

### Companion prompt

`Explain the business/operational question this model answers, what features likely matter, and what errors would be operationally costly.`

### Do this now

- Run `make project-air-traffic`
- Inspect outputs and domain metrics
- Note model limitations in operational context

### Stop condition

You can explain the project in domain language, not only ML language.

## Lesson 13: Ops Anomaly System Project (`docs/projects/OPS_ANOMALY_SYSTEM_MANUAL.md`)

### What this lesson is

Operational anomaly detection workflow with quality, inference, drift, and API/dashboard behavior.

### Why this matters

This is where ML meets operations: monitoring and decision support for live systems.

### Companion prompt

`Explain this system as an operational pipeline: ingestion -> checks -> inference -> monitoring -> alerting/reporting. Where can silent failure happen?`

### Do this now

- Run `make project-ops-system`
- Trace one request/path through the system
- Note what evidence each stage produces

### Stop condition

You can identify where operational ML differs from offline model training.

## Lesson 14: Architecture Docs (`docs/architecture/*.md`)

### What this lesson is

System-level diagrams and flows for the repository and MARL module.

### Why this matters
Architecture understanding prevents local optimizations that break end-to-end systems.

### Companion prompt

`Explain this architecture as a flow of data, decisions, and control. Where are the boundaries, dependencies, and failure points?`

### Do this now

- Read one architecture diagram
- Draw the path manually (data -> model -> evaluation -> reporting)
- Mark where logs/metrics/artifacts are produced

### Stop condition

You can narrate the architecture without looking at the diagram.

## Lesson 15: Algorithm Manual (`algorithm_marl_xgboost/docs/ALGORITHM_MANUAL.md`)

### What this lesson is

Operational guide to the decentralized MARL + XGBoost research module.

### Why this matters

This is the highest-complexity part of the repo and needs careful, reproducible execution habits.

### Companion prompt

`Explain this algorithm module as a reproducible experiment system: what components interact, what outputs are artifacts, and what settings must stay controlled.`

### Do this now

- Run `make algo-run`
- Record config, seed, and output artifact paths

### Stop condition

You can rerun the same experiment and explain why results should be comparable.

## Lesson 16: Parameter Study Manual (`algorithm_marl_xgboost/docs/experiments/PARAMETER_STUDY_MANUAL.md`)

### What this lesson is

Structured repeated experiments across parameter combinations to evaluate performance patterns.

### Why this matters

Single-run results can be misleading; parameter studies reveal stability and sensitivity.

### Companion prompt

`Explain what parameter is changing, what outcome is measured, and how repeated runs improve confidence in conclusions.`

### Do this now

- Run `make algo-study` (or a small subset)
- Record parameter grid, repeats, and summary outputs

### Stop condition

You can distinguish a one-off good result from a repeatable trend.

## Lesson 17: Simplified Concepts + Math Intuition (`CONCEPTS_SIMPLIFIED.md`, `MATH_INTUITION.md`)

### What this lesson is

Human-readable explanations of the algorithm concepts and math intuition.

### Why this matters

Complex algorithms become usable only when you can explain them without equations first.

### Companion prompt

`Explain the algorithm idea without math first, then add only the minimum math intuition needed to understand the behavior.`

### Do this now

- Read a concept section
- Rewrite it in your own words in 5-8 lines
- Add one concrete example from this repo

### Stop condition

You can teach the concept to a beginner without reading directly from the file.

## Lesson 18: Reproducibility / Threat Model / Protocol (`REPRODUCIBILITY_AND_AUDIT.md`, `THREAT_MODEL_AND_LIMITS.md`, `EXPERIMENT_PROTOCOL.md`)

### What this lesson is

Research governance for experiment integrity, limitations, and safe interpretation.

### Why this matters

Strong research work documents limits and threats, not only positive results.

### Companion prompt

`Explain what could invalidate these experiment results, how the protocol reduces that risk, and what limitations must be stated honestly.`

### Do this now

- Read one governance doc at a time
- List risks, controls, and remaining limitations

### Stop condition

You can describe both the strengths and limits of the research process.

## Lesson 19: Portfolio Mapping / Status Docs (`PORTFOLIO_SKILL_MAPPING.md`, `CV_READY_SUMMARY.md`, `REPOSITORY_STATUS_REPORT.md`)

### What this lesson is

Translation layer from technical work to professional evidence and communication.

### Why this matters

Good engineering work must also be explainable to reviewers, hiring panels, and stakeholders.

### Companion prompt

`Explain how this repo demonstrates concrete skills, what evidence supports each claim, and where limitations should be stated to remain credible.`

### Do this now

- Map one project/module to a skill claim
- Link the claim to code/docs/artifacts

### Stop condition

You can describe your work with evidence instead of generic claims.

## Memorization Method (Data/ML Version)

After each lesson, write these 6 lines:

1. `Question this lesson answers`
2. `Inputs used`
3. `Process/model/check performed`
4. `Output produced`
5. `How I verified it`
6. `Main limitation or risk`
