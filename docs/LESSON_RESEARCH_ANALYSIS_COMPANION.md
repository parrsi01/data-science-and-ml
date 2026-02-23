# Data Science Lesson Research Analysis Companion (Beginner Definitions)

Author: Simon Parris + Codex companion notes  
Date: 2026-02-23

This companion teaches you how to read the repo like a research analyst while staying beginner-friendly.

## Why This Companion Exists

Data/ML courses often move quickly from code to results. This note slows the process so you can understand:

- what question a module is answering
- what evidence supports the result
- what assumptions are hidden
- what terms mean in plain language
- what risks/limitations should be stated

## Research-Analyst Reading Method (Simple Version)

For every lesson, ask:

1. `What decision or question is this module trying to support?`
2. `What data is being used?`
3. `What transformation or model is applied?`
4. `What assumptions are required?`
5. `What output is produced?`
6. `How do we validate trustworthiness?`
7. `What could go wrong or be overstated?`

## Universal Beginner Definitions (Data + ML + Research)

- `Dataset`: A collection of rows/records and columns/fields used for analysis.
- `Feature`: An input variable used by a model.
- `Label` / `Target`: The output value the model tries to predict.
- `Model`: A mathematical system that learns patterns from data.
- `Training`: The process of fitting a model to data.
- `Evaluation`: Measuring how well the model performs.
- `Metric`: A numeric measure of performance (accuracy, precision, recall, etc.).
- `Bias` (model context): Systematic error or unfair pattern in predictions.
- `Drift`: Data or behavior changes over time that make prior models less reliable.
- `Pipeline`: A sequence of steps that transform data and produce outputs.
- `Experiment`: A repeatable run with defined settings and recorded results.
- `Reproducibility`: Ability to run again and get comparable results.
- `Artifact`: A saved output (model file, plot, metrics JSON, report).
- `Assumption`: A condition believed to be true for a method/result to hold.
- `Limitation`: A known boundary where results may not generalize.

## Lesson-by-Lesson Analysis Lens

## Lesson 1: Project Manual

### Research lens

Treat this as the experiment environment control document. It defines the conditions required for valid and repeatable work.

### Terms to define while reading

- `venv`: isolated Python environment for project dependencies.
- `dependency`: external library/package required by the code.
- `audit`: documented review of process/evidence.
- `provenance`: record of origin and changes.

### What to analyze

- Which setup steps are mandatory for reproducibility?
- Which steps are convenience-only?
- What failures later in the repo can be traced to setup mistakes?

## Lesson 2: Core Concepts

### Research lens

This is a vocabulary normalization layer. Read it as a glossary that reduces future reasoning errors.

### What to analyze

- Which terms are often confused?
- Which terms change meaning by context (for example, “drift” in data vs model behavior)?

## Lesson 3: Foundations Guide

### Research lens

Foundations are capability prerequisites. Weak foundations create silent mistakes in later code.

### Terms to define while reading

- `data type`: category of value (integer, float, string, etc.).
- `shape`: dimensions of a dataset/array.
- `deterministic`: same input gives same output each time.

### What to analyze

- Which later modules depend on this exact skill?
- What error would appear if this foundation is weak?

## Lesson 4: Statistics Manual

### Research lens

Statistics provides evidence discipline. It answers whether observed patterns are meaningful or possibly noise.

### Terms to define while reading

- `distribution`: pattern of how values are spread.
- `hypothesis test`: method to check if evidence supports a claim.
- `p-value`: measure used in many tests to evaluate evidence against a null hypothesis.
- `confidence interval`: range likely to contain a parameter estimate.

### What to analyze

- What question is the test answering?
- What assumptions are required?
- What wrong conclusion is possible if assumptions fail?

## Lesson 5: Data Pipelines Manual

### Research lens

Treat pipelines as evidence manufacturing systems. If the pipeline is wrong, every downstream metric is suspect.

### Terms to define while reading

- `ingestion`: bringing data into the system.
- `schema`: formal structure of fields/types.
- `transform`: change data format or values.
- `lineage`: trace of where data came from and what changed.

### What to analyze

- Where can data corruption or mis-mapping happen?
- What validation catches it early?

## Lesson 6: Data Quality Manual

### Research lens

This module protects validity before modeling. It should be read as a gatekeeper, not a cleanup afterthought.

### Terms to define while reading

- `completeness`: whether required data is missing.
- `validity`: whether values follow expected rules/ranges.
- `consistency`: whether data agrees across fields/sources.
- `quality threshold`: rule that decides pass/fail.

### What to analyze

- Which quality failures are blocking vs warning-only?
- What downstream model behavior would each failure distort?

## Lesson 7: ML Core Manual

### Research lens

Analyze the baseline model as a reference experiment. The baseline is not “simple only”; it is the comparison anchor.

### Terms to define while reading

- `baseline`: first reliable model/process used for comparison.
- `split`: dividing data into train/validation/test sets.
- `overfitting`: model learns training details too specifically and performs poorly on new data.
- `generalization`: how well the model performs on unseen data.

### What to analyze

- Are dataset splits appropriate?
- Are metrics aligned with the task?
- Are outputs reproducible given configs/seeds?

## Lesson 8: ML Advanced Manual

### Research lens

Advanced methods should be treated as controlled experiments against the baseline, not automatic upgrades.

### Terms to define while reading

- `class imbalance`: one class appears much more than another.
- `hyperparameter`: model setting chosen before training (not learned directly from data).
- `tuning`: searching for better hyperparameter settings.
- `explainability`: methods for understanding model behavior/predictions.

### What to analyze

- What exact problem is the advanced method solving?
- What new risks are introduced (instability, leakage, complexity)?
- Is improvement measured fairly against the same baseline conditions?

## Lesson 9: Evaluation Manual

### Research lens

Evaluation is decision-quality analysis, not only score reporting.

### Terms to define while reading

- `precision`: among predicted positives, how many were correct.
- `recall`: among actual positives, how many were found.
- `threshold`: cutoff used to convert scores into decisions.
- `calibration`: how well predicted probabilities match reality.
- `fairness/group metric`: performance comparison across subgroups.

### What to analyze

- Which errors are most costly in this domain?
- Does the chosen threshold match that cost structure?
- Are subgroup differences operationally acceptable?

## Lesson 10: Scaling Manual

### Research lens

Scaling analysis is performance science: measure bottlenecks, test interventions, verify correctness retention.

### Terms to define while reading

- `throughput`: amount of work completed per unit time.
- `latency`: time taken for one operation.
- `benchmark`: controlled performance measurement.
- `parallelism`: doing multiple tasks at the same time.

### What to analyze

- What is the current bottleneck (CPU, memory, IO)?
- What metric proves improvement?
- Did the optimization change results or only speed?

## Lesson 11: Humanitarian Optimization Project

### Research lens

Read this as constrained decision-making under resource scarcity and accountability requirements.

### Terms to define while reading

- `objective`: what the optimization tries to maximize/minimize.
- `constraint`: rule/limit the solution must obey.
- `decision variable`: value the solver chooses.
- `feasible`: satisfies all constraints.

### What to analyze

- Are constraints realistic?
- What tradeoffs are being made?
- What happens when constraints tighten?

## Lesson 12: Air Traffic Delay Project

### Research lens

Analyze this as operational forecasting/analytics where errors affect timing, planning, and system coordination.

### Terms to define while reading

- `forecast`: prediction of future values/events.
- `feature importance`: estimate of which inputs influenced predictions more.
- `operational metric`: domain metric used by practitioners (not only ML score).

### What to analyze

- Which errors matter most operationally?
- Does the model capture domain seasonality/structure?
- What external factors may not be represented?

## Lesson 13: Ops Anomaly System Project

### Research lens

This is ML in an operations setting. Analyze both model behavior and system behavior (quality checks, API, monitoring, drift).

### Terms to define while reading

- `anomaly`: unusual pattern that may indicate a problem.
- `inference`: using a trained model to score new data.
- `drift monitoring`: tracking changes that may reduce model reliability.
- `API endpoint`: URL path where a service receives requests.

### What to analyze

- What evidence exists for each stage (input checks, inference, drift)?
- What silent failure modes could produce false confidence?

## Lesson 14: Architecture Docs

### Research lens

Architecture docs define system boundaries and evidence flows. Use them to understand where artifacts, logs, and controls belong.

### Terms to define while reading

- `component`: a distinct part of the system.
- `interface`: how components communicate.
- `dependency`: required upstream/downstream relationship.
- `failure point`: place where breakdown can interrupt the flow.

### What to analyze

- Where are critical dependencies?
- Where are observability points placed?
- What component failures have the largest blast radius?

## Lesson 15: Algorithm Manual (MARL + XGBoost)

### Research lens

Treat the algorithm module as a research program with stricter experiment control needs than ordinary app code.

### Terms to define while reading

- `agent`: decision-making unit in reinforcement learning.
- `reward`: feedback signal used to guide agent behavior.
- `decentralized`: agents operate without one central controller making all decisions.
- `federated` (broad sense here): combining distributed learning/updates across nodes/agents.

### What to analyze

- What components are deterministic vs stochastic?
- Which settings materially affect comparability?
- What artifact set is required to reproduce a result claim?

## Lesson 16: Parameter Study Manual

### Research lens

This module is about sensitivity and robustness, not only top-line score maximization.

### Terms to define while reading

- `parameter grid`: selected combinations of settings to test.
- `repeat`: rerunning under the same configuration to measure stability.
- `variance`: amount results differ across runs.
- `significance`: evidence that an observed difference is unlikely due to random variation alone (under a chosen method).

### What to analyze

- Which parameters strongly change outcomes?
- Are results stable across repeats?
- Are conclusions robust or fragile?

## Lesson 17: Concepts Simplified + Math Intuition

### Research lens

These are translation documents. Their job is to reduce cognitive load without losing conceptual truth.

### Terms to define while reading

- `intuition`: mental model that helps predict behavior before formal proof.
- `formalism`: exact mathematical or technical expression.
- `approximation`: simplified representation that preserves useful behavior.

### What to analyze

- Does the simplified explanation preserve the correct cause-effect story?
- What important detail is intentionally omitted for clarity?

## Lesson 18: Reproducibility / Threat Model / Experiment Protocol

### Research lens

This is credibility infrastructure. It explains why conclusions should be trusted and where trust should stop.

### Terms to define while reading

- `threat model`: structured list of ways results/system could fail or be manipulated.
- `protocol`: fixed procedure for running experiments consistently.
- `confounder`: hidden factor that can distort conclusions.
- `audit trail`: record of actions, settings, and outputs.

### What to analyze

- What risks are controlled?
- What risks remain open?
- Are limitations clearly stated and testable?

## Lesson 19: Portfolio / CV / Status Docs

### Research lens

These files translate technical evidence into professional claims. Read them as argument documents supported by artifacts.

### Terms to define while reading

- `evidence-backed claim`: statement tied to code/tests/docs/results.
- `scope`: what the repository actually demonstrates.
- `limitation statement`: explicit boundary preventing overclaiming.

### What to analyze

- Does each skill claim point to proof?
- Are limitations disclosed clearly?
- Is the communication accurate for a technical reviewer?

## Reusable Study Prompt (Any Data Science Module)

`Explain this module like a research analyst teaching a beginner. Define the important terms first. Then explain the question being answered, the data/pipeline/model used, the assumptions, the evidence produced, the risks/limitations, and exactly what I should do next in the terminal and why.`
