# Humanitarian Logistics Optimization (UN-Style)

- Author: Simon Parris
- Date: 2026-02-22

## What linear programming is (simple)

Linear programming (LP) is a method for finding the best allocation of limited resources while obeying fixed rules. It is useful when decisions must be transparent and repeatable.

## What an objective function is

The objective function is the formula the optimizer tries to improve, such as minimizing unmet demand or increasing coverage for high-priority regions.

## What constraints are

Constraints are hard limits the solution must obey, such as budget caps, shipment limits, and minimum policy coverage rules.

## Why fairness matters in allocation

A cost-efficient solution can still over-serve some regions and under-serve others. Fairness indicators help reveal uneven allocation patterns for human review.

## Institutional implications (UN context)

UN-style planning often balances urgency, cost, risk, and equity. A reproducible optimization baseline provides a defensible starting point for coordination and audit review.

## How to rebuild without AI

1. Create a YAML config with regions, resources, and policy constraints
2. Generate reproducible regional demand data with a fixed seed
3. Build a PuLP LP model with one allocation variable per region
4. Add budget, unit, demand-cap, and priority-share constraints
5. Solve with CBC and export the allocation table
6. Run budget and priority-weight sensitivity scenarios
7. Save JSON and PNG artifacts for offline review
8. Add tests for constraints and artifact creation
9. Run the CLI and review the executive summary outputs

## JIRA-Style Ticket Examples

### HOPT-101: Implement Baseline Humanitarian Allocation LP

- Type: Story
- Goal: Provide a reproducible LP model for resource allocation with budget and priority constraints.
- Acceptance Criteria:
  - Config-driven demand generation works
  - `allocation_results.csv` is saved with allocation and unmet demand columns
  - CLI prints an executive summary with utilization and under-served region

### HOPT-102: Add Trade-off Sensitivity Analysis and Executive Visuals

- Type: Task
- Goal: Quantify allocation trade-offs under different budgets and priority weights for leadership review.
- Acceptance Criteria:
  - `sensitivity_analysis.json` saved with budget and priority scenarios
  - `budget_vs_unmet_demand.png` and `weight_vs_priority_coverage.png` saved
  - Executive summary Markdown references all produced artifacts

