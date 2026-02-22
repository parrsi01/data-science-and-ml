# Optimization Cheatsheet

## LP structure

- Decision variables: what the model is allowed to choose
- Objective function: what the model tries to optimize
- Constraints: hard limits the solution must satisfy

## Decision variable design

- Use domain names (`x_Africa`, `x_Asia`) for traceability
- Keep units consistent (all allocations in units, all costs in same currency)
- Add non-negativity and upper bounds early

## Constraint design

- Translate policy rules into algebra first
- Use proportions as decimals (`0.30`, not `30`)
- Validate each constraint with small hand-check examples

## Debugging infeasible models

- Print total demand, total units, and budget before solving
- Temporarily relax one constraint at a time
- Inspect whether policy constraints conflict with budget/capacity limits

## Common modeling mistakes

- Mixing percentages and proportions
- Forgetting demand caps (`x <= demand`)
- Overweighting a secondary objective without sensitivity analysis
- Treating optimization output as final policy without human review

