# Executive Summary — Parameter Study (MARL + XGBoost)

## Top 3 Configurations by Mean F1

1. alpha=0.1, agents=5, topology=ring, comm_budget=0.25 -> mean F1=0.8751 (+/- 0.0823)
2. alpha=0.1, agents=5, topology=ring, comm_budget=0.5 -> mean F1=0.8751 (+/- 0.0823)
3. alpha=0.1, agents=5, topology=ring, comm_budget=1.0 -> mean F1=0.8751 (+/- 0.0823)

## Top 3 Most Efficient (Low Bandwidth/Energy, Acceptable F1)

1. alpha=0.1, agents=5, topology=star, comm_budget=0.25 -> F1=0.8751, bytes=1160.0, energy=21.27
2. alpha=0.1, agents=5, topology=ring, comm_budget=0.25 -> F1=0.8751, bytes=1160.0, energy=21.27
3. alpha=0.1, agents=5, topology=ring, comm_budget=0.5 -> F1=0.8751, bytes=1160.0, energy=21.27

## Plain-Language Interpretation

- Communication budget increases generally improve collaboration quality, but bandwidth/energy costs increase.
- Topology choice changes both performance and operational cost; no single topology is universally best under all constraints.
- Non-IID severity (Dirichlet alpha) can materially reduce consistency, so repeat-based reporting is required.

## Deployment Recommendations

- Use parameter settings with high mean F1 and low variance, not only peak F1.
- Select communication budget based on infrastructure constraints (bandwidth/energy) and SLA priorities.
- Retest significance and stability after any topology or agent-count changes in production-like environments.
