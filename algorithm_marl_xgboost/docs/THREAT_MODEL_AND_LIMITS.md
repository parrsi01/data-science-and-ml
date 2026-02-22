# Threat Model and Limits

## Honest-but-Faulty Nodes

These nodes are not malicious, but may produce poor updates due to bad local data, label skew, or unstable local training. Trust-weighted aggregation is intended to reduce their impact over time.

## Byzantine / Malicious Nodes (Conceptual Only)

This module does not implement Byzantine-robust defenses. A malicious node could send intentionally misleading updates. Future work may add robust aggregation, anomaly scoring on updates, or signed communication policies.

## Privacy Limits

- No differential privacy implemented
- No secure aggregation implemented
- No encryption implemented in this simulation

The module demonstrates decentralized behavior and auditability, not formal privacy guarantees.

## Data Leakage Risks

- Leakage can occur if validation data is reused improperly across rounds or baselines
- Per-agent SMOTE must only be applied on training partitions
- Synthetic data convenience can hide real-world preprocessing leakage risks

## Evaluation Pitfalls

- Comparing methods with different communication budgets is unfair
- Using only one seed can overstate improvements
- Reporting only F1 without traffic/energy cost hides operational tradeoffs
- Non-IID severity (`alpha`) strongly affects outcomes and must be reported

