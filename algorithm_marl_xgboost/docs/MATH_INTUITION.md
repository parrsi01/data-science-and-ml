# Math Intuition (Plain Language)

## What “non-IID” means

IID means data points are sampled from the same distribution. Non-IID means each agent may see a different distribution, which is realistic in institutional systems (different regions, sites, or detectors).

## Why Dirichlet partitioning models non-IID

For each class label, Dirichlet sampling produces different proportions for each agent. Small alpha values create more uneven splits; larger alpha values make agent distributions more similar.

Simple intuition:
- `alpha` low -> agents specialize (high heterogeneity)
- `alpha` high -> agents look more similar (low heterogeneity)

## What rewards are optimizing (tradeoff)

The reward is designed to improve detection quality while controlling communication and energy.

Conceptually:

```text
reward = (F1 improvement)
         - (communication penalty)
         - (energy penalty)
```

This does not directly optimize global loss; it shapes agent behavior toward useful collaboration.

## What trust-weighted aggregation is (simple)

Each agent receives peer updates and combines them using trust scores.

If neighbor A has been consistently helpful and neighbor B has not, A gets more weight in the aggregate.

Conceptually:

```text
aggregate_update = sum( trust_i * update_i ) / sum( trust_i )
```

This helps reduce the influence of weak or noisy neighbors without needing a central server.

