# Topology Cheatsheet

## Supported Topologies

- `ring`: each agent connects to two neighbors
- `star`: one hub connects to all others
- `fully_connected`: every agent can talk to every other agent
- `random`: probabilistic graph with deterministic seed

## Typical Errors + Fixes

- Isolated nodes in random graph: ensure fallback connectivity or use ring/star
- Comparing topologies unfairly: keep data partitions and seed fixed
- Hidden budget differences: report communication budget and neighbor counts

## Quick Definitions

- Topology: who can communicate
- Adjacency list: mapping of node -> neighbors
- Communication budget: how many neighbors can be selected each round

## Rebuild Without AI (Quick)

1. Start with deterministic ring/star/fully-connected
2. Add seeded random graph generation
3. Normalize adjacency list and sort neighbor IDs
4. Add tests for expected node degrees

