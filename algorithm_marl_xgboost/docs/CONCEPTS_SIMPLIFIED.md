# Concepts Simplified (MARL + XGBoost Institutional Module)

## Anomaly Detection

Finding records or events that look abnormal compared with usual behavior.

## Federated Learning (Decentralized)

Multiple agents train locally and exchange limited updates directly with peers instead of sending all data to one server.

## Non-IID Data

Data distributions differ across agents (for example, one site sees mostly one traffic type while another sees different patterns).

## Dirichlet Partitioning

A controlled way to split data so agents get different label and feature distributions.

## MARL (Multi-Agent Reinforcement Learning)

Multiple agents learn decision policies (here: who to communicate with) based on rewards over rounds.

## Reward

A score that encourages useful behavior (better F1) and discourages costly behavior (too much communication/energy).

## Trust Score

A running estimate of how useful or reliable a neighbor's updates have been.

## Trust-Weighted Aggregation

Combining peer updates by giving more weight to neighbors with higher trust scores.

## Topology

The communication graph describing which agents can talk to which other agents.

## Communication Budget

A limit on how much peer communication an agent can use per round.

## Feature Importance Vector

A compact summary of which input features mattered most in a local XGBoost model.

## Baseline

A simpler method used for comparison (for example, local-only training or naive peer averaging).

## Drift Snapshot

A check of how one distribution differs from another (for example, agent partition vs global data).

## Auditability

The ability to inspect configs, logs, and artifacts to understand what happened in an experiment.

## Determinism (Where Possible)

Using fixed seeds and stable ordering so repeated runs produce the same results unless external libraries introduce minor variation.

