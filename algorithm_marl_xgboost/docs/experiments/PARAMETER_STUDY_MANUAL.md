# Parameter Studies & Statistical Rigor

- Author: Simon Parris
- Date: 2026-02-22

## What a parameter study is (simple)

A parameter study runs the same algorithm under different settings (for example, topology or communication budget) to measure how performance and costs change.

## Why repeats matter

Single runs can be misleading because random seeds affect partitions, training, and agent decisions. Repeats help estimate average behavior and variability.

## What a boxplot shows

A boxplot summarizes a distribution across repeats:
- center line: median
- box: middle 50% of values
- whiskers/outliers: spread and extreme values

This helps compare stability, not only mean performance.

## Why statistical significance matters

Significance testing helps estimate whether observed differences are likely due to real differences rather than random variation across repeats.

## Which test to use and why (simple)

- Wilcoxon signed-rank: good default for paired repeat results when normality is uncertain
- t-test (paired): useful when differences are approximately normal
- Mann-Whitney U: fallback when pairing is not valid

The harness compares MARL trust-weighted results against a naive decentralized baseline for both F1 and operational cost metrics.

## How to rebuild without AI

1. Create a study YAML with parameter grid + repeat count
2. Add a repeat runner that wraps the main experiment and stores repeat JSONs
3. Aggregate repeat results into CSV (mean/std/sem)
4. Add significance testing for paired baseline comparisons
5. Add plotting for boxplots and parameter curves
6. Generate executive and research summaries from the CSV/JSON artifacts
7. Add tests for grid generation, stats, and plotting

