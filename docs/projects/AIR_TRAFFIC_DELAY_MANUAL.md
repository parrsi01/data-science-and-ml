# Air Traffic Flow & Delay Forecasting (IATA-Style)

- Author: Simon Parris
- Date: 2026-02-22

## What graph centrality means (simple)

Graph centrality measures how important an airport is in the route network. High-centrality airports often connect many routes or sit on key transfer paths.

## Why bottlenecks matter operationally

When traffic is concentrated through a few airports, delays can spread across the network. Bottleneck detection helps operations teams prioritize staffing, slot management, and recovery planning.

## Why forecasting is uncertainty management

Forecasting does not remove uncertainty; it helps teams plan for likely ranges of delay conditions. The goal is better resource allocation, not perfect prediction.

## How this maps to IATA-type analytics work

IATA-style analytics often combines operational event data, network flow analysis, and predictive models to support delay mitigation, capacity planning, and airport/airline coordination.

## How to rebuild without AI

1. Define a YAML config for simulation, modeling, and forecasting options
2. Simulate airports, routes, and flight operations with reproducible seeds
3. Build a route graph and compute centrality metrics
4. Join node metrics back to flight records
5. Train a delay model (classification or regression) with preprocessing
6. Save metrics, plots, and model artifacts
7. Aggregate daily delays and run Prophet (or ARIMA fallback)
8. Produce an executive summary with bottlenecks, predictors, and forecast trend
9. Add tests with a small config for fast validation

## JIRA-Style Ticket Examples

### AIROPS-101: Build Route Graph Bottleneck Analytics and Delay Predictor

- Type: Story
- Goal: Deliver a reproducible route-network + delay-model pipeline for operational monitoring.
- Acceptance Criteria:
  - `graph_metrics.csv` and `route_graph.png` generated
  - `model_metrics.json` and model artifact saved
  - CLI summary prints top-line metrics and bottleneck airports

### AIROPS-102: Add Daily Delay Forecasting and Executive Recommendations

- Type: Task
- Goal: Produce an operational planning forecast and plain-language bottleneck recommendations.
- Acceptance Criteria:
  - `delay_forecast.csv` and `delay_forecast.png` generated
  - `executive_summary.md` includes bottlenecks, feature importance, and forecast trend
  - Prophet fallback to ARIMA is handled without pipeline failure

