# Metrics Cheatsheet (Performance + Traffic + Energy)

## Core Model Metrics

- Accuracy: overall correctness (can mislead under imbalance)
- Precision: fraction of predicted anomalies that were actually anomalies
- Recall: fraction of actual anomalies detected
- F1: balance of precision and recall
- ROC-AUC: ranking quality across thresholds

## Operational Cost Metrics

- `bytes_sent_total`: total communication payload size per round
- `avg_latency_ms`: simulated latency across peer exchanges
- `packet_loss_mean`: simulated message loss rate
- `energy_total`: simulated communication/training energy cost

## Typical Errors + Fixes

- Reporting only F1: add traffic/energy to show tradeoffs
- Comparing different rounds counts: normalize protocol first
- Ignoring seed variance: run repeats and summarize mean/std

## Rebuild Without AI (Quick)

1. Compute local metrics per agent
2. Aggregate mean metrics per round
3. Simulate traffic/energy events per communication action
4. Save per-round JSON/TXT/PNG and final CSV/summary

