# Executive Summary: Air Traffic Flow & Delay Forecasting

## Busiest / Bottleneck Airports (Degree + Centrality)

| Airport | In Degree | Out Degree | Betweenness | PageRank | Clustering |
|---|---:|---:|---:|---:|---:|
| A09 | 3 | 5 | 0.2089 | 0.0297 | 0.1071 |
| A04 | 7 | 3 | 0.2087 | 0.1049 | 0.2222 |
| A12 | 4 | 8 | 0.2034 | 0.1023 | 0.3111 |
| A17 | 4 | 7 | 0.1827 | 0.0377 | 0.2545 |
| A19 | 4 | 5 | 0.1661 | 0.0554 | 0.3056 |

## Delay Model Metrics

- Accuracy: 0.6599
- Precision: 0.6717
- Recall: 0.6523
- F1: 0.6619
- ROC-AUC: 0.7166

## Top Delay Predictors (XGBoost Feature Importance)

- `numerical__congestion_index`: 0.0947
- `numerical__distance_km`: 0.0712
- `categorical__dep_A11`: 0.0513
- `categorical__dep_A12`: 0.0423
- `numerical__arr_in_degree`: 0.0297
- `numerical__arr_out_degree`: 0.0279
- `categorical__dep_A18`: 0.0247
- `categorical__dep_A05`: 0.0198

## Forecast Trend Summary

- Forecast method: prophet
- Horizon (days): 14
- Trend label: rising
- Trend delta (minutes): 1.152
- Notes: Prophet forecast completed

## Operational Recommendations

- Prioritize flow-management staffing or slot coordination at A09 and A04 due to high betweenness/PageRank bottleneck risk.
- Increase focus on recall-oriented threshold tuning for delay alerts to reduce missed delay-risk flights.
- Use forecast trend and bottleneck airports together to pre-position gate, ramp, and turnaround resources during expected congestion windows.