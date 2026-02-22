# Drift Report

## Numeric Drift

| Feature | Mean Shift | Std Shift | KS Statistic | KS p-value |
|---|---:|---:|---:|---:|
| metric_a | -0.072962 | 0.073391 | 0.016250 | 0.362852 |
| metric_b | 0.254595 | -0.111937 | 0.024813 | 0.038239 |
| ratio_ab | -0.013707 | -0.001913 | 0.017375 | 0.285513 |
| rolling_mean_a | 0.117828 | 0.031003 | 0.021062 | 0.115280 |
| lag_metric_a | 0.457737 | -0.094275 | 0.021625 | 0.098826 |
| rolling_std_a | 0.085045 | -0.007519 | 0.019437 | 0.175775 |
| interaction_a_b | 19.599196 | -17.712940 | 0.016688 | 0.331276 |

## Categorical Drift

| Feature | TV Distance | Chi-square | p-value |
|---|---:|---:|---:|
| category | 0.001063 | 0.017720 | 0.999376 |