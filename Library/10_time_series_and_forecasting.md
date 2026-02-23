# Time Series and Forecasting

---

> **Field** — Statistical Forecasting, Temporal Analytics
> **Scope** — Time-indexed data, trend and seasonality
> decomposition, error metrics, and forecasting models
> including ARIMA, Prophet, and smoothing methods

---

## Overview

A time series is any data measured at regular time
intervals: daily temperatures, monthly sales, hourly
server traffic. Forecasting uses patterns in past data
to predict future values. The core challenge is
separating meaningful patterns (trend, seasonality)
from random noise, then projecting those patterns
forward in time.

---

## Definitions

### `Time Series`

**Definition.**
A time series is a sequence of data points ordered
by time. Each observation has a timestamp and a
value. The ordering matters: the value at time t
depends on what happened before it.

**Context.**
Time series data is everywhere: stock prices, weather
readings, website traffic, heart rate monitors, and
sensor readings. Standard machine learning models
treat rows as independent, but time series data is
inherently sequential. You need specialized methods
that respect the time ordering.

**Example.**
```python
import pandas as pd

# Create a simple time series
dates = pd.date_range(
    start="2025-01-01",
    periods=7,
    freq="D"
)
values = [100, 105, 98, 110, 107, 115, 120]

ts = pd.Series(values, index=dates)
print(ts)
# 2025-01-01    100
# 2025-01-02    105
# 2025-01-03     98
# 2025-01-04    110
# 2025-01-05    107
# 2025-01-06    115
# 2025-01-07    120
```

The key feature: the index is a datetime, and the
order of observations carries meaning.

---

### `Forecasting`

**Definition.**
Forecasting is the process of predicting future
values based on patterns observed in historical
data. A forecast extends the time series beyond the
last known observation into the future.

**Context.**
Forecasting drives business decisions: how much
inventory to order, how many staff to schedule,
whether to expand capacity. Good forecasts reduce
waste and missed opportunities. The quality of a
forecast is measured by how far off its predictions
are from reality (using error metrics like MAE and
RMSE).

**Example.**
```python
import pandas as pd
from prophet import Prophet

# Prepare data (Prophet requires 'ds' and 'y')
df = pd.DataFrame({
    "ds": pd.date_range("2024-01-01",
                         periods=365, freq="D"),
    "y": range(365)  # simplified trend
})

# Fit and forecast
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# The last 30 rows are predictions
print(forecast[["ds", "yhat"]].tail())
```

---

### `Trend`

**Definition.**
A trend is the long-term direction of a time series.
It shows whether values are generally going up, going
down, or staying flat over an extended period. Trend
captures the "big picture" movement, ignoring short-
term fluctuations.

**Context.**
Identifying the trend is usually the first step in
time series analysis. A company needs to know if
sales are growing or declining before worrying about
seasonal spikes. Trend removal (detrending) makes it
easier to analyze other patterns in the data.

**Example.**
```python
import pandas as pd
import numpy as np

# Generate data with an upward trend
np.random.seed(42)
n = 100
trend = np.linspace(0, 50, n)
noise = np.random.normal(0, 3, n)
values = trend + noise

ts = pd.Series(values)

# Simple trend estimate: rolling mean
trend_estimate = ts.rolling(window=20).mean()

# Or use numpy polyfit for linear trend
slope, intercept = np.polyfit(range(n), values, 1)
print(f"Trend: +{slope:.2f} per time step")
```

A positive slope means an upward trend.
A slope near zero means no trend (flat).

---

### `Seasonality`

**Definition.**
Seasonality is a pattern that repeats at regular,
predictable intervals. Examples include higher retail
sales every December, more ice cream sold every
summer, or increased web traffic every Monday morning.
The pattern has a fixed period (daily, weekly, yearly).

**Context.**
Seasonality must be accounted for in forecasts, or
your predictions will be systematically wrong at
certain times. For example, if you ignore the December
sales spike, your forecast will underestimate December
and overestimate January every year.

**Example.**
```python
import pandas as pd
import numpy as np

# Generate data with weekly seasonality
days = 365
daily = np.sin(
    2 * np.pi * np.arange(days) / 7
) * 10  # weekly cycle

trend = np.linspace(100, 150, days)
noise = np.random.normal(0, 2, days)
values = trend + daily + noise

ts = pd.Series(
    values,
    index=pd.date_range("2025-01-01",
                         periods=days)
)

# Detect seasonality by day of week
by_day = ts.groupby(ts.index.dayofweek).mean()
print("Average by day of week:")
print(by_day)
```

If values are consistently higher on certain days,
that is weekly seasonality.

---

### `MAE (Mean Absolute Error)`

**Definition.**
MAE is the average of the absolute differences
between predicted values and actual values. It tells
you how far off your predictions are, on average,
in the same units as your data. Lower MAE means
better predictions.

**Context.**
MAE is the most intuitive error metric. If your MAE
is 5 and you are predicting daily sales in units,
your forecast is off by about 5 units per day on
average. MAE treats all errors equally, unlike RMSE
which penalizes large errors more heavily.

**Example.**
```python
import numpy as np
from sklearn.metrics import mean_absolute_error

actual = np.array([100, 110, 105, 120, 115])
predicted = np.array([102, 108, 107, 118, 112])

mae = mean_absolute_error(actual, predicted)
print(f"MAE: {mae:.1f}")
# MAE: 2.4

# Manual calculation
errors = np.abs(actual - predicted)
print(f"Individual errors: {errors}")
# [2, 2, 2, 2, 3]
print(f"Mean: {errors.mean():.1f}")
# Mean: 2.2
```

---

### `RMSE (Root Mean Squared Error)`

**Definition.**
RMSE is the square root of the average of squared
differences between predictions and actual values.
Like MAE, it measures prediction accuracy in the
same units as your data. Unlike MAE, it penalizes
large errors more heavily because of the squaring
step.

**Context.**
RMSE is the most commonly reported error metric in
forecasting competitions and academic papers. Because
it squares errors before averaging, a single very
wrong prediction will inflate RMSE much more than
MAE. Use RMSE when large errors are particularly
costly (like predicting hospital demand).

**Example.**
```python
import numpy as np
from sklearn.metrics import (
    mean_squared_error
)

actual = np.array([100, 110, 105, 120, 115])
predicted = np.array([102, 108, 107, 118, 112])

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)
print(f"RMSE: {rmse:.2f}")

# Manual calculation
errors = actual - predicted
squared = errors ** 2
mse = squared.mean()
rmse_manual = np.sqrt(mse)
print(f"RMSE (manual): {rmse_manual:.2f}")
```

RMSE is always greater than or equal to MAE.
If RMSE is much larger than MAE, it means you
have a few predictions that are very far off.

---

### `Prophet`

**Definition.**
Prophet is an open-source forecasting library created
by Meta (Facebook). It automatically detects trend
changes, yearly/weekly/daily seasonality, and the
effects of holidays. You give it a table with dates
and values, and it produces forecasts with uncertainty
intervals.

**Context.**
Prophet is popular because it works well "out of the
box" with minimal tuning. It is designed for business
time series that have strong seasonal patterns and
missing data. It is not the most accurate model for
every situation, but it is fast, interpretable, and
easy to use.

**Example.**
```python
from prophet import Prophet
import pandas as pd

# Prophet requires columns named 'ds' and 'y'
df = pd.read_csv("daily_sales.csv")
df.columns = ["ds", "y"]

# Fit
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.fit(df)

# Forecast 90 days ahead
future = model.make_future_dataframe(
    periods=90
)
forecast = model.predict(future)

# Plot
model.plot(forecast)
model.plot_components(forecast)
```

Install with:

```bash
pip install prophet
```

---

### `ARIMA`

**Definition.**
ARIMA stands for AutoRegressive Integrated Moving
Average. It is a classical statistical model for time
series forecasting. It has three components: AR
(using past values), I (differencing to remove
trends), and MA (using past forecast errors). The
model is specified by three numbers: (p, d, q).

**Context.**
ARIMA has been the workhorse of time series
forecasting for decades. It works best on univariate
(single variable) time series without complex
seasonality. For seasonal data, SARIMA adds seasonal
components. ARIMA requires stationary data (constant
mean and variance), which is why the "I" (integrated)
step performs differencing.

**Example.**
```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Load time series
ts = pd.read_csv(
    "monthly_data.csv",
    index_col="date",
    parse_dates=True
)["value"]

# Fit ARIMA(1,1,1)
# p=1: one autoregressive term
# d=1: one differencing step
# q=1: one moving average term
model = ARIMA(ts, order=(1, 1, 1))
result = model.fit()

# Summary
print(result.summary())

# Forecast next 12 periods
forecast = result.forecast(steps=12)
print(forecast)
```

Choosing p, d, q values is often done with
auto_arima from the pmdarima library:

```bash
pip install pmdarima
```

```python
from pmdarima import auto_arima
best = auto_arima(ts, seasonal=False)
print(best.order)  # e.g., (2, 1, 1)
```

---

### `Aggregation (temporal)`

**Definition.**
Temporal aggregation means combining fine-grained
time series data into coarser intervals. For example,
converting hourly data to daily data by summing or
averaging. This reduces noise and makes patterns
easier to see.

**Context.**
Raw data is often too granular for forecasting. Hourly
sensor readings might have too much noise; daily or
weekly aggregation smooths this out. Aggregation also
reduces computational cost when dealing with very long
time series. The choice of aggregation level (hourly,
daily, weekly, monthly) depends on the decision you
are supporting.

**Example.**
```python
import pandas as pd

# Hourly data
dates = pd.date_range(
    "2025-01-01", periods=168, freq="h"
)
values = range(168)
hourly = pd.Series(values, index=dates)

# Aggregate to daily
daily_sum = hourly.resample("D").sum()
daily_mean = hourly.resample("D").mean()

print("Daily sums:")
print(daily_sum)

# Aggregate to weekly
weekly = hourly.resample("W").sum()
print("\nWeekly sums:")
print(weekly)
```

Common resample frequencies:
- `"h"` = hourly
- `"D"` = daily
- `"W"` = weekly
- `"ME"` = month end
- `"QE"` = quarter end

---

### `Stationarity`

**Definition.**
A time series is stationary if its statistical
properties (mean, variance, autocorrelation) do not
change over time. In a stationary series, the data
fluctuates around a constant level with constant
spread. There is no trend and no changing seasonality.

**Context.**
Most classical forecasting models (like ARIMA) require
stationary data. If your data has a trend or changing
variance, you must transform it first. Common
transformations include differencing (subtracting
each value from the previous one) and log transforms.
The Augmented Dickey-Fuller test checks stationarity
statistically.

**Example.**
```python
from statsmodels.tsa.stattools import adfuller
import numpy as np

# Non-stationary: upward trend
trend_data = np.cumsum(
    np.random.normal(0.1, 1, 100)
)

result = adfuller(trend_data)
print(f"ADF statistic: {result[0]:.3f}")
print(f"p-value: {result[1]:.4f}")
# High p-value = NOT stationary

# Make it stationary by differencing
differenced = np.diff(trend_data)
result2 = adfuller(differenced)
print(f"\nAfter differencing:")
print(f"ADF statistic: {result2[0]:.3f}")
print(f"p-value: {result2[1]:.4f}")
# Low p-value (< 0.05) = stationary
```

Rule of thumb: if the ADF p-value is below 0.05,
the series is stationary.

---

### `Autocorrelation`

**Definition.**
Autocorrelation measures how much a time series value
is correlated with its own past values. High
autocorrelation at lag 1 means today's value strongly
predicts tomorrow's value. High autocorrelation at
lag 7 means today's value predicts next week's value.

**Context.**
Autocorrelation reveals the memory structure of your
data. It tells you how many past observations you
need to look at to make a good prediction. The
autocorrelation function (ACF) plot is one of the
first things you examine when analyzing a time series.
It also helps you choose the parameters for ARIMA
models.

**Example.**
```python
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import (
    plot_acf, plot_pacf
)
import matplotlib.pyplot as plt

# Generate data with strong lag-1 correlation
np.random.seed(42)
n = 200
data = [0]
for i in range(1, n):
    data.append(0.8 * data[-1]
                + np.random.normal())

ts = pd.Series(data)

# Plot autocorrelation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ts, ax=axes[0], lags=20)
plot_pacf(ts, ax=axes[1], lags=20)
plt.tight_layout()
plt.show()
```

If the ACF decays slowly, the series has strong
memory. If it cuts off sharply, the memory is
short. Seasonal data shows spikes at the seasonal
lag (e.g., lag 7 for weekly patterns).

---

### `Moving Average`

**Definition.**
A moving average replaces each data point with the
average of its surrounding points within a window.
It smooths out short-term fluctuations and highlights
the underlying trend. The window size controls how
much smoothing occurs.

**Context.**
Moving averages are used for two purposes: as a
simple forecasting method (predict the next value as
the average of the last N values) and as a data
preprocessing step to remove noise before analysis.
Larger windows produce smoother curves but respond
more slowly to real changes.

**Example.**
```python
import pandas as pd
import numpy as np

np.random.seed(42)
values = np.cumsum(np.random.randn(100)) + 50
ts = pd.Series(values)

# Simple moving average (window = 7)
sma_7 = ts.rolling(window=7).mean()

# Larger window (window = 21)
sma_21 = ts.rolling(window=21).mean()

# The first (window-1) values will be NaN
print("Original vs smoothed:")
print(pd.DataFrame({
    "original": ts,
    "SMA_7": sma_7,
    "SMA_21": sma_21
}).iloc[20:25])
```

Variations:
- **Simple Moving Average (SMA):** all points
  in the window weighted equally
- **Weighted Moving Average:** recent points
  get higher weight
- **Centered Moving Average:** window is
  centered on the current point (used for
  decomposition, not forecasting)

---

### `Exponential Smoothing`

**Definition.**
Exponential smoothing is a forecasting method that
gives more weight to recent observations and less
weight to older ones. The weights decrease
exponentially as observations get older. A smoothing
parameter (alpha) controls how quickly old data is
forgotten.

**Context.**
Exponential smoothing is one of the most widely used
forecasting methods in industry. It is fast, simple,
and often surprisingly accurate. The Holt-Winters
variant handles both trend and seasonality, making it
a strong baseline for many forecasting problems.

**Example.**
```python
from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    SimpleExpSmoothing
)
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n = 100
data = 50 + np.cumsum(
    np.random.randn(n) * 0.5
)
ts = pd.Series(data)

# Simple Exponential Smoothing
ses = SimpleExpSmoothing(ts).fit(
    smoothing_level=0.3
)
forecast_ses = ses.forecast(10)

# Holt-Winters (trend + seasonality)
hw = ExponentialSmoothing(
    ts,
    trend="add",
    seasonal=None
).fit()
forecast_hw = hw.forecast(10)

print("Next 10 predictions:")
print(forecast_hw)
```

Alpha values:
- **alpha close to 0:** slow adaptation,
  heavy smoothing, relies on history
- **alpha close to 1:** fast adaptation,
  little smoothing, relies on recent data
- **Typical range:** 0.1 to 0.3

---

## See Also

- [Statistical Foundations](./01_statistical_foundations.md)
- [Python and Numerical Computing](./02_python_and_numerical_computing.md)
- [Scaling and Distributed Processing](./11_scaling_and_distributed_processing.md)
- [Anomaly Detection and Operational ML](./12_anomaly_detection_and_operational_ml.md)

---

> **Author** — Simon Parris | Data Science Reference Library
