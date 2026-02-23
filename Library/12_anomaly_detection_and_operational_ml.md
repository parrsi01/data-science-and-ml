# Anomaly Detection and Operational ML

---

> **Field** — Applied Machine Learning, MLOps
> **Scope** — Detecting unusual data points, deploying
> models as APIs, building dashboards, and operating
> machine learning systems in production

---

## Overview

Anomaly detection finds data points that do not match
the expected pattern. Operational ML is the practice of
running machine learning models in production, where
they serve predictions, monitor data quality, and alert
operators when something goes wrong. This topic bridges
the gap between building a model in a notebook and
running it reliably in a real system.

---

## Definitions

### `Anomaly Detection`

**Definition.**
Anomaly detection is the task of identifying data
points, events, or observations that deviate
significantly from the expected pattern. An anomaly
(also called an outlier) is something that does not
fit the normal behavior of the system.

**Context.**
Anomaly detection is used in fraud detection (unusual
transactions), cybersecurity (unusual network traffic),
manufacturing (defective products), and health
monitoring (unusual vital signs). The core challenge
is defining what "normal" means so you can recognize
what is not normal.

**Example.**
Using Isolation Forest for anomaly detection:

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Normal data + anomalies
np.random.seed(42)
normal = np.random.randn(100, 2)
anomalies = np.array([[5, 5], [-4, 4],
                       [4, -4]])
data = np.vstack([normal, anomalies])

# Fit detector
detector = IsolationForest(
    contamination=0.03,
    random_state=42
)
labels = detector.fit_predict(data)

# -1 = anomaly, 1 = normal
n_anomalies = (labels == -1).sum()
print(f"Found {n_anomalies} anomalies")
```

Other common methods:
- Local Outlier Factor (LOF)
- One-Class SVM
- Autoencoder reconstruction error
- Statistical methods (z-score, IQR)

---

### `Anomaly Score`

**Definition.**
An anomaly score is a numerical value that indicates
how unusual a data point is. Higher scores (or more
negative scores, depending on the method) mean more
anomalous. Instead of a binary "normal/anomaly"
label, scores provide a continuous measure of
abnormality.

**Context.**
Scores are more useful than binary labels because they
let you set different thresholds for different
situations. A hospital might flag anything above 0.7
for review, while a factory might only alert above
0.95. Scores also let you rank anomalies by severity
and prioritize the most suspicious ones.

**Example.**
```python
from sklearn.ensemble import IsolationForest
import numpy as np

np.random.seed(42)
data = np.random.randn(100, 2)

detector = IsolationForest(random_state=42)
detector.fit(data)

# Get anomaly scores
# More negative = more anomalous
scores = detector.score_samples(data)

print(f"Score range: {scores.min():.3f} "
      f"to {scores.max():.3f}")

# Find most anomalous point
worst_idx = scores.argmin()
print(f"Most anomalous: index {worst_idx}")
print(f"  Score: {scores[worst_idx]:.3f}")
print(f"  Values: {data[worst_idx]}")
```

---

### `Operational ML`

**Definition.**
Operational ML (also called MLOps) is the practice of
deploying, monitoring, and maintaining machine learning
models in production systems. It covers everything that
happens after model training: serving predictions,
monitoring performance, detecting data drift, and
updating models.

**Context.**
Most ML value comes from production deployment, not
notebook experiments. A fraud detection model is
worthless if it cannot score transactions in real time.
Operational ML ensures models are reliable, fast, and
accurate in the real world, not just in training.

**Example.**
A typical operational ML pipeline:

```
1. Train model (offline, batch)
2. Package model as API endpoint
3. Deploy to server (FastAPI + Uvicorn)
4. Monitor incoming data for drift
5. Log predictions and ground truth
6. Alert when performance degrades
7. Retrain and redeploy as needed
```

Key metrics to monitor:
- Prediction latency (milliseconds)
- Throughput (requests per second)
- Error rate (failed predictions)
- Data drift (input distribution shift)
- Model accuracy (when labels available)

---

### `Quality + Drift Preconditions`

**Definition.**
Quality and drift preconditions are checks that run
before a model makes predictions. Quality checks
verify that input data is valid (no missing values,
correct types, within expected ranges). Drift checks
verify that the input distribution has not shifted
away from what the model was trained on.

**Context.**
A model trained on data from 2023 may give bad
predictions if 2025 data looks completely different.
Precondition checks catch these problems before they
cause harm. They are the guardrails of operational ML,
preventing the model from silently producing garbage
outputs.

**Example.**
```python
import numpy as np

def check_preconditions(input_data, reference):
    """Check data quality and drift before
    making predictions."""
    issues = []

    # Quality: check for missing values
    if np.any(np.isnan(input_data)):
        issues.append("Missing values detected")

    # Quality: check value ranges
    if np.any(input_data < 0):
        issues.append(
            "Negative values unexpected"
        )

    # Drift: compare means
    input_mean = np.mean(input_data)
    ref_mean = np.mean(reference)
    drift = abs(input_mean - ref_mean) / ref_mean

    if drift > 0.2:  # 20% threshold
        issues.append(
            f"Mean drift: {drift:.1%}"
        )

    # Drift: compare standard deviations
    input_std = np.std(input_data)
    ref_std = np.std(reference)
    std_drift = abs(input_std - ref_std) / ref_std

    if std_drift > 0.3:
        issues.append(
            f"Variance drift: {std_drift:.1%}"
        )

    return issues

# Usage
issues = check_preconditions(
    new_data, training_data
)
if issues:
    print("WARNING:", issues)
else:
    predictions = model.predict(new_data)
```

---

### `FastAPI`

**Definition.**
FastAPI is a modern Python web framework for building
APIs. It is designed for speed (both development speed
and runtime speed) and automatically generates
interactive documentation. It is the most popular
choice for serving ML models as HTTP endpoints.

**Context.**
FastAPI turns your Python function into an API that
any application can call over the network. Instead of
running a model in a notebook, you wrap it in a
FastAPI endpoint so a web app, mobile app, or other
service can send data and get predictions back. It
handles input validation, error responses, and
documentation automatically.

**Example.**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Anomaly Detection API")

class InputData(BaseModel):
    values: list[float]

class Prediction(BaseModel):
    is_anomaly: bool
    score: float

@app.post("/predict", response_model=Prediction)
def predict(data: InputData):
    arr = np.array(data.values).reshape(1, -1)
    score = detector.score_samples(arr)[0]
    return Prediction(
        is_anomaly=score < -0.5,
        score=float(score)
    )
```

Install with:

```bash
pip install fastapi
```

---

### `Dashboard`

**Definition.**
A dashboard is a visual display that shows key
metrics and status information at a glance. In
operational ML, dashboards show model performance,
data quality metrics, anomaly counts, and system
health in real time.

**Context.**
Dashboards let non-technical stakeholders see what the
ML system is doing. They answer questions like "how
many anomalies did we detect today?" and "is the model
still accurate?" without requiring anyone to run code.
Common dashboard tools include Plotly Dash, Streamlit,
Grafana, and custom HTML pages served by FastAPI.

**Example.**
A simple Plotly Dash dashboard:

```python
import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

# Load anomaly detection results
df = pd.read_csv("anomaly_results.csv")

fig = px.scatter(
    df, x="timestamp", y="score",
    color="is_anomaly",
    title="Anomaly Scores Over Time"
)

app.layout = html.Div([
    html.H1("Anomaly Detection Dashboard"),
    html.P(
        f"Total anomalies: "
        f"{df['is_anomaly'].sum()}"
    ),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run(debug=True, port=8050)
```

A minimal FastAPI-served HTML dashboard:

```python
from fastapi.responses import HTMLResponse

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return f"""
    <html><body>
    <h1>System Status</h1>
    <p>Anomalies today: {count}</p>
    <p>Model accuracy: {accuracy:.1%}</p>
    </body></html>
    """
```

---

### `Uvicorn`

**Definition.**
Uvicorn is a lightning-fast ASGI server for Python. It
runs your FastAPI application and handles incoming HTTP
requests. Think of it as the engine that actually
serves your API to the network.

**Context.**
FastAPI defines what your API does; Uvicorn actually
runs it. In development, you start Uvicorn from the
command line. In production, you run multiple Uvicorn
workers behind a process manager like Gunicorn to
handle high traffic.

**Example.**
Starting a FastAPI app with Uvicorn:

```bash
# Development (single worker, auto-reload)
uvicorn app:app --reload --port 8000

# Production (multiple workers)
uvicorn app:app --host 0.0.0.0 \
    --port 8000 --workers 4
```

Or start Uvicorn from Python:

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

Install with:

```bash
pip install uvicorn
```

---

### `ASGI`

**Definition.**
ASGI (Asynchronous Server Gateway Interface) is a
standard interface between Python web applications
and web servers. It is the modern replacement for
WSGI, adding support for asynchronous code, WebSockets,
and long-lived connections.

**Context.**
You do not interact with ASGI directly; it works
behind the scenes. FastAPI is an ASGI application, and
Uvicorn is an ASGI server. Understanding ASGI matters
when you need to choose between frameworks (Flask uses
WSGI, FastAPI uses ASGI) or when configuring deployment
infrastructure.

**Example.**
The relationship between components:

```
Client (browser, app)
    |
    v
Uvicorn (ASGI server)
    |
    v
FastAPI (ASGI application)
    |
    v
Your code (model, database, etc.)
```

A minimal ASGI app without any framework:

```python
async def app(scope, receive, send):
    if scope["type"] == "http":
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                [b"content-type",
                 b"text/plain"]
            ]
        })
        await send({
            "type": "http.response.body",
            "body": b"Hello from ASGI"
        })
```

In practice, you never write raw ASGI code.
FastAPI handles all of this for you.

---

### `Endpoint`

**Definition.**
An endpoint is a specific URL path in your API that
performs a specific function. Each endpoint accepts
certain inputs and returns certain outputs. For
example, `/predict` might accept data and return
predictions, while `/health` returns system status.

**Context.**
Well-designed endpoints make your API easy to use. In
ML APIs, common endpoints include `/predict` (run the
model), `/health` (check if the service is running),
`/metrics` (get performance statistics), and `/retrain`
(trigger model retraining).

**Example.**
```python
from fastapi import FastAPI

app = FastAPI()

# GET endpoint (retrieve information)
@app.get("/health")
def health():
    return {"status": "healthy"}

# POST endpoint (send data, get result)
@app.post("/predict")
def predict(data: dict):
    result = model.predict(data["features"])
    return {"prediction": result.tolist()}

# GET with path parameters
@app.get("/anomalies/{date}")
def get_anomalies(date: str):
    return {"date": date, "count": 42}
```

After starting the server, access endpoints at:
- `http://localhost:8000/health`
- `http://localhost:8000/docs` (auto docs)

---

### `JSON Response`

**Definition.**
A JSON response is data sent back from an API in JSON
(JavaScript Object Notation) format. JSON uses key-
value pairs and is readable by virtually every
programming language. It is the standard format for
API communication.

**Context.**
When your ML API returns a prediction, it sends a JSON
response. The client (web app, mobile app, another
service) parses this JSON to extract the results.
FastAPI automatically converts Python dictionaries and
Pydantic models to JSON responses.

**Example.**
FastAPI automatically returns JSON:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/status")
def status():
    # This dict is automatically
    # converted to JSON
    return {
        "model_version": "2.1",
        "predictions_today": 1453,
        "avg_latency_ms": 12.3,
        "anomalies_detected": 7
    }
```

The client receives:

```json
{
    "model_version": "2.1",
    "predictions_today": 1453,
    "avg_latency_ms": 12.3,
    "anomalies_detected": 7
}
```

For custom responses:

```python
from fastapi.responses import JSONResponse

@app.get("/custom")
def custom():
    return JSONResponse(
        content={"result": "ok"},
        status_code=200,
        headers={"X-Model": "v2"}
    )
```

---

### `Health Check Endpoint`

**Definition.**
A health check endpoint is an API route (typically
`/health` or `/healthz`) that returns whether the
service is running and functioning correctly. It is
the simplest possible endpoint: it receives a request
and says "I am alive."

**Context.**
Health checks are critical for production systems.
Load balancers, container orchestrators (Kubernetes),
and monitoring tools poll the health endpoint to
decide whether to send traffic to this instance or
restart it. A health check that returns errors
triggers automatic recovery.

**Example.**
```python
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

startup_time = datetime.now()

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "uptime_seconds": (
            datetime.now() - startup_time
        ).total_seconds(),
        "model_loaded": model is not None
    }
```

A more thorough health check:

```python
@app.get("/health")
def health():
    checks = {}

    # Check model is loaded
    checks["model"] = model is not None

    # Check database connection
    try:
        db.ping()
        checks["database"] = True
    except Exception:
        checks["database"] = False

    healthy = all(checks.values())
    return {
        "status": "healthy" if healthy
                  else "unhealthy",
        "checks": checks
    }
```

---

### `Batch Inference`

**Definition.**
Batch inference means running a model on a large
collection of data points at once, rather than one
at a time. Instead of processing each request
individually, you gather many inputs and score them
all in a single operation.

**Context.**
Batch inference is used when you do not need real-time
predictions. Examples: scoring all customers overnight
for a marketing campaign, detecting anomalies across
yesterday's transactions, or generating
recommendations for all users weekly. It is more
efficient than real-time inference because it avoids
per-request overhead.

**Example.**
```python
import numpy as np
import pandas as pd

def batch_predict(model, input_path, output_path):
    """Run model on an entire dataset."""
    df = pd.read_parquet(input_path)

    # Score all rows at once (vectorized)
    features = df[["f1", "f2", "f3"]].values
    predictions = model.predict(features)
    scores = model.predict_proba(features)[:, 1]

    # Attach predictions to data
    df["prediction"] = predictions
    df["score"] = scores

    # Save results
    df.to_parquet(output_path)
    print(f"Scored {len(df)} rows")
    return df

# Usage
batch_predict(
    model,
    "data/customers.parquet",
    "data/customers_scored.parquet"
)
```

Batch vs real-time trade-offs:
- **Batch:** higher throughput, lower cost,
  results available after processing completes
- **Real-time:** immediate results, higher cost,
  one request at a time

---

### `Triage`

**Definition.**
Triage is the process of sorting and prioritizing
detected anomalies or alerts based on their severity,
urgency, or potential impact. Not all anomalies are
equally important; triage helps you focus on the ones
that matter most.

**Context.**
An anomaly detection system might flag hundreds of
items per day. Without triage, operators face alert
fatigue (too many alerts, so they ignore all of them).
Triage assigns severity levels, groups related alerts,
and surfaces the most critical issues first.

**Example.**
```python
import pandas as pd

def triage_anomalies(anomalies_df):
    """Sort and prioritize detected anomalies."""

    # Assign severity based on score
    def severity(score):
        if score > 0.9:
            return "critical"
        elif score > 0.7:
            return "high"
        elif score > 0.5:
            return "medium"
        else:
            return "low"

    df = anomalies_df.copy()
    df["severity"] = df["score"].apply(severity)

    # Sort: critical first, then by score
    severity_order = {
        "critical": 0, "high": 1,
        "medium": 2, "low": 3
    }
    df["sort_key"] = df["severity"].map(
        severity_order
    )
    df = df.sort_values(
        ["sort_key", "score"],
        ascending=[True, False]
    )

    # Summary
    print("Triage summary:")
    print(df["severity"].value_counts())

    return df.drop(columns=["sort_key"])
```

---

### `Alert Threshold`

**Definition.**
An alert threshold is the boundary value above (or
below) which an anomaly score triggers an alert.
Scores beyond the threshold cause the system to
notify operators. Setting the right threshold balances
catching real problems (sensitivity) with avoiding
false alarms (specificity).

**Context.**
Threshold selection is one of the hardest parts of
anomaly detection. Too low a threshold and you get
flooded with false alarms (alert fatigue). Too high
and you miss real anomalies. The right threshold
depends on the cost of missing a real anomaly versus
the cost of investigating a false alarm.

**Example.**
```python
import numpy as np

def set_threshold(
    scores, method="percentile", value=95
):
    """Determine alert threshold."""

    if method == "percentile":
        # Alert on top 5% most anomalous
        threshold = np.percentile(scores, value)

    elif method == "std":
        # Alert beyond N standard deviations
        mean = np.mean(scores)
        std = np.std(scores)
        threshold = mean + value * std

    elif method == "fixed":
        # Use a manually set value
        threshold = value

    return threshold

# Usage
scores = model.score_samples(data)
threshold = set_threshold(
    scores, method="percentile", value=95
)

alerts = data[scores > threshold]
print(f"Threshold: {threshold:.3f}")
print(f"Alerts: {len(alerts)}")
```

Best practice: start with a high threshold (few
alerts) and gradually lower it as you build
confidence in the system. Log all scores so you
can retroactively analyze what you missed.

---

## See Also

- [Scaling and Distributed Processing](./11_scaling_and_distributed_processing.md)
- [Time Series and Forecasting](./10_time_series_and_forecasting.md)
- [Reproducibility and Governance](./14_reproducibility_and_governance.md)
- [Federated and Distributed Learning](./13_federated_and_distributed_learning.md)

---

> **Author** — Simon Parris | Data Science Reference Library
