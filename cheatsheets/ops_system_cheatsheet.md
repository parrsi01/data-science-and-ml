# Ops System Cheatsheet

## FastAPI basics

- Define app: `app = FastAPI()`
- JSON endpoint: `@app.get("/metrics")`
- HTML page: return `HTMLResponse`
- Local run: `uvicorn module:app --host 0.0.0.0 --port 8000`

## Operational logging basics

- Log batch start/end with row counts
- Log quality failures before inference
- Log drift flags with thresholds and top offending feature
- Keep logs structured (JSONL) for auditability

## Anomaly triage best practices

- Review top anomalies with source fields + probability
- Check quality and drift flags before acting on alerts
- Use thresholds as policy settings, not model truth
- Track false positives/negatives from operator feedback

## Common deployment mistakes

- Serving predictions without health endpoints
- Ignoring schema changes in upstream data
- Using hard-coded thresholds without review
- Running inference before validation/drift checks

