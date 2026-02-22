"""Data quality metric calculations for institutional quality gates."""

from __future__ import annotations

from typing import Any


def _to_records(df: Any) -> list[dict[str, Any]]:
    if hasattr(df, "to_dict") and df.__class__.__name__ == "DataFrame":
        return [dict(row) for row in df.to_dict(orient="records")]
    if isinstance(df, list):
        return [dict(row) for row in df]
    raise TypeError("Expected pandas DataFrame or list of dictionaries")


def _numeric_columns(records: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in records:
        keys.update(row.keys())
    numeric_cols: list[str] = []
    for key in sorted(keys):
        sample_values = [row.get(key) for row in records if row.get(key) is not None]
        if sample_values and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in sample_values
        ):
            numeric_cols.append(key)
    return numeric_cols


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    variance = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = index - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def _missing_rate(records: list[dict[str, Any]]) -> dict[str, float]:
    columns: set[str] = set()
    for row in records:
        columns.update(row.keys())
    total = len(records) or 1
    return {
        col: sum(1 for row in records if row.get(col) is None) / total
        for col in sorted(columns)
    }


def _duplicate_rate(records: list[dict[str, Any]], primary_id_col: str) -> float:
    if not records:
        return 0.0
    seen: set[Any] = set()
    duplicate_count = 0
    for row in records:
        key = row.get(primary_id_col)
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
    return duplicate_count / len(records)


def _outlier_rate(records: list[dict[str, Any]]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for col in _numeric_columns(records):
        values = [float(row[col]) for row in records if row.get(col) is not None]
        if len(values) < 4:
            rates[col] = 0.0
            continue
        sorted_values = sorted(values)
        q1 = _percentile(sorted_values, 0.25)
        q3 = _percentile(sorted_values, 0.75)
        iqr = q3 - q1
        if iqr == 0:
            rates[col] = 0.0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = sum(1 for v in values if v < lower or v > upper)
        rates[col] = outliers / len(values)
    return rates


def _drift_snapshot(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    snapshot: dict[str, dict[str, float]] = {}
    for col in _numeric_columns(records):
        values = [float(row[col]) for row in records if row.get(col) is not None]
        snapshot[col] = {
            "mean": _mean(values),
            "std": _sample_std(values),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    return snapshot


def compute_quality_metrics(
    *,
    dataset_name: str,
    raw_df: Any,
    valid_df: Any,
    schema_invalid_df: Any,
    domain_invalid_df: Any,
    primary_id_col: str,
) -> dict[str, Any]:
    """Compute quality metrics and return a JSON-serializable dictionary."""

    raw_records = _to_records(raw_df)
    valid_records = _to_records(valid_df)
    schema_invalid_records = _to_records(schema_invalid_df)
    domain_invalid_records = _to_records(domain_invalid_df)

    total_count = len(raw_records)
    schema_violation_count = len(schema_invalid_records)
    domain_violation_count = len(domain_invalid_records)
    valid_count = len(valid_records)

    metrics = {
        "dataset_name": dataset_name,
        "row_counts": {
            "total": total_count,
            "valid": valid_count,
            "schema_invalid": schema_violation_count,
            "domain_invalid": domain_violation_count,
        },
        "missing_rate_per_column": _missing_rate(valid_records),
        "duplicate_rate": _duplicate_rate(valid_records, primary_id_col),
        "outlier_rate_per_numeric_column": _outlier_rate(valid_records),
        "schema_violation_rate": (schema_violation_count / total_count) if total_count else 0.0,
        "domain_violation_rate": (domain_violation_count / total_count) if total_count else 0.0,
        "drift_snapshot": _drift_snapshot(valid_records),
    }
    return metrics

