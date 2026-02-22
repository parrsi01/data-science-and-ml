"""Row-level and domain-level validation for institutional quality gates."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import logging
from typing import Any, Iterable

from data_quality.logging_config import get_logger, log_event


LOGGER = get_logger("data_quality.validators")


def _is_pandas_dataframe(value: Any) -> bool:
    return hasattr(value, "to_dict") and value.__class__.__name__ == "DataFrame"


def _to_records(df: Any) -> list[dict[str, Any]]:
    if _is_pandas_dataframe(df):
        return [dict(row) for row in df.to_dict(orient="records")]
    if isinstance(df, list):
        return [dict(row) for row in df]
    raise TypeError("Expected pandas DataFrame or list of dictionaries")


def _from_records_like(original: Any, records: list[dict[str, Any]]) -> Any:
    if _is_pandas_dataframe(original):
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except Exception:
            return records
        return pd.DataFrame(records)
    return records


def validate_dataframe(df: Any, schema_model: Any, dataset_name: str) -> tuple[Any, Any]:
    """Validate rows against a schema model and split valid/invalid rows.

    Invalid rows are preserved with a ``_validation_error`` field so failures are
    always reported and can be audited later.
    """

    records = _to_records(df)
    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    reasons = Counter()

    for row in records:
        try:
            validated = schema_model.model_validate(row)
            dumped = validated.model_dump() if hasattr(validated, "model_dump") else dict(row)
            valid_records.append(dict(dumped))
        except Exception as exc:
            reason = str(exc)
            reasons[reason] += 1
            invalid_row = dict(row)
            invalid_row["_validation_error"] = reason
            invalid_records.append(invalid_row)

    log_event(
        LOGGER,
        logging.INFO,
        event="schema_validation_summary",
        message="Completed schema validation",
        dataset_name=dataset_name,
        row_count=len(records),
    )
    log_event(
        LOGGER,
        logging.INFO,
        event="schema_validation_counts",
        message=(
            f"valid_count={len(valid_records)} invalid_count={len(invalid_records)} "
            f"invalid_reasons_top5={reasons.most_common(5)}"
        ),
        dataset_name=dataset_name,
        row_count=len(records),
    )

    return (
        _from_records_like(df, valid_records),
        _from_records_like(df, invalid_records),
    )


def _domain_rule_reason(dataset_name: str, row: dict[str, Any]) -> str | None:
    if dataset_name == "flights":
        if row.get("dep_airport") == row.get("arr_airport"):
            return "dep_airport must differ from arr_airport"
        actual_dep = row.get("actual_dep")
        scheduled_dep = row.get("scheduled_dep")
        if isinstance(actual_dep, datetime) and isinstance(scheduled_dep, datetime):
            if actual_dep < scheduled_dep:
                return "actual_dep must be >= scheduled_dep"
    elif dataset_name == "humanitarian_shipments":
        allowed = {"pending", "in_transit", "delivered", "cancelled"}
        if row.get("status") not in allowed:
            return "status not in allowed set"
    elif dataset_name == "cern_events":
        energy = row.get("energy_gev")
        if isinstance(energy, (int, float)) and not (0 <= float(energy) <= 1e7):
            return "energy_gev outside realistic simulation range"
    return None


def enforce_domain_rules(df: Any, dataset_name: str) -> tuple[Any, Any]:
    """Apply domain-specific rules and split valid/invalid rows."""

    records = _to_records(df)
    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    reasons = Counter()

    for row in records:
        reason = _domain_rule_reason(dataset_name, row)
        if reason is None:
            valid_records.append(dict(row))
        else:
            reasons[reason] += 1
            invalid_row = dict(row)
            invalid_row["_domain_rule_error"] = reason
            invalid_records.append(invalid_row)

    if invalid_records:
        log_event(
            LOGGER,
            logging.WARNING,
            event="domain_rule_violations",
            message=f"Domain rule violations detected: {reasons.most_common(5)}",
            dataset_name=dataset_name,
            row_count=len(records),
            error_code="DOMAIN_RULE",
        )
    else:
        log_event(
            LOGGER,
            logging.INFO,
            event="domain_rule_violations",
            message="No domain rule violations detected",
            dataset_name=dataset_name,
            row_count=len(records),
        )

    return (
        _from_records_like(df, valid_records),
        _from_records_like(df, invalid_records),
    )

