"""Schema models for institutional data quality validation.

Uses Pydantic when available. In offline environments without Pydantic,
provides a compatible fallback with ``model_validate`` and ``model_dump``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

try:  # pragma: no cover - environment dependent
    from pydantic import BaseModel, Field, field_validator  # type: ignore[import-not-found]

    class FlightRow(BaseModel):
        flight_id: str
        dep_airport: str = Field(min_length=3, max_length=3)
        arr_airport: str = Field(min_length=3, max_length=3)
        scheduled_dep: datetime
        actual_dep: Optional[datetime] = None
        delay_minutes: Optional[int] = Field(default=None, ge=0)
        passenger_count: int = Field(ge=0)
        fuel_consumption_kg: float = Field(ge=0)

        @field_validator("dep_airport", "arr_airport")
        @classmethod
        def _airport_upper(cls, value: str) -> str:
            return value.upper()

    class HumanitarianShipmentRow(BaseModel):
        shipment_id: str
        region: str
        item_type: str
        quantity: int = Field(ge=0)
        priority: int = Field(ge=1, le=5)
        status: str

    class CERNEventRow(BaseModel):
        event_id: str
        detector: str
        energy_gev: float = Field(ge=0)
        is_rare_event: bool
        recorded_at: datetime

except Exception:  # pragma: no cover - main path in this environment

    @dataclass(slots=True)
    class _ValidatedRow:
        data: dict[str, Any]

        def model_dump(self) -> dict[str, Any]:
            return dict(self.data)


    class _SchemaBase:
        required_fields: dict[str, tuple[type[Any] | tuple[type[Any], ...], dict[str, Any]]] = {}

        @classmethod
        def _parse_datetime(cls, value: Any, field: str) -> datetime:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    # Accept ISO-ish strings, including trailing Z.
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError(f"{field}: invalid datetime format") from exc
            raise ValueError(f"{field}: expected datetime")

        @classmethod
        def _coerce(cls, field: str, value: Any, expected: Any) -> Any:
            if value is None:
                return None
            if expected is datetime:
                return cls._parse_datetime(value, field)
            if expected is str:
                if not isinstance(value, str):
                    raise ValueError(f"{field}: expected str")
                return value
            if expected is int:
                if isinstance(value, bool):
                    raise ValueError(f"{field}: expected int")
                if isinstance(value, int):
                    return value
                raise ValueError(f"{field}: expected int")
            if expected is float:
                if isinstance(value, bool):
                    raise ValueError(f"{field}: expected float")
                if isinstance(value, (int, float)):
                    return float(value)
                raise ValueError(f"{field}: expected float")
            if expected is bool:
                if isinstance(value, bool):
                    return value
                raise ValueError(f"{field}: expected bool")
            raise ValueError(f"{field}: unsupported expected type")

        @classmethod
        def model_validate(cls, data: dict[str, Any]) -> _ValidatedRow:
            if not isinstance(data, dict):
                raise ValueError("row must be a dict")
            normalized: dict[str, Any] = {}
            for field, (expected_type, rules) in cls.required_fields.items():
                if field not in data:
                    raise ValueError(f"{field}: missing")
                value = data[field]
                optional = bool(rules.get("optional", False))
                if value is None:
                    if optional:
                        normalized[field] = None
                        continue
                    raise ValueError(f"{field}: null not allowed")

                coerced = cls._coerce(field, value, expected_type)
                if expected_type is str:
                    text = coerced
                    min_len = rules.get("min_length")
                    max_len = rules.get("max_length")
                    if min_len is not None and len(text) < min_len:
                        raise ValueError(f"{field}: min_length={min_len}")
                    if max_len is not None and len(text) > max_len:
                        raise ValueError(f"{field}: max_length={max_len}")
                    if rules.get("upper"):
                        text = text.upper()
                    allowed = rules.get("allowed")
                    if allowed is not None and text not in allowed:
                        raise ValueError(f"{field}: unexpected value")
                    normalized[field] = text
                    continue

                if isinstance(coerced, (int, float)):
                    ge = rules.get("ge")
                    le = rules.get("le")
                    if ge is not None and coerced < ge:
                        raise ValueError(f"{field}: must be >= {ge}")
                    if le is not None and coerced > le:
                        raise ValueError(f"{field}: must be <= {le}")

                normalized[field] = coerced

            return _ValidatedRow(normalized)


    class FlightRow(_SchemaBase):
        required_fields = {
            "flight_id": (str, {}),
            "dep_airport": (str, {"min_length": 3, "max_length": 3, "upper": True}),
            "arr_airport": (str, {"min_length": 3, "max_length": 3, "upper": True}),
            "scheduled_dep": (datetime, {}),
            "actual_dep": (datetime, {"optional": True}),
            "delay_minutes": (int, {"optional": True, "ge": 0}),
            "passenger_count": (int, {"ge": 0}),
            "fuel_consumption_kg": (float, {"ge": 0}),
        }


    class HumanitarianShipmentRow(_SchemaBase):
        required_fields = {
            "shipment_id": (str, {}),
            "region": (str, {}),
            "item_type": (str, {}),
            "quantity": (int, {"ge": 0}),
            "priority": (int, {"ge": 1, "le": 5}),
            "status": (str, {}),
        }


    class CERNEventRow(_SchemaBase):
        required_fields = {
            "event_id": (str, {}),
            "detector": (str, {}),
            "energy_gev": (float, {"ge": 0}),
            "is_rare_event": (bool, {}),
            "recorded_at": (datetime, {}),
        }

