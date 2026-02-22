"""Structured JSONL logging for data quality workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import logging
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs" / "data_quality.jsonl"


class JsonLineFormatter(logging.Formatter):
    """Format log records as JSON lines with a fixed institutional schema."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "event": getattr(record, "event", None),
            "message": record.getMessage(),
            "dataset_name": getattr(record, "dataset_name", None),
            "row_count": getattr(record, "row_count", None),
            "error_code": getattr(record, "error_code", None),
        }
        return json.dumps(payload, default=str, ensure_ascii=True)


def get_logger(name: str) -> logging.Logger:
    """Return a configured JSONL logger writing to ``logs/data_quality.jsonl``."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    existing = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == LOG_PATH
    ]
    if not existing:
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(JsonLineFormatter())
        logger.addHandler(handler)
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    *,
    event: str,
    message: str,
    dataset_name: str | None = None,
    row_count: int | None = None,
    error_code: str | None = None,
) -> None:
    """Emit a structured log record with standard institutional fields."""

    logger.log(
        level,
        message,
        extra={
            "event": event,
            "dataset_name": dataset_name,
            "row_count": row_count,
            "error_code": error_code,
        },
    )

