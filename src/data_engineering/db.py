"""Database utilities for local PostgreSQL pipelines.

This module reads ``DATABASE_URL`` from the environment and provides SQLAlchemy
engine/session helpers. Imports are lazy so the repository remains importable in
offline environments where SQLAlchemy/psycopg2 are not installed yet.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
from typing import Any, Iterator


DATABASE_URL_ENV = "DATABASE_URL"
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg2://ds_user:ds_password@localhost:5432/institutional_lab"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQL_SCHEMA_PATH = PROJECT_ROOT / "scripts" / "sql" / "001_create_schema.sql"

_ENGINE: Any | None = None


def load_env_file(path: Path | None = None) -> None:
    """Load simple ``KEY=VALUE`` pairs from a local env file if present."""

    env_path = path or (PROJECT_ROOT / "configs" / "db.env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_database_url() -> str:
    """Return the database URL from environment or project default."""

    load_env_file()
    return os.environ.get(DATABASE_URL_ENV, DEFAULT_DATABASE_URL)


def _sqlalchemy_api() -> tuple[Any, Any, Any]:
    """Import SQLAlchemy pieces lazily and raise a clear error if unavailable."""

    try:
        from sqlalchemy import create_engine, text  # type: ignore[import-not-found]
        from sqlalchemy.orm import sessionmaker  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "SQLAlchemy is required for database operations. Install `sqlalchemy` "
            "and `psycopg2-binary`, then retry."
        ) from exc
    return create_engine, text, sessionmaker


def get_engine(*, echo: bool = False, refresh: bool = False) -> Any:
    """Create (or reuse) a SQLAlchemy engine for ``DATABASE_URL``."""

    global _ENGINE
    if _ENGINE is not None and not refresh:
        return _ENGINE

    create_engine, _, _ = _sqlalchemy_api()
    _ENGINE = create_engine(get_database_url(), echo=echo, future=True)
    return _ENGINE


@contextmanager
def get_session() -> Iterator[Any]:
    """Yield a SQLAlchemy session and handle commit/rollback safely."""

    _, _, sessionmaker = _sqlalchemy_api()
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def database_healthcheck() -> tuple[bool, str]:
    """Return ``(healthy, message)`` for DB connectivity."""

    try:
        _, text, _ = _sqlalchemy_api()
        engine = get_engine()
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, "Database reachable"
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, str(exc)


def run_sql_script(sql_path: Path | None = None) -> str:
    """Execute a SQL script file against the configured database."""

    path = sql_path or DEFAULT_SQL_SCHEMA_PATH
    _, text, _ = _sqlalchemy_api()
    engine = get_engine()
    sql_text = path.read_text(encoding="utf-8")
    with engine.begin() as connection:
        connection.execute(text(sql_text))
    return f"Schema SQL executed: {path}"

