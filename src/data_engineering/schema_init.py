"""Schema initialization entry point for the PostgreSQL pipeline."""

from __future__ import annotations

from pathlib import Path

from data_engineering.db import DEFAULT_SQL_SCHEMA_PATH, run_sql_script


def initialize_schema(sql_path: Path | None = None) -> str:
    """Execute the main schema SQL file and return a confirmation string."""

    return run_sql_script(sql_path or DEFAULT_SQL_SCHEMA_PATH)


def main() -> None:
    """CLI entry point for ``make db-init``."""

    print(initialize_schema())


if __name__ == "__main__":
    main()
