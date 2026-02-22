from __future__ import annotations

from pathlib import Path

from foundations.python_core import InstitutionalDataset
from foundations.numerical_computing import (
    DATASET_EXPORT_PATH,
    matrix_multiplication_demo,
    pandas_aviation_dataset_demo,
)


def test_institutional_dataset_describe_returns_string() -> None:
    dataset = InstitutionalDataset(name="test", size=10, source="unit")
    description = dataset.describe()
    assert isinstance(description, str)
    assert "InstitutionalDataset" in description


def test_schema_validation_works() -> None:
    dataset = InstitutionalDataset(name="test", size=10, source="unit")
    assert dataset.validate_schema(
        actual_columns=["flight_id", "delay_minutes"],
        required_columns=["flight_id"],
        column_types={"flight_id": "str"},
    )
    assert not dataset.validate_schema(
        actual_columns=["flight_id"],
        required_columns=["flight_id", "fuel_consumption"],
        column_types={"flight_id": "str", "fuel_consumption": "float"},
    )


def test_matrix_multiplication_dimensions_correct() -> None:
    result = matrix_multiplication_demo()
    assert result.left_shape == (2, 3)
    assert result.right_shape == (3, 2)
    assert result.output_shape == (2, 2)


def test_dataset_export_exists() -> None:
    result = pandas_aviation_dataset_demo()
    export_path = Path(result["export_path"])
    assert export_path == DATASET_EXPORT_PATH
    assert export_path.exists()
    assert export_path.stat().st_size > 0
