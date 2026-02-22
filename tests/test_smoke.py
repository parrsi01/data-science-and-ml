from pathlib import Path


def test_required_directories_exist() -> None:
    required = [
        "docs",
        "cheatsheets",
        "src",
        "tests",
        "models",
        "datasets",
        "notebooks",
        "reports",
        "configs",
        "docker",
        "mlops",
        "algorithm_marl_xgboost",
    ]
    for name in required:
        assert Path(name).exists(), f"Missing required directory: {name}"
