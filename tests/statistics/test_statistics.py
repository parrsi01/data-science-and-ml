from __future__ import annotations

from pathlib import Path

from statistics.probability_models import (
    generate_synthetic_flight_delays,
    normal_pdf,
    summarize_synthetic_delays,
)
from statistics.hypothesis_testing import mean_confidence_interval, welch_t_test
from statistics.monte_carlo import REPORT_PATH, run_monte_carlo_simulations


def test_normal_distribution_integrates_approximately_to_one() -> None:
    step = 0.001
    x = -6.0
    total = 0.0
    while x <= 6.0:
        total += normal_pdf(x) * step
        x += step
    assert 0.995 <= total <= 1.005


def test_confidence_interval_bounds_make_sense() -> None:
    delays = generate_synthetic_flight_delays(count=1000, mean_delay=12.0, std_dev=8.0)
    summary = summarize_synthetic_delays(delays)
    ci = mean_confidence_interval(delays, confidence=0.95)
    assert ci["lower_bound"] < ci["upper_bound"]
    assert ci["lower_bound"] <= summary["mean_delay"] <= ci["upper_bound"]


def test_monte_carlo_returns_expected_shape() -> None:
    result = run_monte_carlo_simulations(iterations=10_000)
    congestion = result["congestion"]
    fuel = result["fuel_cost"]
    assert len(congestion["demand_samples"]) == 10_000
    assert len(fuel["cost_samples"]) == 10_000
    figure_path = Path(result["figure_path"])
    assert figure_path == REPORT_PATH
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_t_test_returns_float_p_value() -> None:
    sample_a = [12.0, 14.0, 11.0, 13.0, 15.0, 12.0]
    sample_b = [20.0, 22.0, 19.0, 21.0, 23.0, 20.0]
    result = welch_t_test(sample_a, sample_b)
    assert isinstance(result["p_value"], float)
    assert 0.0 <= result["p_value"] <= 1.0
