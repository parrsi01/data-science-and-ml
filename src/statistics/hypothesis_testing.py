"""Hypothesis testing and confidence interval utilities.

This module provides offline-safe statistical tests with numerical integration
for p-value approximation when SciPy is unavailable.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("At least one value is required")
    return sum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("At least two values are required")
    mu = _mean(values)
    return sum((x - mu) ** 2 for x in values) / (len(values) - 1)


def _sample_std(values: Sequence[float]) -> float:
    return math.sqrt(_sample_variance(values))


def _simpson_integrate(func, start: float, end: float, steps: int = 4000) -> float:
    """Numerically integrate ``func`` on [start, end] with Simpson's rule."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if steps % 2 == 1:
        steps += 1
    if end == start:
        return 0.0
    h = (end - start) / steps
    total = func(start) + func(end)
    for i in range(1, steps):
        x = start + i * h
        total += (4 if i % 2 else 2) * func(x)
    return total * h / 3.0


def _student_t_pdf(x: float, degrees_of_freedom: float) -> float:
    """Student t distribution PDF."""

    v = degrees_of_freedom
    numerator = math.gamma((v + 1.0) / 2.0)
    denominator = math.sqrt(v * math.pi) * math.gamma(v / 2.0)
    return numerator / denominator * (1.0 + (x * x) / v) ** (-(v + 1.0) / 2.0)


def _student_t_cdf(x: float, degrees_of_freedom: float) -> float:
    """Student t distribution CDF via symmetry and numerical integration."""

    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if x == 0:
        return 0.5
    abs_x = abs(x)
    integral = _simpson_integrate(
        lambda t: _student_t_pdf(t, degrees_of_freedom),
        0.0,
        abs_x,
        steps=4000,
    )
    cdf_pos = min(max(0.5 + integral, 0.0), 1.0)
    return cdf_pos if x > 0 else 1.0 - cdf_pos


def _chi_square_pdf(x: float, degrees_of_freedom: int) -> float:
    """Chi-square distribution PDF."""

    if x < 0:
        return 0.0
    k = float(degrees_of_freedom)
    coef = 1.0 / ((2.0 ** (k / 2.0)) * math.gamma(k / 2.0))
    return coef * (x ** (k / 2.0 - 1.0)) * math.exp(-x / 2.0)


def _chi_square_cdf(x: float, degrees_of_freedom: int) -> float:
    """Chi-square CDF via numerical integration."""

    if x <= 0:
        return 0.0
    start = 1e-9 if degrees_of_freedom <= 2 else 0.0
    integral = _simpson_integrate(
        lambda t: _chi_square_pdf(t, degrees_of_freedom),
        start,
        x,
        steps=5000,
    )
    return min(max(integral, 0.0), 1.0)


def welch_t_test(sample_a: Sequence[float], sample_b: Sequence[float]) -> dict[str, float]:
    """Perform a two-sided Welch t-test for difference in means.

    Returns a dictionary with means, t-statistic, degrees of freedom, and
    approximate p-value.
    """

    if len(sample_a) < 2 or len(sample_b) < 2:
        raise ValueError("Each sample must contain at least two values")

    mean_a = _mean(sample_a)
    mean_b = _mean(sample_b)
    var_a = _sample_variance(sample_a)
    var_b = _sample_variance(sample_b)
    n_a = float(len(sample_a))
    n_b = float(len(sample_b))

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        raise ValueError("Standard error is zero; samples may be constant")

    t_statistic = (mean_a - mean_b) / se
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = ((var_a / n_a) ** 2) / (n_a - 1.0) + ((var_b / n_b) ** 2) / (n_b - 1.0)
    degrees_of_freedom = numerator / denominator if denominator else n_a + n_b - 2.0

    cdf_at_abs_t = _student_t_cdf(abs(t_statistic), degrees_of_freedom)
    p_value = max(0.0, min(1.0, 2.0 * (1.0 - cdf_at_abs_t)))

    return {
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "t_statistic": float(t_statistic),
        "degrees_of_freedom": float(degrees_of_freedom),
        "p_value": float(p_value),
    }


def chi_square_test(
    observed_counts: Sequence[int],
    expected_counts: Sequence[float] | None = None,
) -> dict[str, float]:
    """Perform a chi-square goodness-of-fit test."""

    if len(observed_counts) < 2:
        raise ValueError("At least two categories are required")
    if any(count < 0 for count in observed_counts):
        raise ValueError("Observed counts must be non-negative")

    total = float(sum(observed_counts))
    if total <= 0:
        raise ValueError("Observed counts must sum to a positive value")

    if expected_counts is None:
        expected = [total / len(observed_counts)] * len(observed_counts)
    else:
        if len(expected_counts) != len(observed_counts):
            raise ValueError("Expected counts length must match observed counts length")
        expected = [float(x) for x in expected_counts]

    statistic = 0.0
    for observed, exp in zip(observed_counts, expected):
        if exp <= 0:
            raise ValueError("Expected counts must be positive")
        statistic += ((observed - exp) ** 2) / exp

    df = len(observed_counts) - 1
    p_value = 1.0 - _chi_square_cdf(statistic, df)
    return {
        "chi_square_statistic": float(statistic),
        "degrees_of_freedom": float(df),
        "p_value": float(max(0.0, min(1.0, p_value))),
    }


def mean_confidence_interval(
    data: Sequence[float],
    confidence: float = 0.95,
) -> dict[str, float]:
    """Calculate a confidence interval for the population mean.

    Uses a normal critical value approximation. For 95% confidence, ``z=1.96``.
    """

    if len(data) < 2:
        raise ValueError("At least two values are required")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")

    # Keep a small supported map to avoid external dependencies.
    z_lookup = {0.90: 1.644854, 0.95: 1.959964, 0.99: 2.575829}
    z = z_lookup.get(round(confidence, 2), 1.959964)

    mu = _mean(data)
    std = _sample_std(data)
    margin = z * std / math.sqrt(len(data))
    return {
        "mean": float(mu),
        "confidence": float(confidence),
        "lower_bound": float(mu - margin),
        "upper_bound": float(mu + margin),
        "margin_of_error": float(margin),
    }


def p_value_explanation(p_value: float, alpha: float = 0.05) -> str:
    """Return a plain-language p-value explanation."""

    significance_text = (
        "Statistically significant at the selected threshold."
        if p_value < alpha
        else "Not statistically significant at the selected threshold."
    )
    return (
        f"p-value = {p_value:.6f}. This is the probability of observing results at least "
        "this extreme if there were truly no difference/effect under the test assumptions. "
        f"{significance_text}"
    )


def print_p_value_explanation(p_value: float, alpha: float = 0.05) -> None:
    """Print a clear p-value explanation for notebooks or CLI scripts."""

    print(p_value_explanation(p_value, alpha=alpha))


def interpret_statistical_significance(p_value: float, alpha: float = 0.05) -> str:
    """Return a plain-language interpretation message."""

    if p_value < alpha:
        return (
            "Statistically significant: the observed difference is unlikely to be due "
            "to random variation alone under the model assumptions."
        )
    return (
        "Not statistically significant: the observed difference may be explained by "
        "random variation under the model assumptions."
    )


def example_flight_delay_t_test() -> dict[str, object]:
    """Run an example Welch t-test comparing two synthetic flight-delay groups."""

    # Group A and B could represent two routes, operating periods, or airports.
    sample_a = [12.0, 15.0, 10.0, 14.0, 17.0, 13.0, 16.0, 11.0]
    sample_b = [20.0, 24.0, 18.0, 21.0, 23.0, 19.0, 22.0, 25.0]
    result = welch_t_test(sample_a, sample_b)
    ci = mean_confidence_interval(sample_a, confidence=0.95)
    result["p_value_explanation"] = p_value_explanation(result["p_value"])
    result["interpretation"] = interpret_statistical_significance(result["p_value"])
    result["confidence_interval_sample_a"] = ci
    return result
