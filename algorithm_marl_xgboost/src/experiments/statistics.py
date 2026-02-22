"""Statistical comparison utilities for MARL+XGBoost experimental repeats."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if len(diff) < 2:
        return 0.0
    std = float(np.std(diff, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(diff) / std)


def _rank_biserial_from_paired_diffs(diff: np.ndarray) -> float:
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    return float((n_pos - n_neg) / len(nonzero))


def compare_methods(a: list[float], b: list[float], test: str = "wilcoxon") -> dict[str, Any]:
    """Compare two method score lists and return significance + effect size."""

    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.size == 0 or arr_b.size == 0:
        raise ValueError("Input score lists must be non-empty")

    test = str(test).lower()
    paired = arr_a.size == arr_b.size

    if test == "ttest":
        if paired:
            statistic, p_value = stats.ttest_rel(arr_a, arr_b, nan_policy="omit")
            effect_size = _cohens_d_paired(arr_a, arr_b)
            test_used = "ttest_rel"
        else:
            statistic, p_value = stats.ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit")
            pooled = float(np.sqrt((np.var(arr_a, ddof=1) + np.var(arr_b, ddof=1)) / 2.0)) if min(len(arr_a), len(arr_b)) > 1 else 0.0
            effect_size = float((np.mean(arr_a) - np.mean(arr_b)) / pooled) if pooled else 0.0
            test_used = "ttest_ind"
    elif test == "wilcoxon":
        if paired:
            diff = arr_a - arr_b
            try:
                statistic, p_value = stats.wilcoxon(arr_a, arr_b, zero_method="wilcox")
                effect_size = _rank_biserial_from_paired_diffs(diff)
                test_used = "wilcoxon"
            except ValueError:
                statistic, p_value = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
                effect_size = float((2.0 * statistic) / (len(arr_a) * len(arr_b)) - 1.0)
                test_used = "mannwhitneyu_fallback"
        else:
            statistic, p_value = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
            effect_size = float((2.0 * statistic) / (len(arr_a) * len(arr_b)) - 1.0)
            test_used = "mannwhitneyu"
    else:
        raise ValueError(f"Unsupported significance test: {test}")

    return {
        "test_requested": test,
        "test_used": test_used,
        "n_a": int(arr_a.size),
        "n_b": int(arr_b.size),
        "mean_a": float(np.mean(arr_a)),
        "mean_b": float(np.mean(arr_b)),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "paired": bool(paired),
    }

