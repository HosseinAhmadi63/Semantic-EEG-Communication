"""Subject-level statistical comparisons used in the manuscript."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy import stats


def mean_confidence_interval(values: Iterable[float]) -> tuple[float, float, float, float, float]:
    """Return the mean, dispersion, and two-sided 95% t confidence interval."""

    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        raise ValueError("A confidence interval requires at least one value")
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1)) if array.size > 1 else math.nan
    standard_error = standard_deviation / math.sqrt(array.size) if array.size > 1 else math.nan
    quantile = float(stats.t.ppf(0.975, array.size - 1)) if array.size > 1 else math.nan
    return (
        mean,
        standard_deviation,
        standard_error,
        mean - quantile * standard_error,
        mean + quantile * standard_error,
    )


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Apply Holm's step-down adjustment to one family of p-values."""

    values = np.asarray(tuple(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def group_summary(subject_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize subject-level balanced accuracy by method, budget, and SNR."""

    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    labels: dict[int, tuple[str, float]] = {}
    for row in subject_rows:
        grouped[(row["method"], row["budget_k"], row["snr_order"])].append(
            row["balanced_accuracy"]
        )
        labels[row["snr_order"]] = (row["snr_label"], row["snr_db"])
    output: list[dict[str, Any]] = []
    for (method, budget, snr_order), values in sorted(grouped.items()):
        mean, standard_deviation, standard_error, lower, upper = mean_confidence_interval(values)
        output.append(
            {
                "method": method,
                "budget_k": budget,
                "snr_order": snr_order,
                "snr_label": labels[snr_order][0],
                "snr_db": labels[snr_order][1],
                "n_subjects": len(values),
                "balanced_accuracy_mean": mean,
                "balanced_accuracy_sd": standard_deviation,
                "balanced_accuracy_sem": standard_error,
                "balanced_accuracy_ci95_low": lower,
                "balanced_accuracy_ci95_high": upper,
            }
        )
    return output


def comparison_family(
    lookup: dict[tuple[str, int, int, int], float],
    family: str,
    method_a: str,
    method_b: str,
    budgets: tuple[int, ...],
    subjects: tuple[int, ...],
    snr_labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Compute one prespecified family of paired subject-level comparisons."""

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for snr_order, snr_label in enumerate(snr_labels):
            first = np.asarray(
                [lookup[(method_a, subject, budget, snr_order)] for subject in subjects], dtype=float
            )
            second = np.asarray(
                [lookup[(method_b, subject, budget, snr_order)] for subject in subjects], dtype=float
            )
            difference = first - second
            mean, standard_deviation, standard_error, lower, upper = mean_confidence_interval(difference)
            shapiro = stats.shapiro(difference)
            if float(shapiro.pvalue) >= 0.05:
                test_name = "paired_t_test"
                test_result = stats.ttest_rel(first, second)
            else:
                test_name = "wilcoxon_signed_rank"
                test_result = (
                    stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
                    if np.any(difference != 0)
                    else None
                )
            rows.append(
                {
                    "holm_family": family,
                    "comparison": f"{method_a} vs {method_b}",
                    "method_a": method_a,
                    "method_b": method_b,
                    "budget_k": budget,
                    "snr_order": snr_order,
                    "snr_label": snr_label,
                    "n_subjects": len(subjects),
                    "mean_method_a": float(first.mean()),
                    "mean_method_b": float(second.mean()),
                    "mean_paired_difference": mean,
                    "mean_paired_difference_pp": 100.0 * mean,
                    "median_paired_difference": float(np.median(difference)),
                    "sd_paired_difference": standard_deviation,
                    "sem_paired_difference": standard_error,
                    "ci95_difference_low": lower,
                    "ci95_difference_high": upper,
                    "ci95_difference_low_pp": 100.0 * lower,
                    "ci95_difference_high_pp": 100.0 * upper,
                    "cohen_dz": mean / standard_deviation if standard_deviation > 0 else math.nan,
                    "subject_wins_method_a": int((difference > 0).sum()),
                    "subject_ties": int((difference == 0).sum()),
                    "subject_losses_method_a": int((difference < 0).sum()),
                    "shapiro_wilk_statistic": float(shapiro.statistic),
                    "shapiro_wilk_pvalue": float(shapiro.pvalue),
                    "selected_test": test_name,
                    "test_statistic": float(test_result.statistic) if test_result is not None else 0.0,
                    "pvalue_raw": float(test_result.pvalue) if test_result is not None else 1.0,
                    "pvalue_holm": math.nan,
                    "significant_holm_0_05": False,
                }
            )
    for row, adjusted in zip(rows, holm_adjust(row["pvalue_raw"] for row in rows), strict=True):
        row["pvalue_holm"] = adjusted
        row["significant_holm_0_05"] = adjusted < 0.05
    return rows
