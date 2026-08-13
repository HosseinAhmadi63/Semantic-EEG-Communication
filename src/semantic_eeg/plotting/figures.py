"""Generate Figures 2--5 directly from frozen subject-level results."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

_plot_cache = Path(tempfile.gettempdir()) / "semantic-eeg-plot-cache"
_plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_plot_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_plot_cache / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from semantic_eeg.plotting.style import (
    COLORS,
    MARKERS,
    METHOD_FBCSP,
    METHOD_RECEIVER_ONLY,
    METHOD_RECONSTRUCTION,
    METHOD_SEMANTIC,
    apply_publication_style,
)

SNR_DISPLAY = ("-10", "-5", "0", "5", "10", "15", "20", "NF")
BUDGETS = (16, 32, 64)
SUBJECTS = tuple(range(1, 10))


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{stem}.pdf")
    figure.savefig(output / f"{stem}.svg")
    figure.savefig(output / f"{stem}.png", dpi=400)
    plt.close(figure)


def _group_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {(row["method"], int(row["budget_k"]), int(row["snr_order"])): row for row in rows}


def _figure_2(group_rows: list[dict[str, Any]], output: Path) -> None:
    lookup = _group_lookup(group_rows)
    figure, axis = plt.subplots(figsize=(7.4, 4.9))
    x = np.arange(8)
    axis.axvspan(-0.35, 2.35, color="#EAF2F8", alpha=0.65, zorder=0)
    for method in (METHOD_RECONSTRUCTION, METHOD_FBCSP, METHOD_RECEIVER_ONLY, METHOD_SEMANTIC):
        rows = [lookup[(method, 32, order)] for order in range(8)]
        values = np.asarray([row["balanced_accuracy_mean"] for row in rows])
        linestyle = "--" if method in (METHOD_RECONSTRUCTION, METHOD_RECEIVER_ONLY) else "-"
        axis.plot(
            x,
            values,
            label=method,
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.4 if method == METHOD_SEMANTIC else 1.8,
            linestyle=linestyle,
            markersize=5.5,
        )
        if method in (METHOD_SEMANTIC, METHOD_FBCSP):
            lower = np.asarray([row["balanced_accuracy_ci95_low"] for row in rows])
            upper = np.asarray([row["balanced_accuracy_ci95_high"] for row in rows])
            axis.fill_between(x, lower, upper, color=COLORS[method], alpha=0.12, linewidth=0)
    axis.axhline(0.25, color="#555555", linestyle=":", linewidth=1.2, label="Chance level")
    axis.set_xticks(x, SNR_DISPLAY)
    axis.set_xlabel("Channel SNR (dB; NF = noise-free)")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.20, 0.69)
    axis.set_title("Task-oriented semantic refinement at K = 32")
    axis.text(1.0, 0.675, "Low-SNR region", ha="center", va="top", color="#315A77", fontsize=10.0)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, ncol=3)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    _save(figure, output, "figure_2_main_performance_k32")


def _figure_3(group_rows: list[dict[str, Any]], output: Path) -> None:
    lookup = _group_lookup(group_rows)
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 4.45), sharex=True, sharey=True)
    x = np.arange(8)
    for axis, budget in zip(axes, BUDGETS, strict=True):
        axis.axvspan(-0.35, 2.35, color="#EAF2F8", alpha=0.65, zorder=0)
        for method in (METHOD_RECONSTRUCTION, METHOD_FBCSP, METHOD_SEMANTIC):
            rows = [lookup[(method, budget, order)] for order in range(8)]
            values = np.asarray([row["balanced_accuracy_mean"] for row in rows])
            axis.plot(
                x,
                values,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.3 if method == METHOD_SEMANTIC else 1.8,
                linestyle="--" if method == METHOD_RECONSTRUCTION else "-",
                markersize=5.2,
                label=method,
            )
            if method in (METHOD_SEMANTIC, METHOD_FBCSP):
                lower = np.asarray([row["balanced_accuracy_ci95_low"] for row in rows])
                upper = np.asarray([row["balanced_accuracy_ci95_high"] for row in rows])
                axis.fill_between(x, lower, upper, color=COLORS[method], alpha=0.11, linewidth=0)
        axis.axhline(0.25, color="#555555", linestyle=":", linewidth=1.1)
        axis.set_title(f"K = {budget} channel uses", fontsize=13.5)
        axis.set_xticks(x, SNR_DISPLAY)
        axis.set_xlabel("SNR (dB)")
        axis.set_ylim(0.20, 0.69)
    axes[0].set_ylabel("Balanced accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    figure.suptitle("Balanced accuracy across communication budgets", y=1.11, fontsize=15.0)
    figure.tight_layout()
    _save(figure, output, "figure_3_performance_by_budget")


def _figure_4(primary_rows: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.75))
    x = np.arange(8)
    budget_colors = {16: "#56B4E9", 32: "#0072B2", 64: "#332288"}
    budget_markers = {16: "o", 32: "s", 64: "^"}
    for budget in BUDGETS:
        rows = sorted(
            (row for row in primary_rows if int(row["budget_k"]) == budget),
            key=lambda row: int(row["snr_order"]),
        )
        values = np.asarray([row["mean_paired_difference_pp"] for row in rows])
        lower = np.asarray([row["ci95_difference_low_pp"] for row in rows])
        upper = np.asarray([row["ci95_difference_high_pp"] for row in rows])
        axis.plot(
            x,
            values,
            color=budget_colors[budget],
            marker=budget_markers[budget],
            linewidth=2.0,
            markersize=5.4,
            label=f"K = {budget}",
        )
        axis.fill_between(x, lower, upper, color=budget_colors[budget], alpha=0.10, linewidth=0)
        for index, row in enumerate(rows):
            if bool(row["significant_holm_0_05"]):
                axis.text(index, values[index] + 0.28, "*", color=budget_colors[budget], ha="center", fontsize=12)
    axis.axhline(0, color="#444444", linewidth=1)
    axis.axvspan(-0.35, 2.35, color="#EAF2F8", alpha=0.65, zorder=0)
    axis.set_xticks(x, SNR_DISPLAY)
    axis.set_xlabel("Channel SNR (dB; NF = noise-free)")
    axis.set_ylabel("Semantic minus FBCSP-PCA (percentage points)")
    axis.set_title("Semantic advantage is concentrated under channel noise")
    axis.legend(frameon=False, ncol=3, loc="upper right")
    axis.text(
        0.01,
        0.01,
        "* Holm-adjusted p < 0.05; bands show unadjusted 95% CIs",
        transform=axis.transAxes,
        fontsize=9.3,
        color="#555555",
    )
    figure.tight_layout()
    _save(figure, output, "figure_4_semantic_gain_vs_snr")


def _figure_5(
    subject_lookup: dict[tuple[str, int, int, int], float],
    output: Path,
) -> list[dict[str, Any]]:
    columns = [(budget, order) for budget in BUDGETS for order in (0, 1, 2)]
    data = np.asarray(
        [
            [
                100.0
                * (
                    subject_lookup[(METHOD_SEMANTIC, subject, budget, order)]
                    - subject_lookup[(METHOD_FBCSP, subject, budget, order)]
                )
                for budget, order in columns
            ]
            for subject in SUBJECTS
        ]
    )
    maximum = max(abs(float(data.min())), abs(float(data.max())))
    figure, axis = plt.subplots(figsize=(10.0, 5.3))
    image = axis.imshow(
        data,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum),
        aspect="auto",
    )
    labels = [f"K={budget}\n{SNR_DISPLAY[order]} dB" for budget, order in columns]
    axis.set_xticks(np.arange(len(columns)), labels, fontsize=10.5)
    axis.set_yticks(np.arange(9), [f"S{subject}" for subject in SUBJECTS], fontsize=11.0)
    axis.set_title("Subject-level semantic-minus-FBCSP-PCA difference in the low-SNR region", fontsize=14.0)
    axis.set_ylabel("Subject")
    for row_index in range(data.shape[0]):
        for column_index in range(data.shape[1]):
            color = "white" if abs(data[row_index, column_index]) > 0.58 * maximum else "black"
            axis.text(
                column_index,
                row_index,
                f"{data[row_index, column_index]:+.1f}",
                ha="center",
                va="center",
                fontsize=10.0,
                weight="bold",
                color=color,
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Semantic minus FBCSP-PCA (percentage points)", fontsize=11.0)
    colorbar.ax.tick_params(labelsize=10.5)
    figure.tight_layout()
    _save(figure, output, "figure_5_subject_level_low_snr_heatmap")
    return [
        {
            "subject": subject,
            "budget_k": budget,
            "snr_label": ("-10_dB", "-5_dB", "0_dB")[order],
            "semantic_minus_fbcsp_pp": float(data[subject - 1, column]),
        }
        for subject in SUBJECTS
        for column, (budget, order) in enumerate(columns)
    ]


def generate_publication_figures(
    group_rows: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
    subject_lookup: dict[tuple[str, int, int, int], float],
    output_directory: str | Path,
) -> list[dict[str, Any]]:
    """Regenerate the four data-driven figures used in the final manuscript."""

    output = Path(output_directory)
    if any(not math.isfinite(float(row["balanced_accuracy_mean"])) for row in group_rows):
        raise ValueError("Group summary contains a non-finite balanced accuracy")
    apply_publication_style()
    _figure_2(group_rows, output)
    _figure_3(group_rows, output)
    _figure_4(primary_rows, output)
    return _figure_5(subject_lookup, output)
