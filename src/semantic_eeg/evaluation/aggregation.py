"""Subject-preserving aggregation of detailed channel-realization scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from semantic_eeg.config import ExperimentConfig
from semantic_eeg.utils.io import atomic_csv, read_csv
from semantic_eeg.utils.run import RunContext


def aggregate_detailed_results(
    detail_path: str | Path,
    output_prefix: str,
    config: ExperimentConfig,
    context: RunContext,
) -> dict[str, Path]:
    """Create direction/seed, subject, group, and completeness summaries."""

    detailed = read_csv(detail_path)
    detailed = detailed[detailed["config_hash"].astype(str) == context.config_hash].copy()
    if detailed.empty:
        raise ValueError(f"No detailed results for configuration {context.config_hash}")

    base = [
        "protocol_hash",
        "config_hash",
        "method",
        "subject",
        "budget_k",
        "snr_order",
        "snr_label",
    ]
    direction_seed = (
        detailed.groupby(
            base + ["direction", "direction_index", "train_session", "test_session", "seed"],
            dropna=False,
        )
        .agg(
            snr_db=("snr_db", "first"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            n_channel_realizations=("channel_realization", "nunique"),
            n_test_trials=("n_test_trials", "first"),
            mean_transmit_power=("mean_transmit_power", "mean"),
        )
        .reset_index()
    )
    subject = (
        detailed.groupby(base, dropna=False)
        .agg(
            snr_db=("snr_db", "first"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            n_observations=("balanced_accuracy", "size"),
            n_directions=("direction", "nunique"),
            n_model_seeds=("seed", "nunique"),
            n_channel_realizations=("channel_realization", "nunique"),
            mean_transmit_power=("mean_transmit_power", "mean"),
        )
        .reset_index()
    )
    group = (
        subject.groupby(
            ["protocol_hash", "config_hash", "method", "budget_k", "snr_order", "snr_label"],
            dropna=False,
        )
        .agg(
            snr_db=("snr_db", "first"),
            balanced_accuracy_mean=("balanced_accuracy_mean", "mean"),
            balanced_accuracy_std_across_subjects=("balanced_accuracy_mean", "std"),
            n_subjects=("subject", "nunique"),
            mean_transmit_power=("mean_transmit_power", "mean"),
        )
        .reset_index()
    )
    group["balanced_accuracy_sem_across_subjects"] = (
        group["balanced_accuracy_std_across_subjects"] / np.sqrt(group["n_subjects"])
    )
    audit = (
        detailed.groupby(
            ["protocol_hash", "config_hash", "method", "subject", "direction", "seed", "budget_k"],
            dropna=False,
        )
        .agg(
            n_rows=("balanced_accuracy", "size"),
            n_snr_conditions=("snr_label", "nunique"),
            n_channel_realizations=("channel_realization", "nunique"),
            minimum_score=("balanced_accuracy", "min"),
            maximum_score=("balanced_accuracy", "max"),
        )
        .reset_index()
    )
    expected_conditions = len(config.snr_specs)
    expected_realizations = int(config.section("communication")["channel_realizations"])
    audit["is_complete"] = (
        (audit["n_rows"] == expected_conditions * expected_realizations)
        & (audit["n_snr_conditions"] == expected_conditions)
        & (audit["n_channel_realizations"] == expected_realizations)
    )

    paths = {
        "direction_seed": context.csv_dir / f"{output_prefix}_results_direction_seed_summary.csv",
        "subject": context.csv_dir / f"{output_prefix}_results_subject_summary.csv",
        "group": context.csv_dir / f"{output_prefix}_results_group_summary.csv",
        "completion": context.csv_dir / f"{output_prefix}_completion_audit.csv",
    }
    for frame, path in (
        (direction_seed, paths["direction_seed"]),
        (subject, paths["subject"]),
        (group, paths["group"]),
        (audit, paths["completion"]),
    ):
        atomic_csv(frame, path)
    if not audit["is_complete"].all():
        raise RuntimeError(f"Incomplete jobs detected in {paths['completion']}")
    return paths
