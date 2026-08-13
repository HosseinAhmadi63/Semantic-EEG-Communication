"""Seed-locked held-out-session evaluation for transmitted EEG messages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from semantic_eeg.communication.awgn import channel_seed
from semantic_eeg.communication.power import mean_transmit_power, normalize_numpy
from semantic_eeg.config import ExperimentConfig
from semantic_eeg.data.splits import CrossSessionSplit
from semantic_eeg.evaluation.metrics import predict_receiver
from semantic_eeg.results.writer import write_detailed_results
from semantic_eeg.utils.run import RunContext, utc_now


def evaluate_transmitted_messages(
    receiver: nn.Module,
    messages: np.ndarray,
    labels: np.ndarray,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
    model_seed: int,
    checkpoint_path: str | Path,
    result_path: str | Path,
    method: str,
    config: ExperimentConfig,
    context: RunContext,
    secondary_checkpoint_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one fitted job at every SNR and paired channel realization."""

    transmitted = normalize_numpy(messages)
    expected = np.asarray(labels, dtype=np.int64).reshape(-1)
    if transmitted.shape != (len(expected), budget):
        raise ValueError(
            f"Expected messages with shape ({len(expected)}, {budget}), received {transmitted.shape}"
        )
    power = mean_transmit_power(transmitted)
    communication = config.section("communication")
    realizations = int(communication["channel_realizations"])
    seed_base = int(communication["channel_seed_base"])
    evaluation_batch_size = int(config.section("training")["evaluation_batch_size"])
    rows: list[dict[str, Any]] = []
    for snr_order, (snr_label, snr_db) in enumerate(config.snr_specs):
        for realization in range(realizations):
            noise_seed = channel_seed(
                seed_base,
                subject,
                split.direction_index,
                budget,
                snr_order,
                realization,
            )
            prediction = predict_receiver(
                receiver,
                transmitted,
                snr_db,
                noise_seed,
                context.device,
                evaluation_batch_size,
                normalize=False,
            )
            rows.append(
                {
                    "run_id": context.run_id,
                    "protocol_hash": context.protocol_hash,
                    "config_hash": context.config_hash,
                    "method": method,
                    "subject": subject,
                    "direction": split.direction,
                    "direction_index": split.direction_index,
                    "train_session": split.train_session,
                    "test_session": split.test_session,
                    "seed": model_seed,
                    "budget_k": budget,
                    "snr_order": snr_order,
                    "snr_label": snr_label,
                    "snr_db": np.nan if snr_db is None else snr_db,
                    "channel_realization": realization,
                    "channel_seed": noise_seed,
                    "n_test_trials": len(expected),
                    "balanced_accuracy": float(balanced_accuracy_score(expected, prediction)),
                    "mean_transmit_power": power,
                    "checkpoint_path": str(checkpoint_path),
                    "secondary_checkpoint_path": (
                        "" if secondary_checkpoint_path is None else str(secondary_checkpoint_path)
                    ),
                    "evaluated_at_utc": utc_now(),
                }
            )
    write_detailed_results(result_path, rows)
    return rows


def job_is_complete(
    result_path: str | Path,
    subject: int,
    direction: str,
    model_seed: int,
    budget: int,
    config_hash: str,
    config: ExperimentConfig,
) -> bool:
    """Return whether an exact subject/direction/seed/budget result grid exists."""

    path = Path(result_path)
    if not path.exists():
        return False
    import pandas as pd

    frame = pd.read_csv(path)
    selected = frame[
        (frame["config_hash"].astype(str) == config_hash)
        & (frame["subject"] == subject)
        & (frame["direction"].astype(str) == direction)
        & (frame["seed"] == model_seed)
        & (frame["budget_k"] == budget)
    ]
    expected_rows = len(config.snr_specs) * int(
        config.section("communication")["channel_realizations"]
    )
    return bool(
        len(selected) == expected_rows
        and selected["snr_label"].nunique() == len(config.snr_specs)
        and selected["channel_realization"].nunique()
        == int(config.section("communication")["channel_realizations"])
    )
