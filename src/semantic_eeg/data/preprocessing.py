"""Training-only EEG standardization for the reconstruction comparator."""

from __future__ import annotations

import numpy as np

from semantic_eeg.constants import N_CHANNELS
from semantic_eeg.data.splits import CrossSessionSplit
from semantic_eeg.utils.io import upsert_csv
from semantic_eeg.utils.run import RunContext


def standardize_wideband_trials(
    subject: int,
    split: CrossSessionSplit,
    array: np.ndarray,
    context: RunContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize EEG trials using training-only channel statistics and audit them."""

    train = np.asarray(array[split.train_indices], dtype=np.float32)
    validation = np.asarray(array[split.validation_indices], dtype=np.float32)
    test = np.asarray(array[split.test_indices], dtype=np.float32)
    mean = train.mean((0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    standard_deviation = np.maximum(
        train.std((0, 2), keepdims=True, dtype=np.float64).astype(np.float32),
        1e-7,
    )
    train = (train - mean) / standard_deviation
    validation = (validation - mean) / standard_deviation
    test = (test - mean) / standard_deviation
    rows = [
        {
            "protocol_hash": context.protocol_hash,
            "config_hash": context.config_hash,
            "method": context.method,
            "subject": subject,
            "direction": split.direction,
            "direction_index": split.direction_index,
            "channel_index": channel,
            "training_mean": float(mean[0, channel, 0]),
            "training_std": float(standard_deviation[0, channel, 0]),
        }
        for channel in range(N_CHANNELS)
    ]
    upsert_csv(
        context.csv_dir / "preprocessing_parameters.csv",
        rows,
        ["config_hash", "subject", "direction", "channel_index"],
    )
    return train, validation, test
