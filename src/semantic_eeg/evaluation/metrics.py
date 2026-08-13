"""Deterministic classification metrics for noisy transmitted messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from semantic_eeg.communication.awgn import add_awgn, channel_seed
from semantic_eeg.communication.power import normalize_numpy


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Return four-class balanced accuracy after validating aligned inputs."""
    expected = np.asarray(labels, dtype=np.int64).reshape(-1)
    observed = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if expected.shape != observed.shape:
        raise ValueError(
            f"Labels and predictions must have equal shapes, received "
            f"{expected.shape} and {observed.shape}"
        )
    if expected.size == 0:
        raise ValueError("Balanced accuracy is undefined for an empty input")
    return float(balanced_accuracy_score(expected, observed))


@torch.inference_mode()
def predict_receiver(
    receiver: nn.Module,
    messages: np.ndarray,
    snr_db: float | None,
    noise_seed: int,
    device: torch.device,
    batch_size: int,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Predict labels after applying one seed-locked channel realization."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    transmitted = normalize_numpy(messages) if normalize else np.asarray(messages, dtype=np.float32)
    received = add_awgn(transmitted, snr_db, noise_seed)
    receiver.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(received), batch_size):
        batch = torch.from_numpy(received[start : start + batch_size]).to(device)
        predictions.append(receiver(batch).argmax(dim=1).cpu().numpy())
    if not predictions:
        raise ValueError("Cannot predict an empty message array")
    return np.concatenate(predictions).astype(np.int64, copy=False)


def validation_balanced_accuracies(
    receiver: nn.Module,
    messages: np.ndarray,
    labels: np.ndarray,
    snr_specs: Sequence[tuple[str, float | None]],
    subject: int,
    direction_index: int,
    budget: int,
    realizations: int,
    seed_base: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate fixed validation conditions with paired deterministic noise."""
    if realizations < 1:
        raise ValueError("realizations must be positive")
    transmitted = normalize_numpy(messages)
    result: dict[str, float] = {}
    for snr_order, (label, snr_db) in enumerate(snr_specs):
        if snr_db is None:
            prediction = predict_receiver(
                receiver,
                transmitted,
                None,
                0,
                device,
                batch_size,
                normalize=False,
            )
            result[label] = balanced_accuracy(labels, prediction)
            continue
        scores = []
        for realization in range(realizations):
            noise_seed = channel_seed(
                seed_base,
                subject,
                direction_index,
                budget,
                snr_order,
                realization,
            )
            prediction = predict_receiver(
                receiver,
                transmitted,
                snr_db,
                noise_seed,
                device,
                batch_size,
                normalize=False,
            )
            scores.append(balanced_accuracy(labels, prediction))
        result[label] = float(np.mean(scores))
    return result


def semantic_validation_utility(scores: Mapping[str, float]) -> float:
    """Return the frozen low-SNR utility used to select semantic checkpoints."""
    required = ("-10_dB", "-5_dB", "0_dB", "noise_free")
    missing = [label for label in required if label not in scores]
    if missing:
        raise KeyError(f"Missing semantic validation conditions: {missing}")
    severe = 0.5 * (float(scores["-10_dB"]) + float(scores["-5_dB"]))
    clean_support = 0.5 * (float(scores["0_dB"]) + float(scores["noise_free"]))
    return severe + 0.10 * clean_support


def passes_clean_guard(
    scores: Mapping[str, float],
    baseline_scores: Mapping[str, float],
    tolerance: float,
) -> bool:
    """Check that 0-dB and noise-free accuracy remain near the baseline."""
    return bool(
        float(scores["0_dB"]) >= float(baseline_scores["0_dB"]) - tolerance
        and float(scores["noise_free"])
        >= float(baseline_scores["noise_free"]) - tolerance
    )
