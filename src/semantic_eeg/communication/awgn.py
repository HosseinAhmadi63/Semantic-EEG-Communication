"""Seed-locked additive white Gaussian noise operations."""

from __future__ import annotations

import numpy as np
import torch


def channel_seed(
    base: int,
    subject: int,
    direction_index: int,
    budget: int,
    snr_order: int,
    realization: int,
) -> int:
    """Derive the deterministic channel seed for one experimental condition."""

    return int(
        base
        + subject * 1_000_000
        + direction_index * 100_000
        + budget * 1_000
        + snr_order * 20
        + realization
    )


def add_awgn(message: np.ndarray, snr_db: float | None, seed: int) -> np.ndarray:
    """Apply seed-locked AWGN at the requested SNR to a NumPy message array."""

    values = np.asarray(message, dtype=np.float32)
    if snr_db is None:
        return values.copy()
    standard_deviation = 10.0 ** (-float(snr_db) / 20.0)
    noise = np.random.default_rng(seed).normal(0.0, standard_deviation, values.shape)
    return values + noise.astype(np.float32)


def training_awgn(message: torch.Tensor, snr_values: tuple[float, ...]) -> torch.Tensor:
    """Add AWGN after sampling one configured SNR independently per batch example."""

    choices = torch.as_tensor(snr_values, dtype=message.dtype, device=message.device)
    selected = choices[torch.randint(0, len(choices), (len(message),), device=message.device)]
    standard_deviation = torch.pow(
        torch.tensor(10.0, dtype=message.dtype, device=message.device),
        -selected / 20.0,
    )
    return message + torch.randn_like(message) * standard_deviation[:, None]
