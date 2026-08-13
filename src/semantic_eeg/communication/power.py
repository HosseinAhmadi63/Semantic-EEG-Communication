"""Per-message normalization to unit mean power per real-valued channel use."""

from __future__ import annotations

import math

import numpy as np
import torch


def normalize_numpy(message: np.ndarray) -> np.ndarray:
    """Normalize each NumPy message to unit mean power per transmitted value."""

    values = np.asarray(message, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a two-dimensional message array, received {values.shape}")
    norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    return (math.sqrt(values.shape[1]) * values / norms).astype(np.float32)


def normalize_torch(message: torch.Tensor) -> torch.Tensor:
    """Normalize each Torch message to unit mean power per transmitted value."""

    if message.ndim != 2:
        raise ValueError(f"Expected a two-dimensional message tensor, received {tuple(message.shape)}")
    norms = torch.linalg.vector_norm(message, dim=1, keepdim=True).clamp_min(1e-8)
    return math.sqrt(message.shape[1]) * message / norms


def mean_transmit_power(message: np.ndarray) -> float:
    """Validate and return the mean squared power of normalized messages."""

    value = float(np.mean(np.square(np.asarray(message), dtype=np.float64)))
    if abs(value - 1.0) > 5e-4:
        raise AssertionError(f"Mean transmit power is {value:.8f}, expected approximately 1")
    return value
