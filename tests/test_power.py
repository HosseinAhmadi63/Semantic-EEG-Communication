"""Checks for the matched transmission-power constraint."""

import numpy as np
import torch

from semantic_eeg.communication.power import mean_transmit_power, normalize_numpy, normalize_torch


def test_numpy_normalization_has_unit_mean_power() -> None:
    messages = np.arange(1, 49, dtype=np.float32).reshape(3, 16)
    normalized = normalize_numpy(messages)
    assert normalized.shape == messages.shape
    assert np.allclose(np.mean(normalized**2, axis=1), 1.0, atol=1e-6)
    assert np.isclose(mean_transmit_power(normalized), 1.0, atol=1e-6)


def test_torch_normalization_matches_numpy() -> None:
    messages = np.arange(1, 97, dtype=np.float32).reshape(3, 32)
    observed = normalize_torch(torch.from_numpy(messages)).numpy()
    assert np.allclose(observed, normalize_numpy(messages), atol=1e-6)
