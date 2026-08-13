"""Checks for reconstruction architecture dimensions."""

import torch

from semantic_eeg.models.autoencoder import ReconstructionAutoencoder


def test_autoencoder_shapes_for_every_budget() -> None:
    trials = torch.zeros(2, 22, 560)
    for budget in (16, 32, 64):
        model = ReconstructionAutoencoder(budget, (-10.0, -5.0, 0.0))
        message = model.encoder(trials)
        reconstructed = model.decoder(message)
        assert message.shape == (2, budget)
        assert reconstructed.shape == trials.shape
