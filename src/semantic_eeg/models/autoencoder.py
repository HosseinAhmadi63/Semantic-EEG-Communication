"""Convolutional autoencoder used by the reconstruction-oriented comparator."""

from __future__ import annotations

import torch
from torch import nn

from semantic_eeg.communication.awgn import training_awgn
from semantic_eeg.communication.power import normalize_torch
from semantic_eeg.constants import N_TIMES


class ReconstructionEncoder(nn.Module):
    """Encode a standardized EEG epoch as a fixed-budget latent message."""

    def __init__(self, budget: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(22, 32, 15, 2, 7),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Conv1d(32, 64, 9, 2, 4),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Conv1d(64, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool1d(8),
        )
        self.projection = nn.Linear(64 * 8, budget)

    def forward(self, trial: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(trial).flatten(1))


class ReconstructionDecoder(nn.Module):
    """Decode a received latent message into a standardized EEG epoch."""

    def __init__(self, budget: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(budget, 64 * 70), nn.ELU())
        self.network = nn.Sequential(
            nn.ConvTranspose1d(64, 64, 7, 2, 3, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.ConvTranspose1d(64, 32, 9, 2, 4, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.ConvTranspose1d(32, 22, 15, 2, 7, output_padding=1),
        )

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        reconstructed = self.network(self.projection(message).view(-1, 64, 70))
        if reconstructed.shape[-1] != N_TIMES:
            raise RuntimeError(f"Decoder produced {tuple(reconstructed.shape)}")
        return reconstructed


class ReconstructionAutoencoder(nn.Module):
    """Train the reconstruction comparator through a stochastic AWGN channel."""

    def __init__(self, budget: int, training_snrs: tuple[float, ...]) -> None:
        super().__init__()
        self.encoder = ReconstructionEncoder(budget)
        self.decoder = ReconstructionDecoder(budget)
        self.training_snrs = training_snrs

    def forward(self, trial: torch.Tensor) -> torch.Tensor:
        message = normalize_torch(self.encoder(trial))
        return self.decoder(training_awgn(message, self.training_snrs))
