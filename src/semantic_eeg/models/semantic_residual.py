"""Baseline-preserving residual refinement of normalized FBCSP–PCA messages."""

from __future__ import annotations

import torch
from torch import nn

from semantic_eeg.communication.power import normalize_torch
from semantic_eeg.models.receiver import Receiver


class SemanticResidualModel(nn.Module):
    """Refine a normalized FBCSP--PCA message for task-oriented transmission."""

    def __init__(
        self,
        budget: int,
        hidden_units: int = 96,
        residual_scale: float = 0.35,
    ) -> None:
        super().__init__()
        self.budget = budget
        self.residual_scale = residual_scale
        self.residual = nn.Sequential(
            nn.LayerNorm(budget),
            nn.Linear(budget, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, budget),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.receiver = Receiver(budget)

    def semantic_message(self, base_message: torch.Tensor) -> torch.Tensor:
        correction = self.residual_scale * torch.tanh(self.residual(base_message))
        return normalize_torch(base_message + correction)

    def forward(self, base_message: torch.Tensor) -> torch.Tensor:
        return self.receiver(self.semantic_message(base_message))
