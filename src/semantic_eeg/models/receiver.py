"""Common four-class receiver used after every transmitted representation."""

from __future__ import annotations

import torch
from torch import nn


class Receiver(nn.Module):
    """Map a received fixed-budget message to four motor-imagery logits."""

    def __init__(self, budget: int, hidden_units: int = 64, dropout: float = 0.25) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(budget, hidden_units),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 4),
        )

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        return self.network(message)
