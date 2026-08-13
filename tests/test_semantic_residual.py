"""Checks for baseline-preserving semantic initialization."""

import torch

from semantic_eeg.communication.power import normalize_torch
from semantic_eeg.models.semantic_residual import SemanticResidualModel


def test_zero_initialized_residual_preserves_base_message() -> None:
    generator = torch.Generator().manual_seed(2026)
    base = normalize_torch(torch.randn(7, 32, generator=generator))
    model = SemanticResidualModel(32)
    semantic = model.semantic_message(base)
    assert torch.allclose(semantic, base, atol=1e-6, rtol=0.0)
