"""Deterministic seed construction and process initialization."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Initialize Python, NumPy, and Torch for deterministic execution."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def semantic_initialization_seed(base: int, subject: int, direction: int, budget: int, seed: int) -> int:
    """Derive a condition-specific initialization seed for the semantic model."""

    value = int(base + subject * 1_000_000 + direction * 100_000 + budget * 1_000 + seed)
    if not 0 <= value < 2**32:
        raise ValueError(f"Invalid semantic initialization seed: {value}")
    return value


def receiver_initialization_seed(base: int, subject: int, direction: int, budget: int, seed: int) -> int:
    """Derive a condition-specific initialization seed for a receiver model."""

    value = int(base + subject * 1_000_000 + direction * 100_000 + budget * 1_000 + seed)
    if not 0 <= value < 2**32:
        raise ValueError(f"Invalid receiver initialization seed: {value}")
    return value
