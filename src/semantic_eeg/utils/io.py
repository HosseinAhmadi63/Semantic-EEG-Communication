"""Atomic persistence helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV file atomically after creating its parent directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def read_csv(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV file or return an empty frame when no persisted rows exist."""

    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    return pd.read_csv(source)


def upsert_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    keys: list[str],
    columns: list[str] | None = None,
) -> None:
    """Atomically merge rows into a CSV file using the specified unique keys."""

    if not rows:
        return
    old = read_csv(path, columns)
    new = pd.DataFrame(rows)
    combined = new if old.empty else pd.concat([old, new], ignore_index=True, sort=False)
    atomic_csv(combined.drop_duplicates(keys, keep="last"), path)


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Persist a Torch checkpoint atomically."""

    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def atomic_joblib_save(payload: Any, path: str | Path) -> None:
    """Persist a compressed joblib artifact atomically."""

    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, destination)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
