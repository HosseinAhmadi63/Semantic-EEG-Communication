"""Deterministic bidirectional cross-session partitioning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from semantic_eeg.config import ExperimentConfig
from semantic_eeg.utils.io import upsert_csv
from semantic_eeg.utils.run import RunContext


@dataclass(frozen=True)
class CrossSessionSplit:
    """Indices and session metadata for one leakage-free evaluation direction."""

    direction: str
    direction_index: int
    train_session: str
    test_session: str
    validation_run: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def _choose_validation_run(
    source_indices: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    seed: int,
) -> tuple[str, int]:
    runs = sorted(metadata.iloc[source_indices]["run"].astype(str).unique())
    if len(runs) != 6:
        raise ValueError(f"Expected six source-session runs, received {runs}")
    order = np.asarray(runs)[np.random.default_rng(seed).permutation(6)]
    for rank, run in enumerate(order):
        validation_mask = metadata.iloc[source_indices]["run"].astype(str).to_numpy() == str(run)
        train_indices = source_indices[~validation_mask]
        validation_indices = source_indices[validation_mask]
        if set(labels[train_indices]) == set(range(4)) and set(labels[validation_indices]) == set(range(4)):
            return str(run), rank
    raise ValueError("No whole-run validation split retained all four classes")


def make_cross_session_splits(
    subject: int,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    config: ExperimentConfig,
    context: RunContext,
) -> tuple[CrossSessionSplit, CrossSessionSplit]:
    """Create and audit both subject-specific cross-session data partitions."""

    metadata = metadata.copy().reset_index(drop=True)
    metadata["session"] = metadata["session"].astype(str)
    metadata["run"] = metadata["run"].astype(str)
    sessions = sorted(metadata["session"].unique())
    if len(sessions) != 2:
        raise ValueError(f"Expected two recording sessions, received {sessions}")

    split_seed = int(config.section("dataset")["validation_split_seed"])
    splits: list[CrossSessionSplit] = []
    manifest: list[dict[str, object]] = []
    for direction_index, (source_session, target_session) in enumerate(
        ((sessions[0], sessions[1]), (sessions[1], sessions[0]))
    ):
        source = np.flatnonzero(metadata["session"].to_numpy() == source_session)
        target = np.flatnonzero(metadata["session"].to_numpy() == target_session)
        validation_run, rank = _choose_validation_run(source, labels, metadata, split_seed)
        validation_mask = metadata.iloc[source]["run"].astype(str).to_numpy() == validation_run
        train = source[~validation_mask]
        validation = source[validation_mask]
        direction = f"{source_session}_to_{target_session}"
        if set(train) & set(validation) or set(train) & set(target) or set(validation) & set(target):
            raise AssertionError("Training, validation, and test indices overlap")
        split = CrossSessionSplit(
            direction=direction,
            direction_index=direction_index,
            train_session=source_session,
            test_session=target_session,
            validation_run=validation_run,
            train_indices=train,
            validation_indices=validation,
            test_indices=target,
        )
        splits.append(split)
        for role, indices in (("train", train), ("validation", validation), ("test", target)):
            for index in indices:
                manifest.append(
                    {
                        "protocol_hash": context.protocol_hash,
                        "config_hash": context.config_hash,
                        "method": context.method,
                        "subject": subject,
                        "direction": direction,
                        "direction_index": direction_index,
                        "train_session": source_session,
                        "test_session": target_session,
                        "validation_run": validation_run,
                        "validation_candidate_rank": rank,
                        "trial_index": int(index),
                        "session": metadata.iloc[index]["session"],
                        "run": metadata.iloc[index]["run"],
                        "label_id": int(labels[index]),
                        "role": role,
                    }
                )
    upsert_csv(
        context.csv_dir / "split_manifest.csv",
        manifest,
        ["config_hash", "subject", "direction", "trial_index"],
    )
    frame = pd.DataFrame(manifest)
    counts = (
        frame.groupby(
            [
                "protocol_hash",
                "config_hash",
                "method",
                "subject",
                "direction",
                "role",
                "session",
                "run",
                "label_id",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n_trials")
    )
    upsert_csv(
        context.csv_dir / "split_counts.csv",
        counts.to_dict("records"),
        ["config_hash", "subject", "direction", "role", "session", "run", "label_id"],
    )
    return splits[0], splits[1]
