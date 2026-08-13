"""Version-stable BNCI2014-001 loading with source artifact rejection."""

from __future__ import annotations

import inspect
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from moabb.datasets import BNCI2014_001
from moabb.paradigms import FilterBankMotorImagery, MotorImagery
from scipy.io import loadmat

from semantic_eeg.config import ExperimentConfig
from semantic_eeg.constants import LABEL_TO_ID, N_CHANNELS, N_TIMES
from semantic_eeg.utils.io import upsert_csv
from semantic_eeg.utils.run import RunContext


def make_dataset() -> BNCI2014_001:
    """Create a BNCI2014-001 loader compatible with supported MOABB APIs."""

    parameters = inspect.signature(BNCI2014_001).parameters
    if "artifact_handling" in parameters:
        return BNCI2014_001(artifact_handling="ignore")
    return BNCI2014_001()


def encode_labels(labels: np.ndarray) -> np.ndarray:
    """Map the four motor-imagery labels to integer class identifiers."""

    names = np.asarray(labels).astype(str)
    unknown = set(names) - set(LABEL_TO_ID)
    if unknown:
        raise ValueError(f"Unexpected motor-imagery labels: {sorted(unknown)}")
    return np.asarray([LABEL_TO_ID[name] for name in names], dtype=np.int64)


def trim_epoch(array: np.ndarray, time_axis: int) -> np.ndarray:
    """Trim a preprocessed epoch array to the frozen 560-sample interval."""

    if array.shape[time_axis] < N_TIMES:
        raise ValueError(f"Expected at least {N_TIMES} time samples, received {array.shape}")
    slices = [slice(None)] * array.ndim
    slices[time_axis] = slice(0, N_TIMES)
    return np.asarray(array[tuple(slices)], dtype=np.float32)


def _flatten_paths(value: Any) -> list[Path]:
    if isinstance(value, (str, os.PathLike)):
        return [Path(value)]
    paths: list[Path] = []
    for item in value:
        paths.extend(_flatten_paths(item))
    return paths


def _source_trial_runs(path: Path) -> list[dict[str, Any]]:
    payload = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    if "data" not in payload:
        raise KeyError(f"No data variable in {path}")
    source_runs: list[dict[str, Any]] = []
    for source_run_index, run in enumerate(np.atleast_1d(payload["data"]).ravel()):
        if not hasattr(run, "trial"):
            continue
        trials = np.atleast_1d(run.trial).ravel()
        if trials.size == 0:
            continue
        flags = np.asarray(getattr(run, "artifacts", np.zeros(trials.size)), dtype=np.int64).ravel()
        labels = np.asarray(getattr(run, "y", np.empty(0)), dtype=np.int64).ravel()
        if flags.size != trials.size:
            raise ValueError(f"Artifact count mismatch in {path}, source run {source_run_index}")
        if labels.size not in (0, trials.size):
            raise ValueError(f"Label count mismatch in {path}, source run {source_run_index}")
        source_runs.append(
            {
                "source_run_index": source_run_index,
                "flags": flags.astype(bool),
                "labels": labels,
            }
        )
    if not source_runs:
        raise ValueError(f"No motor-imagery trial runs found in {path}")
    return source_runs


def reject_source_artifacts(
    subject: int,
    dataset: BNCI2014_001,
    array: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    context: RunContext,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Map source artifact flags to MOABB epochs, audit them, and retain clean trials."""

    metadata = metadata.copy().reset_index(drop=True)
    metadata["session"] = metadata["session"].astype(str)
    metadata["run"] = metadata["run"].astype(str)
    paths = _flatten_paths(dataset.data_path(subject))
    sessions = list(dict.fromkeys(metadata["session"].tolist()))
    if len(paths) != len(sessions):
        raise ValueError(f"Expected {len(sessions)} session files, received {paths}")

    by_kind: dict[str, Path] = {}
    for path in paths:
        stem = path.stem.upper()
        kind = "T" if stem.endswith("T") else "E" if stem.endswith("E") else None
        if kind is not None:
            by_kind[kind] = path

    session_paths: dict[str, Path] = {}
    for position, session in enumerate(sessions):
        lower = session.lower()
        kind = "T" if "train" in lower else "E" if "test" in lower else None
        session_paths[session] = by_kind.get(kind, paths[position])

    keep = np.ones(len(metadata), dtype=bool)
    mapped = np.zeros(len(metadata), dtype=bool)
    audit_rows: list[dict[str, Any]] = []
    for session in sessions:
        session_mask = metadata["session"].to_numpy() == session
        run_names = list(dict.fromkeys(metadata.loc[session_mask, "run"].tolist()))
        source_runs = _source_trial_runs(session_paths[session])
        if len(source_runs) != len(run_names):
            raise ValueError(
                f"Run mismatch for Subject {subject}, {session}: "
                f"MOABB={run_names}, source={len(source_runs)}"
            )
        for run_name, source in zip(run_names, source_runs, strict=True):
            rows = np.flatnonzero(session_mask & (metadata["run"].to_numpy() == run_name))
            flags = source["flags"]
            if len(rows) != len(flags):
                raise ValueError(
                    f"Trial mismatch for Subject {subject}, {session}, {run_name}: "
                    f"MOABB={len(rows)}, source={len(flags)}"
                )
            source_labels = source["labels"]
            if source_labels.size and not np.array_equal(labels[rows], source_labels - 1):
                raise ValueError(f"Label-order mismatch for Subject {subject}, {session}, {run_name}")
            keep[rows] = ~flags
            mapped[rows] = True
            for within_run, (row, flag) in enumerate(zip(rows, flags, strict=True)):
                audit_rows.append(
                    {
                        "protocol_hash": context.protocol_hash,
                        "config_hash": context.config_hash,
                        "method": context.method,
                        "subject": subject,
                        "original_trial_index": int(row),
                        "session": session,
                        "run": run_name,
                        "source_run_index": int(source["source_run_index"]),
                        "trial_in_run": within_run,
                        "label_id": int(labels[row]),
                        "artifact_flag": bool(flag),
                        "kept": bool(not flag),
                        "source_file": str(session_paths[session]),
                    }
                )
    if not mapped.all():
        raise AssertionError(f"Artifact audit failed to map {int((~mapped).sum())} trials")

    upsert_csv(
        context.csv_dir / "artifact_trial_audit.csv",
        audit_rows,
        ["config_hash", "subject", "original_trial_index"],
    )
    summary = (
        pd.DataFrame(audit_rows)
        .groupby(["protocol_hash", "config_hash", "method", "subject", "session"], dropna=False)
        .agg(
            n_trials_source=("kept", "size"),
            n_trials_rejected=("artifact_flag", "sum"),
            n_trials_retained=("kept", "sum"),
        )
        .reset_index()
    )
    upsert_csv(
        context.csv_dir / "artifact_rejection_summary.csv",
        summary.to_dict("records"),
        ["config_hash", "subject", "session"],
    )
    context.log(
        "INFO",
        "artifact_rejection",
        f"Subject {subject}: rejected {int((~keep).sum())}/{len(keep)} source-flagged trials",
        subject=subject,
    )
    return np.asarray(array[keep]), np.asarray(labels[keep]), metadata.loc[keep].reset_index(drop=True)


def _load_with_retries(
    subject: int,
    paradigm_factory: Callable[[], Any],
    expected_dimensions: int,
    time_axis: int,
    context: RunContext,
    attempts: int = 3,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            context.log("INFO", "data_load", f"Subject {subject}, attempt {attempt}", subject=subject)
            dataset = make_dataset()
            array, names, metadata = paradigm_factory().get_data(dataset=dataset, subjects=[subject])
            array = np.asarray(array)
            if array.ndim != expected_dimensions or array.shape[1] != N_CHANNELS:
                raise ValueError(f"Unexpected preprocessed data shape: {array.shape}")
            array = trim_epoch(array, time_axis)
            labels = encode_labels(names)
            metadata = metadata.copy().reset_index(drop=True)
            if not (len(array) == len(labels) == len(metadata)) or not np.isfinite(array).all():
                raise ValueError("Loaded data are incomplete or non-finite")
            return reject_source_artifacts(subject, dataset, array, labels, metadata, context)
        except BaseException as error:
            last_error = error
            context.log(
                "WARNING",
                "data_load_attempt_failed",
                f"{type(error).__name__}: {error}",
                subject=subject,
            )
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Unable to load Subject {subject}") from last_error


def load_filterbank_subject(
    subject: int,
    config: ExperimentConfig,
    context: RunContext,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one subject as five-band EEG trials with source-flag rejection."""

    dataset = config.section("dataset")
    bands = config.frequency_bands

    def factory() -> FilterBankMotorImagery:
        return FilterBankMotorImagery(
            n_classes=4,
            filters=list(bands),
            tmin=float(dataset["epoch_start"]),
            tmax=float(dataset["epoch_stop"]),
            baseline=None,
            resample=float(dataset["sampling_frequency"]),
        )

    array, labels, metadata = _load_with_retries(subject, factory, 4, 2, context)
    if array.shape[-1] != len(bands):
        raise ValueError(f"Expected {len(bands)} filter-bank arrays, received {array.shape}")
    return array, labels, metadata


def load_wideband_subject(
    subject: int,
    config: ExperimentConfig,
    context: RunContext,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one subject as 8--30 Hz EEG trials with source-flag rejection."""

    dataset = config.section("dataset")
    bandpass = config.section("reconstruction")["bandpass"]

    def factory() -> MotorImagery:
        return MotorImagery(
            n_classes=4,
            fmin=float(bandpass[0]),
            fmax=float(bandpass[1]),
            tmin=float(dataset["epoch_start"]),
            tmax=float(dataset["epoch_stop"]),
            baseline=None,
            resample=float(dataset["sampling_frequency"]),
        )

    return _load_with_retries(subject, factory, 3, 2, context)
