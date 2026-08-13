"""Conventional FBCSP-PCA transmitter and noise-aware receiver experiment."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch

from semantic_eeg.config import ExperimentConfig
from semantic_eeg.constants import METHOD_CONVENTIONAL
from semantic_eeg.data.bnci2014_001 import load_filterbank_subject
from semantic_eeg.data.splits import CrossSessionSplit, make_cross_session_splits
from semantic_eeg.evaluation.aggregation import aggregate_detailed_results
from semantic_eeg.features.fbcsp import FBCSPPCA, fit_fbcsp_pca, transform_exact_fbcsp
from semantic_eeg.models.receiver import Receiver
from semantic_eeg.training.common import evaluate_receiver, load_receiver, train_receiver
from semantic_eeg.utils.io import atomic_joblib_save, read_csv, upsert_csv
from semantic_eeg.utils.run import RunContext


FRONT_END_TOLERANCE = 2e-6


@dataclass(frozen=True)
class PreparedFBCSPMessages:
    """Conventional and differentiable messages for one cross-session split."""

    pipeline: FBCSPPCA
    checkpoint_path: Path
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    torch_train: np.ndarray
    torch_validation: np.ndarray
    torch_test: np.ndarray


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return resolved


def _select(values: Iterable[int] | None, allowed: tuple[int, ...], name: str) -> tuple[int, ...]:
    selected = allowed if values is None else tuple(int(value) for value in values)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"{name} must contain distinct values")
    unsupported = set(selected) - set(allowed)
    if unsupported:
        raise ValueError(f"Unsupported {name}: {sorted(unsupported)}")
    return selected


def _pipeline_checkpoint(
    context: RunContext,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
) -> Path:
    return context.checkpoint_dir / f"fbcsp_s{subject:02d}_{split.direction}_k{budget}.joblib"


def _load_pipeline(path: Path, budget: int, context: RunContext) -> FBCSPPCA:
    payload = joblib.load(path)
    if payload.get("protocol_hash") != context.protocol_hash:
        raise ValueError("FBCSP checkpoint protocol mismatch")
    if payload.get("config_hash") != context.config_hash:
        raise ValueError("FBCSP checkpoint configuration mismatch")
    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, FBCSPPCA) or pipeline.budget != budget:
        raise ValueError("FBCSP checkpoint message-budget mismatch")
    return pipeline


def prepare_fbcsp_messages(
    subject: int,
    split: CrossSessionSplit,
    trials: np.ndarray,
    labels: np.ndarray,
    budget: int,
    config: ExperimentConfig,
    context: RunContext,
) -> PreparedFBCSPMessages:
    """Fit or load FBCSP-PCA and create leakage-free split messages.

    The function also verifies that the frozen Torch front end used by the
    semantic system reproduces the MNE/scikit-learn messages within the
    numerical tolerance applied in the publication experiment.
    """
    path = _pipeline_checkpoint(context, subject, split, budget)
    resume = bool(config.section("project")["resume"])
    pipeline: FBCSPPCA
    if resume and path.exists():
        try:
            pipeline = _load_pipeline(path, budget, context)
        except (OSError, ValueError, TypeError, KeyError, EOFError):
            pipeline = fit_fbcsp_pca(
                trials[split.train_indices],
                labels[split.train_indices],
                budget,
                config.frequency_bands,
                int(config.section("conventional")["csp_components_per_class"]),
            )
            atomic_joblib_save(
                {
                    "pipeline": pipeline,
                    "protocol_hash": context.protocol_hash,
                    "config_hash": context.config_hash,
                    "budget_k": budget,
                },
                path,
            )
    else:
        pipeline = fit_fbcsp_pca(
            trials[split.train_indices],
            labels[split.train_indices],
            budget,
            config.frequency_bands,
            int(config.section("conventional")["csp_components_per_class"]),
        )
        atomic_joblib_save(
            {
                "pipeline": pipeline,
                "protocol_hash": context.protocol_hash,
                "config_hash": context.config_hash,
                "budget_k": budget,
            },
            path,
        )

    indices_by_role = (
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
    )
    conventional_messages: list[np.ndarray] = []
    torch_messages: list[np.ndarray] = []
    audit_rows: list[dict[str, object]] = []
    front_end = pipeline.to_torch(context.device)
    for role, indices in indices_by_role:
        role_trials = trials[indices]
        raw_reference = pipeline.raw_features(role_trials)
        reference = pipeline.transform(role_trials)
        torch_message, torch_raw = transform_exact_fbcsp(
            front_end,
            role_trials,
            context.device,
            int(config.section("training")["evaluation_batch_size"]),
            include_raw_features=True,
        )
        raw_error = float(np.max(np.abs(torch_raw - raw_reference)))
        message_error = float(np.max(np.abs(torch_message - reference)))
        passed = raw_error <= FRONT_END_TOLERANCE and message_error <= FRONT_END_TOLERANCE
        audit_rows.append(
            {
                "protocol_hash": context.protocol_hash,
                "config_hash": context.config_hash,
                "method": context.method,
                "subject": subject,
                "direction": split.direction,
                "direction_index": split.direction_index,
                "budget_k": budget,
                "role": role,
                "n_trials": len(indices),
                "max_abs_raw_feature_error": raw_error,
                "max_abs_pca_latent_error": message_error,
                "tolerance": FRONT_END_TOLERANCE,
                "reproduction_pass": passed,
            }
        )
        conventional_messages.append(reference)
        torch_messages.append(np.asarray(torch_message, dtype=np.float32))

    upsert_csv(
        context.csv_dir / "frontend_reproduction_audit.csv",
        audit_rows,
        ["config_hash", "subject", "direction", "budget_k", "role"],
    )
    if not all(bool(row["reproduction_pass"]) for row in audit_rows):
        raise AssertionError(
            "The differentiable FBCSP-PCA front end failed for "
            f"Subject {subject}, {split.direction}"
        )

    manifest_rows = [
        {
            "protocol_hash": context.protocol_hash,
            "config_hash": context.config_hash,
            "method": context.method,
            "subject": subject,
            "direction": split.direction,
            **row,
        }
        for row in pipeline.feature_manifest
    ]
    upsert_csv(
        context.csv_dir / "fbcsp_feature_manifest.csv",
        manifest_rows,
        ["config_hash", "subject", "direction", "feature_index"],
    )
    upsert_csv(
        context.csv_dir / "fbcsp_summary.csv",
        [
            {
                "protocol_hash": context.protocol_hash,
                "config_hash": context.config_hash,
                "method": context.method,
                "subject": subject,
                "direction": split.direction,
                "budget_k": budget,
                "raw_feature_count": pipeline.raw_feature_count,
                "transmitted_values": budget,
                "checkpoint_path": str(path),
            }
        ],
        ["config_hash", "subject", "direction", "budget_k"],
    )
    return PreparedFBCSPMessages(
        pipeline=pipeline,
        checkpoint_path=path,
        train=conventional_messages[0],
        validation=conventional_messages[1],
        test=conventional_messages[2],
        torch_train=torch_messages[0],
        torch_validation=torch_messages[1],
        torch_test=torch_messages[2],
    )


def _result_complete(
    context: RunContext,
    subject: int,
    direction: str,
    seed: int,
    budget: int,
    config: ExperimentConfig,
) -> bool:
    if not bool(config.section("project")["resume"]):
        return False
    detailed = read_csv(context.csv_dir / "baseline_results_detailed.csv")
    if detailed.empty:
        return False
    selected = detailed[
        (detailed["config_hash"].astype(str) == context.config_hash)
        & (detailed["subject"].astype(int) == subject)
        & (detailed["direction"].astype(str) == direction)
        & (detailed["seed"].astype(int) == seed)
        & (detailed["budget_k"].astype(int) == budget)
    ]
    expected = len(config.snr_specs) * int(config.section("communication")["channel_realizations"])
    return len(selected[["snr_label", "channel_realization"]].drop_duplicates()) == expected


def run_conventional_experiment(
    config: ExperimentConfig,
    *,
    subjects: Iterable[int] | None = None,
    budgets: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    device: str | torch.device | None = None,
) -> RunContext:
    """Run the complete conventional experiment and write publication-format CSV files."""
    selected_subjects = _select(subjects, config.subjects, "subjects")
    selected_budgets = _select(budgets, config.budgets, "budgets")
    selected_seeds = _select(seeds, config.seeds, "seeds")
    publication = config.section("publication")
    context = RunContext.create(
        METHOD_CONVENTIONAL,
        config,
        str(publication["semantic_protocol_hash"]),
        str(publication["semantic_config_hash"]),
        _resolve_device(device),
    )
    fail_fast = bool(config.section("project")["fail_fast"])
    failures = 0
    for subject in selected_subjects:
        try:
            trials, labels, metadata = load_filterbank_subject(subject, config, context)
            splits = make_cross_session_splits(subject, labels, metadata, config, context)
        except BaseException as error:
            failures += 1
            context.record_failure("subject_setup", error, subject=subject)
            if fail_fast:
                raise
            continue

        for split in splits:
            for budget in selected_budgets:
                try:
                    messages = prepare_fbcsp_messages(
                        subject,
                        split,
                        trials,
                        labels,
                        budget,
                        config,
                        context,
                    )
                except BaseException as error:
                    failures += 1
                    context.record_failure(
                        "fbcsp_preparation",
                        error,
                        subject=subject,
                        direction=split.direction,
                        budget=budget,
                    )
                    if fail_fast:
                        raise
                    continue

                train_labels = labels[split.train_indices]
                validation_labels = labels[split.validation_indices]
                test_labels = labels[split.test_indices]
                for seed in selected_seeds:
                    if _result_complete(context, subject, split.direction, seed, budget, config):
                        context.log(
                            "INFO",
                            "job_skipped",
                            "The result is already complete",
                            subject,
                            split.direction,
                            seed,
                            budget,
                        )
                        continue
                    receiver_path = (
                        context.checkpoint_dir
                        / (
                            f"conventional_receiver_s{subject:02d}_{split.direction}_"
                            f"k{budget}_seed{seed}.pt"
                        )
                    )
                    try:
                        receiver: Receiver
                        if bool(config.section("project")["resume"]) and receiver_path.exists():
                            try:
                                receiver = load_receiver(
                                    receiver_path,
                                    budget,
                                    context.protocol_hash,
                                    context.config_hash,
                                    context.device,
                                    expected_stage="conventional_receiver",
                                    hidden_units=int(
                                        config.section("conventional")["receiver_hidden_units"]
                                    ),
                                    dropout=float(
                                        config.section("conventional")["receiver_dropout"]
                                    ),
                                )
                            except (OSError, ValueError, KeyError, RuntimeError):
                                receiver = train_receiver(
                                    messages.train,
                                    train_labels,
                                    messages.validation,
                                    validation_labels,
                                    subject=subject,
                                    split=split,
                                    budget=budget,
                                    model_seed=seed,
                                    phase="conventional_receiver",
                                    config=config,
                                    context=context,
                                    checkpoint_path=receiver_path,
                                ).receiver
                        else:
                            receiver = train_receiver(
                                messages.train,
                                train_labels,
                                messages.validation,
                                validation_labels,
                                subject=subject,
                                split=split,
                                budget=budget,
                                model_seed=seed,
                                phase="conventional_receiver",
                                config=config,
                                context=context,
                                checkpoint_path=receiver_path,
                            ).receiver
                        evaluate_receiver(
                            receiver,
                            messages.test,
                            test_labels,
                            subject=subject,
                            split=split,
                            budget=budget,
                            model_seed=seed,
                            checkpoint_path=receiver_path,
                            secondary_checkpoint_path=messages.checkpoint_path,
                            config=config,
                            context=context,
                            output_filename="baseline_results_detailed.csv",
                        )
                        context.log(
                            "INFO",
                            "job_complete",
                            "Saved conventional test results",
                            subject,
                            split.direction,
                            seed,
                            budget,
                        )
                    except BaseException as error:
                        failures += 1
                        context.record_failure(
                            "conventional_job",
                            error,
                            subject,
                            split.direction,
                            seed,
                            budget,
                        )
                        if fail_fast:
                            raise
                    finally:
                        gc.collect()
        del trials, labels, metadata
        gc.collect()

    detailed_path = context.csv_dir / "baseline_results_detailed.csv"
    if detailed_path.exists() and detailed_path.stat().st_size:
        aggregate_detailed_results(detailed_path, "baseline", config, context)
    status = "completed" if failures == 0 else "completed_with_failures"
    context.write_metadata(status)
    return context
