"""Training of the reconstruction-oriented latent-transmission comparator."""

from __future__ import annotations

import gc
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_eeg.communication.awgn import add_awgn, channel_seed
from semantic_eeg.communication.power import normalize_numpy
from semantic_eeg.config import ExperimentConfig
from semantic_eeg.constants import METHOD_RECONSTRUCTION
from semantic_eeg.data.bnci2014_001 import load_wideband_subject
from semantic_eeg.data.preprocessing import standardize_wideband_trials
from semantic_eeg.data.splits import CrossSessionSplit, make_cross_session_splits
from semantic_eeg.evaluation.aggregation import aggregate_detailed_results
from semantic_eeg.models.autoencoder import ReconstructionAutoencoder, ReconstructionEncoder
from semantic_eeg.training.common import evaluate_receiver, load_receiver, train_receiver
from semantic_eeg.utils.io import atomic_torch_save, read_csv, upsert_csv
from semantic_eeg.utils.randomness import seed_everything
from semantic_eeg.utils.run import RunContext


@dataclass(frozen=True)
class ReconstructionTrainingResult:
    """Best reconstruction checkpoint and its complete epoch history."""

    model: ReconstructionAutoencoder
    checkpoint_path: Path
    history: pd.DataFrame
    best_epoch: int
    best_validation_loss: float
    duration_seconds: float


def _validate_trials(trials: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(trials, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (22, 560):
        raise ValueError(f"{name} must have shape (trials, 22, 560), received {values.shape}")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite, nonempty trial data")
    return values


@torch.inference_mode()
def encode_trials(
    encoder: ReconstructionEncoder,
    trials: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode standardized EEG trials without applying channel noise."""
    values = _validate_trials(trials, "trials")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    encoder.eval()
    batches: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(values[start : start + batch_size]).to(device)
        batches.append(encoder(batch).cpu().numpy().astype(np.float32, copy=False))
    messages = np.concatenate(batches)
    if not np.isfinite(messages).all():
        raise FloatingPointError("The reconstruction encoder produced non-finite messages")
    return messages


@torch.inference_mode()
def _decode_messages(
    model: ReconstructionAutoencoder,
    messages: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.decoder.eval()
    values = np.asarray(messages, dtype=np.float32)
    batches: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(values[start : start + batch_size]).to(device)
        batches.append(model.decoder(batch).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(batches)


def reconstruction_validation_mse(
    model: ReconstructionAutoencoder,
    validation_trials: np.ndarray,
    subject: int,
    direction_index: int,
    budget: int,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    """Return mean validation MSE over the seven noisy channel conditions."""
    trials = _validate_trials(validation_trials, "validation_trials")
    evaluation_batch_size = int(config.section("training")["evaluation_batch_size"])
    messages = normalize_numpy(encode_trials(model.encoder, trials, device, evaluation_batch_size))
    validation_seed_base = int(config.section("communication")["validation_channel_seed_base"])
    losses = []
    for snr_order, (_, snr_db) in enumerate(config.snr_specs[:-1]):
        if snr_db is None:
            raise ValueError("The noisy validation conditions must have finite SNR values")
        noise_seed = channel_seed(
            validation_seed_base,
            subject,
            direction_index,
            budget,
            snr_order,
            0,
        )
        received = add_awgn(messages, snr_db, noise_seed)
        reconstructed = _decode_messages(model, received, device, evaluation_batch_size)
        error = reconstructed.astype(np.float64) - trials.astype(np.float64)
        losses.append(float(np.mean(np.square(error))))
    return float(np.mean(losses))


def train_reconstruction_autoencoder(
    train_trials: np.ndarray,
    validation_trials: np.ndarray,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
    model_seed: int,
    config: ExperimentConfig,
    context: RunContext,
    checkpoint_path: str | Path | None = None,
) -> ReconstructionTrainingResult:
    """Train and select the reconstruction autoencoder used in the paper."""
    train = _validate_trials(train_trials, "train_trials")
    validation = _validate_trials(validation_trials, "validation_trials")
    if budget not in config.budgets:
        raise ValueError(f"Unsupported message budget: {budget}")
    if model_seed not in config.seeds:
        raise ValueError(f"Unsupported model seed: {model_seed}")

    training = config.section("training")
    settings = config.section("reconstruction")
    batch_size = int(training["batch_size"])
    maximum_epochs = int(settings["maximum_epochs"])
    patience = int(settings["early_stopping_patience"])
    minimum_improvement = float(training["minimum_improvement"])
    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else context.checkpoint_dir
        / f"reconstruction_s{subject:02d}_{split.direction}_k{budget}_seed{model_seed}.pt"
    )

    seed_everything(model_seed)
    generator = torch.Generator().manual_seed(model_seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=context.device.type == "cuda",
        generator=generator,
    )
    model = ReconstructionAutoencoder(budget, config.training_snrs).to(context.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
        min_lr=float(training["minimum_learning_rate"]),
    )

    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for (batch,) in loader:
            batch = batch.to(context.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite reconstruction loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            examples += len(batch)

        validation_loss = reconstruction_validation_mse(
            model,
            validation,
            subject,
            split.direction_index,
            budget,
            config,
            context.device,
        )
        scheduler.step(validation_loss)
        history.append(
            {
                "run_id": context.run_id,
                "protocol_hash": context.protocol_hash,
                "config_hash": context.config_hash,
                "method": context.method,
                "subject": subject,
                "direction": split.direction,
                "direction_index": split.direction_index,
                "seed": model_seed,
                "budget_k": budget,
                "phase": "reconstruction_autoencoder",
                "epoch": epoch,
                "train_loss": total_loss / examples,
                "validation_loss": validation_loss,
                "validation_balanced_accuracy": np.nan,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if validation_loss < best_loss - minimum_improvement:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch_save(
                {
                    "stage": "reconstruction_autoencoder",
                    "model_state_dict": model.state_dict(),
                    "budget_k": budget,
                    "model_seed": model_seed,
                    "best_epoch": epoch,
                    "best_validation_reconstruction_loss": validation_loss,
                    "protocol_hash": context.protocol_hash,
                    "config_hash": context.config_hash,
                },
                destination,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_epoch == 0:
        raise RuntimeError("Reconstruction training did not produce a valid checkpoint")
    model = load_reconstruction_autoencoder(
        destination,
        budget,
        context.protocol_hash,
        context.config_hash,
        context.device,
    )
    duration = time.perf_counter() - started
    history_frame = pd.DataFrame(history)
    upsert_csv(
        context.csv_dir / "training_history.csv",
        history,
        ["config_hash", "subject", "direction", "seed", "budget_k", "phase", "epoch"],
    )
    upsert_csv(
        context.csv_dir / "training_summary.csv",
        [
            {
                "run_id": context.run_id,
                "protocol_hash": context.protocol_hash,
                "config_hash": context.config_hash,
                "method": context.method,
                "subject": subject,
                "direction": split.direction,
                "direction_index": split.direction_index,
                "seed": model_seed,
                "budget_k": budget,
                "phase": "reconstruction_autoencoder",
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "best_validation_balanced_accuracy": np.nan,
                "best_validation_loss": best_loss,
                "duration_seconds": duration,
                "checkpoint_path": str(destination),
            }
        ],
        ["config_hash", "subject", "direction", "seed", "budget_k", "phase"],
    )
    return ReconstructionTrainingResult(
        model=model,
        checkpoint_path=destination,
        history=history_frame,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        duration_seconds=duration,
    )


def load_reconstruction_autoencoder(
    checkpoint_path: str | Path,
    budget: int,
    protocol_hash: str,
    config_hash: str,
    device: torch.device,
) -> ReconstructionAutoencoder:
    """Load a validated reconstruction checkpoint for inference."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if checkpoint.get("stage") != "reconstruction_autoencoder":
        raise ValueError("The checkpoint is not a reconstruction autoencoder")
    if checkpoint.get("protocol_hash") != protocol_hash:
        raise ValueError("Reconstruction checkpoint protocol mismatch")
    if checkpoint.get("config_hash") != config_hash:
        raise ValueError("Reconstruction checkpoint configuration mismatch")
    if int(checkpoint.get("budget_k", -1)) != budget:
        raise ValueError("Reconstruction checkpoint message-budget mismatch")
    model = ReconstructionAutoencoder(budget, ()).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _select(
    values: Iterable[int] | None,
    allowed: tuple[int, ...],
    name: str,
) -> tuple[int, ...]:
    selected = allowed if values is None else tuple(int(value) for value in values)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(f"{name} must contain distinct values")
    unsupported = set(selected) - set(allowed)
    if unsupported:
        raise ValueError(f"Unsupported {name}: {sorted(unsupported)}")
    return selected


def _resolve_device(device: str | torch.device | None) -> torch.device:
    resolved = torch.device("cpu") if device is None else torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return resolved


def _phase_recorded(
    context: RunContext,
    subject: int,
    direction: str,
    seed: int,
    budget: int,
    phase: str,
) -> bool:
    summary = read_csv(context.csv_dir / "training_summary.csv")
    if summary.empty:
        return False
    selected = summary[
        (summary["config_hash"].astype(str) == context.config_hash)
        & (summary["subject"].astype(int) == subject)
        & (summary["direction"].astype(str) == direction)
        & (summary["seed"].astype(int) == seed)
        & (summary["budget_k"].astype(int) == budget)
        & (summary["phase"].astype(str) == phase)
    ]
    return len(selected) == 1


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
    detailed = read_csv(context.csv_dir / "results_detailed.csv")
    if detailed.empty:
        return False
    selected = detailed[
        (detailed["config_hash"].astype(str) == context.config_hash)
        & (detailed["subject"].astype(int) == subject)
        & (detailed["direction"].astype(str) == direction)
        & (detailed["seed"].astype(int) == seed)
        & (detailed["budget_k"].astype(int) == budget)
    ]
    expected = len(config.snr_specs) * int(
        config.section("communication")["channel_realizations"]
    )
    return len(selected[["snr_label", "channel_realization"]].drop_duplicates()) == expected


def _rename_reconstruction_summaries(paths: dict[str, Path], context: RunContext) -> None:
    destinations = {
        "direction_seed": context.csv_dir / "results_direction_seed_summary.csv",
        "subject": context.csv_dir / "results_subject_summary.csv",
        "group": context.csv_dir / "results_group_summary.csv",
        "completion": context.csv_dir / "completion_audit.csv",
    }
    for name, source in paths.items():
        source.replace(destinations[name])


def run_reconstruction_experiment(
    config: ExperimentConfig,
    *,
    subjects: Iterable[int] | None = None,
    budgets: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    device: str | torch.device | None = None,
) -> RunContext:
    """Run the reconstruction comparator and write publication-format CSV files."""
    selected_subjects = _select(subjects, config.subjects, "subjects")
    selected_budgets = _select(budgets, config.budgets, "budgets")
    selected_seeds = _select(seeds, config.seeds, "seeds")
    publication = config.section("publication")
    context = RunContext.create(
        METHOD_RECONSTRUCTION,
        config,
        str(publication["reconstruction_protocol_hash"]),
        str(publication["reconstruction_config_hash"]),
        _resolve_device(device),
    )
    fail_fast = bool(config.section("project")["fail_fast"])
    resume = bool(config.section("project")["resume"])
    failures = 0
    evaluation_batch_size = int(config.section("training")["evaluation_batch_size"])

    for subject in selected_subjects:
        try:
            trials, labels, metadata = load_wideband_subject(subject, config, context)
            splits = make_cross_session_splits(subject, labels, metadata, config, context)
        except BaseException as error:
            failures += 1
            context.record_failure("subject_setup", error, subject=subject)
            if fail_fast:
                raise
            continue

        for split in splits:
            try:
                train_trials, validation_trials, test_trials = standardize_wideband_trials(
                    subject,
                    split,
                    trials,
                    context,
                )
                train_labels = labels[split.train_indices]
                validation_labels = labels[split.validation_indices]
                test_labels = labels[split.test_indices]
            except BaseException as error:
                failures += 1
                context.record_failure(
                    "preprocessing",
                    error,
                    subject=subject,
                    direction=split.direction,
                )
                if fail_fast:
                    raise
                continue

            for budget in selected_budgets:
                for seed in selected_seeds:
                    if _result_complete(context, subject, split.direction, seed, budget, config):
                        context.log(
                            "INFO",
                            "job_skipped",
                            "The reconstruction result is already complete",
                            subject,
                            split.direction,
                            seed,
                            budget,
                        )
                        continue
                    autoencoder_path = (
                        context.checkpoint_dir
                        / (
                            f"reconstruction_autoencoder_s{subject:02d}_{split.direction}_"
                            f"k{budget}_seed{seed}.pt"
                        )
                    )
                    receiver_path = (
                        context.checkpoint_dir
                        / (
                            f"reconstruction_receiver_s{subject:02d}_{split.direction}_"
                            f"k{budget}_seed{seed}.pt"
                        )
                    )
                    try:
                        if (
                            resume
                            and autoencoder_path.exists()
                            and _phase_recorded(
                                context,
                                subject,
                                split.direction,
                                seed,
                                budget,
                                "reconstruction_autoencoder",
                            )
                        ):
                            autoencoder = load_reconstruction_autoencoder(
                                autoencoder_path,
                                budget,
                                context.protocol_hash,
                                context.config_hash,
                                context.device,
                            )
                        else:
                            autoencoder = train_reconstruction_autoencoder(
                                train_trials,
                                validation_trials,
                                subject,
                                split,
                                budget,
                                seed,
                                config,
                                context,
                                autoencoder_path,
                            ).model

                        train_messages = encode_trials(
                            autoencoder.encoder,
                            train_trials,
                            context.device,
                            evaluation_batch_size,
                        )
                        validation_messages = encode_trials(
                            autoencoder.encoder,
                            validation_trials,
                            context.device,
                            evaluation_batch_size,
                        )
                        test_messages = encode_trials(
                            autoencoder.encoder,
                            test_trials,
                            context.device,
                            evaluation_batch_size,
                        )
                        if (
                            resume
                            and receiver_path.exists()
                            and _phase_recorded(
                                context,
                                subject,
                                split.direction,
                                seed,
                                budget,
                                "reconstruction_receiver",
                            )
                        ):
                            receiver = load_receiver(
                                receiver_path,
                                budget,
                                context.protocol_hash,
                                context.config_hash,
                                context.device,
                                expected_stage="reconstruction_receiver",
                            )
                        else:
                            receiver = train_receiver(
                                train_messages,
                                train_labels,
                                validation_messages,
                                validation_labels,
                                subject=subject,
                                split=split,
                                budget=budget,
                                model_seed=seed,
                                phase="reconstruction_receiver",
                                config=config,
                                context=context,
                                checkpoint_path=receiver_path,
                            ).receiver
                        evaluate_receiver(
                            receiver,
                            test_messages,
                            test_labels,
                            subject=subject,
                            split=split,
                            budget=budget,
                            model_seed=seed,
                            checkpoint_path=receiver_path,
                            secondary_checkpoint_path=autoencoder_path,
                            config=config,
                            context=context,
                        )
                        context.log(
                            "INFO",
                            "job_complete",
                            "Saved reconstruction test results",
                            subject,
                            split.direction,
                            seed,
                            budget,
                        )
                    except BaseException as error:
                        failures += 1
                        context.record_failure(
                            "reconstruction_job",
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

    detailed_path = context.csv_dir / "results_detailed.csv"
    if detailed_path.exists() and detailed_path.stat().st_size:
        generated = aggregate_detailed_results(
            detailed_path,
            "reconstruction",
            config,
            context,
        )
        _rename_reconstruction_summaries(generated, context)
    context.write_metadata("completed" if failures == 0 else "completed_with_failures")
    return context
