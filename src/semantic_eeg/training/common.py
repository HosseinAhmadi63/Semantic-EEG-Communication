"""Shared receiver training and paired noisy-channel evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_eeg.communication.awgn import add_awgn, channel_seed, training_awgn
from semantic_eeg.communication.power import mean_transmit_power, normalize_numpy
from semantic_eeg.config import ExperimentConfig
from semantic_eeg.data.splits import CrossSessionSplit
from semantic_eeg.models.receiver import Receiver
from semantic_eeg.results.writer import write_detailed_results
from semantic_eeg.utils.io import atomic_torch_save, upsert_csv
from semantic_eeg.utils.randomness import seed_everything
from semantic_eeg.utils.run import RunContext, utc_now


@dataclass(frozen=True)
class ReceiverTrainingResult:
    """Selected receiver checkpoint and the complete training history."""

    receiver: Receiver
    checkpoint_path: Path
    history: pd.DataFrame
    best_epoch: int
    best_validation_balanced_accuracy: float
    duration_seconds: float


def _validate_messages(
    messages: np.ndarray,
    labels: np.ndarray | None,
    budget: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    values = np.asarray(messages, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != budget:
        raise ValueError(f"{name} must have shape (trials, {budget}), received {values.shape}")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite, nonempty messages")
    if labels is None:
        return values, None
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(targets) != len(values):
        raise ValueError(f"{name} has {len(values)} messages but {len(targets)} labels")
    return values, targets


def make_message_loader(
    messages: np.ndarray,
    labels: np.ndarray,
    budget: int,
    batch_size: int,
    seed: int,
    pin_memory: bool = False,
) -> DataLoader:
    """Create the deterministic shuffled loader used for receiver fitting."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values, targets = _validate_messages(messages, labels, budget, "messages")
    if targets is None:
        raise AssertionError("Validated receiver labels are unavailable")
    dataset = TensorDataset(
        torch.from_numpy(normalize_numpy(values)),
        torch.from_numpy(targets),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def predict_classes(
    receiver: nn.Module,
    messages: np.ndarray,
    snr_db: float | None,
    noise_seed: int,
    device: torch.device | str,
    batch_size: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """Predict classes after one deterministic AWGN realization."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = np.asarray(messages, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(
            f"messages must be a finite two-dimensional array, received {values.shape}"
        )
    transmitted = normalize_numpy(values) if normalize else values
    received = add_awgn(transmitted, snr_db, noise_seed)
    receiver = receiver.to(torch.device(device)).eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(received), batch_size):
        batch = torch.from_numpy(received[start : start + batch_size]).to(torch.device(device))
        predictions.append(receiver(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions).astype(np.int64, copy=False)


def validation_balanced_accuracy(
    receiver: nn.Module,
    messages: np.ndarray,
    labels: np.ndarray,
    *,
    subject: int,
    direction_index: int,
    budget: int,
    snr_specs: Sequence[tuple[str, float | None]],
    channel_seed_base: int,
    device: torch.device | str,
    batch_size: int = 256,
    realizations: int = 1,
) -> dict[str, float]:
    """Evaluate fixed validation conditions without selecting on test data."""
    if realizations < 1:
        raise ValueError("realizations must be positive")
    values, targets = _validate_messages(messages, labels, budget, "validation_messages")
    if targets is None:
        raise AssertionError("Validated receiver labels are unavailable")
    transmitted = normalize_numpy(values)
    scores: dict[str, float] = {}
    for snr_order, (label, snr_db) in enumerate(snr_specs):
        if snr_db is None:
            predictions = predict_classes(
                receiver,
                transmitted,
                None,
                0,
                device,
                batch_size,
                normalize=False,
            )
            scores[label] = float(balanced_accuracy_score(targets, predictions))
            continue
        realizations_scores = []
        for realization in range(realizations):
            noise_seed = channel_seed(
                channel_seed_base,
                subject,
                direction_index,
                budget,
                snr_order,
                realization,
            )
            predictions = predict_classes(
                receiver,
                transmitted,
                snr_db,
                noise_seed,
                device,
                batch_size,
                normalize=False,
            )
            realizations_scores.append(balanced_accuracy_score(targets, predictions))
        scores[label] = float(np.mean(realizations_scores))
    return scores


def mean_noisy_validation_accuracy(
    receiver: nn.Module,
    messages: np.ndarray,
    labels: np.ndarray,
    *,
    subject: int,
    direction_index: int,
    budget: int,
    config: ExperimentConfig,
    device: torch.device | str,
) -> float:
    """Return the conventional selection score over the seven noisy SNRs."""
    noisy_specs = tuple((label, snr) for label, snr in config.snr_specs if snr is not None)
    scores = validation_balanced_accuracy(
        receiver,
        messages,
        labels,
        subject=subject,
        direction_index=direction_index,
        budget=budget,
        snr_specs=noisy_specs,
        channel_seed_base=int(config.section("communication")["validation_channel_seed_base"]),
        device=device,
        batch_size=int(config.section("training")["evaluation_batch_size"]),
        realizations=1,
    )
    return float(np.mean(tuple(scores.values())))


def load_receiver(
    checkpoint_path: str | Path,
    budget: int,
    protocol_hash: str,
    config_hash: str,
    device: torch.device | str,
    *,
    expected_stage: str | None = None,
    hidden_units: int = 64,
    dropout: float = 0.25,
) -> Receiver:
    """Load and validate a receiver checkpoint."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=torch.device(device), weights_only=False
    )
    if checkpoint.get("protocol_hash") != protocol_hash:
        raise ValueError("Receiver checkpoint protocol mismatch")
    if checkpoint.get("config_hash") != config_hash:
        raise ValueError("Receiver checkpoint configuration mismatch")
    if int(checkpoint.get("budget_k", -1)) != budget:
        raise ValueError("Receiver checkpoint message-budget mismatch")
    if expected_stage is not None and checkpoint.get("stage") != expected_stage:
        received_stage = checkpoint.get("stage")
        raise ValueError(f"Expected a {expected_stage} checkpoint, received {received_stage}")
    receiver = Receiver(budget, hidden_units=hidden_units, dropout=dropout).to(torch.device(device))
    receiver.load_state_dict(checkpoint["receiver_state_dict"])
    receiver.eval()
    return receiver


def train_receiver(
    train_messages: np.ndarray,
    train_labels: np.ndarray,
    validation_messages: np.ndarray,
    validation_labels: np.ndarray,
    *,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
    model_seed: int,
    phase: str,
    config: ExperimentConfig,
    context: RunContext,
    checkpoint_path: str | Path | None = None,
) -> ReceiverTrainingResult:
    """Train the shared noise-aware classifier used by conventional messages.

    The same routine is also used after the reconstruction encoder has been
    selected and frozen.
    """
    if budget not in config.budgets:
        raise ValueError(f"Unsupported message budget: {budget}")
    if model_seed not in config.seeds:
        raise ValueError(f"Unsupported model seed: {model_seed}")
    settings = config.section("training")
    conventional = config.section("conventional")
    batch_size = int(settings["batch_size"])
    maximum_epochs = int(conventional["maximum_epochs"])
    patience = int(conventional["early_stopping_patience"])
    minimum_improvement = float(settings["minimum_improvement"])
    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else context.checkpoint_dir
        / f"{phase}_s{subject:02d}_{split.direction}_k{budget}_seed{model_seed}.pt"
    )

    train_values, train_targets = _validate_messages(
        train_messages, train_labels, budget, "train_messages"
    )
    validation_values, validation_targets = _validate_messages(
        validation_messages, validation_labels, budget, "validation_messages"
    )
    if train_targets is None or validation_targets is None:
        raise AssertionError("Validated receiver labels are unavailable")

    seed_everything(model_seed)
    loader = make_message_loader(
        train_values,
        train_targets,
        budget,
        batch_size,
        model_seed,
        pin_memory=context.device.type == "cuda",
    )
    receiver = Receiver(
        budget,
        hidden_units=int(conventional["receiver_hidden_units"]),
        dropout=float(conventional["receiver_dropout"]),
    ).to(context.device)
    optimizer = torch.optim.Adam(
        receiver.parameters(),
        lr=float(conventional["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(settings["scheduler_factor"]),
        patience=int(settings["scheduler_patience"]),
        min_lr=float(settings["minimum_learning_rate"]),
    )

    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        receiver.train()
        total_loss = 0.0
        examples = 0
        for messages, targets in loader:
            messages = messages.to(context.device, non_blocking=True)
            targets = targets.to(context.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = receiver(training_awgn(messages, config.training_snrs))
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite receiver loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            examples += len(targets)

        validation_score = mean_noisy_validation_accuracy(
            receiver,
            validation_values,
            validation_targets,
            subject=subject,
            direction_index=split.direction_index,
            budget=budget,
            config=config,
            device=context.device,
        )
        scheduler.step(validation_score)
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
                "phase": phase,
                "epoch": epoch,
                "train_loss": total_loss / examples,
                "validation_loss": np.nan,
                "validation_balanced_accuracy": validation_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if validation_score > best_score + minimum_improvement:
            best_score = validation_score
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch_save(
                {
                    "stage": phase,
                    "receiver_state_dict": receiver.state_dict(),
                    "budget_k": budget,
                    "model_seed": model_seed,
                    "best_epoch": epoch,
                    "best_validation_balanced_accuracy": validation_score,
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
        raise RuntimeError("Receiver training did not produce a valid checkpoint")
    receiver = load_receiver(
        destination,
        budget,
        context.protocol_hash,
        context.config_hash,
        context.device,
        expected_stage=phase,
        hidden_units=int(conventional["receiver_hidden_units"]),
        dropout=float(conventional["receiver_dropout"]),
    )
    duration = time.perf_counter() - started
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
                "phase": phase,
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "best_validation_balanced_accuracy": best_score,
                "best_validation_loss": np.nan,
                "duration_seconds": duration,
                "checkpoint_path": str(destination),
            }
        ],
        ["config_hash", "subject", "direction", "seed", "budget_k", "phase"],
    )
    return ReceiverTrainingResult(
        receiver=receiver,
        checkpoint_path=destination,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_validation_balanced_accuracy=best_score,
        duration_seconds=duration,
    )


def evaluate_receiver(
    receiver: nn.Module,
    messages: np.ndarray,
    labels: np.ndarray,
    *,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
    model_seed: int,
    checkpoint_path: str | Path,
    config: ExperimentConfig,
    context: RunContext,
    secondary_checkpoint_path: str | Path = "",
    output_filename: str = "results_detailed.csv",
    method: str | None = None,
) -> list[dict[str, object]]:
    """Evaluate all paired test conditions and persist one row per realization."""
    values, targets = _validate_messages(messages, labels, budget, "test_messages")
    if targets is None:
        raise AssertionError("Validated receiver labels are unavailable")
    transmitted = normalize_numpy(values)
    power = mean_transmit_power(transmitted)
    communication = config.section("communication")
    realization_count = int(communication["channel_realizations"])
    seed_base = int(communication["channel_seed_base"])
    batch_size = int(config.section("training")["evaluation_batch_size"])
    rows: list[dict[str, object]] = []
    for snr_order, (snr_label, snr_db) in enumerate(config.snr_specs):
        for realization in range(realization_count):
            noise_seed = channel_seed(
                seed_base,
                subject,
                split.direction_index,
                budget,
                snr_order,
                realization,
            )
            predictions = predict_classes(
                receiver,
                transmitted,
                snr_db,
                noise_seed,
                context.device,
                batch_size,
                normalize=False,
            )
            rows.append(
                {
                    "run_id": context.run_id,
                    "protocol_hash": context.protocol_hash,
                    "config_hash": context.config_hash,
                    "method": context.method if method is None else method,
                    "subject": subject,
                    "direction": split.direction,
                    "direction_index": split.direction_index,
                    "train_session": split.train_session,
                    "test_session": split.test_session,
                    "seed": model_seed,
                    "budget_k": budget,
                    "snr_order": snr_order,
                    "snr_label": snr_label,
                    "snr_db": np.nan if snr_db is None else snr_db,
                    "channel_realization": realization,
                    "channel_seed": noise_seed,
                    "n_test_trials": len(targets),
                    "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
                    "mean_transmit_power": power,
                    "checkpoint_path": str(checkpoint_path),
                    "secondary_checkpoint_path": str(secondary_checkpoint_path),
                    "evaluated_at_utc": utc_now(),
                }
            )
    write_detailed_results(context.csv_dir / output_filename, rows)
    return rows
