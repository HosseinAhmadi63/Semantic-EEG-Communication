"""Baseline-preserving task-oriented residual and receiver-only training."""

from __future__ import annotations

import copy
import gc
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_eeg.communication.power import normalize_numpy
from semantic_eeg.config import ExperimentConfig
from semantic_eeg.constants import METHOD_CONVENTIONAL, METHOD_RECEIVER_ONLY, METHOD_SEMANTIC
from semantic_eeg.data.bnci2014_001 import load_filterbank_subject
from semantic_eeg.data.splits import CrossSessionSplit, make_cross_session_splits
from semantic_eeg.evaluation.aggregation import aggregate_detailed_results
from semantic_eeg.evaluation.metrics import (
    passes_clean_guard,
    semantic_validation_utility,
    validation_balanced_accuracies,
)
from semantic_eeg.models.receiver import Receiver
from semantic_eeg.models.semantic_residual import SemanticResidualModel
from semantic_eeg.training.common import evaluate_receiver, load_receiver
from semantic_eeg.training.conventional import prepare_fbcsp_messages
from semantic_eeg.utils.io import atomic_csv, atomic_torch_save, read_csv, upsert_csv
from semantic_eeg.utils.randomness import (
    receiver_initialization_seed,
    seed_everything,
    semantic_initialization_seed,
)
from semantic_eeg.utils.run import RunContext


@dataclass(frozen=True)
class SemanticPhaseResult:
    """Selected checkpoint and history for one semantic training phase."""

    model: SemanticResidualModel
    checkpoint_path: Path
    history: pd.DataFrame
    best_epoch: int
    best_validation_utility: float
    best_validation_scores: dict[str, float]
    duration_seconds: float


@dataclass(frozen=True)
class SemanticTrainingResult:
    """Outputs from residual warmup followed by joint fine-tuning."""

    model: SemanticResidualModel
    warmup: SemanticPhaseResult
    joint: SemanticPhaseResult


@dataclass(frozen=True)
class ReceiverOnlyTrainingResult:
    """Selected classifier-only control checkpoint and training history."""

    receiver: Receiver
    checkpoint_path: Path
    history: pd.DataFrame
    best_epoch: int
    best_validation_utility: float
    best_validation_scores: dict[str, float]
    duration_seconds: float


def _validate_messages(messages: np.ndarray, budget: int, name: str) -> np.ndarray:
    values = np.asarray(messages, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != budget:
        raise ValueError(f"{name} must have shape (trials, {budget}), received {values.shape}")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite, nonempty messages")
    return normalize_numpy(values)


def _validate_labels(labels: np.ndarray, expected_length: int, name: str) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(values) != expected_length:
        raise ValueError(f"{name} has {len(values)} labels for {expected_length} messages")
    if not set(np.unique(values)).issubset({0, 1, 2, 3}):
        raise ValueError(f"{name} contains labels outside the four motor-imagery classes")
    return values


def _message_loader(
    messages: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(messages), torch.from_numpy(labels))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
    )


def _training_channel(message: torch.Tensor, snrs: tuple[float, ...]) -> torch.Tensor:
    values = torch.as_tensor(snrs, dtype=message.dtype, device=message.device)
    selected = values[torch.randint(0, len(values), (len(message),), device=message.device)]
    scale = torch.pow(
        torch.tensor(10.0, dtype=message.dtype, device=message.device),
        -selected / 20.0,
    )
    return message + torch.randn_like(message) * scale[:, None]


def _distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        nn.functional.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2


@torch.inference_mode()
def semantic_messages(
    model: SemanticResidualModel,
    base_messages: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Transform normalized FBCSP--PCA messages with the learned residual."""
    values = _validate_messages(base_messages, model.budget, "base_messages")
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = torch.from_numpy(values[start : start + batch_size]).to(device)
        output.append(model.semantic_message(batch).cpu().numpy().astype(np.float32, copy=False))
    result = np.concatenate(output)
    if not np.isfinite(result).all():
        raise FloatingPointError("The semantic transmitter produced non-finite messages")
    return result


def initialize_semantic_model(
    baseline_receiver: Receiver,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
    model_seed: int,
    config: ExperimentConfig,
    device: torch.device,
) -> SemanticResidualModel:
    """Create a zero-output residual initialized to the conventional receiver."""
    settings = config.section("semantic")
    initialization_seed = semantic_initialization_seed(
        int(settings["semantic_initialization_seed_base"]),
        subject,
        split.direction_index,
        budget,
        model_seed,
    )
    seed_everything(initialization_seed)
    model = SemanticResidualModel(
        budget,
        hidden_units=int(settings["residual_hidden_units"]),
        residual_scale=float(settings["residual_scale"]),
    ).to(device)
    model.receiver.load_state_dict(baseline_receiver.state_dict())
    return model


def _validation_scores(
    receiver: nn.Module,
    messages: np.ndarray,
    labels: np.ndarray,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    communication = config.section("communication")
    semantic = config.section("semantic")
    return validation_balanced_accuracies(
        receiver,
        messages,
        labels,
        config.snr_specs,
        subject,
        split.direction_index,
        budget,
        int(semantic["validation_realizations"]),
        int(communication["validation_channel_seed_base"]),
        device,
        int(config.section("training")["evaluation_batch_size"]),
    )


def _save_semantic_checkpoint(
    model: SemanticResidualModel,
    destination: Path,
    stage: str,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
    model_seed: int,
    epoch: int,
    scores: dict[str, float],
    baseline_scores: dict[str, float],
    config: ExperimentConfig,
    context: RunContext,
) -> None:
    settings = config.section("semantic")
    initialization_seed = semantic_initialization_seed(
        int(settings["semantic_initialization_seed_base"]),
        subject,
        split.direction_index,
        budget,
        model_seed,
    )
    atomic_torch_save(
        {
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "budget_k": budget,
            "model_seed": model_seed,
            "initialization_seed": initialization_seed,
            "best_epoch": epoch,
            "validation_metrics": scores,
            "baseline_validation_metrics": baseline_scores,
            "protocol_hash": context.protocol_hash,
            "config_hash": context.config_hash,
        },
        destination,
    )


def load_semantic_model(
    checkpoint_path: str | Path,
    budget: int,
    stage: str,
    protocol_hash: str,
    config_hash: str,
    config: ExperimentConfig,
    device: torch.device,
) -> SemanticResidualModel:
    """Load a validated residual checkpoint for evaluation or continuation."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if checkpoint.get("stage") != stage:
        raise ValueError(f"Expected a {stage} checkpoint")
    if checkpoint.get("protocol_hash") != protocol_hash:
        raise ValueError("Semantic checkpoint protocol mismatch")
    if checkpoint.get("config_hash") != config_hash:
        raise ValueError("Semantic checkpoint configuration mismatch")
    if int(checkpoint.get("budget_k", -1)) != budget:
        raise ValueError("Semantic checkpoint message-budget mismatch")
    settings = config.section("semantic")
    model = SemanticResidualModel(
        budget,
        hidden_units=int(settings["residual_hidden_units"]),
        residual_scale=float(settings["residual_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _train_semantic_phase(
    model: SemanticResidualModel,
    baseline_receiver: Receiver,
    train_messages: np.ndarray,
    train_labels: np.ndarray,
    validation_messages: np.ndarray,
    validation_labels: np.ndarray,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
    model_seed: int,
    phase: str,
    checkpoint_path: Path,
    config: ExperimentConfig,
    context: RunContext,
) -> SemanticPhaseResult:
    settings = config.section("semantic")
    training = config.section("training")
    if phase == "semantic_residual_warmup":
        maximum_epochs = int(settings["warmup_maximum_epochs"])
        patience = int(settings["warmup_patience"])
        residual_learning_rate = float(settings["warmup_learning_rate"])
        receiver_learning_rate: float | None = None
        phase_seed = model_seed + 200_000
        checkpoint_stage = "residual_warmup"
    elif phase == "semantic_joint_finetune":
        maximum_epochs = int(settings["joint_maximum_epochs"])
        patience = int(settings["joint_patience"])
        residual_learning_rate = float(settings["joint_residual_learning_rate"])
        receiver_learning_rate = float(settings["joint_receiver_learning_rate"])
        phase_seed = model_seed + 300_000
        checkpoint_stage = "joint_finetune"
    else:
        raise ValueError(f"Unknown semantic training phase: {phase}")

    seed_everything(phase_seed)
    loader = _message_loader(
        train_messages,
        train_labels,
        int(training["batch_size"]),
        phase_seed,
        context.device.type == "cuda",
    )
    model = model.to(context.device)
    teacher = copy.deepcopy(baseline_receiver).to(context.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    if receiver_learning_rate is None:
        for parameter in model.receiver.parameters():
            parameter.requires_grad = False
        optimizer = torch.optim.AdamW(
            model.residual.parameters(),
            lr=residual_learning_rate,
            weight_decay=float(training["weight_decay"]),
        )
    else:
        for parameter in model.receiver.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(
            [
                {"params": model.residual.parameters(), "lr": residual_learning_rate},
                {"params": model.receiver.parameters(), "lr": receiver_learning_rate},
            ],
            weight_decay=float(training["weight_decay"]),
        )

    baseline_scores = _validation_scores(
        teacher,
        validation_messages,
        validation_labels,
        subject,
        split,
        budget,
        config,
        context.device,
    )
    initial_messages = semantic_messages(
        model,
        validation_messages,
        context.device,
        int(training["evaluation_batch_size"]),
    )
    initial_scores = _validation_scores(
        model.receiver,
        initial_messages,
        validation_labels,
        subject,
        split,
        budget,
        config,
        context.device,
    )
    clean_guard = float(settings["clean_guard"])
    if not passes_clean_guard(initial_scores, baseline_scores, clean_guard):
        raise AssertionError("The starting semantic checkpoint violates the clean validation guard")
    _save_semantic_checkpoint(
        model,
        checkpoint_path,
        checkpoint_stage,
        subject,
        split,
        budget,
        model_seed,
        0,
        initial_scores,
        baseline_scores,
        config,
        context,
    )

    best_scores = initial_scores
    best_utility = semantic_validation_utility(initial_scores)
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
        min_lr=float(training["minimum_learning_rate"]),
    )
    started = time.perf_counter()
    severe_snrs = (-10.0, -5.0)
    mixed_snrs = config.training_snrs
    temperature = float(settings["distillation_temperature"])
    minimum_improvement = float(training["minimum_improvement"])

    for epoch in range(1, maximum_epochs + 1):
        model.residual.train()
        model.receiver.eval() if receiver_learning_rate is None else model.receiver.train()
        totals = {
            "loss": 0.0,
            "clean": 0.0,
            "severe": 0.0,
            "mixed": 0.0,
            "anchor": 0.0,
            "distillation": 0.0,
        }
        examples = 0
        for base, labels in loader:
            base = base.to(context.device, non_blocking=True)
            labels = labels.to(context.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            semantic = model.semantic_message(base)
            clean_logits = model.receiver(semantic)
            clean_loss = nn.functional.cross_entropy(
                clean_logits,
                labels,
                label_smoothing=float(settings["label_smoothing"]),
            )
            severe_loss = torch.stack(
                [
                    nn.functional.cross_entropy(
                        model.receiver(_training_channel(semantic, severe_snrs)),
                        labels,
                        label_smoothing=float(settings["label_smoothing"]),
                    )
                    for _ in range(int(settings["severe_views"]))
                ]
            ).mean()
            mixed_loss = torch.stack(
                [
                    nn.functional.cross_entropy(
                        model.receiver(_training_channel(semantic, mixed_snrs)),
                        labels,
                        label_smoothing=float(settings["label_smoothing"]),
                    )
                    for _ in range(int(settings["mixed_views"]))
                ]
            ).mean()
            anchor_loss = nn.functional.mse_loss(semantic, base)
            with torch.no_grad():
                teacher_logits = teacher(base)
            distillation_loss = _distillation_loss(clean_logits, teacher_logits, temperature)
            loss = (
                float(settings["clean_loss_weight"]) * clean_loss
                + float(settings["severe_loss_weight"]) * severe_loss
                + float(settings["mixed_loss_weight"]) * mixed_loss
                + float(settings["latent_anchor_weight"]) * anchor_loss
                + float(settings["logit_distillation_weight"]) * distillation_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite semantic training loss")
            loss.backward()
            parameters = [
                parameter for group in optimizer.param_groups for parameter in group["params"]
            ]
            nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            examples += len(labels)
            for name, value in (
                ("loss", loss),
                ("clean", clean_loss),
                ("severe", severe_loss),
                ("mixed", mixed_loss),
                ("anchor", anchor_loss),
                ("distillation", distillation_loss),
            ):
                totals[name] += float(value.item()) * len(labels)

        validation_semantic = semantic_messages(
            model,
            validation_messages,
            context.device,
            int(training["evaluation_batch_size"]),
        )
        scores = _validation_scores(
            model.receiver,
            validation_semantic,
            validation_labels,
            subject,
            split,
            budget,
            config,
            context.device,
        )
        eligible = passes_clean_guard(scores, baseline_scores, clean_guard)
        utility = semantic_validation_utility(scores)
        scheduler.step(utility if eligible else best_utility - 1.0)
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
                "train_loss": totals["loss"] / examples,
                "train_clean_classification_loss": totals["clean"] / examples,
                "train_severe_classification_loss": totals["severe"] / examples,
                "train_mixed_classification_loss": totals["mixed"] / examples,
                "train_latent_anchor_loss": totals["anchor"] / examples,
                "train_logit_distillation_loss": totals["distillation"] / examples,
                "validation_balanced_accuracy": utility,
                "validation_severe_balanced_accuracy": 0.5
                * (scores["-10_dB"] + scores["-5_dB"]),
                "validation_minus10_balanced_accuracy": scores["-10_dB"],
                "validation_minus5_balanced_accuracy": scores["-5_dB"],
                "validation_zero_db_balanced_accuracy": scores["0_dB"],
                "validation_noise_free_balanced_accuracy": scores["noise_free"],
                "validation_clean_guard_pass": eligible,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "receiver_learning_rate": (
                    np.nan if receiver_learning_rate is None else optimizer.param_groups[1]["lr"]
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if eligible and utility > best_utility + minimum_improvement:
            best_utility = utility
            best_scores = scores
            best_epoch = epoch
            stale_epochs = 0
            _save_semantic_checkpoint(
                model,
                checkpoint_path,
                checkpoint_stage,
                subject,
                split,
                budget,
                model_seed,
                epoch,
                scores,
                baseline_scores,
                config,
                context,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model = load_semantic_model(
        checkpoint_path,
        budget,
        checkpoint_stage,
        context.protocol_hash,
        context.config_hash,
        config,
        context.device,
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
                "best_validation_balanced_accuracy": best_utility,
                "best_validation_robust_balanced_accuracy": 0.5
                * (best_scores["-10_dB"] + best_scores["-5_dB"]),
                "best_validation_zero_db_balanced_accuracy": best_scores["0_dB"],
                "best_validation_noise_free_balanced_accuracy": best_scores["noise_free"],
                "best_validation_loss": np.nan,
                "duration_seconds": duration,
                "checkpoint_path": str(checkpoint_path),
            }
        ],
        ["config_hash", "subject", "direction", "seed", "budget_k", "phase"],
    )
    return SemanticPhaseResult(
        model=model,
        checkpoint_path=checkpoint_path,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_validation_utility=best_utility,
        best_validation_scores=best_scores,
        duration_seconds=duration,
    )


def train_semantic_model(
    baseline_receiver: Receiver,
    train_messages: np.ndarray,
    train_labels: np.ndarray,
    validation_messages: np.ndarray,
    validation_labels: np.ndarray,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
    model_seed: int,
    config: ExperimentConfig,
    context: RunContext,
    warmup_checkpoint_path: str | Path | None = None,
    joint_checkpoint_path: str | Path | None = None,
) -> SemanticTrainingResult:
    """Train the frozen two-phase task-oriented residual procedure."""
    train = _validate_messages(train_messages, budget, "train_messages")
    validation = _validate_messages(validation_messages, budget, "validation_messages")
    train_y = _validate_labels(train_labels, len(train), "train_labels")
    validation_y = _validate_labels(validation_labels, len(validation), "validation_labels")
    if budget not in config.budgets or model_seed not in config.seeds:
        raise ValueError(
            "The message budget or model seed is outside the publication configuration"
        )

    stem = f"s{subject:02d}_{split.direction}_k{budget}_seed{model_seed}"
    warmup_path = (
        Path(warmup_checkpoint_path)
        if warmup_checkpoint_path is not None
        else context.checkpoint_dir / f"semantic_residual_warmup_{stem}.pt"
    )
    joint_path = (
        Path(joint_checkpoint_path)
        if joint_checkpoint_path is not None
        else context.checkpoint_dir / f"semantic_joint_finetune_{stem}.pt"
    )
    model = initialize_semantic_model(
        baseline_receiver,
        subject,
        split,
        budget,
        model_seed,
        config,
        context.device,
    )
    initial_semantic = semantic_messages(
        model,
        validation,
        context.device,
        int(config.section("training")["evaluation_batch_size"]),
    )
    maximum_error = float(np.max(np.abs(initial_semantic - validation)))
    receiver_error = max(
        float(
            (
                model.receiver.state_dict()[name]
                - value.to(model.receiver.state_dict()[name].device)
            )
            .abs()
            .max()
            .cpu()
        )
        for name, value in baseline_receiver.state_dict().items()
    )
    passed = bool(maximum_error <= 3e-6 and receiver_error == 0.0)
    upsert_csv(
        context.csv_dir / "initialization_reproduction_audit.csv",
        [
            {
                "protocol_hash": context.protocol_hash,
                "config_hash": context.config_hash,
                "method": context.method,
                "subject": subject,
                "direction": split.direction,
                "direction_index": split.direction_index,
                "seed": model_seed,
                "budget_k": budget,
                "max_abs_zero_residual_latent_error": maximum_error,
                "max_abs_receiver_state_error": receiver_error,
                "initialization_pass": passed,
            }
        ],
        ["config_hash", "subject", "direction", "seed", "budget_k"],
    )
    if not passed:
        raise AssertionError("Baseline-preserving semantic initialization failed")

    warmup = _train_semantic_phase(
        model,
        baseline_receiver,
        train,
        train_y,
        validation,
        validation_y,
        subject,
        split,
        budget,
        model_seed,
        "semantic_residual_warmup",
        warmup_path,
        config,
        context,
    )
    joint = _train_semantic_phase(
        warmup.model,
        baseline_receiver,
        train,
        train_y,
        validation,
        validation_y,
        subject,
        split,
        budget,
        model_seed,
        "semantic_joint_finetune",
        joint_path,
        config,
        context,
    )
    return SemanticTrainingResult(model=joint.model, warmup=warmup, joint=joint)


def _save_receiver_only_checkpoint(
    receiver: Receiver,
    destination: Path,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
    model_seed: int,
    epoch: int,
    scores: dict[str, float],
    baseline_scores: dict[str, float],
    config: ExperimentConfig,
    context: RunContext,
) -> None:
    settings = config.section("semantic")
    initialization_seed = receiver_initialization_seed(
        int(settings["receiver_initialization_seed_base"]),
        subject,
        split.direction_index,
        budget,
        model_seed,
    )
    atomic_torch_save(
        {
            "stage": "receiver_only",
            "receiver_state_dict": receiver.state_dict(),
            "residual_status": "fixed_zero",
            "budget_k": budget,
            "model_seed": model_seed,
            "initialization_seed": initialization_seed,
            "best_epoch": epoch,
            "validation_metrics": scores,
            "baseline_validation_metrics": baseline_scores,
            "protocol_hash": context.protocol_hash,
            "config_hash": context.config_hash,
        },
        destination,
    )


def load_receiver_only_control(
    checkpoint_path: str | Path,
    budget: int,
    protocol_hash: str,
    config_hash: str,
    device: torch.device,
) -> Receiver:
    """Load a validated classifier-only control checkpoint."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if checkpoint.get("stage") != "receiver_only":
        raise ValueError("The checkpoint is not a receiver-only control")
    if checkpoint.get("protocol_hash") != protocol_hash:
        raise ValueError("Receiver-only checkpoint protocol mismatch")
    if checkpoint.get("config_hash") != config_hash:
        raise ValueError("Receiver-only checkpoint configuration mismatch")
    if int(checkpoint.get("budget_k", -1)) != budget:
        raise ValueError("Receiver-only checkpoint message-budget mismatch")
    receiver = Receiver(budget).to(device)
    receiver.load_state_dict(checkpoint["receiver_state_dict"])
    receiver.eval()
    return receiver


def train_receiver_only_control(
    baseline_receiver: Receiver,
    train_messages: np.ndarray,
    train_labels: np.ndarray,
    validation_messages: np.ndarray,
    validation_labels: np.ndarray,
    subject: int,
    split: CrossSessionSplit,
    model_seed: int,
    config: ExperimentConfig,
    context: RunContext,
    checkpoint_path: str | Path | None = None,
) -> ReceiverOnlyTrainingResult:
    """Train the classifier-only control while keeping the message fixed."""
    settings = config.section("semantic")
    training = config.section("training")
    budget = int(settings["receiver_only_budget"])
    train = _validate_messages(train_messages, budget, "train_messages")
    validation = _validate_messages(validation_messages, budget, "validation_messages")
    train_y = _validate_labels(train_labels, len(train), "train_labels")
    validation_y = _validate_labels(validation_labels, len(validation), "validation_labels")
    initialization_seed = receiver_initialization_seed(
        int(settings["receiver_initialization_seed_base"]),
        subject,
        split.direction_index,
        budget,
        model_seed,
    )
    seed_everything(initialization_seed)
    receiver = Receiver(budget).to(context.device)
    receiver.load_state_dict(baseline_receiver.state_dict())
    teacher = copy.deepcopy(baseline_receiver).to(context.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    state_error = max(
        float(
            (
                receiver.state_dict()[name]
                - value.to(receiver.state_dict()[name].device)
            )
            .abs()
            .max()
            .cpu()
        )
        for name, value in baseline_receiver.state_dict().items()
    )
    initialization_pass = state_error == 0.0
    upsert_csv(
        context.csv_dir / "receiver_only_initialization_audit.csv",
        [
            {
                "protocol_hash": context.protocol_hash,
                "config_hash": context.config_hash,
                "method": METHOD_RECEIVER_ONLY,
                "subject": subject,
                "direction": split.direction,
                "direction_index": split.direction_index,
                "seed": model_seed,
                "budget_k": budget,
                "initialization_seed": initialization_seed,
                "residual_status": "fixed_zero",
                "max_abs_receiver_state_error": state_error,
                "initialization_pass": initialization_pass,
            }
        ],
        ["config_hash", "subject", "direction", "seed", "budget_k"],
    )
    if not initialization_pass:
        raise AssertionError("Receiver-only baseline initialization failed")

    phase_seed = model_seed + 300_000
    seed_everything(phase_seed)
    loader = _message_loader(
        train,
        train_y,
        int(training["batch_size"]),
        phase_seed,
        context.device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        receiver.parameters(),
        lr=float(settings["joint_receiver_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
        min_lr=float(training["minimum_learning_rate"]),
    )
    baseline_scores = _validation_scores(
        teacher,
        validation,
        validation_y,
        subject,
        split,
        budget,
        config,
        context.device,
    )
    initial_scores = _validation_scores(
        receiver,
        validation,
        validation_y,
        subject,
        split,
        budget,
        config,
        context.device,
    )
    clean_guard = float(settings["clean_guard"])
    if not passes_clean_guard(initial_scores, baseline_scores, clean_guard):
        raise AssertionError("The receiver-only initialization violates the clean validation guard")
    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else context.checkpoint_dir
        / f"receiver_only_s{subject:02d}_{split.direction}_k{budget}_seed{model_seed}.pt"
    )
    _save_receiver_only_checkpoint(
        receiver,
        destination,
        subject,
        split,
        budget,
        model_seed,
        0,
        initial_scores,
        baseline_scores,
        config,
        context,
    )

    best_scores = initial_scores
    best_utility = semantic_validation_utility(initial_scores)
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    temperature = float(settings["distillation_temperature"])
    severe_snrs = (-10.0, -5.0)
    minimum_improvement = float(training["minimum_improvement"])

    for epoch in range(1, int(settings["joint_maximum_epochs"]) + 1):
        receiver.train()
        totals = {"loss": 0.0, "clean": 0.0, "severe": 0.0, "mixed": 0.0, "distillation": 0.0}
        examples = 0
        for base, labels in loader:
            base = base.to(context.device, non_blocking=True)
            labels = labels.to(context.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            clean_logits = receiver(base)
            clean_loss = nn.functional.cross_entropy(
                clean_logits,
                labels,
                label_smoothing=float(settings["label_smoothing"]),
            )
            severe_loss = torch.stack(
                [
                    nn.functional.cross_entropy(
                        receiver(_training_channel(base, severe_snrs)),
                        labels,
                        label_smoothing=float(settings["label_smoothing"]),
                    )
                    for _ in range(int(settings["severe_views"]))
                ]
            ).mean()
            mixed_loss = torch.stack(
                [
                    nn.functional.cross_entropy(
                        receiver(_training_channel(base, config.training_snrs)),
                        labels,
                        label_smoothing=float(settings["label_smoothing"]),
                    )
                    for _ in range(int(settings["mixed_views"]))
                ]
            ).mean()
            with torch.no_grad():
                teacher_logits = teacher(base)
            distillation_loss = _distillation_loss(clean_logits, teacher_logits, temperature)
            loss = (
                float(settings["clean_loss_weight"]) * clean_loss
                + float(settings["severe_loss_weight"]) * severe_loss
                + float(settings["mixed_loss_weight"]) * mixed_loss
                + float(settings["logit_distillation_weight"]) * distillation_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite receiver-only training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(receiver.parameters(), 5.0)
            optimizer.step()
            examples += len(labels)
            for name, value in (
                ("loss", loss),
                ("clean", clean_loss),
                ("severe", severe_loss),
                ("mixed", mixed_loss),
                ("distillation", distillation_loss),
            ):
                totals[name] += float(value.item()) * len(labels)

        scores = _validation_scores(
            receiver,
            validation,
            validation_y,
            subject,
            split,
            budget,
            config,
            context.device,
        )
        eligible = passes_clean_guard(scores, baseline_scores, clean_guard)
        utility = semantic_validation_utility(scores)
        scheduler.step(utility if eligible else best_utility - 1.0)
        history.append(
            {
                "run_id": context.run_id,
                "protocol_hash": context.protocol_hash,
                "config_hash": context.config_hash,
                "method": METHOD_RECEIVER_ONLY,
                "subject": subject,
                "direction": split.direction,
                "direction_index": split.direction_index,
                "seed": model_seed,
                "budget_k": budget,
                "phase": "receiver_only",
                "epoch": epoch,
                "train_loss": totals["loss"] / examples,
                "train_clean_classification_loss": totals["clean"] / examples,
                "train_severe_classification_loss": totals["severe"] / examples,
                "train_mixed_classification_loss": totals["mixed"] / examples,
                "train_latent_anchor_loss": 0.0,
                "train_logit_distillation_loss": totals["distillation"] / examples,
                "validation_balanced_accuracy": utility,
                "validation_severe_balanced_accuracy": 0.5
                * (scores["-10_dB"] + scores["-5_dB"]),
                "validation_minus10_balanced_accuracy": scores["-10_dB"],
                "validation_minus5_balanced_accuracy": scores["-5_dB"],
                "validation_zero_db_balanced_accuracy": scores["0_dB"],
                "validation_noise_free_balanced_accuracy": scores["noise_free"],
                "validation_clean_guard_pass": eligible,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "receiver_learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if eligible and utility > best_utility + minimum_improvement:
            best_utility = utility
            best_scores = scores
            best_epoch = epoch
            stale_epochs = 0
            _save_receiver_only_checkpoint(
                receiver,
                destination,
                subject,
                split,
                budget,
                model_seed,
                epoch,
                scores,
                baseline_scores,
                config,
                context,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= int(settings["joint_patience"]):
            break

    receiver = load_receiver_only_control(
        destination,
        budget,
        context.protocol_hash,
        context.config_hash,
        context.device,
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
                "method": METHOD_RECEIVER_ONLY,
                "subject": subject,
                "direction": split.direction,
                "direction_index": split.direction_index,
                "seed": model_seed,
                "budget_k": budget,
                "initialization_seed": initialization_seed,
                "phase": "receiver_only",
                "epochs_completed": len(history),
                "best_epoch": best_epoch,
                "best_validation_balanced_accuracy": best_utility,
                "best_validation_robust_balanced_accuracy": 0.5
                * (best_scores["-10_dB"] + best_scores["-5_dB"]),
                "best_validation_zero_db_balanced_accuracy": best_scores["0_dB"],
                "best_validation_noise_free_balanced_accuracy": best_scores["noise_free"],
                "best_validation_loss": np.nan,
                "duration_seconds": duration,
                "checkpoint_path": str(destination),
                "residual_status": "fixed_zero",
            }
        ],
        ["config_hash", "subject", "direction", "seed", "budget_k", "phase"],
    )
    return ReceiverOnlyTrainingResult(
        receiver=receiver,
        checkpoint_path=destination,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_validation_utility=best_utility,
        best_validation_scores=best_scores,
        duration_seconds=duration,
    )


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


def _method_slug(method: str) -> str:
    return method.lower().replace(" ", "_").replace("–", "-").replace("-", "_")


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
    filename: str,
    subject: int,
    direction: str,
    seed: int,
    budget: int,
    config: ExperimentConfig,
) -> bool:
    if not bool(config.section("project")["resume"]):
        return False
    detailed = read_csv(context.csv_dir / filename)
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


def _conventional_paths(config: ExperimentConfig) -> tuple[Path, Path]:
    root = config.output_root / _method_slug(METHOD_CONVENTIONAL)
    return root / "csv", root / "checkpoints"


def _import_conventional_results(config: ExperimentConfig, context: RunContext) -> None:
    conventional_csv, _ = _conventional_paths(config)
    source = conventional_csv / "baseline_results_detailed.csv"
    if not source.exists():
        raise FileNotFoundError(
            "Conventional results are required before semantic training; "
            "run the conventional experiment first"
        )
    detailed = read_csv(source)
    detailed = detailed[detailed["config_hash"].astype(str) == context.config_hash].copy()
    if detailed.empty:
        raise ValueError("The conventional results do not match the semantic configuration")
    atomic_csv(detailed, context.csv_dir / "baseline_results_detailed.csv")


def _prepare_semantic_messages(
    subject: int,
    split: CrossSessionSplit,
    trials: np.ndarray,
    labels: np.ndarray,
    budget: int,
    config: ExperimentConfig,
    context: RunContext,
):
    _, conventional_checkpoints = _conventional_paths(config)
    conventional_pipeline = (
        conventional_checkpoints / f"fbcsp_s{subject:02d}_{split.direction}_k{budget}.joblib"
    )
    semantic_pipeline = (
        context.checkpoint_dir / f"fbcsp_s{subject:02d}_{split.direction}_k{budget}.joblib"
    )
    if not conventional_pipeline.exists():
        raise FileNotFoundError(
            f"Missing conventional FBCSP-PCA checkpoint: {conventional_pipeline}"
        )
    if not semantic_pipeline.exists():
        shutil.copy2(conventional_pipeline, semantic_pipeline)
    return prepare_fbcsp_messages(subject, split, trials, labels, budget, config, context)


def _baseline_receiver_path(
    config: ExperimentConfig,
    subject: int,
    split: CrossSessionSplit,
    budget: int,
    seed: int,
) -> Path:
    _, checkpoints = _conventional_paths(config)
    return checkpoints / (
        f"conventional_receiver_s{subject:02d}_{split.direction}_k{budget}_seed{seed}.pt"
    )


def run_semantic_experiment(
    config: ExperimentConfig,
    *,
    subjects: Iterable[int] | None = None,
    budgets: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    device: str | torch.device | None = None,
) -> RunContext:
    """Run semantic residual transmission and its classifier-only control."""
    selected_subjects = _select(subjects, config.subjects, "subjects")
    selected_budgets = _select(budgets, config.budgets, "budgets")
    selected_seeds = _select(seeds, config.seeds, "seeds")
    publication = config.section("publication")
    context = RunContext.create(
        METHOD_SEMANTIC,
        config,
        str(publication["semantic_protocol_hash"]),
        str(publication["semantic_config_hash"]),
        _resolve_device(device),
    )
    fail_fast = bool(config.section("project")["fail_fast"])
    resume = bool(config.section("project")["resume"])
    receiver_only_budget = int(config.section("semantic")["receiver_only_budget"])
    evaluation_batch_size = int(config.section("training")["evaluation_batch_size"])
    failures = 0
    try:
        _import_conventional_results(config, context)
    except BaseException as error:
        context.record_failure("conventional_import", error)
        context.write_metadata("failed")
        raise

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
            train_labels = labels[split.train_indices]
            validation_labels = labels[split.validation_indices]
            test_labels = labels[split.test_indices]
            for budget in selected_budgets:
                try:
                    messages = _prepare_semantic_messages(
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

                for seed in selected_seeds:
                    baseline_path = _baseline_receiver_path(
                        config,
                        subject,
                        split,
                        budget,
                        seed,
                    )
                    stem = f"s{subject:02d}_{split.direction}_k{budget}_seed{seed}"
                    warmup_path = context.checkpoint_dir / f"semantic_residual_warmup_{stem}.pt"
                    joint_path = context.checkpoint_dir / f"semantic_joint_finetune_{stem}.pt"
                    try:
                        baseline_receiver = load_receiver(
                            baseline_path,
                            budget,
                            context.protocol_hash,
                            context.config_hash,
                            context.device,
                            expected_stage="conventional_receiver",
                        )
                        if not _result_complete(
                            context,
                            "results_detailed.csv",
                            subject,
                            split.direction,
                            seed,
                            budget,
                            config,
                        ):
                            if (
                                resume
                                and joint_path.exists()
                                and _phase_recorded(
                                    context,
                                    subject,
                                    split.direction,
                                    seed,
                                    budget,
                                    "semantic_joint_finetune",
                                )
                            ):
                                semantic_model = load_semantic_model(
                                    joint_path,
                                    budget,
                                    "joint_finetune",
                                    context.protocol_hash,
                                    context.config_hash,
                                    config,
                                    context.device,
                                )
                            else:
                                semantic_model = train_semantic_model(
                                    baseline_receiver,
                                    messages.torch_train,
                                    train_labels,
                                    messages.torch_validation,
                                    validation_labels,
                                    subject,
                                    split,
                                    budget,
                                    seed,
                                    config,
                                    context,
                                    warmup_path,
                                    joint_path,
                                ).model
                            test_semantic = semantic_messages(
                                semantic_model,
                                messages.torch_test,
                                context.device,
                                evaluation_batch_size,
                            )
                            evaluate_receiver(
                                semantic_model.receiver,
                                test_semantic,
                                test_labels,
                                subject=subject,
                                split=split,
                                budget=budget,
                                model_seed=seed,
                                checkpoint_path=joint_path,
                                secondary_checkpoint_path=baseline_path,
                                config=config,
                                context=context,
                            )

                        if budget == receiver_only_budget and not _result_complete(
                            context,
                            "receiver_only_results_detailed.csv",
                            subject,
                            split.direction,
                            seed,
                            budget,
                            config,
                        ):
                            receiver_only_path = context.checkpoint_dir / (
                                f"receiver_only_s{subject:02d}_{split.direction}_"
                                f"k{budget}_seed{seed}.pt"
                            )
                            if (
                                resume
                                and receiver_only_path.exists()
                                and _phase_recorded(
                                    context,
                                    subject,
                                    split.direction,
                                    seed,
                                    budget,
                                    "receiver_only",
                                )
                            ):
                                receiver_only = load_receiver_only_control(
                                    receiver_only_path,
                                    budget,
                                    context.protocol_hash,
                                    context.config_hash,
                                    context.device,
                                )
                            else:
                                receiver_only = train_receiver_only_control(
                                    baseline_receiver,
                                    messages.torch_train,
                                    train_labels,
                                    messages.torch_validation,
                                    validation_labels,
                                    subject,
                                    split,
                                    seed,
                                    config,
                                    context,
                                    receiver_only_path,
                                ).receiver
                            evaluate_receiver(
                                receiver_only,
                                messages.torch_test,
                                test_labels,
                                subject=subject,
                                split=split,
                                budget=budget,
                                model_seed=seed,
                                checkpoint_path=receiver_only_path,
                                secondary_checkpoint_path=baseline_path,
                                config=config,
                                context=context,
                                output_filename="receiver_only_results_detailed.csv",
                                method=METHOD_RECEIVER_ONLY,
                            )
                        context.log(
                            "INFO",
                            "job_complete",
                            "Saved semantic test results",
                            subject,
                            split.direction,
                            seed,
                            budget,
                        )
                    except BaseException as error:
                        failures += 1
                        context.record_failure(
                            "semantic_job",
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

    result_files = (
        ("baseline_results_detailed.csv", "baseline"),
        ("results_detailed.csv", "semantic"),
        ("receiver_only_results_detailed.csv", "receiver_only"),
    )
    for filename, prefix in result_files:
        path = context.csv_dir / filename
        if path.exists() and path.stat().st_size:
            aggregate_detailed_results(path, prefix, config, context)
    expected_semantic_jobs = (
        len(selected_subjects)
        * 2
        * len(selected_seeds)
        * len(selected_budgets)
    )
    expected_receiver_jobs = (
        len(selected_subjects)
        * 2
        * len(selected_seeds)
        * int(receiver_only_budget in selected_budgets)
    )
    completion_rows = []
    for prefix, expected_jobs in (
        ("baseline", expected_semantic_jobs),
        ("semantic", expected_semantic_jobs),
        ("receiver_only", expected_receiver_jobs),
    ):
        audit_path = context.csv_dir / f"{prefix}_completion_audit.csv"
        audit = read_csv(audit_path)
        if not audit.empty:
            audit = audit[
                audit["subject"].astype(int).isin(selected_subjects)
                & audit["seed"].astype(int).isin(selected_seeds)
                & audit["budget_k"].astype(int).isin(selected_budgets)
            ]
        completed_jobs = int(audit["is_complete"].astype(bool).sum()) if not audit.empty else 0
        completion_rows.append(
            {
                "check_name": f"{prefix}_jobs_complete",
                "expected": expected_jobs,
                "observed": completed_jobs,
                "check_pass": completed_jobs == expected_jobs,
            }
        )
    integrity = pd.DataFrame(completion_rows)
    atomic_csv(integrity, context.csv_dir / "full_experiment_integrity_audit.csv")
    status = "COMPLETE" if failures == 0 and integrity["check_pass"].all() else "INCOMPLETE"
    atomic_csv(
        pd.DataFrame(
            [
                {
                    "status": status,
                    "failures": failures,
                    "checks_passed": int(integrity["check_pass"].sum()),
                    "checks_total": len(integrity),
                    "protocol_hash": context.protocol_hash,
                    "config_hash": context.config_hash,
                }
            ]
        ),
        context.csv_dir / "full_experiment_status.csv",
    )
    context.write_metadata("completed" if status == "COMPLETE" else "completed_with_failures")
    return context
