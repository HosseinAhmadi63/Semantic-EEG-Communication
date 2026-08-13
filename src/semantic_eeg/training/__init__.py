"""Training routines for the conventional, reconstruction, and semantic systems."""

from semantic_eeg.training.common import (
    ReceiverTrainingResult,
    evaluate_receiver,
    load_receiver,
    predict_classes,
    train_receiver,
    validation_balanced_accuracy,
)
from semantic_eeg.training.conventional import (
    PreparedFBCSPMessages,
    prepare_fbcsp_messages,
    run_conventional_experiment,
)
from semantic_eeg.training.reconstruction import (
    ReconstructionTrainingResult,
    encode_trials,
    run_reconstruction_experiment,
    train_reconstruction_autoencoder,
)
from semantic_eeg.training.semantic import (
    ReceiverOnlyTrainingResult,
    SemanticPhaseResult,
    SemanticTrainingResult,
    run_semantic_experiment,
    semantic_messages,
    train_receiver_only_control,
    train_semantic_model,
)

__all__ = [
    "PreparedFBCSPMessages",
    "ReceiverOnlyTrainingResult",
    "ReceiverTrainingResult",
    "ReconstructionTrainingResult",
    "SemanticPhaseResult",
    "SemanticTrainingResult",
    "encode_trials",
    "evaluate_receiver",
    "load_receiver",
    "predict_classes",
    "prepare_fbcsp_messages",
    "run_conventional_experiment",
    "run_reconstruction_experiment",
    "run_semantic_experiment",
    "semantic_messages",
    "train_receiver_only_control",
    "train_reconstruction_autoencoder",
    "train_receiver",
    "train_semantic_model",
    "validation_balanced_accuracy",
]
