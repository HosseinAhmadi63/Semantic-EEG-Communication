"""Stable labels shared by training, evaluation, and analysis."""

METHOD_CONVENTIONAL = "FBCSP-PCA"
METHOD_SEMANTIC = "Semantic residual"
METHOD_RECONSTRUCTION = "Reconstruction latent"
METHOD_RECEIVER_ONLY = "Receiver-only"

LABEL_TO_ID = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
N_CLASSES = 4
N_CHANNELS = 22
N_TIMES = 560

DETAIL_COLUMNS = [
    "run_id",
    "protocol_hash",
    "config_hash",
    "method",
    "subject",
    "direction",
    "direction_index",
    "train_session",
    "test_session",
    "seed",
    "budget_k",
    "snr_order",
    "snr_label",
    "snr_db",
    "channel_realization",
    "channel_seed",
    "n_test_trials",
    "balanced_accuracy",
    "mean_transmit_power",
    "checkpoint_path",
    "secondary_checkpoint_path",
    "evaluated_at_utc",
]
