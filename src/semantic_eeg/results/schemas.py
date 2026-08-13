"""Column definitions shared by experiment runners and analysis."""

DETAILED_RESULT_COLUMNS = [
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

DETAILED_RESULT_KEYS = [
    "config_hash",
    "subject",
    "direction",
    "seed",
    "budget_k",
    "snr_label",
    "channel_realization",
]

TRAINING_HISTORY_KEYS = [
    "config_hash",
    "subject",
    "direction",
    "seed",
    "budget_k",
    "phase",
    "epoch",
]

TRAINING_SUMMARY_KEYS = [
    "config_hash",
    "subject",
    "direction",
    "seed",
    "budget_k",
    "phase",
]
