# Result schema

All numerical results required for plots and statistical analysis are stored in CSV files. Checkpoints are needed only to resume or inspect model training; publication analysis does not read them.

New experiment outputs are written beneath:

```text
results/runs/<method>/
├── checkpoints/
└── csv/
```

The method slug identifies conventional FBCSP--PCA transmission, reconstruction-oriented transmission, or semantic residual transmission. The semantic directory also contains the classifier-only control.

## Identifying columns

Files use the following identifiers wherever they apply:

| Column | Meaning |
|---|---|
| `run_id` | Unique identifier for one launched execution |
| `protocol_hash` | Hash of publication-defining protocol content |
| `config_hash` | Hash of the complete effective configuration |
| `method` | Scientific method name |
| `subject` | BNCI2014-001 subject number, 1--9 |
| `direction` | Source-to-target session direction |
| `train_session` | Session supplying training and validation runs |
| `test_session` | Held-out session used only for final evaluation |
| `seed` | Model repetition seed: 2026, 2027, or 2028 |
| `budget_k` | Number of transmitted real values: 16, 32, or 64 |
| `snr_order` | Stable order from -10 dB through noise-free |
| `snr_label` | Machine-readable SNR label, including `noise_free` |
| `snr_db` | Numerical SNR in dB; empty for noise-free |
| `channel_realization` | Test-noise realization index |
| `channel_seed` | Deterministic seed used to generate the disturbance |

## Run management files

### `run_metadata.csv`

One row describes the active or completed run. It records status, start and update times, method, hashes, Python and package versions, device, serialized configuration, and output path.

### `job_log.csv`

One row records each structured event. In addition to the identifying columns, it contains:

- `timestamp_utc`;
- `level`;
- `event`; and
- `message`.

The log is suitable for filtering by subject, direction, seed, and budget. Console text is not needed to audit the run.

### `failed_jobs.csv`

One row records each failed unit with its stage, exception type, message, and traceback. A successful run may contain an empty file with only the schema header.

### `full_experiment_status.csv`

The semantic runner records its terminal status, failure count, and passed completion checks here. The frozen publication source retains the corresponding status from the original execution. Publication analysis requires a completed status and matching counts.

## Data-audit files

### `artifact_rejection_summary.csv`

One row per subject and session records the original trial count, number flagged by the data provider, number removed, and number retained. The publication totals are 5,184 original trials, 488 removed trials, and 4,696 retained trials.

`artifact_trial_audit.csv` retains the trial-level mapping between provider flags and MOABB metadata.

### `split_counts.csv`

Rows identify subject, direction, session, run, class, partition role, and trial count. This file demonstrates that acquisition runs remain intact and that the held-out session never enters fitting or model selection.

`split_manifest.csv` records the selected training, validation, and test runs for every subject and direction. `preprocessing_parameters.csv` records the training-derived electrode standardization parameters used by the reconstruction path.

## Training and selection files

### FBCSP preparation audits

The conventional and semantic paths write:

- `fbcsp_feature_manifest.csv`, identifying the fitted frontend and cached message arrays;
- `fbcsp_summary.csv`, recording feature dimensions and finite-value checks; and
- `frontend_reproduction_audit.csv`, verifying that the differentiable FBCSP implementation reproduces the fitted conventional frontend within tolerance.

### `training_history.csv`

One row per epoch and training stage records the identifying fields together with:

- stage and epoch;
- optimizer learning rate;
- training objective;
- validation objective or balanced accuracy;
- model-selection score;
- preservation eligibility when applicable;
- phase-specific validation components; and
- elapsed operational time.

Reconstruction rows retain validation reconstruction loss. Semantic rows retain the clean, severe-noise, mixed-noise, anchor, distillation, and combined objectives, together with the low-SNR selection utility.

### `training_summary.csv`

One row per fitted phase records epochs completed, selected epoch, selection value, elapsed training time, and checkpoint path. Elapsed time is retained as an operational log field; it is not analyzed as a computational benchmark. The held-out test session does not contribute to model-selection fields.

## Detailed evaluation files

### Detailed result files

The fundamental evaluation row represents one method, subject, direction, model seed, budget, SNR, and channel realization. It contains the complete identifying columns plus:

- `n_test_trials`;
- `balanced_accuracy`;
- `mean_transmit_power`;
- selected and secondary checkpoint paths; and
- the UTC evaluation time.

Every condition has 20 indexed rows per subject, direction, seed, and budget. At a noisy SNR, they represent 20 independent disturbances. The noise-free rows are identical because no disturbance is generated; their 20 indices preserve a rectangular 160-row job schema. They are bookkeeping records, not independent observations, and their mean equals the unique noise-free score. This granularity permits aggregation in a different order while retaining the correct independent unit.

The conventional execution writes `baseline_results_detailed.csv`. The reconstruction execution writes `results_detailed.csv`. The semantic execution writes its proposed-method rows to `results_detailed.csv`, retains corresponding conventional rows in `baseline_results_detailed.csv`, and writes classifier-only rows to `receiver_only_results_detailed.csv`. Their detailed keys and channel seeds are checked before paired analysis.

## Aggregated experiment files

### `<prefix>_results_direction_seed_summary.csv`

Balanced accuracy is averaged across channel realizations, leaving one row per method, subject, direction, seed, budget, and SNR.

### `<prefix>_results_subject_summary.csv`

Direction-seed rows are averaged across the two directions and three model seeds, leaving one row per method, subject, budget, and SNR. Principal fields are:

- `balanced_accuracy_mean`;
- `n_observations`;
- `n_directions`;
- `n_model_seeds`; and
- `n_channel_realizations`.

These subject rows are the inputs to group statistics. Trials, channel realizations, seeds, and directions are not treated as independent inferential samples.

### Completion and integrity audits

Each aggregated method output has a `<prefix>_completion_audit.csv`. The semantic runner also writes `full_experiment_integrity_audit.csv`, which checks completion of its conventional, semantic, and classifier-only grids. These files record pass/fail checks for expected subjects, directions, seeds, budgets, SNRs, channel counts, paired channel seeds, finite metrics, and absence of duplicated detailed keys. The frozen publication source retains the corresponding original audits.

## Publication source and generated outputs

`results/publication/source/` contains compressed immutable CSV records and a checksum manifest. `scripts/reproduce_paper_analysis.py` validates those files before creating `results/publication/generated/`.

Generated tables include:

- experimental configuration;
- group balanced accuracy;
- primary semantic-versus-conventional statistics;
- classifier-only comparisons;
- semantic-versus-reconstruction statistics;
- conventional-versus-reconstruction statistics;
- subject-level results; and
- a complete pairwise-statistics supplement.

Generated figure-data files include:

- `figure_data_group_summary.csv`;
- `figure_data_semantic_gain.csv`; and
- `figure_data_subject_heatmap.csv`.

Each group row records the mean, standard deviation, standard error, two-sided 95% confidence interval, and subject count. Each paired-comparison row records the mean paired difference, its unadjusted 95% confidence interval, the selected test, Shapiro--Wilk result, raw and Holm-adjusted $p$-values, and subject win counts.

## Units

Balanced accuracy is stored on the 0--1 scale in detailed and subject-summary files. Publication tables may additionally express it as a percentage. Paired differences labeled with `_pp` are percentage points. SNR is in dB. `budget_k` is a count of transmitted real values, not bits or bit rate.
