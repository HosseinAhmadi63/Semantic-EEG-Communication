# Manuscript-to-code map

This map identifies the implementation responsible for each methodological component. Numerical experiment settings are centralized in `configs/paper.yaml` rather than duplicated in entry scripts.

| Manuscript component | Configuration | Implementation | Entry point or output |
|---|---|---|---|
| BNCI2014-001 access | `dataset.name`, `dataset.subjects` | `src/semantic_eeg/data/bnci2014_001.py` | `scripts/cache_data.py` |
| Provider artifact-flag audit | `dataset.artifact_policy` | `src/semantic_eeg/data/bnci2014_001.py` | `artifact_rejection_summary.csv` |
| Filtering, resampling, and epoching | `dataset.sampling_frequency`, `dataset.epoch_start`, `dataset.epoch_stop`; method frequency bands | `src/semantic_eeg/data/preprocessing.py` | cached subject/session arrays |
| Bidirectional cross-session partition | `dataset.validation_split_seed` | `src/semantic_eeg/data/splits.py` | `split_counts.csv` |
| Per-vector power normalization | communication budgets | `src/semantic_eeg/communication/power.py` | used by every experiment |
| AWGN and deterministic channel seeds | SNR lists, realization count, seed bases | `src/semantic_eeg/communication/awgn.py` | detailed result channel fields |
| FBCSP and PCA representation | `conventional.frequency_bands`, `conventional.csp_components_per_class` | `src/semantic_eeg/features/fbcsp.py` | `scripts/run_conventional.py` |
| Common receiver | receiver width and dropout | `src/semantic_eeg/models/receiver.py` | all three experiment paths |
| Reconstruction encoder and decoder | `reconstruction` section | `src/semantic_eeg/models/autoencoder.py` | `scripts/run_reconstruction.py` |
| Semantic residual transmitter | hidden width and residual scale | `src/semantic_eeg/models/semantic_residual.py` | `scripts/run_semantic.py` |
| Shared receiver training, validation, and held-out evaluation | receiver and communication settings | `src/semantic_eeg/training/common.py` | receiver checkpoints and detailed result rows |
| Conventional experiment orchestration | `conventional` and `training` sections | `src/semantic_eeg/training/conventional.py` | `baseline_results_detailed.csv` and conventional summaries |
| Autoencoder and reconstruction-receiver training | `reconstruction` and `training` sections | `src/semantic_eeg/training/reconstruction.py` | reconstruction checkpoints and CSV files |
| Semantic warm-up, joint refinement, and classifier-only control | `semantic` and `training` sections | `src/semantic_eeg/training/semantic.py` | semantic and receiver-only checkpoints and CSV files |
| Held-out SNR and channel-realization grid | budgets, SNRs, seeds, channel realizations | `src/semantic_eeg/evaluation/experiment.py`; shared conventional evaluation in `src/semantic_eeg/training/common.py` | method-level detailed result CSV files |
| Subject and group aggregation | aggregation rules fixed by the paper | `src/semantic_eeg/evaluation/aggregation.py` | subject and group summary CSV files |
| Paired tests and Holm adjustment | statistical families fixed by the paper | `src/semantic_eeg/evaluation/statistics.py` | statistical comparison CSV files |
| Frozen-source validation and publication analysis | publication source manifest | `src/semantic_eeg/evaluation/publication.py` | manuscript tables and analysis audit |
| Figures 2--5 | publication analysis settings | `src/semantic_eeg/plotting/figures.py` | `results/publication/generated/figures/` |
| CSV definitions and safe writing | output schemas | `src/semantic_eeg/results/` | all structured outputs |
| Configuration validation and command routing | complete YAML document | `src/semantic_eeg/config.py`, `src/semantic_eeg/cli.py` | `semantic-eeg` and `scripts/` |

## Experiment entry points

| Scientific path | PyCharm-friendly script | Installed command |
|---|---|---|
| Dataset download and audit | `scripts/cache_data.py` | `semantic-eeg cache` |
| Conventional FBCSP--PCA system | `scripts/run_conventional.py` | `semantic-eeg conventional` |
| Reconstruction-oriented system | `scripts/run_reconstruction.py` | `semantic-eeg reconstruction` |
| Semantic residual system and classifier-only control | `scripts/run_semantic.py` | `semantic-eeg semantic` |
| Complete sequence | `scripts/run_all.py` | `semantic-eeg all` |
| Publication tables, statistics, and figures | `scripts/reproduce_paper_analysis.py` | `semantic-eeg analysis` |

## Figure inputs

The publication-analysis command consumes detailed and subject-level CSV records, not checkpoints. Its generated figure-data tables provide the exact plotted values:

- `figure_data_group_summary.csv`: balanced-accuracy curves and confidence intervals;
- `figure_data_semantic_gain.csv`: semantic-minus-conventional differences and adjusted significance;
- `figure_data_subject_heatmap.csv`: subject-level low-SNR differences.

Figures can therefore be regenerated without dataset access, model code execution, or manual transcription of values.

## Scope of the implementation

The package reproduces the controlled communication experiment described in the manuscript. It is not an implementation of every motor-imagery decoder and is not intended to benchmark state-of-the-art classification architectures. Computational runtime and complexity measurements are intentionally absent because they were not outcomes of the study.
