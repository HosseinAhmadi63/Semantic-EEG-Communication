# Semantic EEG Communication

This repository contains the complete Python implementation and frozen analysis inputs for the study:

> **Task-Oriented Semantic Feature Transmission for Robust EEG Motor Imagery Decoding under Additive White Gaussian Noise**  
> Hossein Ahmadi and Luca Mesin

The study asks whether a compact motor-imagery EEG message can be made more robust to communication noise by optimizing its content for the remote classification task. It compares three equally sized, power-normalized transmitted representations:

- conventional filter bank common spatial pattern features compressed by principal component analysis (FBCSP--PCA);
- a convolutional autoencoder latent vector optimized for waveform reconstruction; and
- a baseline-preserving semantic vector obtained by learning a bounded residual refinement of FBCSP--PCA.

A classifier-only control at $K=32$ keeps the conventional transmitted vector fixed while refining only the receiver. This control separates changes to the message from additional receiver training.

The objective is a controlled comparison of representation goals under matched communication conditions, not a state-of-the-art motor-imagery classification benchmark. The systems are deliberately compact to keep reproduction and transfer to other datasets and paradigms tractable. Execution time and computational complexity were not benchmarked because they are outside the study scope.

## Experimental protocol

The frozen configuration is stored in [`configs/paper.yaml`](configs/paper.yaml). Its principal settings are:

- **Dataset:** BNCI2014-001, also known as BCI Competition IV Dataset 2a.
- **Participants and task:** nine subjects and four motor-imagery classes: left hand, right hand, both feet, and tongue.
- **Evaluation:** subject-specific, bidirectional cross-session evaluation. One session supplies training and validation data; the other is held out for testing, and the direction is then reversed.
- **Source-session split:** five complete runs for training and one complete run for validation, selected deterministically with seed 2026.
- **Signal interval:** 0.5--4.0 s after cue onset, resampled from 250 Hz to 160 Hz. Each prepared trial has 22 EEG electrodes and 560 samples.
- **Artifact handling:** the provider's trial-level artifact flags are mapped to MOABB epochs and removed before fitting or evaluation. The same retained trials are used by every method.
- **Message dimensions:** $K\in\{16,32,64\}$ real-valued channel uses per trial.
- **Communication model:** per-vector power normalization followed by additive white Gaussian noise (AWGN).
- **Evaluation conditions:** $-10,-5,0,5,10,15,20$ dB and a noise-free reference.
- **Repetitions:** seeds 2026, 2027, and 2028.
- **Channel repetitions:** 20 paired noise realizations for every noisy test condition.
- **Outcome:** balanced accuracy. Subjects are the independent units for group statistics.
- **Control:** classifier-only refinement at $K=32$.

The analysis uses paired subject-level comparisons, conditionally selects a paired $t$-test or Wilcoxon signed-rank test after a Shapiro--Wilk check, and applies Holm correction within the comparison families defined in the manuscript.

## Repository structure

```text
configs/                     Frozen manuscript configuration
data/                        Dataset and cache instructions
docs/                        Implementation and reproducibility documentation
results/
  publication/source/        Immutable compressed CSV inputs used in the paper
  publication/generated/     Recreated tables, statistics, and figures
  runs/                      Outputs from new executions
scripts/                     PyCharm-friendly experiment entry points
src/semantic_eeg/
  communication/             Power normalization and AWGN operations
  data/                      BNCI2014-001 loading, artifact audit, and splits
  features/                  FBCSP and PCA processing
  models/                    Receiver, autoencoder, and semantic residual model
  training/                  Training and model-selection procedures
  evaluation/                Test execution, aggregation, and statistics
  plotting/                  Publication figure generation
  results/                   Result schemas and CSV writers
  utils/                     Determinism, logging, and validation utilities
```

Generated dataset files, checkpoints, and new run outputs are excluded from Git. The compact publication CSV inputs remain versioned so the reported statistics and figures can be regenerated without retraining.

## Installation

Python 3.11 or 3.12 is supported. The recorded publication executions used Python 3.12.13; Python 3.12 is recommended for the closest environment match.

```bash
git clone https://github.com/HosseinAhmadi63/Semantic-EEG-Communication.git
cd Semantic-EEG-Communication
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

An equivalent Conda environment is provided in [`environment.yml`](environment.yml).

## Reproduce the publication analysis

The reported statistical tables and Figures 2--5 can be regenerated directly from the immutable publication CSV inputs. Dataset download and model training are not required for this step.

```bash
python scripts/reproduce_paper_analysis.py
```

Generated files are written to `results/publication/generated/`. The source archives are validated before analysis, and the analysis stops if their contents do not match the recorded manifest.

## Run the complete experiments

MOABB downloads BNCI2014-001 on first use. No EEG recordings are distributed with this repository.

```bash
python scripts/cache_data.py --config configs/paper.yaml
python scripts/run_conventional.py --config configs/paper.yaml --device cpu
python scripts/run_reconstruction.py --config configs/paper.yaml --device cpu
python scripts/run_semantic.py --config configs/paper.yaml --device cpu
python scripts/reproduce_paper_analysis.py
```

The semantic command runs both the residual-transmission experiment for $K=16,32,64$ and the classifier-only control at $K=32$. It uses the matching conventional messages and receiver checkpoints, so run the conventional experiment first. The dataset and model-experiment sequence can also be launched with:

```bash
python scripts/run_all.py --config configs/paper.yaml --device cpu
```

Run `scripts/reproduce_paper_analysis.py` separately to regenerate the immutable publication analysis.

Runs are checkpointed and resumable. Detailed progress, failures, metadata, model-selection histories, and test results are written as structured CSV files under `results/runs/`. The full CPU execution is computationally substantial; no runtime claim is made because runtime benchmarking was not part of the study.

The installed command exposes the same workflow:

```bash
semantic-eeg cache --config configs/paper.yaml
semantic-eeg conventional --config configs/paper.yaml --device cpu
semantic-eeg reconstruction --config configs/paper.yaml --device cpu
semantic-eeg semantic --config configs/paper.yaml --device cpu
semantic-eeg analysis
```

## PyCharm

Open the repository root as the project, select the `.venv` interpreter, and use the files in `scripts/` as run configurations. Exact settings are provided in [`docs/PYCHARM.md`](docs/PYCHARM.md).

## Outputs and traceability

Every detailed result row records the method, subject, cross-session direction, seed, communication budget, SNR, and channel realization. Run metadata record the configuration and software versions. The CSV columns and aggregation levels are documented in [`docs/RESULT_SCHEMA.md`](docs/RESULT_SCHEMA.md), while [`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md) maps manuscript components to their implementations.

## Data

BNCI2014-001 is publicly available through MOABB and its original data provider. Dataset licenses and access conditions remain those of the provider. Raw EEG recordings and generated caches are not included in this repository.

## Citation

Citation metadata are supplied in [`CITATION.cff`](CITATION.cff). The final journal citation and DOI will be added after publication.

## License

The code is released under the [MIT License](LICENSE). The dataset is not redistributed and remains subject to its original terms.

## Repository

<https://github.com/HosseinAhmadi63/Semantic-EEG-Communication>
