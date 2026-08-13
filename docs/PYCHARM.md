# PyCharm setup

## Open and install the project

1. Open the repository root, `Semantic-EEG-Communication`, as the PyCharm project. Do not open `src/semantic_eeg` as a separate project.
2. Create a Python 3.11 virtual environment named `.venv` inside the repository.
3. Select `.venv` as the project interpreter.
4. Open the PyCharm terminal at the repository root and run:

```bash
python -m pip install --upgrade pip
pip install -e .
```

The editable installation makes `semantic_eeg` importable while preserving live source editing. PyCharm should recognize `src` automatically through the installed package. If it does not, right-click `src` and select **Mark Directory as → Sources Root**.

## Run configurations

Create Python run configurations with the repository root as the working directory. The configuration file can be passed explicitly in every experiment run.

| Configuration name | Script path | Parameters |
|---|---|---|
| Cache and audit dataset | `scripts/cache_data.py` | `--config configs/paper.yaml` |
| Conventional transmission | `scripts/run_conventional.py` | `--config configs/paper.yaml --device cpu` |
| Reconstruction transmission | `scripts/run_reconstruction.py` | `--config configs/paper.yaml --device cpu` |
| Semantic transmission and control | `scripts/run_semantic.py` | `--config configs/paper.yaml --device cpu` |
| Complete experiment | `scripts/run_all.py` | `--config configs/paper.yaml --device cpu` |
| Reproduce paper analysis | `scripts/reproduce_paper_analysis.py` | no parameters required |

The complete publication experiment includes nine subjects, two cross-session directions, three model seeds, three message dimensions, seven noisy SNRs, a noise-free condition, and 20 noisy test realizations. It is designed to resume completed units after interruption. Runtime was not benchmarked and depends on the computer, software stack, and available device.

Run the conventional configuration before the semantic configuration. Semantic refinement is deliberately initialized from the corresponding conventional FBCSP--PCA message and receiver checkpoint. The `run_all.py` configuration enforces the required order automatically.

## Optional diagnostic runs

The experiment runners accept restrictions that are useful for verifying an installation before starting the complete run. For example:

```text
--subjects 1 --budgets 32 --seeds 2026 --device cpu
```

Such a restricted execution is a diagnostic experiment and does not reproduce the reported group results. Remove all restrictions, or use only `--config configs/paper.yaml --device cpu`, for the complete protocol.

## Data location

MOABB downloads BNCI2014-001 to `data/mne_data/`. This directory is excluded from Git. If the download is interrupted, rerun the cache configuration; completed and valid files are reused.

## Output location

New results are written under `results/runs/<method>/`. Each method directory contains:

- CSV metadata and event logs;
- detailed evaluation rows;
- training and model-selection histories;
- integrity and failure records; and
- resumable checkpoints.

Generated checkpoints and run outputs are excluded from Git. Publication source CSV archives are stored separately under `results/publication/source/`.

## CPU and GPU use

The reported workflow can be executed on CPU, and the configurations above request CPU explicitly. A CUDA device may be selected with `--device cuda` when a compatible PyTorch installation and GPU are available. CPU and GPU libraries can differ in low-level floating-point behavior; preserve the same configuration, seeds, data partitions, and channel seeds when comparing outputs.

## Reproducing only the published tables and figures

Model training is unnecessary when the goal is to regenerate the reported analysis. Run `scripts/reproduce_paper_analysis.py`; it reads the immutable publication CSV inputs and writes validated outputs to `results/publication/generated/`.
