# Reproducibility guide

Two reproducibility levels are supported: regenerating the reported analysis from frozen CSV records and repeating the complete EEG experiments.

## Level 1: regenerate the reported analysis

This is the shortest path to the manuscript tables, statistical comparisons, and Figures 2--5.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python scripts/reproduce_paper_analysis.py
```

The command reads only `results/publication/source/`. It verifies the source manifest, file hashes, run status, detailed-key completeness, paired channel seeds, subject counts, and aggregation counts before producing output. It never changes the immutable source files.

Generated artifacts are written to:

```text
results/publication/generated/
├── figures/
├── tables/
│   └── source_provenance.csv
└── analysis_summary.json
```

No EEG download, checkpoint, or model retraining is required for this level.

## Level 2: repeat the complete experiment

Install the package as above, then run:

```bash
python scripts/cache_data.py --config configs/paper.yaml
python scripts/run_conventional.py --config configs/paper.yaml --device cpu
python scripts/run_reconstruction.py --config configs/paper.yaml --device cpu
python scripts/run_semantic.py --config configs/paper.yaml --device cpu
```

The same sequence is available through:

```bash
python scripts/run_all.py --config configs/paper.yaml --device cpu
```

The first command downloads BNCI2014-001 through MOABB, maps the provider artifact flags to epochs, validates counts, and caches deterministic inputs. The three experiment commands then fit and evaluate the compared systems. Completed units and checkpoints are reused when a run is resumed.

The full execution is computationally substantial. Runtime was not a study outcome and was not benchmarked, so no completion-time estimate should be interpreted as a scientific result.

## Frozen protocol

`configs/paper.yaml` fixes:

- Subjects 1--9;
- both session directions;
- validation split seed 2026;
- model seeds 2026, 2027, and 2028;
- message dimensions 16, 32, and 64;
- seven noisy training and test SNRs;
- a noise-free test condition;
- 20 paired test disturbances per noisy condition;
- channel-seed derivation bases;
- architecture, optimizer, loss, and selection settings; and
- hashes identifying the authoritative execution protocols.

The configuration validator rejects an altered set of publication-defining values. To conduct a new experiment, copy the configuration, choose a new output directory, and retain the modified file with its outputs. Do not replace the publication source archives with results from a modified protocol.

## Leakage prevention

The following checks are enforced separately for every subject and session direction:

1. The held-out session is assigned before any data-dependent transformation is fitted.
2. Five complete source-session runs form training and one complete run forms validation.
3. CSP filters, FBCSP feature scaling, PCA, and electrode standardization use training data only.
4. Validation selects checkpoints but does not contribute to preprocessing refits.
5. The target session is used only for final evaluation.
6. The same retained trials are used by all methods.

`split_counts.csv` and `artifact_rejection_summary.csv` provide an independent CSV audit of these decisions.

## Determinism and pairing

Python, NumPy, and PyTorch random generators are seeded before fitting. Model initialization, batch ordering, validation disturbances, semantic training views, and test disturbances use deterministic seeds derived from the fixed identifiers of each experimental unit.

Corresponding methods use identical test channel seeds. This pairing is verified from the detailed CSV keys before statistical analysis. The noise-free condition does not generate a disturbance. Its identical result is represented at 20 realization indices to maintain the same CSV shape as the noisy conditions; these rows do not represent independent noise-free measurements.

The pinned dependencies in `requirements.txt` and `environment.yml` describe the reference software environment. CPU execution provides the closest reproduction path. Other platforms or GPUs can introduce small floating-point differences even when the experimental design and random seeds are identical; reproducibility should therefore be judged from validated numerical summaries and statistical conclusions rather than byte-identical checkpoint tensors.

## Expected coverage

For each principal method, a complete detailed evaluation contains 25,920 rows: $9$ subjects $\times2$ directions $\times3$ seeds $\times3$ budgets $\times8$ conditions $\times20$ realization indices. The 20 noise-free rows within a job are identical bookkeeping records for the unique noise-free condition.

The classifier-only control at $K=32$ contains 8,640 rows: $9\times2\times3\times1\times8\times20$.

After averaging realizations, seeds, and directions, each principal method has 216 subject-condition rows: $9$ subjects $\times3$ budgets $\times8$ test conditions. The classifier-only control has 72.

The analysis checks these counts, all nine-subject group cells, the 24-comparison primary family, absence of duplicate detailed keys, finite balanced-accuracy values, and exact pairing of test sizes and channel seeds.

## Independent statistical unit

The subject is the independent unit. Channel realizations, seeds, directions, and trials are repeated measurements within a subject and are averaged before group inference. Recomputing tests from individual trial or channel rows as if they were independent would not reproduce the reported analysis.

## Verifying an installation

Before the complete run, a diagnostic unit can be executed with:

```bash
python scripts/run_conventional.py \
  --config configs/paper.yaml \
  --subjects 1 \
  --budgets 32 \
  --seeds 2026 \
  --device cpu
```

This command checks data access, fitting, channel simulation, checkpointing, and CSV writing. It is not a substitute for the complete publication protocol.

## Failure recovery

Errors are recorded in `failed_jobs.csv` together with the failed stage and identifying fields. The job log and valid completed CSV rows remain available. Correct the reported cause and relaunch the same command; resumable units are discovered from their completed records and checkpoints.

Publication analysis fails rather than silently omitting missing or inconsistent conditions. A successful exit therefore indicates that the required source files and integrity checks were present.
