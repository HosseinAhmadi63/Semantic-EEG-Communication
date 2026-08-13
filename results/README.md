# Results

New executions are written to `results/runs/` and are ignored by Git because checkpoints and detailed CSV files can be large.

`publication/source/` contains compressed, immutable CSV records from the executions used in the paper. `publication/generated/` contains the verified tables and Figures 2--5 recreated by `scripts/reproduce_paper_analysis.py`.

The immutable files retain their original run identifiers and checkpoint-path strings as execution provenance. These non-analytic fields include historical internal run labels and Colab/Drive paths; the public API uses scientific method names. Scores, experimental keys, channel seeds, test sizes, and statistical inputs are unchanged, and checksums are verified before analysis.
