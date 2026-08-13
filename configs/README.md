# Configuration

`paper.yaml` is the frozen experiment specification used for the reported analysis. The command-line runners read this file by default. Every fitted quantity is estimated within the source session of the relevant subject and direction; the opposite session is used only for evaluation.

Changing this file defines a new experiment and should be accompanied by a new output directory. The publication hashes identify the two authoritative executions whose compressed CSV records are included under `results/publication/source/`.
