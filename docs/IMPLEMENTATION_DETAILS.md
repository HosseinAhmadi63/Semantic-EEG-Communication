# Implementation details

This document describes how the software realizes the experimental protocol. The authoritative numerical settings are in `configs/paper.yaml`; the code validates the publication-defining values before a run begins.

## System definition

One EEG trial is converted into a vector $z\in\mathbb{R}^{K}$, where $K\in\{16,32,64\}$. Every method applies the same per-vector power normalization:

$$
\widetilde{z}=\sqrt{K}\,\frac{z}{\max(\lVert z\rVert_2,10^{-8})}.
$$

For nonzero vectors, the mean squared transmitted value is therefore one. A noisy received vector is generated as:

$$
r=\widetilde{z}+n,\qquad
n\sim\mathcal{N}(0,\sigma_n^2 I_K),\qquad
\sigma_n^2=10^{-\mathrm{SNR}_{\mathrm{dB}}/10}.
$$

Each scalar in the vector is counted as one real-valued channel use. The implementation does not model quantization, bits, modulation, coding, latency, energy consumption, a wireless standard, or a physical radio. An EEG electrode channel and a communication channel use are distinct concepts.

The noisy conditions are $-10,-5,0,5,10,15,20$ dB. The noise-free reference bypasses noise generation and represents one unique test condition. Its identical prediction result is stored under 20 realization indices so that every condition has the same rectangular CSV key structure; these records are not independent noise-free evaluations.

## Dataset and artifact audit

The loader uses MOABB 1.5.0 to access BNCI2014-001. The dataset has nine subjects, two sessions per subject, six runs per session, 48 trials per run, 22 EEG electrodes, and four balanced motor-imagery classes.

The original MATLAB files contain trial-level artifact flags. The audit maps these flags to MOABB epochs using subject, session, run, within-run trial order, and class label. Flagged epochs are removed before any model is fitted or evaluated. The publication run removed 488 of 5,184 trials and retained 4,696. No additional amplitude threshold or automated rejection rule is applied.

Signals are resampled from 250 Hz to 160 Hz. Epochs contain the interval $0.5\leq t<4.0$ s after cue onset. Each prepared trial therefore has shape $22\times560$.

The reconstruction path uses an 8--30 Hz signal. The conventional and semantic paths use five separately filtered bands: 8--12, 12--16, 16--20, 20--24, and 24--30 Hz.

## Subject-specific cross-session partition

No subject contributes data to another subject's fitted system. Each subject is evaluated in both directions:

1. Session 1 supplies training and validation data, and Session 2 is the held-out test set.
2. Session 2 supplies training and validation data, and Session 1 is the held-out test set.

Within the source session, five complete acquisition runs are assigned to training and one complete run is assigned to validation. The validation run is selected deterministically with seed 2026, subject to all four classes being present in both partitions. Runs are never split across training and validation.

Every data-dependent transformation is fitted on the training partition only. Validation selects a parameter state; it is not used to refit preprocessing. The target session is not accessed until final testing.

## Conventional transmitted representation

The conventional system constructs an 80-dimensional FBCSP feature vector:

- five frequency bands;
- four one-versus-rest class comparisons per band;
- four CSP components per comparison; and
- log average power for each retained component.

CSP covariance matrices are estimated from concatenated training trials with Ledoit--Wolf regularization, no trace normalization, and mutual-information component ordering. The 80 features are standardized using training-set statistics. Principal component analysis with full singular value decomposition is fitted on the standardized training features and reduces them to $K$ values. The resulting vector is power-normalized before AWGN is applied.

The receiver contains a fully connected $K\rightarrow64$ layer, an exponential linear unit, dropout 0.25, and a fully connected $64\rightarrow4$ output layer. It is trained with categorical cross-entropy while each training example receives an SNR sampled uniformly from the seven noisy training conditions.

At the end of each epoch, validation balanced accuracy is evaluated at the seven noisy SNRs with fixed validation disturbances. The retained state maximizes the mean validation balanced accuracy across those conditions.

## Reconstruction-oriented transmitted representation

The reconstruction system receives standardized 8--30 Hz trials. Each electrode is standardized using its mean and standard deviation over training trials and samples; the same values are applied to validation and test data.

The encoder is:

1. Conv1D $22\rightarrow32$, kernel 15, stride 2, padding 7;
2. Conv1D $32\rightarrow64$, kernel 9, stride 2, padding 4;
3. Conv1D $64\rightarrow64$, kernel 7, stride 2, padding 3;
4. batch normalization, exponential linear unit, and dropout 0.25 after every convolution;
5. adaptive average pooling to eight temporal positions; and
6. a fully connected projection $512\rightarrow K$.

The decoder is:

1. a fully connected projection $K\rightarrow4480$ and exponential linear unit;
2. transposed Conv1D $64\rightarrow64$, kernel 7, stride 2, padding 3, output padding 1;
3. transposed Conv1D $64\rightarrow32$, kernel 9, stride 2, padding 4, output padding 1; and
4. transposed Conv1D $32\rightarrow22$, kernel 15, stride 2, padding 7, output padding 1.

Batch normalization and exponential linear units follow the first two transposed convolutions. The autoencoder minimizes mean squared reconstruction error after its power-normalized latent vector has passed through AWGN. Model selection minimizes mean validation reconstruction error over the seven noisy SNRs.

After selection, the encoder is frozen and the decoder is discarded. The same receiver architecture used by the conventional system is then trained on noisy reconstruction latents. Reconstruction error is a training and selection quantity, not a reported study outcome.

## Baseline-preserving semantic representation

The semantic transmitter starts from the power-normalized FBCSP--PCA vector $b$. A residual network applies layer normalization, a $K\rightarrow96$ fully connected layer, a Gaussian error linear unit, and a $96\rightarrow K$ fully connected layer. The message is:

$$
s=\operatorname{PN}\left[b+0.35\tanh(h(b))\right].
$$

The last residual layer is initialized with zero weights and biases, and the receiver is initialized from the trained conventional receiver. The initial semantic system therefore exactly reproduces the corresponding conventional system before learning begins.

Semantic training combines five quantities:

- clean-message classification loss;
- mean classification loss from two independent severe-noise views sampled from $-10$ and $-5$ dB;
- classification loss from one view sampled across all seven noisy SNRs;
- a mean squared anchor penalty between the semantic and conventional messages; and
- temperature-scaled Kullback--Leibler distillation from the fixed conventional system, with temperature 2.

The objective is:

$$
L=0.75L_{\mathrm{clean}}+0.75L_{\mathrm{severe}}+0.50L_{\mathrm{mixed}}
+0.35L_{\mathrm{anchor}}+0.50L_{\mathrm{distill}}.
$$

Classification targets use label smoothing 0.02. Training has two stages. During residual warm-up, the receiver is frozen and the residual network is optimized with AdamW at $5\times10^{-4}$ for at most 160 epochs. During joint refinement, the residual network and receiver are optimized together for at most 220 epochs, with learning rates $2\times10^{-4}$ and $5\times10^{-5}$, respectively.

Semantic validation uses three fixed noise realizations at each noisy SNR and a noise-free evaluation. The selection utility is:

$$
U=\frac{\mathrm{BA}_{-10}+\mathrm{BA}_{-5}}{2}
+0.1\frac{\mathrm{BA}_{0}+\mathrm{BA}_{\mathrm{NF}}}{2}.
$$

A state is eligible only when its balanced accuracy at 0 dB and noise-free is no more than 0.01 below the corresponding conventional result. The unchanged initialization is eligible, so a learned refinement is retained only if it respects this preservation condition.

The classifier-only control is run at $K=32$. It keeps the conventional message fixed, updates only the receiver, uses the clean, severe, mixed, and distillation terms, and follows the same selection utility and preservation condition. It omits the anchor term because the transmitted vector cannot change.

## Shared optimization settings

The reported repetitions use seeds 2026, 2027, and 2028. Adam or AdamW stages use batch size 32, weight decay $10^{-4}$, a minimum improvement of $10^{-4}$, and learning-rate reduction by a factor of 0.5 after 10 epochs without improvement, down to $10^{-5}$. Conventional receiver, reconstruction autoencoder, reconstruction receiver, and semantic warm-up patience are 30 epochs. Semantic joint refinement and classifier-only patience are 40 epochs. Semantic and classifier-only gradient norms are clipped at 5.

The maximum epoch counts are stopping limits rather than fixed training lengths. Only the state selected from the source-session validation set is evaluated on the held-out session.

## Test evaluation and aggregation

Each selected system is tested at all seven noisy SNRs and noise-free. At a noisy SNR, 20 independently generated disturbances are applied to each held-out trial. Corresponding methods use the same channel seeds and therefore receive paired disturbances. The noise-free result is identical at all 20 schema indices because no disturbance is generated; its average is the unique noise-free score.

The detailed CSV layer retains one result for every method, subject, direction, repetition seed, budget, SNR, and channel realization. Balanced accuracy is first averaged across the 20 realizations, then across the three repetition seeds and two directions, producing one value per subject and condition. Group summaries are computed from the nine subject values.

The group mean is accompanied by a two-sided 95% $t$-interval. Paired subject-level differences are checked with Shapiro--Wilk. A paired $t$-test is used when its $p$-value is at least 0.05; otherwise, a Wilcoxon signed-rank test is used. Holm adjustment is applied independently within each prespecified comparison family.

## Reliability features

Each run writes configuration and software metadata, detailed event logs, recoverable checkpoints, failure records, and CSV outputs. Existing completed units are detected when resuming. Atomic writes prevent a partially written CSV from replacing a valid file. Publication analysis validates source hashes, expected condition counts, paired keys, channel seeds, and subject-level aggregation before producing tables or figures.

## Computational scope

The three models are intentionally compact because the study isolates the effect of the transmitted representation under matched conditions. It does not compare against state-of-the-art EEG classifiers, and it does not claim that the selected architectures maximize classification accuracy. Runtime, memory use, floating-point operations, energy consumption, and hardware efficiency were not benchmarked because computational performance is outside the study scope.
