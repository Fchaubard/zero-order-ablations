# Experimental protocol (single source of truth for code and paper)

**Question.** Which zero-order optimizer wins across standard small architectures/tasks?

## Tasks / architectures (~40k parameters each, Xavier-uniform init, zero biases)
| task key          | data                    | model                                                    | params |
|-------------------|-------------------------|----------------------------------------------------------|--------|
| `mlp_mnist`       | MNIST (55k/5k/10k)      | MLP 784-48-32-10, ReLU                                   | 39,578 |
| `conv_cifar10`    | CIFAR-10 (45k/5k/10k)   | conv16-conv32-conv32 (3x3, pool) - fc48 - 10, ReLU       | 39,450 |
| `lstm_ptb`        | PTB char-level          | emb32 - LSTM80 - V                                       | ~41.8k |
| `transformer_ptb` | PTB char-level          | 2 pre-LN blocks, d=48, 4 heads, d_ff=64, learned pos, T=64 | ~39.3k |

Splits: last 5k of the MNIST/CIFAR train sets are validation.  PTB uses the standard
train/valid/test files, character level (vocabulary built from the training text).
Classification batch = 128 images; LM batch = 64 sequences x 64 characters.  No
augmentation, no dropout, no weight decay.  Inputs standardised (MNIST, CIFAR per channel).

## Optimizers (8)
| key    | method                                      | forward passes / step |
|--------|---------------------------------------------|-----------------------|
| `bp`   | backprop (reference)                        | 1 (+1 backward)       |
| `spsa` | SPSA, central difference, Gaussian probes   | 16 = 8 antithetic pairs |
| `rs`   | random search (jumping), best of 15 probes  | 16 = 1 clean + 15     |
| `np`   | classic node perturbation                   | 16 = 1 clean + 15 noisy |
| `anp`  | activity-based node perturbation            | 16 = 1 clean + 15 noisy |
| `inp`  | iterative node perturbation (one layer per pass) | 16 = 1 clean + 15 single-layer noisy |
| `danp` | decorrelated activity-based node perturbation | 16 = 1 clean + 15 noisy |
| `ff`   | Hinton's forward-forward (local goodness)   | 2 (positive + negative) |

* All probes are Gaussian (unit variance, scaled by epsilon).
* **epsilon == learning rate** for the whole run, swept over {0.1, 0.01, 0.001}.
  epsilon is the parameter-probe scale (SPSA, RS) or the node-noise std sigma (NP family).
* Every method writes a gradient (or pseudo-gradient) into `p.grad` and takes an
  **Adam** step with betas (0.9, 0.99), lr = epsilon, no schedule.
* Loss: cross-entropy; per-example for classification, per-sequence mean over the 64
  next-character predictions for LM.  Zero-order estimates use per-example loss
  differences where the method allows it (node perturbation) and batch-mean losses
  where it does not (SPSA, RS).
* Budget: **1 hour of training wall-clock per run** on one RTX 5090 (evaluation time
  is excluded from the clock).  Evaluate on validation + test every 5 minutes of train
  time and at the end.  A run whose loss is NaN, or whose training loss is > 2x its initial
  value or whose validation loss is > 2x its step-0 value at two consecutive evaluations,
  is stopped early and marked diverged (recorded at its last evaluation).
* Model selection: for each (task, optimizer) pick the epsilon with the lowest final
  **validation** loss; report that run's final **test** loss (and test accuracy for
  classification; bits-per-character = loss / ln 2 for LM).
* 4 tasks x 8 optimizers x 3 epsilons = **96 runs**, 8 in parallel (one per GPU).
* Logging: wandb project `zoo-survey`, run name `{task}-{opt}-lr{eps}`; every run also
  writes `results/{task}-{opt}-lr{eps}.json` with config, history and final metrics.

## Stage 2: replication and ablations (after the review of the stage-1 paper)

Stage 1 is the 96-run grid above (`results/`, seed 0, evaluations every 5 minutes); it is kept
as the **epsilon screening** sweep.  Stage 2 (`results_v2/`, `results_bp/`, `results_decouple/`)
re-runs the selected configurations and adds the ablations the reviews asked for.  Differences
from stage 1, all applied to every stage-2 run:

* **Evaluation every 60 s** of train time (evaluation stays outside the clock).
* **Primary metric = best validation checkpoint**: for each (task, optimizer) the epsilon and
  the evaluation with the lowest validation loss over the whole run are selected; that
  checkpoint's test loss / accuracy / bits-per-character is reported (mean +- s.d. over seeds).
  The final-after-one-hour value is reported as a secondary stability metric.  Forward-forward
  on the image tasks is selected by goodness-inference validation accuracy (Hinton's
  label-accumulation inference over all blocks but the first) and reported with that accuracy;
  the linear readout is reported next to it.
* **Divergence rule = validation only**: a run is stopped when its clean validation loss exceeds
  10x its step-0 value at two consecutive evaluations, or when any loss is non-finite (stage-2
  runs before 2026-11-03 01:00 UTC used 2x, which also stopped over-confident but accurate runs
  such as backprop on CIFAR-10 at 67 % accuracy; runs the 2x rule did not stop are identical).  The
  stage-1 training-loss clause compared each method's own training loss with its first-step
  value, which for SPSA is a probe mean, i.e. it was method-dependent; it is dropped.
* **Precision**: full fp32 (TF32 disabled for cuDNN convolutions and matmuls; stage 1 ran with
  PyTorch's default, TF32 on for convolutions).
* **Seeds 0, 1, 2** for the selected epsilon of every (task, optimizer).  Seed 0 keeps the
  historical run name; seeds 1 and 2 append `-s1` / `-s2`.
* **Checkpoints** at the best validation loss, at 5 minutes and at the end (`*.pt`, uploaded
  to wandb) so that estimator diagnostics can be run on trained networks.
* Run names record non-protocol settings: `-sig{sigma}` (probe scale decoupled from the learning
  rate), `-K{passes}` (forward-pass budget), `-wd{weight decay}` (AdamW), `-tf32`.

Ablations (single seed unless stated): backprop at lr 1e-4 and with AdamW weight decay 1e-2 at
lr 1e-3 / 1e-4 (`results_bp/`); probe scale decoupled from the step size, lr = 1e-3 with
sigma in {1e-2, 1e-4} for the six probe methods and lr = 1e-2 with sigma = 1e-3 for SPSA / NP
/ ANP / INP, 30 minutes each (`results_decouple/`); random search with direct jumps and no Adam
(`rsj`, `results_v2/`); estimator / true-gradient alignment versus number of probes, per layer,
at initialisation and on trained checkpoints, in fp32 / TF32 / fp64 (`results_align/`,
`scripts/gradient_alignment.py`).  The job list is generated by `scripts/make_stage2_jobs.py`
(`jobs/stage2.txt`) and run with `python -m zoo.sweep --jobs jobs/stage2.txt --gpus 0,...,7`.

### Stage-2 additions after the interim review (2026-11-03)

* epsilon = 1e-4 (tied) screened for every (task, method) under the stage-2 settings (`results_v2/`).
* Untied grid for every probe method: lr x sigma in {1e-2, 1e-3, 1e-4}^2, 30-minute screening at
  1-minute granularity (`results_decouple/`, `-sig` tags), then 1 h x 3 seeds at the best untied
  setting where it differs from the tied one (`results_v2/`, `-sig` tags).  The paper reports the
  tied protocol and the best untied setting side by side (equal compute, no schedules).
* The 1.5-order rules under their own recipe (`results_15/`): plain SGD (`--optim sgd`), fp64 loss
  evaluation (`--loss64`); 1.5-SPSA with lr = eps tied in {1e-2, 3e-3, 1e-3, 3e-4, 1e-4} and 8 or
  64 antithetic pairs (`--passes 17 / 129`) with the alpha = 0 (1SPSA) control; 1.5-ANP with lr in
  {0.05, 0.02, 0.01, 0.005} decoupled from sigma in {1e-2, 1e-3}, lambda 1e-4 on the LSTM / 1e-3
  elsewhere, alpha 0.5, with the alpha = 0 control, a lambda = 1e-2 floor and the spec's 0.5
  recurrent-state damping (`--damp 0.5`) on the LSTM; 30-minute screening, then 1 h x 3 seeds.
* Controls under the protocol: 1.5-ANP with alpha = 0 and with fp64 losses, 1.5-SPSA with
  alpha = 0.5 / lambda = 1e-3.
* Backprop reference per task: the (lr, weight decay) with the lowest validation loss among
  lr in {1e-2, 1e-3, 1e-4} x AdamW decay in {0, 1e-2}, replicated with 3 seeds.
* Resources: every selected configuration rerun alone on one GPU for 150 s (`results_resources/`)
  for isolated throughput and peak GPU memory; every run records seconds/step and peak memory.
* Reporting additions: best-validation-accuracy checkpoints for the classification tasks,
  bootstrap intervals (over seeds) for the mean ranks, equal-pass curves as the best checkpoint
  within a forward-pass budget.

### Stage-2 additions after the eps 1e-4 screening (2026-11-03, 03:30 UTC)

* The eps = 1e-4 screening selected 1e-4 for 8 of the 10 methods on both image tasks (DANP keeps
  1e-3, FF 1e-2), so the tied grid is extended once more: eps = lr = 1e-5 for those 8 methods on
  MNIST and CIFAR-10 (1 h, seed 0).  The selected eps must have a worse neighbour on each side;
  seeds 1, 2 are run at the selected value (`make_stage2_jobs.py --eps-seeds`).  On the LM tasks
  the step size is decided between 1e-3 and 1e-4 from the same screening (at 1e-5 the LSTM would
  travel < 1 unit per coordinate in an hour).
* Seed replications queued at a superseded eps are pruned from a running sweep through its
  `--skip-jobs` file (re-read before every job) instead of restarting it.
* 1.5-ANP at the selected step size (group X, `results_v2/`, `-sig ... -l64` tags): Adam lr 1e-4,
  sigma in {1e-2, 1e-3} decoupled from lr, fp64 losses, alpha in {0.5, 0, 0.25, 1}, and the LSTM
  with the spec's 0.5 damping.  The untied grid gains a sigma = 0.1 row at lr 1e-4 (group Y).
* The untied-best seed replications (group K) are run only where the best untied cell beats the
  best tied diagonal cell by at least 3 % (both at the 30-minute checkpoint), at most 12 pairs.
* Gradient alignment is measured on a GPU in fp32 (TF32 off) for sigma in {1e-1 ... 1e-5}
  (`results_align_gpu/`); the CPU run (`results_align/`) covers {1e-1, 1e-2, 1e-3}.
