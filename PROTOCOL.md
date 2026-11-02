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
