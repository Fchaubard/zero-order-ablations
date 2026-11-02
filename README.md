# A ZOO Deep Learning Survey

Which zero-order optimizer wins across standard small architectures and tasks?

**10 optimizers** (`bp` backprop reference, `spsa`, `spsa15` (1.5-SPSA), `rs` random search,
`np` / `anp` / `inp` / `danp` node-perturbation variants, `anp15` (1.5-ANP), `ff` forward-forward)
**x 4 tasks** (`mlp_mnist`, `conv_cifar10`, `lstm_ptb`, `transformer_ptb`, ~40k parameters each)
**x 3 epsilons** (0.1, 0.01, 0.001; epsilon == learning rate), one hour of training wall-clock each.
Stage 1 screens epsilon (one seed, `results/`); stage 2 replicates the selected configurations
with three seeds and 1-minute evaluations (`results_v2/`) and adds ablations (`results_bp/`,
`results_decouple/`, `results_align/`).  The experimental protocol is the single source of
truth: **[PROTOCOL.md](PROTOCOL.md)**.

```
zoo/perturb.py      perturbable layers + the global forward "mode" every optimizer drives
zoo/models.py       the four models (TASKS, build_model)
zoo/optimizers/     one file per method (curvature.py = spsa15 / anp15); get_optimizer(name, model, lr, task_kind, n_classes, ...)
zoo/data.py         MNIST / CIFAR-10 / PTB-char, resident on the device; python -m zoo.data --download-all
zoo/train.py        one run: timed training loop, eval every N seconds, results JSON + wandb
zoo/sweep.py        tasks x opts x lrs over a GPU pool (resumable);  zoo/status.py  tables + grid
scripts/            setup_remote.sh (fresh CUDA box), launch_sweep.sh (tmux), make_stage2_jobs.py,
                    collect_results.py (stage 1), collect_stage2.py (stage 2), gradient_alignment.py
jobs/               stage2.txt / stage2_extra.txt: the stage-2 job lists (one zoo.train line per run)
results/            {task}-{opt}-lr{eps}[-sig..][-K..][-wd..][-tf32][-s{seed}].json per run      paper/   the write-up
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m zoo.data --root ./data --download-all      # MNIST, CIFAR-10, PTB (cached under data/)
python -m pytest tests -q                            # CPU, no network
```

## Run one experiment

```bash
export WANDB_API_KEY=...                             # optional; without it runs log offline
python -m zoo.train --task mlp_mnist --opt spsa --lr 0.01 --budget 3600 --eval-every 300 \
    --seed 0 --out results --wandb-project zoo-survey --data-root ./data
```

`--budget` counts only seconds spent inside `opt.step` (evaluation and logging are excluded).
The run writes `results/mlp_mnist-spsa-lr0.01.json` after every evaluation (`history`, then
`final`), saves checkpoints at the best validation loss / 5 minutes / the end (`*.pt`), stops
early and sets `diverged` when a loss is non-finite or the validation loss exceeds 2x its step-0
value at two consecutive evaluations, and logs to wandb with `train_time` as the x-axis.
Ablation flags (each recorded in the run name): `--sigma` (probe scale decoupled from the
learning rate), `--passes` (forward-pass budget), `--weight-decay` (AdamW), `--tf32` (default:
full fp32), `--seed`.  Add `--no-wandb` to skip wandb, `--synthetic` for a smoke run on random
data, `--budget 20 --eval-every 5` for a quick look.

## Run the sweep (8 GPUs)

```bash
bash scripts/setup_remote.sh                         # fresh Ubuntu + CUDA image: deps, data, GPU list
bash scripts/launch_sweep.sh                         # detached tmux session "sweep", log in runs/sweep.log
tmux attach -t sweep
# or by hand:
python -m zoo.sweep --gpus 0,1,2,3,4,5,6,7 --tasks all --opts all --lrs 0.1,0.01,0.001 \
    --budget 3600 --eval-every 300 --out results --logdir runs [--dry-run] [--order interleaved]
```

The sweep is resumable: configs whose JSON already has a `final` block and no `error` are
skipped (`--force` reruns them).  Each run's stdout/stderr goes to `runs/{name}.log`.
Stage 2 uses the job-list mode: `python -m zoo.sweep --jobs jobs/stage2.txt --extra-jobs jobs/extra.txt
--gpus 0,...,7 --logdir runs_v2` runs one `zoo.train` argument line per job over the GPU pool;
lines appended to the `--extra-jobs` file while the sweep runs are picked up with priority.

## Inspecting results

```bash
python -m zoo.status --out results                   # every run + best epsilon per (task, opt) by val loss
python -m zoo.status --out results --grid            # 4 x 8 test-loss grid
python -m zoo.status --out results --grid --csv paper/grid.csv --metric test_acc
```

Model selection follows the protocol: for each (task, optimizer) the epsilon with the lowest
final validation loss is selected and that run's test loss (test accuracy for classification,
bits-per-character for language modelling) is reported.  The write-up lives in `paper/`.

## Results (all 96 runs complete)

Final test loss after one hour, epsilon selected by final validation loss (the paper's Table 3), and the
early-stopping oracle (Table 4: best validation-loss evaluation over all epsilon and times).  Second rows
of the paper's tables give accuracy / bits-per-character.  Headline: iterative node perturbation (INP) is
the most consistent zero-order rule (best on both sequence tasks, mean rank 2.0 of 7); ANP / DANP win the
image tasks; the best zero-order rule trails backprop by about one bit per character on the LSTM and
Transformer and by 1-7 accuracy points on MNIST / CIFAR-10 at its best evaluation.

```
final test loss (selected eps):
task                     bp     spsa       rs       np      anp      inp     danp       ff
mlp_mnist             1.857    1.039    1.412    2.141    0.381    0.916    0.751    2.127
conv_cifar10          2.402    2.468    2.406    1.613    1.177    1.339    1.285    1.886
lstm_ptb              1.231    2.048    2.124    2.015    3.232    1.955    2.608    2.688
transformer_ptb       1.262    2.181    2.279    2.357    3.412    2.151    2.525    2.959

early-stopping oracle test loss (best val-loss evaluation over all eps and times):
task                     bp     spsa       rs       np      anp      inp     danp       ff
mlp_mnist             1.260    0.391    0.487    0.167    0.137    0.162    0.321    1.051
conv_cifar10          1.319    1.642    1.742    1.434    1.139    1.341    1.085    1.819
lstm_ptb              1.231    2.048    2.124    2.015    2.969    1.955    2.608    2.658
transformer_ptb       1.260    2.181    2.279    2.342    3.065    2.151    2.513    2.925
```

The paper (`paper/main.pdf`, NeurIPS format, all plots in TikZ) is built with `make` in `paper/` after
`python scripts/collect_results.py`.  Raw per-run JSON results are in `results/` (plus four 10-minute
backprop runs with 30-second evaluations in `results_supp/`), the sweep logs in `logs/`, and every run is
on Weights & Biases (project `zoo-survey`).
