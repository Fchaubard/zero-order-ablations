#!/usr/bin/env python3
"""Generate the stage-2 job list (see PROTOCOL.md, "Stage 2") from the stage-1 summary.

    python scripts/make_stage2_jobs.py --summary results/summary.json --out jobs/stage2.txt

Stage 1 (results/, 96 runs, seed 0, 5-minute evaluations) screens epsilon.  Stage 2 replicates
the selected configurations with 3 seeds and 1-minute evaluations and adds the ablations the
reviews asked for.  Every line is a ``zoo.train`` argument string consumed by
``python -m zoo.sweep --jobs``.  Groups (in queue order, most important first):

  A. replication   selected (task, method, eps*) x seeds {0,1,2}, 1 h, eval every 60 s -> results_v2/
                   eps* = eps with the lowest validation loss over all stage-1 evaluations (best
                   checkpoint); where the final-loss selection differs and is not degenerate
                   (DANP on both image tasks) both eps are replicated; BP uses eps = 0.001 on every
                   task (and 0.01 on CIFAR-10 as well); FF on the image tasks is replicated at all
                   eps in {0.01, 0.001} because its selection uses goodness-inference validation accuracy.
                   The LM tasks keep their stage-1 seed-0 run (monotone curves; the 5-minute grid
                   loses nothing) and add seeds 1, 2.
  B. bp reference  BP x {lr 1e-4 Adam, lr 1e-3 AdamW wd 1e-2, lr 1e-4 AdamW wd 1e-2} x 4 tasks, 1 h -> results_bp/
  C. decoupling    lr = 1e-3 with sigma in {1e-2, 1e-4} for the 6 probe methods x 4 tasks, and
                   lr = 1e-2 with sigma = 1e-3 for SPSA / NP / ANP / INP x 4 tasks; 30 min -> results_decouple/
  D. rs jump       direct-jump random search (rsj) at the selected RS eps x 4 tasks, 1 h -> results_v2/
  E. 1.5-order     spsa15 / anp15 (17 passes: 1 clean + 8 antithetic pairs), added after stage 1:
                   eps screening {0.1, 0.01, 0.001} x 4 tasks under the stage-2 protocol (--extra, injected
                   into the running sweep), then seeds 1, 2 at the selected eps (--extra-seeds, once the
                   screening is in results_v2/)
"""
from __future__ import annotations

import argparse
import json
import os

TASKS = ["mlp_mnist", "conv_cifar10", "lstm_ptb", "transformer_ptb"]
IMAGE = ["mlp_mnist", "conv_cifar10"]
OPTS = ["bp", "spsa", "rs", "np", "anp", "inp", "danp", "ff"]
PROBE = ["spsa", "rs", "np", "anp", "inp", "danp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--out", default="jobs/stage2.txt")
    ap.add_argument("--extra", default=None, help="write group E (1.5-order eps screening) to this file instead")
    ap.add_argument("--extra-seeds", default=None, help="write group E seeds 1, 2 at the eps selected from --v2")
    ap.add_argument("--v2", default="results_v2")
    ap.add_argument("--node2", default=None,
                    help="write the review-driven extra sweeps (second node) to this file: eps 1e-4 for every pair, "
                         "alpha = 0 / fp64 / non-inert controls, and the 1.5-order rules under their own SGD recipe "
                         "(grids recommended by the code audit)")
    ap.add_argument("--node3", default=None,
                    help="write the untied (lr, sigma) grid for every probe method (third node): the combinations of "
                         "lr, sigma in {1e-2, 1e-3, 1e-4}^2 not covered by the tied runs or the node-1 decoupling group")
    ap.add_argument("--resources", default=None,
                    help="write a job list that re-runs every selected (task, method, eps) of results_v2/summary_v2.json ALONE on one GPU "
                         "for 150 s (no wandb) into results_resources/, to measure isolated throughput and peak GPU memory")
    a = ap.parse_args()
    s = json.load(open(a.summary))
    lines = []
    if a.resources:
        v2 = json.load(open(os.path.join(a.v2, "summary_v2.json")))
        lines.append("# R. isolated throughput / peak memory: each selected configuration alone on a GPU for 150 s (run the sweep with --gpus 0)")
        for k, v in v2["selected"].items():
            t, o = k.split("/")
            lines.append(f"--task {t} --opt {o} --lr {float(v['eps']):g} --seed 0 --budget 150 --eval-every 600 --no-wandb --no-save-ckpt --out results_resources")
        os.makedirs(os.path.dirname(a.resources) or ".", exist_ok=True)
        with open(a.resources, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{len(lines) - 1} jobs -> {a.resources}")
        return
    if a.node3:
        H = "--seed 0 --budget 1800 --eval-every 60 --out results_decouple"
        probe = ["spsa", "spsa15", "rs", "np", "anp", "anp15", "inp", "danp"]
        lines.append("# J. untied (lr, sigma) grid, 30-minute screening at 1-minute granularity; the tied diagonal is in results_v2 (1 h) and "
                     "lr 1e-3 x sigma {1e-2, 1e-4} and lr 1e-2 x sigma 1e-3 (spsa, np, anp, inp) ran on node 1")
        for t in TASKS:
            for o in probe:
                for sig in ("0.01", "0.001"):
                    lines.append(f"--task {t} --opt {o} --lr 0.0001 --sigma {sig} {H}")
                lines.append(f"--task {t} --opt {o} --lr 0.01 --sigma 0.0001 {H}")
                if o in ("rs", "danp", "spsa15", "anp15"):
                    lines.append(f"--task {t} --opt {o} --lr 0.01 --sigma 0.001 {H}")
        os.makedirs(os.path.dirname(a.node3) or ".", exist_ok=True)
        with open(a.node3, "w") as f:
            f.write("\n".join(lines) + "\n")
        n = sum(1 for l in lines if not l.startswith("#"))
        print(f"{n} jobs, {n * 0.5:.0f} GPU-hours -> {a.node3}")
        return
    if a.node2:
        A = "--seed 0 --budget 3600 --eval-every 60"
        H = "--seed 0 --budget 1800 --eval-every 60"
        lines.append("# F. tied eps = 1e-4 for every (task, method): the stage-1 grid's lower boundary was selected by 26 of 32 pairs")
        for t in TASKS:
            for o in ["bp", "spsa", "spsa15", "rs", "np", "anp", "anp15", "inp", "danp", "ff"]:
                lines.append(f"--task {t} --opt {o} --lr 0.0001 {A} --out results_v2")
        lines.append("# G. controls for the 1.5-order rules under the protocol (Adam, eps = lr = 1e-3; LSTM anp15 at 1e-2)")
        for t in TASKS:
            e = "0.01" if t == "lstm_ptb" else "0.001"
            lines.append(f"--task {t} --opt anp15 --lr {e} --alpha 0 {A} --out results_v2")             # token-local central-difference ANP
            lines.append(f"--task {t} --opt anp15 --lr {e} --loss64 {A} --out results_v2")             # fp64 second differences
            lines.append(f"--task {t} --opt spsa15 --lr 0.001 --alpha 0.5 --lam 0.001 {A} --out results_v2")  # non-inert curvature factor
        lines.append("# H. 1.5-SPSA under its own recipe (plain SGD, lambda = eps tied, alpha 0.1, lambda_reg 1, fp64 losses; 8 or 64 pairs) and the alpha = 0 (1SPSA-SGD) control; 30 min screening")
        for t in TASKS:
            for e in ("0.01", "0.003", "0.001", "0.0003", "0.0001"):
                for K in (17, 129):
                    lines.append(f"--task {t} --opt spsa15 --lr {e} --optim sgd --passes {K} --loss64 {H} --out results_15")
                    lines.append(f"--task {t} --opt spsa15 --lr {e} --optim sgd --passes {K} --loss64 --alpha 0 {H} --out results_15")
        lines.append("# I. 1.5-ANP under its own recipe (plain SGD, lr decoupled from sigma, alpha 0.5, lambda 1e-3 / 1e-4 on the LSTM, 8 pairs, fp64 losses), the alpha = 0 control and a lambda = 1e-2 floor; 30 min screening")
        for t in TASKS:
            lam = "0.0001" if t == "lstm_ptb" else "0.001"
            for lr in ("0.05", "0.02", "0.01", "0.005"):
                for sig in ("0.01", "0.001"):
                    lines.append(f"--task {t} --opt anp15 --lr {lr} --sigma {sig} --optim sgd --lam {lam} --loss64 {H} --out results_15")
                    lines.append(f"--task {t} --opt anp15 --lr {lr} --sigma {sig} --optim sgd --lam {lam} --loss64 --alpha 0 {H} --out results_15")
                lines.append(f"--task {t} --opt anp15 --lr {lr} --sigma 0.001 --optim sgd --lam 0.01 --loss64 {H} --out results_15")
        os.makedirs(os.path.dirname(a.node2) or ".", exist_ok=True)
        with open(a.node2, "w") as f:
            f.write("\n".join(lines) + "\n")
        n = sum(1 for l in lines if not l.startswith("#"))
        h = sum((3600 if "--budget 3600" in l else 1800) / 3600 for l in lines if not l.startswith("#"))
        print(f"{n} jobs, {h:.0f} GPU-hours -> {a.node2}")
        return
    if a.extra or a.extra_seeds:
        new = ["spsa15", "anp15"]
        if a.extra:
            lines.append("# E. 1.5-order rules: eps screening under the stage-2 protocol (1 h, 1-minute evaluations)")
            for e in (0.1, 0.01, 0.001):
                for t in TASKS:
                    for o in new:
                        lines.append(f"--task {t} --opt {o} --lr {e:g} --seed 0 --budget 3600 --eval-every 60 --out results_v2")
            path = a.extra
        else:
            import glob
            import math
            lines.append("# E. 1.5-order rules: seeds 1, 2 at the eps with the lowest best-validation loss of the screening")
            for t in TASKS:
                for o in new:
                    best = None
                    for e in (0.1, 0.01, 0.001):
                        f = os.path.join(a.v2, f"{t}-{o}-lr{e:g}.json")
                        if not os.path.exists(f):
                            continue
                        r = json.load(open(f))
                        if "final" not in r:
                            continue
                        v = min((h["val_loss"] for h in r["history"] if h.get("val_loss") is not None and not math.isnan(h["val_loss"])), default=None)
                        if v is not None and (best is None or v < best[0]):
                            best = (v, e)
                    if best is None:
                        print(f"  {t}/{o}: screening incomplete, skipped")
                        continue
                    for seed in (1, 2):
                        lines.append(f"--task {t} --opt {o} --lr {best[1]:g} --seed {seed} --budget 3600 --eval-every 60 --out results_v2")
            path = a.extra_seeds
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{sum(1 for l in lines if not l.startswith('#'))} jobs -> {path}")
        return

    def job(task, opt, lr, seed=0, out="results_v2", budget=3600, extra=""):
        lines.append(f"--task {task} --opt {opt} --lr {lr:g} --seed {seed} --budget {budget} --eval-every 60 --out {out}{(' ' + extra) if extra else ''}")

    # ---- A. replication -----------------------------------------------------------------
    sel = {}
    for t in TASKS:
        for o in OPTS:
            eps = {float(s["early_stop"][f"{t}/{o}"]["eps"])}
            fin = float(s["main"][f"{t}/{o}"]["eps"])
            if o == "bp":
                eps = {0.001} | ({0.01} if t == "conv_cifar10" else set())
            elif o == "ff" and t in IMAGE:
                eps = {0.01, 0.001}          # eps = 0.1 was clearly worst in stage 1 (goodness 92.2 %)
            elif o == "danp" and t in IMAGE:
                eps = eps | {fin}
            sel[(t, o)] = sorted(eps, reverse=True)
    lines.append("# A. replication: selected eps x seeds, 1 h, 1-minute evaluations")
    for seed in (0, 1, 2):
        for t in TASKS:
            for o in OPTS:
                for e in sel[(t, o)]:
                    if seed == 0 and t not in IMAGE and e == float(s["early_stop"][f"{t}/{o}"]["eps"]) and o != "bp" and not (o == "ff"):
                        continue          # LM tasks: stage-1 seed-0 run is kept (results/)
                    if seed == 0 and t not in IMAGE and o == "bp" and e == 0.001:
                        continue
                    if seed == 0 and t not in IMAGE and o == "ff":
                        continue
                    job(t, o, e, seed)
    # ---- B. backprop reference variants ---------------------------------------------------
    lines.append("# B. backprop reference: lower lr and decoupled weight decay, 1 h")
    for t in TASKS:
        job(t, "bp", 1e-4, out="results_bp")
        job(t, "bp", 1e-3, out="results_bp", extra="--weight-decay 0.01")
        job(t, "bp", 1e-4, out="results_bp", extra="--weight-decay 0.01")
    # ---- D. direct-jump random search ------------------------------------------------------
    lines.append("# D. random search with direct jumps (no Adam) at the selected RS eps, 1 h")
    for t in TASKS:
        job(t, "rsj", sel[(t, "rs")][0])

    # ---- C. decoupling probe scale from step size ------------------------------------------
    lines.append("# C. decoupling: lr 1e-3 with sigma {1e-2, 1e-4}; lr 1e-2 with sigma 1e-3; 30 min")
    for t in TASKS:
        for o in PROBE:
            for sig in (0.01, 0.0001):
                job(t, o, 1e-3, out="results_decouple", budget=1800, extra=f"--sigma {sig:g}")
    for t in TASKS:
        for o in ("spsa", "np", "anp", "inp"):
            job(t, o, 1e-2, out="results_decouple", budget=1800, extra="--sigma 0.001")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    n = sum(1 for l in lines if not l.startswith("#"))
    hours = sum((3600 if "--budget 3600" in l else 1800) / 3600 for l in lines if not l.startswith("#"))
    print(f"{n} jobs, {hours:.0f} GPU-hours -> {a.out}")
    for (t, o), e in sel.items():
        print(f"  {t:16s} {o:5s} eps {e}")


if __name__ == "__main__":
    main()
