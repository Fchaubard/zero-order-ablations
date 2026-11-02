#!/usr/bin/env python3
"""Estimator-quality diagnostic: cosine similarity between each zero-order gradient estimate
and the true (backprop) gradient as a function of the number of forward passes, per layer,
for every model, at initialisation and after a short backprop warm-up, at every probe scale
of the protocol, and (for the precision check) in fp32-with-TF32-off, fp32-with-TF32-on and fp64.

    python scripts/gradient_alignment.py --out results_align --tasks all [--precision-only]

Writes results_align/alignment.csv with one row per
    task, state, precision, estimator, sigma, passes, layer, cosine, norm_ratio, mean_abs_dloss
where ``layer`` is a PLayer index ("all" = whole flat gradient) and ``norm_ratio`` is
||ghat|| / ||g||.  Estimators use the optimizer classes of zoo/optimizers (same code as the
sweep); the number of forward passes P means: SPSA P/2 antithetic pairs, RS P/16 rounds of
(1 clean + 15 candidates) averaged, NP / ANP / DANP P noisy passes, INP max(1, P // L) draws
per layer (DANP = ANP after 500 decorrelation updates of R at eta_R = 1e-3).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoo.data import Data  # noqa: E402
from zoo.models import build_model  # noqa: E402
from zoo.optimizers import get_optimizer, per_sample_loss  # noqa: E402
from zoo.perturb import ctx, layer_params, players  # noqa: E402

TASKS = ["mlp_mnist", "conv_cifar10", "lstm_ptb", "transformer_ptb"]
ESTIMATORS = ["spsa", "rs", "np", "anp", "inp", "danp"]
SIGMAS = [0.1, 0.01, 0.001]
PASSES = [16, 64, 256, 1024, 4096]
PRECISIONS = ["fp32", "tf32", "fp64"]


def set_precision(prec: str):
    tf32 = prec == "tf32"
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])


def true_grad(model, x, y):
    ctx.mode = "bp"
    for p in model.parameters():
        p.grad = None
    per_sample_loss(model(x), y).mean().backward()
    g = [p.grad.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = None
    return g


def cosine(a, b):
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan"), float("nan")
    return float((a @ b) / (na * nb)), float(na / nb)


def warm_up(model, data, steps, lr=1e-3, seed=0):
    """Backprop warm-up to an early-training operating point (the sweep saved no checkpoints)."""
    torch.manual_seed(seed)
    opt = get_optimizer("bp", model, lr, data.kind, data.n_classes)
    for _ in range(steps):
        x, y = data.train_batch()
        opt.step(x, y)
    for p in model.parameters():
        p.grad = None


def rs_pseudo_grad(opt, x, y, rounds):
    """Average over ``rounds`` RS steps of the pseudo-gradient fed to Adam (0 when no candidate improved)."""
    g = None
    for _ in range(rounds):
        _, ghat, _, _ = opt._search(x, y)
        g = ghat.clone() if g is None else g + ghat
    return g / rounds


def estimate(name, model, data, x, y, sigma, passes, danp_steps=500):
    """Return the flat estimate as a list of per-parameter tensors (same layout as model.parameters())."""
    opt = get_optimizer(name, model, sigma, data.kind, data.n_classes)   # lr == sigma, Adam unused here
    params = list(model.parameters())
    if name == "danp":
        # train R with the local decorrelation rule on clean passes (R does not depend on the weights)
        ctx.set_layers(opt.layers)
        for _ in range(danp_steps):
            xb, _ = data.train_batch()
            ctx.mode, ctx.noise_layer = "clean", None
            ctx.reset()
            with torch.no_grad():
                model(xb)
            for layer in opt.layers:
                layer.decorrelation_update(opt.lr_decor)
        ctx.reset()
    if name == "spsa":
        g = opt.estimate_grad(x, y, n_pairs=max(1, passes // 2))
        return [g[a:b].view_as(p) for p, (a, b) in zip(opt.flat.params, opt.flat.slices)], None
    if name == "rs":
        g = rs_pseudo_grad(opt, x, y, rounds=max(1, passes // 16))
        return [g[a:b].view_as(p) for p, (a, b) in zip(opt.flat.params, opt.flat.slices)], None
    # node-perturbation family: estimate_grad writes into p.grad
    n = max(1, passes // len(opt.layers)) if name == "inp" else passes
    for p in params:
        p.grad = None
    opt.estimate_grad(x, y, n_passes=n)
    g = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in params]
    for p in params:
        p.grad = None
    dloss = float(opt._last_abs_dloss) if opt._last_abs_dloss is not None else None
    return g, dloss


def rows_for(task, state, prec, model, data, x, y, estimators, sigmas, passes_list, writer, f):
    params = list(model.parameters())
    layers = players(model)
    layer_of = {}
    for li, layer in enumerate(layers):
        for p in layer_params(layer):
            layer_of[id(p)] = li
    gt = true_grad(model, x, y)
    gt_flat = torch.cat([g.reshape(-1) for g in gt])
    for est in estimators:
        for sigma in sigmas:
            for passes in passes_list:
                torch.manual_seed(1234)            # identical probes across estimators / precisions
                if est == "danp":
                    # R must be trained on a fresh copy so the other estimators keep R = I
                    import copy
                    ctx.set_layers(layers)
                    ctx.reset()                    # drop recorded activities (they hold graphs) before copying
                    m2 = copy.deepcopy(model)
                    ctx.set_layers(players(m2))
                    gh, dl = estimate(est, m2, data, x, y, sigma, passes)
                    # DANP estimates the gradient of the *decorrelated* model (a = W R h), so its
                    # reference is that model's backprop gradient, not the undecorrelated one.
                    gt_ref = true_grad(m2, x, y)
                    ctx.set_layers(layers)
                else:
                    gh, dl = estimate(est, model, data, x, y, sigma, passes)
                    gt_ref = gt
                gt_ref_flat = torch.cat([g.reshape(-1) for g in gt_ref])
                gh_flat = torch.cat([g.reshape(-1) for g in gh])
                c, r = cosine(gh_flat, gt_ref_flat)
                if gh_flat.norm() == 0:            # e.g. random search found no improving candidate
                    c, r = 0.0, 0.0
                c_all = c
                writer.writerow([task, state, prec, est, sigma, passes, "all", f"{c:.4f}", f"{r:.4g}", "" if dl is None else f"{dl:.3e}"])
                # per layer (weights + bias of that PLayer)
                for li in range(len(layers)):
                    idx = [i for i, p in enumerate(params) if layer_of.get(id(p)) == li]
                    if not idx:
                        continue
                    a = torch.cat([gh[i].reshape(-1) for i in idx])
                    b = torch.cat([gt_ref[i].reshape(-1) for i in idx])
                    c, r = cosine(a, b)
                    if a.norm() == 0:
                        c, r = 0.0, 0.0
                    writer.writerow([task, state, prec, est, sigma, passes, li, f"{c:.4f}", f"{r:.4g}", ""])
                f.flush()
                print(f"  {task:16s} {state:7s} {prec:4s} {est:4s} sigma={sigma:<6g} P={passes:<5d} cos={c_all:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_align")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--precision-only", action="store_true", help="only the fp32 / tf32 / fp64 check (trained state, P=1024)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--passes", default=None, help="comma list overriding the pass counts (tests)")
    ap.add_argument("--sigmas", default=None, help="comma list overriding the probe scales (tests)")
    ap.add_argument("--estimators", default=None, help="comma list overriding the estimators (tests)")
    a = ap.parse_args()
    passes_list = [int(v) for v in a.passes.split(",")] if a.passes else PASSES
    sigmas = [float(v) for v in a.sigmas.split(",")] if a.sigmas else SIGMAS
    estimators = a.estimators.split(",") if a.estimators else ESTIMATORS
    tasks = TASKS if a.tasks == "all" else a.tasks.split(",")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "precision.csv" if a.precision_only else "alignment.csv")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["task", "state", "precision", "estimator", "sigma", "passes", "layer", "cosine", "norm_ratio", "mean_abs_dloss"])
    t0 = time.time()
    for task in tasks:
        data = Data(task, device, root=a.data_root)
        torch.manual_seed(0)
        x, y = data.train_batch()                      # one protocol-sized batch, fixed for every estimator
        for state in (["trained"] if a.precision_only else ["init", "trained"]):
            set_precision("fp32")
            torch.manual_seed(0)
            model = build_model(task, vocab=data.n_classes).to(device)
            ctx.set_layers(players(model))
            if state == "trained":
                warm_up(model, data, a.warmup_steps)
            with torch.no_grad():
                ctx.mode = "bp"
                l = per_sample_loss(model(x), y).mean().item()
            print(f"[{task}] {state}: batch loss {l:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if a.precision_only:
                for prec in PRECISIONS:
                    set_precision(prec)
                    m = model.double() if prec == "fp64" else model.float()
                    xx = x.double() if (prec == "fp64" and x.is_floating_point()) else x
                    ctx.set_layers(players(m))
                    rows_for(task, state, prec, m, data if prec != "fp64" else _DoubleData(data), xx, y,
                             [e for e in estimators if e in ("np", "anp", "inp", "spsa")] or ["np"],
                             [s_ for s_ in sigmas if s_ <= 0.01] or [0.001], passes_list if a.passes else [1024], w, f)
                    model = m.float()
                set_precision("fp32")
            else:
                rows_for(task, state, "fp32", model, data, x, y, estimators, sigmas, passes_list, w, f)
    f.close()
    print(f"wrote {path} in {time.time() - t0:.0f}s")


class _DoubleData:
    """Wrap a Data object so train batches come out as float64 (for the fp64 precision check)."""

    def __init__(self, data):
        self.d = data
        self.kind, self.n_classes = data.kind, data.n_classes

    def train_batch(self):
        x, y = self.d.train_batch()
        return (x.double() if x.is_floating_point() else x), y


if __name__ == "__main__":
    main()
