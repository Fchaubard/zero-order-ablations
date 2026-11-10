#!/usr/bin/env python3
"""1.5-DUST check (Appendix C): gradient alignment of DUST against 1.5-DUST (alpha = 0 and 0.5) at equal passes,
and the LaTeX table of the paper from it and the CPU training pilot.

    python scripts/dust15_alignment.py --out results_dust15            # alignment (CPU, ~10 min)
    python scripts/dust15_alignment.py --out results_dust15 --table    # paper/tables/dust15_table.tex

For each task, at initialisation and after a 2000-step backpropagation warm-up (as scripts/gradient_alignment.py),
on one fixed protocol batch, each estimator is drawn ``--draws`` times at DUST's selected tied eps and its cosine
with the backprop gradient is averaged; ``direct`` restricts both vectors to the parameters of DUST's direct sites
(the only ones whose estimate changes: the readout and q/k/v are excluded), ``all`` is the whole parameter vector.
All three estimators spend 17 passes (DUST: 1 clean + 16 direct draws; 1.5-DUST: 1 clean + 8 antithetic pairs), with
the same local and head draws.  Writes <out>/alignment.csv.  The pilot runs (zoo.train, 10 minutes on the CPU, three
at a time with three threads each) are in <out>/pilot.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoo.data import Data  # noqa: E402
from zoo.models import build_model  # noqa: E402
from zoo.optimizers import get_optimizer, per_sample_loss  # noqa: E402
from zoo.perturb import ctx, players  # noqa: E402

SIG = {"mlp_mnist": 1e-4, "conv_cifar10": 1e-4, "lstm_ptb": 1e-2, "transformer_ptb": 1e-3}   # DUST's selected eps
VARIANTS = [("dust", "dust", dict(n_passes=17), {}),
            ("dust15-a0", "dust15", dict(n_passes=17, loss64=True), {"alpha": 0.0}),
            ("dust15-a0.5", "dust15", dict(n_passes=17, loss64=True), {"alpha": 0.5})]
LABEL = {"mlp_mnist": "MNIST / MLP", "conv_cifar10": "CIFAR-10 / ConvNet", "lstm_ptb": "PTB / char-LSTM",
         "transformer_ptb": "PTB / char-Transformer"}


def _grads(model):
    return {k: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for k, p in model.named_parameters()}


def _cos(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def align(out: str, draws: int, data_root: str):
    os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, "alignment.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["task", "state", "estimator", "sigma", "draws", "cos_all", "se_all", "cos_direct", "se_direct"])
    for task in SIG:
        data = Data(task, torch.device("cpu"), root=data_root)
        torch.manual_seed(0)
        x, y = data.train_batch()
        for state in ("init", "trained"):
            torch.manual_seed(0)
            model = build_model(task, vocab=data.n_classes)
            ctx.set_layers(players(model))
            if state == "trained":
                bp = get_optimizer("bp", model, 1e-3, data.kind, data.n_classes)
                for _ in range(2000):
                    bp.step(*data.train_batch())
            ctx.mode = "bp"
            model.zero_grad()
            per_sample_loss(model(x), y).mean().backward()
            ref = _grads(model)
            model.zero_grad()
            direct = [k for k in ref if not k.startswith("readout") and not any(f".attn.{q}." in k for q in "qkv")]
            r_all = torch.cat([ref[k].flatten() for k in ref])
            r_dir = torch.cat([ref[k].flatten() for k in direct])
            for label, name, kw, attrs in VARIANTS:
                ca, cd = [], []
                t0 = time.time()
                for d in range(draws):
                    opt = get_optimizer(name, model, SIG[task], data.kind, data.n_classes, **kw)
                    for k, v in attrs.items():
                        setattr(opt, k, v)
                    opt.adam.step = lambda: None                    # estimate only
                    torch.manual_seed(1000 + d)
                    opt.step(x, y)
                    g = _grads(model)
                    model.zero_grad()
                    ca.append(_cos(torch.cat([g[k].flatten() for k in ref]), r_all))
                    cd.append(_cos(torch.cat([g[k].flatten() for k in direct]), r_dir))
                se = lambda v: st.stdev(v) / len(v) ** 0.5
                w.writerow([task, state, label, SIG[task], draws, f"{st.mean(ca):.4f}", f"{se(ca):.4f}",
                            f"{st.mean(cd):.4f}", f"{se(cd):.4f}"])
                f.flush()
                print(f"{task:16s} {state:7s} {label:12s} all {st.mean(ca):.4f} direct {st.mean(cd):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    f.close()


def table(out: str, path: str):
    rows = {}
    with open(os.path.join(out, "alignment.csv")) as f:
        for r in csv.DictReader(f):
            rows[(r["task"], r["state"], r["estimator"])] = r
    est = [v[0] for v in VARIANTS]
    L = ["% generated by scripts/dust15_alignment.py -- cosine with the backprop gradient, mean (s.e.) over draws",
         "\\begin{tabular}{l l l ccc ccc}", "\\toprule",
         " & & & \\multicolumn{3}{c}{direct sites} & \\multicolumn{3}{c}{whole model} \\\\",
         "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
         "Task & $\\varepsilon$ & state & \\DUST{} & $\\alpha=0$ & $\\alpha=0.5$ & \\DUST{} & $\\alpha=0$ & $\\alpha=0.5$ \\\\",
         "\\midrule"]
    for task in SIG:
        for state in ("init", "trained"):
            cells = []
            for col in ("direct", "all"):
                vals = [float(rows[(task, state, e)][f"cos_{col}"]) for e in est]
                ses = [float(rows[(task, state, e)][f"se_{col}"]) for e in est]
                best = max(vals)
                for v, s in zip(vals, ses):
                    txt = f"{v:.3f}({round(s * 1000):d})"
                    cells.append(f"\\textbf{{{txt}}}" if v == best else txt)
            sig = f"$10^{{{int(round(__import__('math').log10(SIG[task])))}}}$"
            name = LABEL[task] if state == "init" else ""
            L.append(f"{name} & {sig if state == 'init' else ''} & {'start' if state == 'init' else 'warm'} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
    pil = {}
    for p in glob.glob(os.path.join(out, "pilot", "*.json")):
        r = json.load(open(p))
        c, h = r["config"], r["history"]
        key = "dust" if c["opt"] == "dust" else f"dust15-a{c['alpha']:g}"
        best = min((e for e in h if "val_loss" in e), key=lambda e: e["val_loss"])
        pil[key] = (best["test_loss"], h[-1]["step"])
    if pil:
        cells = " & ".join(f"{pil[e][0]:.3f}" for e in est)
        steps = " & ".join(f"{pil[e][1]:,}".replace(",", "{,}") for e in est)
        L.append(f"\\multicolumn{{3}}{{l}}{{Transformer, 10-min CPU run: test loss}} & {cells} & & & \\\\")
        L.append(f"\\multicolumn{{3}}{{l}}{{\\quad steps}} & {steps} & & & \\\\")
    L += ["\\bottomrule", "\\end{tabular}", ""]
    open(path, "w").write("\n".join(L))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_dust15")
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--table", action="store_true", help="only write paper/tables/dust15_table.tex from <out>")
    a = ap.parse_args()
    if a.table:
        table(a.out, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "tables", "dust15_table.tex"))
        return
    torch.set_num_threads(a.threads)
    align(a.out, a.draws, a.data_root)


if __name__ == "__main__":
    main()
