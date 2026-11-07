#!/usr/bin/env python3
"""Stage-2 analysis: seeds, best-checkpoint selection, ablations -> paper tables / figure data.

    python scripts/collect_stage2.py [--stage1 results] [--v2 results_v2] [--bp results_bp]
                                     [--decouple results_decouple] [--align results_align] [--paper paper]

Inputs
  results/            stage 1: 96 runs, seed 0, 5-minute evaluations (epsilon screening)
  results_v2/         stage 2: selected configurations x seeds 0-2, 1-minute evaluations (+ rsj)
                      (for the LM tasks the seed-0 run of the selected epsilon is the stage-1 run)
  results_bp/         backprop reference variants (lr 1e-4, AdamW)
  results_decouple/   probe scale decoupled from the step size (30-minute runs)
  results_align/      gradient-alignment CSVs from scripts/gradient_alignment.py

Outputs (paper/tables, paper/data, results_v2/summary_v2.json)
  main_table.tex        best-validation checkpoint: test loss, acc % / bits per char, mean +- s.d. over seeds,
                        eps* and the mean minute of the best checkpoint; FF on images selected and reported
                        by goodness-inference accuracy (readout accuracy in parentheses)
  final_table.tex       the same runs at the end of the hour (secondary, stability)
  bp_table.tex          backprop reference variants (best checkpoint / final)
  decouple_table.tex    (lr, sigma) grid per task and method, best checkpoint within 30 minutes
  rsj_table.tex         random search: Adam pseudo-gradient vs direct jump
  throughput_table.tex  steps/h and forward passes/h (mean over seeds)
  curve2_{task}.csv     test loss vs minutes (mean, min, max over seeds) of the selected runs
  passes_{task}.csv     test loss vs forward passes (interpolated; equal-compute view)
  align_{task}_{state}.csv   cosine vs passes per estimator at sigma = 1e-3 (whole model) + per-layer at P=1024
  precision_table.tex   fp32 / TF32 / fp64 cosines on the ConvNet
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import math
import os
import statistics as st
from collections import defaultdict

TASKS = ["mlp_mnist", "conv_cifar10", "lstm_ptb", "transformer_ptb"]
IMAGE = ["mlp_mnist", "conv_cifar10"]
OPTS = ["bp", "spsa", "spsa15", "rs", "np", "anp", "anp15", "inp", "inp15", "danp", "ff"]   # table row order
NO_SIZE = {"inp15"}   # rules without model-size or untied-grid runs (kept out of those tables and figures)
PROBE = ["spsa", "spsa15", "rs", "np", "anp", "anp15", "inp", "inp15", "danp"]
OPT_LABEL = {"bp": "\\BP{}", "spsa": "\\SPSA{}", "spsa15": "\\SPSAb{}", "rs": "\\RS{}", "np": "\\NP{}", "anp": "\\ANP{}",
             "anp15": "\\ANPb{}", "inp": "\\INP{}", "danp": "\\DANP{}", "ff": "\\FF{}", "inp15": "1.5-INP", "rsj": "RS-jump"}
TASK_SHORT = {"mlp_mnist": "MNIST", "conv_cifar10": "CIFAR-10", "lstm_ptb": "LSTM", "transformer_ptb": "Transformer"}
TASK_LABEL = {"mlp_mnist": "MNIST / MLP", "conv_cifar10": "CIFAR-10 / ConvNet",
              "lstm_ptb": "PTB / char-LSTM", "transformer_ptb": "PTB / char-Transformer"}
LN2 = math.log(2)
SEEDS = [0, 1, 2]


# ----------------------------------------------------------------------------- loading
def load_dir(d, partial=False):
    """{(task, opt, eps_str, seed, tag): result}; runs still in progress (status 'running' / 'starting')
    or that crashed ('error') are left out unless ``partial`` is set."""
    runs = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        if os.path.basename(f).startswith("summary"):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        c = r.get("config", {})
        if not r.get("history") or c.get("task") is None:
            continue
        if not partial and r.get("status") in ("running", "starting", "error"):
            continue
        key = (c["task"], c["opt"], f"{float(c['lr']):g}", int(c.get("seed", 0)), c.get("tag", "") or "")
        runs[key] = r
    return runs


def final(r):
    return r.get("final") or r["history"][-1]


def best_eval(r, key="val_loss", maximize=False, t_max=None):
    """The evaluation with the lowest ``key`` (or highest when maximize) within t_max seconds."""
    best = None
    for h in r["history"]:
        v = h.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if t_max is not None and h["train_time"] > t_max + 45:
            continue
        if h.get("test_loss") is None or math.isnan(h["test_loss"]):
            continue
        if best is None or (v > best[0] if maximize else v < best[0]):
            best = (v, h)
    return best[1] if best else None


def ms(vals, nd=3):
    """mean +- sample s.d. over seeds, e.g. 0.140$\\pm$0.005; a single value when n == 1;
    scientific notation for the huge losses of collapsed runs."""
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "--"
    if len(vals) == 1:
        return fmt_loss(vals[0], nd)
    m, s = st.mean(vals), st.stdev(vals)
    if abs(m) >= 1000:
        return fmt_loss(m, nd)
    return f"{m:.{nd}f}$\\pm${s:.{nd}f}"


def fmt_loss(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if abs(v) >= 1000:
        e = int(math.floor(math.log10(abs(v))))
        return f"${v / 10 ** e:.1f}{{\\times}}10^{{{e}}}$"
    return f"{v:.{nd}f}"


def mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return st.mean(vals) if vals else float("nan")


def sd(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return st.stdev(vals) if len(vals) > 1 else 0.0


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join("nan" if (isinstance(v, float) and math.isnan(v)) else (f"{v:.6g}" if isinstance(v, float) else str(v)) for v in row) + "\n")


def best_within(hist, xkey, x, ykey="test_loss", by="val_loss"):
    """Test loss of the best-validation evaluation among those with xkey <= x (nan if none)."""
    cands = [h for h in hist if h.get(xkey) is not None and h[xkey] <= x and h.get(by) is not None and not math.isnan(h[by])
             and h.get(ykey) is not None and not math.isnan(h[ykey])]
    if not cands:
        return float("nan")
    return min(cands, key=lambda h: h[by])[ykey]


def interp_at(hist, xkey, x, ykey="test_loss"):
    """Linear interpolation of ykey at xkey == x over the history (nan outside the range)."""
    pts = [(h[xkey], h[ykey]) for h in hist if h.get(ykey) is not None and not math.isnan(h[ykey])]
    pts.sort()
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return float("nan")
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="results")
    ap.add_argument("--v2", default="results_v2")
    ap.add_argument("--bp", default="results_bp")
    ap.add_argument("--decouple", default="results_decouple")
    ap.add_argument("--align", default="results_align")
    ap.add_argument("--own", default="results_15", help="1.5-order rules under their own recipe (SGD)")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--partial", action="store_true", help="also use runs that are still in progress (interim tables)")
    a = ap.parse_args()
    s1, v2, bp, dc, own = (load_dir(d, a.partial) for d in (a.stage1, a.v2, a.bp, a.decouple, a.own))
    # 1.5-INP enters the main comparison as its alpha = 0, fp64-loss runs (tag -a0-l64), selected and replicated like every rule
    for (t_, o_, e_, s_, tg), r in list(v2.items()):
        if o_ == "inp15" and tg == "-a0-l64":
            v2[(t_, o_, e_, s_, "")] = r
    data_dir, tab_dir = os.path.join(a.paper, "data"), os.path.join(a.paper, "tables")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    summary = {"selected": {}, "main": {}, "final": {}, "bp": {}, "decouple": {}, "rsj": {}, "throughput": {}, "n_runs": {}}

    # ---- the replication set: every (task, opt, eps) present in results_v2 (any seed) ----
    def seeds_of(task, opt, eps):
        """{seed: result} for a replicated configuration; seed 0 falls back to the stage-1 run
        (by design for the LM tasks; for the image tasks only until their stage-2 run exists)."""
        out = {}
        for seed in SEEDS:
            r = v2.get((task, opt, eps, seed, ""))
            if r is None and seed == 0:
                r = s1.get((task, opt, eps, 0, ""))
            if r is not None:
                out[seed] = r
        return out

    rep = defaultdict(set)
    for (task, opt, eps, seed, tag) in v2:
        if tag == "" and opt in OPTS:
            rep[(task, opt)].add(eps)
    # Stage-1 fallback: a pair without any stage-2 run yet (and, by design, the seed-0 run of the
    # LM tasks) uses the stage-1 run at the epsilon with the lowest validation loss over all
    # stage-1 evaluations.  Cells whose seeds are all stage-1 runs are marked in the tables.
    s1_sel = {}
    try:
        s1_summary = json.load(open(os.path.join(a.stage1, "summary.json")))
        for k, v in s1_summary.get("early_stop", {}).items():
            t_, o_ = k.split("/")
            s1_sel[(t_, o_)] = v["eps"]
    except Exception:
        pass
    for (task, opt, eps, seed, tag) in s1:
        if opt in OPTS and tag == "" and seed == 0:
            rep[(task, opt)].add(eps)      # every stage-1 value stays a candidate (seed 0 falls back to its run; the
                                           # stage-1 selection cannot lose to another stage-1 value under the same criterion)
    stage1_only = set()

    # ---- selection: lowest mean (over seeds) best validation loss; FF images: highest goodness val acc ----
    sel, chosen = {}, {}
    for task in TASKS:
        for opt in OPTS:
            cands = []
            for eps in sorted(rep.get((task, opt), []), key=float, reverse=True):
                runs = seeds_of(task, opt, eps)
                if not runs:
                    continue
                # the screening criterion is the seed-0 run (PROTOCOL.md: select, then replicate); the seed mean is
                # used only where seed 0 is missing
                if opt == "ff" and task in IMAGE:
                    evs = {sd_: best_eval(r, "val_acc_goodness", maximize=True) for sd_, r in runs.items()}
                    vals = {sd_: -e["val_acc_goodness"] for sd_, e in evs.items() if e}
                else:
                    evs = {sd_: best_eval(r, "val_loss") for sd_, r in runs.items()}
                    vals = {sd_: e["val_loss"] for sd_, e in evs.items() if e}
                if not vals:
                    continue
                crit = vals[0] if 0 in vals else mean(list(vals.values()))
                cands.append((crit, eps, runs, evs))
            if cands:
                crit, eps, runs, evs = min(cands, key=lambda c: c[0])
                sel[(task, opt)] = eps
                chosen[(task, opt)] = (runs, evs)
                if not any((task, opt, eps, sd_, "") in v2 for sd_ in runs):
                    stage1_only.add((task, opt))
                summary["selected"][f"{task}/{opt}"] = {"eps": eps, "seeds": sorted(runs), "criterion": crit,
                                                        "candidates": {c[1]: c[0] for c in cands}}

    # ---- main table (best checkpoint) and final table: rows = methods, columns = tasks ----
    def table(kind):
        is_final = kind == "final"
        what = "end of the hour" if is_final else "best validation checkpoint"
        lines = [f"% generated by scripts/collect_stage2.py -- {what}; rows: methods sorted by the overall score; per task: test loss and accuracy (%) or bits per char, "
                 "mean +- s.d. over seeds; score = sum of the four test losses; rank = mean rank over the four tasks by loss; best probe method per task in bold",
                 "\\begin{tabular}{l cc cc cc cc c}", "\\toprule",
                 " & \\multicolumn{2}{c}{MNIST / MLP} & \\multicolumn{2}{c}{CIFAR-10 / ConvNet} & \\multicolumn{2}{c}{PTB / char-LSTM} & \\multicolumn{2}{c}{PTB / char-Transformer} & \\\\",
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
                 "Method & loss & acc.\\ (\\%) & loss & acc.\\ (\\%) & loss & bits/char & loss & bits/char & score \\\\", "\\midrule"]
        cells = {o: [] for o in OPTS}
        means = {}
        grid = [f"{'task':18s} " + " ".join(f"{o:>10s}" for o in OPTS)]
        for task in TASKS:
            is_cls = task in IMAGE
            means[task] = {}
            for o in OPTS:
                if (task, o) not in chosen:
                    cells[o] += ["--", "--"]; means[task][o] = float("nan"); continue
                runs, evs = chosen[(task, o)]
                recs = {sd_: (final(r) if is_final else evs[sd_]) for sd_, r in runs.items() if evs.get(sd_)}
                losses = [h["test_loss"] for h in recs.values()]
                means[task][o] = mean(losses)
                dag = "$^\\dagger$" if (is_final and any(r.get("diverged") for r in runs.values())) else ""
                if is_cls:
                    if o == "ff":
                        g = [100 * h["test_acc_goodness"] for h in recs.values()]
                        ro = [100 * h["test_acc"] for h in recs.values()]
                        sec = f"{ms(g, 1)} [{mean(ro):.1f}]"
                    else:
                        sec = ms([100 * h["test_acc"] for h in recs.values()], 1)
                else:
                    sec = ms([h["test_loss"] / LN2 for h in recs.values()], 2)
                s1mark = "$^{\\mathrm{s1}}$" if (task, o) in stage1_only else ""
                cells[o] += [ms(losses) + dag + s1mark, sec]
                minute = mean([h["train_time"] / 60 for h in recs.values()])
                key = f"{task}/{o}"
                summary["final" if is_final else "main"][key] = {
                    "eps": sel[(task, o)], "n_seeds": len(recs),
                    "test_loss_mean": means[task][o], "test_loss_sd": sd(losses),
                    "test_acc_mean": mean([h.get("test_acc") for h in recs.values()]),
                    "test_acc_sd": sd([h.get("test_acc") for h in recs.values()]),
                    "test_bpc_mean": means[task][o] / LN2, "test_bpc_sd": sd(losses) / LN2,
                    "test_acc_goodness_mean": mean([h.get("test_acc_goodness") for h in recs.values()]) if o == "ff" and is_cls else None,
                    "minute_mean": minute, "per_seed": {sd_: {"test_loss": h["test_loss"], "test_acc": h.get("test_acc"),
                                                              "test_acc_goodness": h.get("test_acc_goodness"), "minute": h["train_time"] / 60,
                                                              "step": h["step"], "val_loss": h["val_loss"]} for sd_, h in recs.items()}}
            probe = {o: means[task][o] for o in PROBE if not math.isnan(means[task][o])}
            if probe:
                b = min(probe, key=probe.get)
                ti = TASKS.index(task)
                cells[b][2 * ti] = f"\\textbf{{{cells[b][2 * ti]}}}"
                cells[b][2 * ti + 1] = f"\\textbf{{{cells[b][2 * ti + 1]}}}"
            grid.append(f"{task:18s} " + " ".join(f"{means[task][o]:10.3f}" if not math.isnan(means[task][o]) else f"{'--':>10s}" for o in OPTS))
        # overall score: sum over tasks of the method's loss relative to BP's; mean rank over tasks (all ten methods)
        score, mrank = {}, {}
        for o in OPTS:
            rel = [means[t][o] for t in TASKS if not math.isnan(means[t][o])]   # plain sum of the four test losses
            score[o] = sum(rel) if len(rel) == len(TASKS) else float("nan")
            ranks = []
            for t in TASKS:
                order = sorted([x for x in OPTS if not math.isnan(means[t][x])], key=lambda x: means[t][x])
                if o in order:
                    ranks.append(order.index(o) + 1)
            mrank[o] = mean(ranks) if ranks else float("nan")
        order = sorted(OPTS, key=lambda o: (math.isnan(score[o]), score[o]))
        for o in order:
            sc = "--" if (math.isnan(score[o]) or o == "ff") else f"{score[o]:.2f}"   # FF's readout loss is not comparable
            rk = "--" if math.isnan(mrank[o]) else f"{mrank[o]:.2f}"
            lines.append(f"{OPT_LABEL[o]} & " + " & ".join(cells[o]) + f" & {sc} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(tab_dir, f"{kind}_table.tex"), "w").write("\n".join(lines) + "\n")
        rows = []
        PLAIN = {"bp": "BP", "spsa": "SPSA", "spsa15": "1.5-SPSA", "rs": "RS", "np": "NP", "anp": "ANP", "anp15": "1.5-ANP", "inp": "INP", "inp15": "1.5-INP", "danp": "DANP", "ff": "FF"}
        for o in order:
            if math.isnan(score[o]) or o == "ff":
                continue
            rows.append([o, PLAIN[o], *[means[t][o] for t in TASKS], score[o], mrank[o]])
        write_csv(os.path.join(data_dir, f"overall_{kind}.csv"), ["method", "label", "mnist", "cifar", "lstm", "transformer", "score", "rank"], rows)
        summary.setdefault("overall", {})[kind] = {o: {"score": score[o], "rank": mrank[o]} for o in OPTS}
        return grid

    grid_main = table("main")
    grid_final = table("final")

    # ---- best-validation-accuracy checkpoints of the classification tasks (appendix) ----
    lines = ["% generated by scripts/collect_stage2.py -- image tasks: test accuracy (%) and test loss at the evaluation with the HIGHEST validation accuracy (mean over seeds, s.d. in units of the last digit); best probe method per task in bold",
             "\\begin{tabular}{l cc cc}", "\\toprule",
             " & \\multicolumn{2}{c}{MNIST / MLP} & \\multicolumn{2}{c}{CIFAR-10 / ConvNet} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Method & acc.\\ (\\%) & loss & acc.\\ (\\%) & loss \\\\", "\\midrule"]
    acc_cells = {o: [] for o in OPTS}
    for task in IMAGE:
        accs = {}
        for o in OPTS:
            if (task, o) not in chosen:
                acc_cells[o] += ["--", "--"]; continue
            runs, _ = chosen[(task, o)]
            key = "val_acc_goodness" if o == "ff" else "val_acc"
            evs = [best_eval(r, key, maximize=True) for r in runs.values()]
            evs = [e for e in evs if e]
            tkey = "test_acc_goodness" if o == "ff" else "test_acc"
            a_ = [100 * e[tkey] for e in evs]; l_ = [e["test_loss"] for e in evs]
            accs[o] = mean(a_)
            acc_cells[o] += [ms(a_, 1), ms(l_)]
            summary.setdefault("best_acc", {})[f"{task}/{o}"] = {"test_acc_mean": mean(a_), "test_acc_sd": sd(a_), "test_loss_mean": mean(l_),
                                                                 "minute_mean": mean([e["train_time"] / 60 for e in evs]), "n": len(evs)}
        probe = {o: accs[o] for o in PROBE if o in accs}
        if probe:
            b = max(probe, key=probe.get); ti = IMAGE.index(task)
            acc_cells[b][2 * ti] = f"\\textbf{{{acc_cells[b][2 * ti]}}}"
    for o in OPTS:
        lines.append(f"{OPT_LABEL[o]} & " + " & ".join(acc_cells[o]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "bestacc_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- mean ranks over the probe methods with a bootstrap interval over seeds ----
    import random as _random
    _random.seed(0)
    for kind in ("main", "final"):
        src = summary[kind]
        per_seed = {}
        for t in TASKS:
            for o in PROBE:
                v = src.get(f"{t}/{o}")
                if v:
                    per_seed[(t, o)] = [x["test_loss"] for x in v["per_seed"].values()]
        def ranks_once(draw):
            out = {o: [] for o in PROBE}
            for t in TASKS:
                vals = {o: draw(per_seed[(t, o)]) for o in PROBE if (t, o) in per_seed}
                order = sorted(vals, key=vals.get)
                for i, o in enumerate(order):
                    out[o].append(i + 1)
            return {o: (sum(r) / len(r) if r else float("nan")) for o, r in out.items()}
        point = ranks_once(lambda xs: sum(xs) / len(xs))
        boots = [ranks_once(lambda xs: _random.choice(xs)) for _ in range(1000)]
        summary.setdefault("mean_rank", {})[kind] = {o: {"mean": point[o],
                                                         "lo": sorted(b[o] for b in boots)[25], "hi": sorted(b[o] for b in boots)[974]}
                                                     for o in PROBE}

    # ---- selected eps, minute of the best checkpoint, seeds (appendix) ----
    lines = ["% generated by scripts/collect_stage2.py -- selected eps, mean minute of the best validation checkpoint, number of seeds",
             "\\begin{tabular}{l cc cc cc cc}", "\\toprule",
             " & \\multicolumn{2}{c}{MNIST} & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{LSTM} & \\multicolumn{2}{c}{Transformer} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
             "Method & $\\varepsilon^\\ast$ & min / seeds & $\\varepsilon^\\ast$ & min / seeds & $\\varepsilon^\\ast$ & min / seeds & $\\varepsilon^\\ast$ & min / seeds \\\\", "\\midrule"]
    for o in OPTS:
        cells = []
        for task in TASKS:
            m = summary["main"].get(f"{task}/{o}")
            cells += ["--", "--"] if m is None else [m["eps"], f"{m['minute_mean']:.0f} / {m['n_seeds']}"]
        lines.append(f"{OPT_LABEL[o]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "selected_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- throughput (mean over seeds of the selected runs): rows = methods ----
    lines = ["% generated by scripts/collect_stage2.py -- optimizer steps per hour and forward passes per hour (thousands), mean over seeds of the selected runs",
             "\\begin{tabular}{l cc cc cc cc}", "\\toprule",
             " & \\multicolumn{2}{c}{MNIST} & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{LSTM} & \\multicolumn{2}{c}{Transformer} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
             "Method & steps/h & passes/h & steps/h & passes/h & steps/h & passes/h & steps/h & passes/h \\\\", "\\midrule"]
    for o in OPTS:
        cells = []
        for task in TASKS:
            if (task, o) not in chosen:
                cells += ["--", "--"]; continue
            runs, _ = chosen[(task, o)]
            sph = [final(r)["step"] / max(final(r)["train_time"], 1e-9) * 3600 / 1000 for r in runs.values()]
            fph = [final(r)["step"] * r["config"].get("forward_passes_per_step", 1) / max(final(r)["train_time"], 1e-9) * 3600 / 1000 for r in runs.values()]
            cells += [f"{mean(sph):.1f}", f"{mean(fph):.1f}"]
            summary["throughput"][f"{task}/{o}"] = {"steps_per_hour_mean": mean(sph) * 1000, "steps_per_hour_sd": sd(sph) * 1000,
                                                    "fwd_per_hour_mean": mean(fph) * 1000, "n": len(sph)}
        lines.append(f"{OPT_LABEL[o]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "throughput_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- steps per run, seconds per step, peak GPU memory (rows = methods) ----
    def _res_table(name, header, cell_fn, comment, skip=()):
        lines = [f"% generated by scripts/collect_stage2.py -- {comment}",
                 "\\begin{tabular}{l cc cc cc cc}", "\\toprule",
                 " & \\multicolumn{2}{c}{MNIST} & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{LSTM} & \\multicolumn{2}{c}{Transformer} \\\\",
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
                 "Method & " + " & ".join([header] * 4) + " \\\\", "\\midrule"]
        for o in [x for x in OPTS if x not in skip]:
            cells = []
            for task in TASKS:
                cells += cell_fn(task, o) if (task, o) in chosen else ["--", "--"]
            lines.append(f"{OPT_LABEL[o]} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(tab_dir, name), "w").write("\n".join(lines) + "\n")

    def _steps_cells(task, o):
        runs = chosen[(task, o)][0].values()
        steps = [final(r)["step"] / 1000 for r in runs]
        msps = [1000 * final(r)["train_time"] / max(final(r)["step"], 1) for r in runs]
        summary.setdefault("steps", {})[f"{task}/{o}"] = {"steps_k_mean": mean(steps), "steps_k_sd": sd(steps), "ms_per_step_mean": mean(msps), "ms_per_step_sd": sd(msps), "n": len(steps)}
        return [f"{mean(steps):.0f}" + (f"({sd(steps):.0f})" if len(steps) > 1 else ""), f"{mean(msps):.2f}"]

    res = load_dir("results_resources") if os.path.isdir("results_resources") else {}

    def _mem_cells(task, o):
        iso = [r for (t_, o_, e_, s_, tg), r in res.items() if t_ == task and o_ == o and r.get("max_memory_allocated")]
        runs = iso or [r for r in chosen[(task, o)][0].values() if r.get("max_memory_allocated")]
        if not runs:
            return ["--", "--"]
        alloc = [r["max_memory_allocated"] / 2 ** 20 for r in runs]
        resv = [r["max_memory_reserved"] / 2 ** 20 for r in runs]
        summary.setdefault("memory", {})[f"{task}/{o}"] = {"max_alloc_mb": mean(alloc), "max_reserved_mb": mean(resv), "n": len(runs)}
        return [f"{mean(alloc):.0f}", f"{mean(resv):.0f}"]

    _res_table("steps_table.tex", "steps (k) & ms/step", _steps_cells,
               "optimizer steps completed in the hour (thousands) and milliseconds of training time per step, mean over seeds of the selected runs")
    _res_table("memory_table.tex", "alloc.\\ & reserved", _mem_cells,
               "peak GPU memory of the training process (MB): tensors allocated / reserved by the caching allocator (isolated 150-s measurement when available, else the seeds that recorded it)")

    def _iso_cells(task, o):
        iso = [r for (t_, o_, e_, s_, tg), r in res.items() if t_ == task and o_ == o and r.get("steps")]
        if not iso:
            return ["--", "--"]
        sps = [r["steps"] / max(r["train_time"], 1e-9) for r in iso]
        msps = [1000 * r["train_time"] / max(r["steps"], 1) for r in iso]
        summary.setdefault("isolated", {})[f"{task}/{o}"] = {"steps_per_sec": mean(sps), "ms_per_step": mean(msps)}
        return [f"{mean(sps):.0f}", f"{mean(msps):.2f}"]
    _res_table("isolated_table.tex", "steps/s & ms/step", _iso_cells,
               "isolated throughput: each selected configuration run alone on one RTX 5090 for 150 s (no co-scheduled runs)", skip=NO_SIZE)

    # ---- curves: mean / min / max over seeds vs minute; equal-pass view ----
    for task in TASKS:
        minutes = list(range(0, 61))
        rows = []
        for t in minutes:
            row = [t]
            for o in OPTS:
                if (task, o) not in chosen:
                    row += [float("nan")] * 3; continue
                runs, _ = chosen[(task, o)]
                vals = [interp_at(r["history"], "train_time", t * 60) for r in runs.values()]
                vals = [v for v in vals if not math.isnan(v)]
                row += [mean(vals), min(vals), max(vals)] if vals else [float("nan")] * 3
            rows.append(row)
        write_csv(os.path.join(data_dir, f"curve2_{task}.csv"), ["time", *[f"{o}_{k}" for o in OPTS for k in ("mean", "min", "max")]], rows)
        # seed bands only for the methods that have data (pgfplots' fill between fails on empty paths)
        colors = {"bp": "black", "spsa": "zoospsa", "spsa15": "zoospsa", "rs": "zoors", "np": "zoonp", "anp": "zooanp", "anp15": "zooanp",
                  "inp": "zooinp", "inp15": "zooinp", "danp": "zoodanp", "ff": "zooff"}
        fig_dir = os.path.join(a.paper, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        with open(os.path.join(fig_dir, f"bands_{task}.tex"), "w") as f:
            f.write(f"% generated by scripts/collect_stage2.py -- seed bands of the methods with data for {task}\n")
            for o in OPTS:
                if (task, o) in chosen and len(chosen[(task, o)][0]) > 1:
                    f.write(f"\\zooband{{data/curve2_{task}.csv}}{{{o}}}{{{colors[o]}}}\n")
        # equal forward passes: log grid from 1e3 to the largest count reached
        rows = []
        pmax = max((final(r)["forward_passes"] for o in OPTS if (task, o) in chosen for r in chosen[(task, o)][0].values()), default=0)
        P = 1000.0
        while P <= pmax:
            row = [P]
            for o in OPTS:
                if (task, o) not in chosen:
                    row.append(float("nan")); continue
                vals = [best_within(r["history"], "forward_passes", P) for r in chosen[(task, o)][0].values()]
                vals = [v for v in vals if not math.isnan(v)]
                row.append(mean(vals) if vals else float("nan"))
            rows.append(row)
            P *= 10 ** 0.125
        write_csv(os.path.join(data_dir, f"passes_{task}.csv"), ["passes", *OPTS], rows)

    # ---- backprop reference variants ----
    lines = ["% generated by scripts/collect_stage2.py -- backprop reference: best validation checkpoint (test acc % or bits/char, minute) and end of the hour",
             "\\begin{tabular}{ll cc cc cc cc}", "\\toprule",
             " & & \\multicolumn{2}{c}{MNIST} & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{LSTM} & \\multicolumn{2}{c}{Transformer} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
             "lr & decay & best (min) & final & best (min) & final & best (min) & final & best (min) & final \\\\", "\\midrule"]
    variants = [("0.01", "", "0.01 & 0"), ("0.001", "", "0.001 & 0"), ("0.0001", "", "0.0001 & 0"), ("1e-05", "", "0.00001 & 0"),
                ("0.001", "-wd0.01", "0.001 & 0.01"), ("0.0001", "-wd0.01", "0.0001 & 0.01")]
    for eps, tag, label in variants:
        cells = []
        for task in TASKS:
            runs = []
            if tag == "":
                runs = list(seeds_of(task, "bp", eps).values())
            if not runs:
                r = bp.get((task, "bp", eps, 0, tag))
                runs = [r] if r else []
            if not runs:
                cells += ["--", "--"]; continue
            evs = [best_eval(r, "val_loss") for r in runs]
            if task in IMAGE:
                b = [100 * e["test_acc"] for e in evs]; f_ = [100 * final(r)["test_acc"] for r in runs]
                fmtb = ms(b, 1)
            else:
                b = [e["test_loss"] / LN2 for e in evs]; f_ = [final(r)["test_loss"] / LN2 for r in runs]
                fmtb = ms(b, 2)
            minute = mean([e["train_time"] / 60 for e in evs])
            dag = "$^\\dagger$" if any(r.get("diverged") for r in runs) else ""
            cells += [f"{fmtb} ({minute:.0f})", (ms(f_, 1) if task in IMAGE else ms(f_, 2)) + dag]
            summary["bp"][f"{task}/lr{eps}{tag}"] = {"best": mean(b), "best_sd": sd(b), "final": mean(f_), "minute": minute, "n": len(runs)}
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "bp_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- untied (lr, sigma) grid: best checkpoint within 30 minutes; the tied diagonal from the 1-h runs ----
    LRS, SIGS = ["0.01", "0.001", "0.0001"], ["0.01", "0.001", "0.0001"]
    SIGS_OF = {"0.01": SIGS, "0.001": SIGS, "0.0001": ["0.1"] + SIGS}
    def _grid_cell(task, o, lr, sig):
        """(mean test loss, n, diverged) of the best validation checkpoint within 30 min for (lr, sigma)."""
        if lr == sig:
            runs = list(seeds_of(task, o, lr).values())
            if not runs and lr == "0.01":
                r = s1.get((task, o, lr, 0, "")); runs = [r] if r else []
        else:
            r = dc.get((task, o, lr, 0, f"-sig{sig}")); runs = [r] if r else []
        if not runs:
            return None
        evs = [best_eval(r, "val_loss", t_max=1800) for r in runs]
        vals = [e["test_loss"] for e in evs if e]
        return (mean(vals), len(vals), any(r.get("diverged") for r in runs)) if vals else None
    lines = ["% generated by scripts/collect_stage2.py -- probe scale decoupled from the step size: test loss of the best validation checkpoint within 30 minutes for every (lr, sigma); the tied diagonal is the mean over the stage-2 seeds (the tied 1e-2 cell is the stage-1 run); dagger = diverged",
             "\\begin{tabular}{ll ccc ccc cccc}", "\\toprule",
             " & & \\multicolumn{3}{c}{$\\eta=10^{-2}$} & \\multicolumn{3}{c}{$\\eta=10^{-3}$} & \\multicolumn{4}{c}{$\\eta=10^{-4}$} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-12}",
             "Task & Method & " + " & ".join(["$\\sigma{=}10^{-2}$ & $10^{-3}$ & $10^{-4}$"] * 2) + " & $\\sigma{=}10^{-1}$ & $10^{-2}$ & $10^{-3}$ & $10^{-4}$ \\\\", "\\midrule"]
    for ti, task in enumerate(TASKS):
        for oi, o in enumerate([x for x in PROBE if x not in NO_SIZE]):
            cells = []
            best = None
            for lr in LRS:
                for sig in SIGS_OF[lr]:
                    g = _grid_cell(task, o, lr, sig)
                    if g is None:
                        cells.append("--"); continue
                    v, n, div = g
                    cells.append(fmt_loss(v) + ("$^\\dagger$" if div else ""))
                    summary["decouple"][f"{task}/{o}/lr{lr}/sig{sig}"] = {"test_loss": v, "n": n, "diverged": div}
                    if not div and (best is None or v < best[0]):
                        best = (v, lr, sig, len(cells) - 1)
            if best:
                cells[best[3]] = f"\\textbf{{{cells[best[3]]}}}"
                summary.setdefault("untied_best", {})[f"{task}/{o}"] = {"lr": best[1], "sigma": best[2], "test_loss": best[0],
                                                                        "tied": min([summary["decouple"][f"{task}/{o}/lr{lr}/sig{lr}"]["test_loss"] for lr in LRS
                                                                                     if f"{task}/{o}/lr{lr}/sig{lr}" in summary["decouple"]
                                                                                     and summary["decouple"][f"{task}/{o}/lr{lr}/sig{lr}"]["test_loss"] is not None]
                                                                                    or [None], key=lambda v: float("inf") if v is None else v)}
            lines.append(f"{TASK_LABEL[task] if oi == 0 else ''} & {OPT_LABEL[o]} & " + " & ".join(cells) + " \\\\")
        if ti < len(TASKS) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "decouple_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- random search: Adam pseudo-gradient vs direct jump, at the three model sizes ----
    SZ = [("", "40k"), ("-sz400k", "400k"), ("-sz4m", "4M")]
    sc = load_dir("results_scale") if os.path.isdir("results_scale") else {}
    lines = ["% generated by scripts/collect_stage2.py -- random search with the Adam wrapper (protocol; 40k = mean over seeds, larger = seed 0) vs direct jumps (rsj, seed 0): best-checkpoint test loss (acc % / bpc)",
             "\\begin{tabular}{l l cc}", "\\toprule", "Params & Task & RS (Adam, protocol) & RS direct jump \\\\", "\\midrule"]
    def _bc(r, task):
        e = best_eval(r, "val_loss")
        if not e:
            return "--"
        sec = f"{100 * e['test_acc']:.1f}" if task in IMAGE else f"{e['test_loss'] / LN2:.2f}"
        return f"{e['test_loss']:.3f} ({sec})"
    for tag, lab in SZ:
        for ti, task in enumerate(TASKS):
            if tag == "":
                runs = list(chosen[(task, "rs")][0].values()) if (task, "rs") in chosen else []
                evs = [best_eval(r, "val_loss") for r in runs]
                rs_cell = (ms([e["test_loss"] for e in evs]) + " (" + (ms([100 * e["test_acc"] for e in evs], 1) if task in IMAGE else ms([e["test_loss"] / LN2 for e in evs], 2)) + ")") if runs else "--"
            else:
                cands = [r for (t_, o_, e_, s_, tg), r in sc.items() if t_ == task and o_ == "rs" and tg == tag and s_ == 0 and best_eval(r, "val_loss")]
                r = min(cands, key=lambda r: best_eval(r, "val_loss")["val_loss"]) if cands else None
                rs_cell = _bc(r, task) if r else "--"
            rj = next((r for (t_, o_, e_, s_, tg), r in v2.items() if t_ == task and o_ == "rsj" and tg == tag), None)
            rj_cell = _bc(rj, task) if rj else "--"
            if rj:
                summary["rsj"][f"{task}{tag}/rsj"] = {"best": best_eval(rj, "val_loss")["test_loss"]}
            lines.append(f"{lab if ti == 0 else ''} & {TASK_LABEL[task]} & {rs_cell} & {rj_cell} \\\\")
        if tag != "-sz4m":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "rsj_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- more pairs: 1.5-ANP with 32 pairs (65 passes) vs ANP / INP at 65 passes, Adam lr 1e-4, sigma 1e-3 ----
    lines = ["% generated by scripts/collect_stage2.py -- 65 forward passes per step (32 antithetic pairs for 1.5-ANP, 64 noisy passes for ANP / INP), Adam lr 1e-4, sigma 1e-3, fp64 losses for 1.5-ANP, seed 0: best-checkpoint test loss (acc % / bpc); the 17 / 16-pass columns are the tied-protocol selections of Table 1 (mean over seeds)",
             "\\begin{tabular}{l cccc ccc}", "\\toprule",
             "Task & \\ANPb{} 65 & \\ANPb{} 65, $\\alpha{=}0$ & \\ANP{} 65 & \\INP{} 65 & \\ANPb{} 17 & \\ANP{} 16 & \\INP{} 16 \\\\", "\\midrule"]
    def _k65(task, o, tag):
        r = v2.get((task, o, "0.0001", 0, tag))
        return _bc(r, task) if r else "--"
    for task in TASKS:
        cells = [_k65(task, "anp15", "-sig0.001-K65-l64"), _k65(task, "anp15", "-sig0.001-K65-a0-l64"), _k65(task, "anp", "-sig0.001-K65"), _k65(task, "inp", "-sig0.001-K65")]
        for o in ("anp15", "anp", "inp"):
            ch = chosen.get((task, o))
            vals = [e["test_loss"] for e in ch[1].values() if e] if ch else []
            cells.append(ms(vals) if vals else "--")
        lines.append(f"{TASK_LABEL[task]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "passes_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- 1.5-order rules under their own recipe (SGD screening, 30 min): best config and controls ----
    def _own_best(task, opt, pred):
        """(tag, run, best_eval within 30 min) with the lowest best-validation loss among the runs matching pred."""
        best = None
        for (t_, o_, eps, seed, tag), r in own.items():
            if t_ != task or o_ != opt or seed != 0 or not pred(eps, tag):
                continue
            e = best_eval(r, "val_loss", t_max=1800)
            if e and (best is None or e["val_loss"] < best[2]["val_loss"]):
                best = (f"lr{eps}{tag}", r, e)
        return best
    lines = ["% generated by scripts/collect_stage2.py -- the 1.5-order rules under their own recipe (plain SGD, fp64 losses, 30-minute screening): "
             "best configuration by validation loss, its test loss (acc % / bpc), the alpha = 0 control at the same grid, and the protocol run (Adam, tied eps) at minute 30",
             "\\begin{tabular}{l l l c c c}", "\\toprule",
             "Task & Rule & best config & test loss (acc / bpc) & $\\alpha=0$ control & protocol (Adam) @30 min \\\\", "\\midrule"]
    for task in TASKS:
        is_cls = task in IMAGE
        for o in ("spsa15", "anp15"):
            b = _own_best(task, o, lambda eps, tag: "-a0" not in tag and "-d" not in tag and "-lam0.01" not in tag)
            c0 = _own_best(task, o, lambda eps, tag: "-a0" in tag)
            ctrl = _own_best(task, "spsa" if o == "spsa15" else "anp", lambda eps, tag: True)
            prot = chosen.get((task, o))
            def cell(x):
                if not x:
                    return "--"
                e = x[2]
                sec = f"{100 * e['test_acc']:.1f}" if is_cls else f"{e['test_loss'] / LN2:.2f}"
                return f"{e['test_loss']:.3f} ({sec})"
            pcell = "--"
            if prot:
                evs = [best_eval(r, "val_loss", t_max=1800) for r in prot[0].values()]
                vals = [e["test_loss"] for e in evs if e]
                pcell = ms(vals) if vals else "--"
            lines.append(f"{TASK_LABEL[task] if o == 'spsa15' else ''} & {OPT_LABEL[o]} & {b[0].replace('-', ' ') if b else '--'} & {cell(b)} & {cell(c0)} & {pcell} \\\\")
            if b:
                summary.setdefault("own", {})[f"{task}/{o}"] = {"config": b[0], "test_loss": b[2]["test_loss"], "test_acc": b[2].get("test_acc"),
                                                                "alpha0": c0[2]["test_loss"] if c0 else None, "control": ctrl[2]["test_loss"] if ctrl else None}
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "own_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- eps screening table: best validation loss (mean over seeds) at every tied eps; the selected value in bold ----
    EPS_ALL = ["0.1", "0.01", "0.001", "0.0001", "1e-05"]
    lines = ["% generated by scripts/collect_stage2.py -- tied eps screening: lowest validation loss over the hour (mean over seeds; FF on the "
             "image tasks: minus the goodness validation accuracy), selected value in bold; stage-1 runs (5-minute evaluations) for the "
             "first three values where no stage-2 run exists",
             "\\begin{tabular}{l l " + "c" * len(EPS_ALL) + "}", "\\toprule",
             "Task & Method & " + " & ".join(f"$\\varepsilon={e.replace('1e-05', '10^{-5}')}$" for e in EPS_ALL) + " \\\\", "\\midrule"]
    for task in TASKS:
        for o in OPTS:
            selv = summary["selected"].get(f"{task}/{o}", {})
            cands = selv.get("candidates", {})
            cells = []
            for e in EPS_ALL:
                v = cands.get(e)
                if v is None:
                    cells.append("--"); continue
                txt = fmt_loss(-v, 3) if (o == "ff" and task in IMAGE) else fmt_loss(v, 3)
                cells.append(f"\\textbf{{{txt}}}" if e == selv.get("eps") else txt)
            lines.append(f"{TASK_LABEL[task] if o == OPTS[0] else ''} & {OPT_LABEL[o]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "eps_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- 1.5-ANP at the step size the eps 1e-4 screening selected (group X): Adam lr 1e-4, sigma decoupled, fp64 losses,
    #      the alpha = 0 control, and the LSTM with the spec's recurrent-state damping; best validation checkpoint, seed 0 ----
    def _x_run(task, parts):
        for (t_, o_, eps, seed, tag), r in v2.items():
            if t_ == task and o_ == "anp15" and eps == "0.0001" and seed == 0 and set(tag.strip("-").split("-")) == set(parts):
                return r
        return None
    xcols = [("sig0.01", "l64"), ("sig0.01", "l64", "a0"), ("sig0.001", "l64"), ("sig0.001", "l64", "a0"), ("sig0.001", "l64", "a0.25"), ("sig0.001", "l64", "a1")]
    xhead = ["$\\sigma{=}10^{-2}$", "$10^{-2}$, $\\alpha{=}0$", "$10^{-3}$", "$10^{-3}$, $\\alpha{=}0$", "$10^{-3}$, $\\alpha{=}\\frac14$", "$10^{-3}$, $\\alpha{=}1$"]
    lines = ["% generated by scripts/collect_stage2.py -- 1.5-ANP with Adam at lr 1e-4 (the step size selected on the image tasks), "
             "sigma decoupled from lr, fp64 losses: test loss (acc % / bpc) at the best validation checkpoint, seed 0; "
             "the alpha = 0 control removes the curvature weighting; the 'damped' row is the LSTM with recurrent-state damping 0.5; "
             "the last two columns are the tied-protocol selections of 1.5-ANP and ANP (mean over seeds)",
             "\\begin{tabular}{l cccccc cc}", "\\toprule",
             "Task & " + " & ".join(xhead) + " & \\ANPb{} tied & \\ANP{} tied \\\\", "\\midrule"]
    def _xcell(task, parts):
        r = _x_run(task, parts)
        e = best_eval(r, "val_loss") if r else None
        if not e:
            return "--"
        sec = f"{100 * e['test_acc']:.1f}" if task in IMAGE else f"{e['test_loss'] / LN2:.2f}"
        summary.setdefault("x", {})[f"{task}/{'-'.join(parts)}"] = {"test_loss": e["test_loss"], "test_acc": e.get("test_acc"), "minute": e["train_time"] / 60}
        return f"{e['test_loss']:.3f} ({sec})" + ("$^\\dagger$" if r.get("diverged") else "")
    for task in TASKS:
        rows = [(TASK_LABEL[task], [])]
        if task == "lstm_ptb":
            rows.append((TASK_LABEL[task] + ", damped", ["d0.5"]))
        for label, extra in rows:
            cells = [_xcell(task, tuple(list(parts) + extra)) for parts in xcols]
            for o in ("anp15", "anp"):
                ch = chosen.get((task, o))
                vals = [e["test_loss"] for e in ch[1].values() if e] if ch else []
                cells.append(ms(vals) if vals and not extra else ("--" if extra else "--"))
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "x_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- controls under the protocol at eps = 1e-3 (group G): 1.5-SPSA with a non-inert factor, 1.5-ANP alpha = 0 and fp64 ----
    def _ctl(task, o, eps, tag):
        r = v2.get((task, o, eps, 0, tag))
        e = best_eval(r, "val_loss") if r else None
        if not e:
            return "--"
        sec = f"{100 * e['test_acc']:.1f}" if task in IMAGE else f"{e['test_loss'] / LN2:.2f}"
        return f"{e['test_loss']:.3f} ({sec})"
    def _sel(task, o, eps):
        runs = seeds_of(task, o, eps)
        vals = [best_eval(r, "val_loss")["test_loss"] for r in runs.values() if best_eval(r, "val_loss")]
        return ms(vals) if vals else "--"
    lines = ["% generated by scripts/collect_stage2.py -- controls under the protocol (Adam, eps = lr = 1e-3; LSTM 1.5-ANP at 1e-2 is the tied selection there): "
             "test loss (acc % / bpc) at the best validation checkpoint; seed 0 for the controls, mean over seeds for the tied runs",
             "\\begin{tabular}{l ccc ccc}", "\\toprule",
             "Task & \\SPSA{} & \\SPSAb{} & \\SPSAb{} $\\alpha{=}0.5$, $\\lambda{=}10^{-3}$ & \\ANPb{} & \\ANPb{} $\\alpha{=}0$ & \\ANPb{} fp64 \\\\", "\\midrule"]
    for task in TASKS:
        lines.append(f"{TASK_LABEL[task]} & {_sel(task, 'spsa', '0.001')} & {_sel(task, 'spsa15', '0.001')} & {_ctl(task, 'spsa15', '0.001', '-a0.5-lam0.001')} & "
                     f"{_sel(task, 'anp15', '0.001')} & {_ctl(task, 'anp15', '0.001', '-a0')} & {_ctl(task, 'anp15', '0.001', '-l64')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "controls_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- untied-best replications (group K): 1-h runs with -sig tags and seeds, against the tied selection ----
    krows = {}
    for (t_, o_, eps, seed, tag), r in v2.items():
        if tag.startswith("-sig") and tag.count("-") == 1 and o_ in OPTS and r["config"].get("budget", 3600) >= 3600:
            krows.setdefault((t_, o_, eps, tag), {})[seed] = r
    lines = ["% generated by scripts/collect_stage2.py -- the best untied (lr, sigma) settings replicated for an hour (mean over seeds, best validation checkpoint) "
             "next to the tied selection of the same pair",
             "\\begin{tabular}{l l l c c c}", "\\toprule",
             "Task & Method & untied $(\\eta,\\sigma)$ & untied (seeds) & tied selection (seeds) & $\\varepsilon^\\ast$ \\\\", "\\midrule"]
    for (t_, o_, eps, tag), runs in sorted(krows.items(), key=lambda kv: (TASKS.index(kv[0][0]), OPTS.index(kv[0][1]))):
        evs = [best_eval(r, "val_loss") for r in runs.values()]
        vals = [e["test_loss"] for e in evs if e]
        ch = chosen.get((t_, o_))
        tvals = [e["test_loss"] for e in ch[1].values() if e] if ch else []
        sig = tag[4:]
        lines.append(f"{TASK_LABEL[t_]} & {OPT_LABEL[o_]} & ({float(eps):g}, {float(sig):g}) & {ms(vals)} ({len(vals)}) & {ms(tvals)} ({len(tvals)}) & {sel.get((t_, o_), '--')} \\\\")
        summary.setdefault("untied_rep", {})[f"{t_}/{o_}"] = {"lr": eps, "sigma": sig, "untied": mean(vals) if vals else None, "n": len(vals), "tied": mean(tvals) if tvals else None}
    if len(lines) == 5:
        lines.append("\\multicolumn{6}{l}{(no replicated untied setting)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "untied_rep_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- model-size sweep (results_scale/: -sz400k / -sz4m tags, seed 0, selected eps): per-task losses, summed loss vs size ----
    SOPTS = [o for o in OPTS if o not in NO_SIZE]
    SIZE_N = {"": 40000, "-sz400k": 400000, "-sz4m": 4000000}
    scale_loss, scale_eps = {}, {}   # (size_tag, task, opt) -> best-checkpoint test loss, selected eps (lowest val loss among the eps run at that size)
    best_val = {}
    for (t_, o_, eps, seed, tag), r in sc.items():
        if tag in SIZE_N and seed == 0 and o_ in SOPTS:
            e = best_eval(r, "val_loss")
            if e and (((tag, t_, o_) not in best_val) or e["val_loss"] < best_val[(tag, t_, o_)]):
                best_val[(tag, t_, o_)] = e["val_loss"]
                scale_loss[(tag, t_, o_)] = e["test_loss"]
                scale_eps[(tag, t_, o_)] = eps
    summary["scale_eps"] = {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in scale_eps.items()}
    for t_ in TASKS:
        for o_ in SOPTS:
            if f"{t_}/{o_}" in summary["main"]:
                scale_loss[("", t_, o_)] = summary["main"][f"{t_}/{o_}"]["test_loss_mean"]
    rows = []
    for tag in ("", "-sz400k", "-sz4m"):
        row = [SIZE_N[tag]]
        for o_ in SOPTS:
            vals = [scale_loss.get((tag, t_, o_)) for t_ in TASKS]
            row.append(sum(vals) if all(v is not None for v in vals) else float("nan"))
        rows.append(row)
    write_csv(os.path.join(data_dir, "scale.csv"), ["params", *SOPTS], rows)
    # test loss vs minute for the larger sizes (seed 0), same column layout as curve2_{task}.csv
    for tag, lab in (("-sz400k", "400k"), ("-sz4m", "4m")):
        for t_ in TASKS:
            rows = []
            for t in range(0, 61):
                row = [t]
                for o_ in SOPTS:
                    r = sc.get((t_, o_, scale_eps.get((tag, t_, o_)), 0, tag))
                    v = interp_at(r["history"], "train_time", t * 60) if r else float("nan")
                    row += [v, v, v]
                rows.append(row)
            write_csv(os.path.join(data_dir, f"curve2_{t_}_{lab}.csv"), ["time", *[f"{o}_{k}" for o in SOPTS for k in ("mean", "min", "max")]], rows)
    for t_ in TASKS:   # per-task curves for the four-panel appendix figure
        write_csv(os.path.join(data_dir, f"scale_{t_}.csv"), ["params", *SOPTS],
                  [[SIZE_N[tag], *[scale_loss.get((tag, t_, o_), float("nan")) for o_ in SOPTS]] for tag in ("", "-sz400k", "-sz4m")])
    lines = ["% generated by scripts/collect_stage2.py -- model-size sweep: best-checkpoint test loss per task and summed, seed 0, selected eps; "
             "the 40k column is the stage-2 seed mean",
             "\\begin{tabular}{l l " + "c" * len(SOPTS) + "}", "\\toprule",
             "Params & Task & " + " & ".join(OPT_LABEL[o] for o in SOPTS) + " \\\\", "\\midrule"]
    for tag, label in (("", "40k"), ("-sz400k", "400k"), ("-sz4m", "4M")):
        for ti, t_ in enumerate(TASKS + ["sum"]):
            cells = []
            for o_ in SOPTS:
                if t_ == "sum":
                    vals = [scale_loss.get((tag, tt, o_)) for tt in TASKS]
                    v = sum(vals) if all(x is not None for x in vals) else None
                else:
                    v = scale_loss.get((tag, t_, o_))
                cells.append("--" if v is None else fmt_loss(v))
            lines.append(f"{label if ti == 0 else ''} & {TASK_LABEL[t_] if t_ != 'sum' else 'sum'} & " + " & ".join(cells) + " \\\\")
        if tag != "-sz4m":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "scale_table.tex"), "w").write("\n".join(lines) + "\n")
    summary["scale"] = {f"{k[0] or '-sz40k'}/{k[1]}/{k[2]}": v for k, v in scale_loss.items()}

    # ---- alignment at 4M vs 40k on the sequence models ----
    p4 = "results_align_4m/alignment.csv"
    if os.path.exists(p4) and os.path.exists(os.path.join(a.align, "alignment.csv")):
        r40 = list(csv.DictReader(open(os.path.join(a.align, "alignment.csv")))); r4m = list(csv.DictReader(open(p4)))
        ests = ["spsa", "rs", "np", "anp", "inp", "danp", "spsa15", "anp15"]
        def cosv(rows, task, e, P):
            v = [float(r["cosine"]) for r in rows if r["task"] == task and r["state"] == "trained" and r["layer"] == "all"
                 and r["estimator"] == e and abs(float(r["sigma"]) - 1e-3) < 1e-12 and int(r["passes"]) == P]
            return f"{v[0]:.3f}" if v else "--"
        lines = ["% generated by scripts/collect_stage2.py -- whole-model cosine with the true gradient on the sequence models at 40k and 4M parameters, sigma 1e-3, trained state",
                 "\\begin{tabular}{l l cc cc}", "\\toprule",
                 " & & \\multicolumn{2}{c}{16 passes} & \\multicolumn{2}{c}{1024 passes} \\\\", "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
                 "Model & Estimator & 40k & 4M & 40k & 4M \\\\", "\\midrule"]
        for task in ("lstm_ptb", "transformer_ptb"):
            for ei, e in enumerate(ests):
                lines.append(f"{TASK_LABEL[task] if ei == 0 else ''} & {OPT_LABEL[e]} & {cosv(r40, task, e, 16)} & {cosv(r4m, task, e, 16)} & {cosv(r40, task, e, 1024)} & {cosv(r4m, task, e, 1024)} \\\\")
            if task == "lstm_ptb":
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(tab_dir, "align4m_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- 1.5-INP: the 1.5-ANP estimator with INP's layer-wise probing ----
    # alpha = 0 (tag -a0-l64): eps chosen per task by the seed-0 best-checkpoint validation loss over the bracket,
    # then mean +- s.d. over the seeds at that eps.  Untied runs (tag -sig<s>-a0-l64) are reported separately:
    # the (eps, sigma) pair with the lowest seed-0 validation loss, averaged over its seeds.
    def inp15_runs(task, tag, eps=None):
        return {s_: r for (t_, o_, e_, s_, tg), r in v2.items()
                if t_ == task and o_ == "inp15" and tg == tag and (eps is None or e_ == eps)}
    def inp15_cell(task, runs):
        evs = [best_eval(r, "val_loss") for r in runs.values()]; evs = [e for e in evs if e]
        if not evs:
            return "--", None
        sec = ms([100 * e["test_acc"] for e in evs], 1) if task in IMAGE else ms([e["test_loss"] / LN2 for e in evs], 2)
        return f"{ms([e['test_loss'] for e in evs])} ({sec})", evs
    def eps_tex(e):
        m, x = f"{float(e):.0e}".split("e")
        return f"$10^{{{int(x)}}}$" if m == "1" else f"${m}\\cdot 10^{{{int(x)}}}$"
    bracket = {}
    lines = ["% generated by scripts/collect_stage2.py -- 1.5-INP (fp64 losses): alpha = 0 at the eps with the best seed-0 validation loss, "
             "mean +- s.d. over seeds; best untied (eps, sigma) by seed-0 validation loss; alpha = 0.5 at INP's eps (one seed); tied INP and 1.5-ANP selections (mean over seeds): best-checkpoint test loss (acc % / bpc)",
             "\\begin{tabular}{l c cc cc cc}", "\\toprule",
             "Task & $\\alpha{=}0.5$ & $\\varepsilon$ & $\\alpha{=}0$ & $(\\varepsilon,\\sigma)$ & $\\alpha{=}0$, untied & \\INP{} & \\ANPb{} \\\\", "\\midrule"]
    for task in TASKS:
        cells = [inp15_cell(task, inp15_runs(task, "-l64"))[0]]
        seed0 = {e_: r for (t_, o_, e_, s_, tg), r in v2.items() if t_ == task and o_ == "inp15" and tg == "-a0-l64" and s_ == 0}
        sc0 = {e_: best_eval(r, "val_loss") for e_, r in seed0.items()}; sc0 = {e_: v for e_, v in sc0.items() if v}
        bracket[task] = sc0
        if sc0:
            eps = min(sc0, key=lambda e_: sc0[e_]["val_loss"])
            cell, evs = inp15_cell(task, inp15_runs(task, "-a0-l64", eps))
            cells += [eps_tex(eps), cell]
            summary.setdefault("inp15", {})[f"{task}/-a0-l64"] = {"eps": eps, "test_loss": mean([e["test_loss"] for e in evs]), "n": len(evs),
                                                                  "seed0_val_by_eps": {e_: v["val_loss"] for e_, v in sc0.items()}}
        else:
            cells += ["--", "--"]
        unt = {(e_, tg): best_eval(r, "val_loss") for (t_, o_, e_, s_, tg), r in v2.items()
               if t_ == task and o_ == "inp15" and s_ == 0 and re.fullmatch(r"-sig[0-9.e+-]+-a0-l64", tg)}
        unt = {k: v for k, v in unt.items() if v}
        if unt:
            e_u, tg_u = min(unt, key=lambda k: unt[k]["val_loss"])
            sig = re.fullmatch(r"-sig([0-9.e+-]+)-a0-l64", tg_u).group(1)
            cell, evs = inp15_cell(task, inp15_runs(task, tg_u, e_u))
            cells += [f"({eps_tex(e_u)}, {eps_tex(sig)})", cell]
            summary.setdefault("inp15", {})[f"{task}/untied"] = {"eps": e_u, "sigma": sig, "test_loss": mean([e["test_loss"] for e in evs]), "n": len(evs),
                                                                 "seed0_val": {f"{k[0]}{k[1]}": v["val_loss"] for k, v in unt.items()}}
        else:
            cells += ["--", "--"]
        for o in ("inp", "anp15"):
            ch = chosen.get((task, o))
            vals = [e["test_loss"] for e in ch[1].values() if e] if ch else []
            cells.append(ms(vals) if vals else "--")
        lines.append(f"{TASK_LABEL[task]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "inp15_table.tex"), "w").write("\n".join(lines) + "\n")
    # step-size bracket of 1.5-INP (alpha = 0, seed 0): best-checkpoint test loss per eps; the selected eps in bold
    grid = sorted({e_ for d in bracket.values() for e_ in d}, key=float)
    lines = ["% generated by scripts/collect_stage2.py -- 1.5-INP alpha = 0, seed 0: best-checkpoint test loss for each tied eps (selected by validation loss, in bold)",
             "\\begin{tabular}{l " + "c" * len(grid) + "}", "\\toprule",
             "Task & " + " & ".join(eps_tex(e_) for e_ in grid) + " \\\\", "\\midrule"]
    for task in TASKS:
        d = bracket.get(task, {})
        best = min(d, key=lambda e_: d[e_]["val_loss"]) if d else None
        row = [("\\textbf{%.3f}" if e_ == best else "%.3f") % d[e_]["test_loss"] if e_ in d else "--" for e_ in grid]
        lines.append(f"{TASK_LABEL[task]} & " + " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "inp15_eps_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- DANP decorrelation rate: -dr<rate> runs (seed 0) against the eta_R = 1e-3 run at the same task, eps and size ----
    allr = dict(v2); allr.update(sc)
    lines = ["% generated by scripts/collect_stage2.py -- DANP with the decorrelation rate eta_R changed (seed 0): best-checkpoint test loss (acc % / bpc), its minute, and the end-of-hour loss; the eta_R = 1e-3 row is the protocol run at the same task, eps and size",
             "\\begin{tabular}{l l l l c c c}", "\\toprule",
             "Task & Params & $\\varepsilon$ & $\\eta_R$ & best & minute & final \\\\", "\\midrule"]
    def drrow(r, task, size, eps, rate):
        e = best_eval(r, "val_loss"); f = final(r)
        if not e: return None
        sec = f"{100 * e['test_acc']:.1f}" if task in IMAGE else f"{e['test_loss'] / LN2:.2f}"
        return f"{TASK_LABEL[task]} & {size} & {eps} & {rate} & {e['test_loss']:.3f} ({sec}) & {round(e['train_time'] / 60)} & {f['test_loss']:.3f} \\\\"
    seen = set()
    def drkey(kv):
        tg = kv[0][4]; m = re.search(r"-dr([0-9.e+-]+?)(?=-sz|$)", tg)
        base = tg.replace(m.group(0), "") if m else tg
        return (TASKS.index(kv[0][0]), ["", "-sz400k", "-sz4m"].index(base), -float(m.group(1)) if m else 0)
    for (t_, o_, e_, s_, tg), r in sorted((kv for kv in allr.items() if kv[0][1] == "danp" and kv[0][3] == 0 and "-dr" in kv[0][4]), key=drkey):
        m = re.search(r"-dr([0-9.e+-]+?)(?=-sz|$)", tg); rate = m.group(1); base = tg.replace(m.group(0), "")
        size = {"": "40k", "-sz400k": "400k", "-sz4m": "4M"}[base]
        if (t_, e_, base) not in seen:
            seen.add((t_, e_, base))
            ref = allr.get((t_, "danp", e_, 0, base))
            if ref:
                row = drrow(ref, t_, size, f"$10^{{{round(math.log10(float(e_)))}}}$", "$10^{-3}$")
                if row: lines.append(row)
        row = drrow(r, t_, size, f"$10^{{{round(math.log10(float(e_)))}}}$", f"$10^{{{round(math.log10(float(rate)))}}}$")
        if row: lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "danp_rate_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- gradient alignment ----
    apath = os.path.join(a.align, "alignment.csv")
    if os.path.exists(apath):
        rows = list(csv.DictReader(open(apath)))
        ests = ["spsa", "rs", "np", "anp", "inp", "danp", "spsa15", "anp15"]
        for task in TASKS:
            for state in ("init", "trained"):
                sub = [r for r in rows if r["task"] == task and r["state"] == state and r["layer"] == "all" and float(r["sigma"]) == 0.001]
                passes = sorted({int(r["passes"]) for r in sub})
                out = []
                for P in passes:
                    row = [P]
                    for e in ests:
                        v = [float(r["cosine"]) for r in sub if r["estimator"] == e and int(r["passes"]) == P]
                        row.append(v[0] if v else float("nan"))
                    out.append(row)
                if not out:   # task not measured yet: one nan row keeps the figure compilable
                    out = [[16, *[float("nan")] * len(ests)]]
                write_csv(os.path.join(data_dir, f"align_{task}_{state}.csv"), ["passes", *ests], out)
        # whole-model cosine against sigma at P = 1024, trained
        SIG_ALL = ["0.1", "0.01", "0.001", "0.0001", "1e-05"]
        lines = ["% generated by scripts/collect_stage2.py -- whole-model cosine with the true gradient at 1024 forward passes (trained state) against the probe scale",
                 "\\begin{tabular}{l l " + "c" * len(SIG_ALL) + "}", "\\toprule",
                 "Model & Estimator & " + " & ".join(f"$\\sigma={e.replace('1e-05', '10^{-5}')}$" for e in SIG_ALL) + " \\\\", "\\midrule"]
        for task in TASKS:
            for ei, e in enumerate(ests):
                cells = []
                for sg in SIG_ALL:
                    v = [float(r["cosine"]) for r in rows if r["task"] == task and r["state"] == "trained" and r["layer"] == "all"
                         and r["estimator"] == e and abs(float(r["sigma"]) - float(sg)) < 1e-12 and int(r["passes"]) == 1024]
                    cells.append(f"{v[0]:.2f}" if v else "--")
                lines.append(f"{TASK_LABEL[task] if ei == 0 else ''} & {OPT_LABEL[e]} & " + " & ".join(cells) + " \\\\")
            if task != TASKS[-1]:
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(tab_dir, "align_sigma_table.tex"), "w").write("\n".join(lines) + "\n")
        # per-layer table at P = 1024, sigma = 1e-3, trained
        lines = ["% generated by scripts/collect_stage2.py -- cosine between estimate and true gradient per layer, 1024 forward passes, sigma = 1e-3, after a 2000-step backprop warm-up",
                 "\\begin{tabular}{l l cccccccc}", "\\toprule", "Model & Layer & SPSA & RS & NP & ANP & INP & DANP & 1.5-SPSA & 1.5-ANP \\\\", "\\midrule"]
        for task in TASKS:
            sub = [r for r in rows if r["task"] == task and r["state"] == "trained" and float(r["sigma"]) == 0.001 and int(r["passes"]) == 1024]
            layers = sorted({r["layer"] for r in sub if r["layer"] != "all"}, key=int) + ["all"]
            for li, layer in enumerate(layers):
                cells = []
                for e in ests:
                    v = [float(r["cosine"]) for r in sub if r["estimator"] == e and r["layer"] == layer]
                    cells.append(f"{v[0]:.2f}" if v else "--")
                lines.append(f"{TASK_LABEL[task] if li == 0 else ''} & {('all' if layer == 'all' else layer)} & " + " & ".join(cells) + " \\\\")
            lines.append("\\midrule")
        lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(tab_dir, "align_table.tex"), "w").write("\n".join(lines) + "\n")
    ppath = os.path.join(a.align, "precision.csv")
    if os.path.exists(ppath):
        rows = [r for r in csv.DictReader(open(ppath)) if r["layer"] == "all"]
        lines = ["% generated by scripts/collect_stage2.py -- whole-model cosine with the true gradient at 1024 passes in fp32 (TF32 off), fp32 with TF32 on, and fp64",
                 "\\begin{tabular}{l l c ccc}", "\\toprule", "Model & Estimator & $\\sigma$ & fp32 & TF32 & fp64 \\\\", "\\midrule"]
        for task in TASKS:
            sub = [r for r in rows if r["task"] == task]
            if not sub:
                continue
            first = True
            for e in ("np", "anp", "inp", "spsa"):
                for sig in sorted({r["sigma"] for r in sub}, key=float, reverse=True):
                    cells = []
                    for prec in ("fp32", "tf32", "fp64"):
                        v = [float(r["cosine"]) for r in sub if r["estimator"] == e and r["sigma"] == sig and r["precision"] == prec]
                        cells.append(f"{v[0]:.3f}" if v else "--")
                    lines.append(f"{TASK_LABEL[task] if first else ''} & {e.upper()} & {float(sig):g} & " + " & ".join(cells) + " \\\\")
                    first = False
            lines.append("\\midrule")
        lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(tab_dir, "precision_table.tex"), "w").write("\n".join(lines) + "\n")

    summary["n_runs"] = {"stage1": len(s1), "v2": len(v2), "bp": len(bp), "decouple": len(dc)}
    os.makedirs(a.v2, exist_ok=True)
    json.dump(summary, open(os.path.join(a.v2, "summary_v2.json"), "w"), indent=1)
    print(f"stage1 {len(s1)} runs, v2 {len(v2)}, bp {len(bp)}, decouple {len(dc)}")
    print("\nbest-checkpoint test loss (mean over seeds):")
    print("\n".join(grid_main))
    print("\nfinal test loss (mean over seeds):")
    print("\n".join(grid_final))
    print("\nselected eps / seeds:", {k: (v["eps"], len(v["seeds"])) for k, v in summary["selected"].items()})


if __name__ == "__main__":
    main()
