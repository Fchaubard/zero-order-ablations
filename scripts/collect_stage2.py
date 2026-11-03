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
import math
import os
import statistics as st
from collections import defaultdict

TASKS = ["mlp_mnist", "conv_cifar10", "lstm_ptb", "transformer_ptb"]
IMAGE = ["mlp_mnist", "conv_cifar10"]
OPTS = ["bp", "spsa", "spsa15", "rs", "np", "anp", "anp15", "inp", "danp", "ff"]   # table row order
PROBE = ["spsa", "spsa15", "rs", "np", "anp", "anp15", "inp", "danp"]
OPT_LABEL = {"bp": "\\BP{}", "spsa": "\\SPSA{}", "spsa15": "\\SPSAb{}", "rs": "\\RS{}", "np": "\\NP{}", "anp": "\\ANP{}",
             "anp15": "\\ANPb{}", "inp": "\\INP{}", "danp": "\\DANP{}", "ff": "\\FF{}"}
TASK_SHORT = {"mlp_mnist": "MNIST", "conv_cifar10": "CIFAR-10", "lstm_ptb": "LSTM", "transformer_ptb": "Transformer"}
TASK_LABEL = {"mlp_mnist": "MNIST / MLP", "conv_cifar10": "CIFAR-10 / ConvNet",
              "lstm_ptb": "PTB / char-LSTM", "transformer_ptb": "PTB / char-Transformer"}
LN2 = math.log(2)
SEEDS = [0, 1, 2]


# ----------------------------------------------------------------------------- loading
def load_dir(d):
    """{(task, opt, eps_str, seed, tag): result}"""
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
    """mean with the s.d. over seeds in units of the last digit, e.g. 0.140(5) = 0.140 +- 0.005;
    a single value when n == 1; scientific notation for the huge losses of collapsed runs."""
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "--"
    if len(vals) == 1:
        return fmt_loss(vals[0], nd)
    m, s = st.mean(vals), st.stdev(vals)
    if abs(m) >= 1000:
        return fmt_loss(m, nd)
    u = int(round(s * 10 ** nd))
    return f"{m:.{nd}f}({u})" if u < 100 else f"{m:.{nd}f}({u})"


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
    ap.add_argument("--paper", default="paper")
    a = ap.parse_args()
    s1, v2, bp, dc = load_dir(a.stage1), load_dir(a.v2), load_dir(a.bp), load_dir(a.decouple)
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
    for (task, opt), eps in s1_sel.items():
        if opt in OPTS and (task, opt, eps, 0, "") in s1 and not rep.get((task, opt)):
            rep[(task, opt)].add(eps)
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
                if opt == "ff" and task in IMAGE:
                    evs = {sd_: best_eval(r, "val_acc_goodness", maximize=True) for sd_, r in runs.items()}
                    crit = -mean([e["val_acc_goodness"] for e in evs.values() if e])
                else:
                    evs = {sd_: best_eval(r, "val_loss") for sd_, r in runs.items()}
                    crit = mean([e["val_loss"] for e in evs.values() if e])
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
        lines = [f"% generated by scripts/collect_stage2.py -- {what}; rows: methods; per task: test loss and accuracy (%) or bits per char; "
                 "mean over seeds with the s.d. in units of the last digit in parentheses; best probe method per task (by mean loss) in bold",
                 "\\begin{tabular}{l cc cc cc cc}", "\\toprule",
                 " & \\multicolumn{2}{c}{MNIST / MLP} & \\multicolumn{2}{c}{CIFAR-10 / ConvNet} & \\multicolumn{2}{c}{PTB / char-LSTM} & \\multicolumn{2}{c}{PTB / char-Transformer} \\\\",
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
                 "Method & loss & acc.\\ (\\%) & loss & acc.\\ (\\%) & loss & bits/char & loss & bits/char \\\\", "\\midrule"]
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
        for o in OPTS:
            lines.append(f"{OPT_LABEL[o]} & " + " & ".join(cells[o]) + " \\\\")
            if o in ("bp", "rs", "danp"):
                lines.append("\\addlinespace[2pt]")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(tab_dir, f"{kind}_table.tex"), "w").write("\n".join(lines) + "\n")
        return grid

    grid_main = table("main")
    grid_final = table("final")

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
    def _res_table(name, header, cell_fn, comment):
        lines = [f"% generated by scripts/collect_stage2.py -- {comment}",
                 "\\begin{tabular}{l cc cc cc cc}", "\\toprule",
                 " & \\multicolumn{2}{c}{MNIST} & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{LSTM} & \\multicolumn{2}{c}{Transformer} \\\\",
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
                 "Method & " + " & ".join([header] * 4) + " \\\\", "\\midrule"]
        for o in OPTS:
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

    def _mem_cells(task, o):
        runs = [r for r in chosen[(task, o)][0].values() if r.get("max_memory_allocated")]
        if not runs:
            return ["--", "--"]
        alloc = [r["max_memory_allocated"] / 2 ** 20 for r in runs]
        resv = [r["max_memory_reserved"] / 2 ** 20 for r in runs]
        summary.setdefault("memory", {})[f"{task}/{o}"] = {"max_alloc_mb": mean(alloc), "max_reserved_mb": mean(resv), "n": len(runs)}
        return [f"{mean(alloc):.0f}", f"{mean(resv):.0f}"]

    _res_table("steps_table.tex", "steps (k) & ms/step", _steps_cells,
               "optimizer steps completed in the hour (thousands) and milliseconds of training time per step, mean over seeds of the selected runs")
    _res_table("memory_table.tex", "alloc.\\ & reserved", _mem_cells,
               "peak GPU memory of the training process (MB): tensors allocated / reserved by the caching allocator, mean over the seeds that recorded it")

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
        colors = {"bp": "black", "spsa": "zoospsa", "rs": "zoors", "np": "zoonp", "anp": "zooanp", "inp": "zooinp", "danp": "zoodanp", "ff": "zooff"}
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
                vals = [interp_at(r["history"], "forward_passes", P) for r in chosen[(task, o)][0].values()]
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
    variants = [("0.01", "", "0.01 & 0"), ("0.001", "", "0.001 & 0"), ("0.0001", "", "0.0001 & 0"),
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

    # ---- decoupling: (lr, sigma) grid; best checkpoint within 30 minutes ----
    combos = [("0.001", "0.01"), ("0.001", "0.001"), ("0.001", "0.0001"), ("0.01", "0.001"), ("0.01", "0.01")]
    lines = ["% generated by scripts/collect_stage2.py -- probe scale decoupled from the step size: test loss of the best validation checkpoint within 30 minutes; dagger = diverged; tied cells (lr = sigma) from stage 2 (mean over seeds) / stage 1",
             "\\begin{tabular}{ll ccccc}", "\\toprule",
             " & & \\multicolumn{3}{c}{$\\eta = 10^{-3}$} & \\multicolumn{2}{c}{$\\eta = 10^{-2}$} \\\\", "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
             "Task & Method & $\\sigma=10^{-2}$ & $\\sigma=10^{-3}$ & $\\sigma=10^{-4}$ & $\\sigma=10^{-3}$ & $\\sigma=10^{-2}$ \\\\", "\\midrule"]
    for ti, task in enumerate(TASKS):
        for oi, o in enumerate(PROBE):
            cells = []
            for lr, sig in combos:
                if lr == sig:
                    runs = list(seeds_of(task, o, lr).values()) if lr == "0.001" else []
                    if not runs:
                        r = s1.get((task, o, lr, 0, ""))
                        runs = [r] if r else []
                else:
                    r = dc.get((task, o, lr, 0, f"-sig{sig}"))
                    runs = [r] if r else []
                if not runs:
                    cells.append("--"); continue
                evs = [best_eval(r, "val_loss", t_max=1800) for r in runs]
                vals = [e["test_loss"] for e in evs if e]
                dag = "$^\\dagger$" if any(r.get("diverged") for r in runs) else ""
                cells.append(ms(vals) + dag)
                summary["decouple"][f"{task}/{o}/lr{lr}/sig{sig}"] = {"test_loss": mean(vals), "n": len(vals), "diverged": any(r.get("diverged") for r in runs)}
            lines.append(f"{TASK_LABEL[task] if oi == 0 else ''} & {OPT_LABEL[o]} & " + " & ".join(cells) + " \\\\")
        if ti < len(TASKS) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "decouple_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- random search: Adam pseudo-gradient vs direct jump ----
    lines = ["% generated by scripts/collect_stage2.py -- random search with the Adam wrapper (protocol, mean over seeds) vs direct jumps (rsj, one seed): best checkpoint test loss, acc % / bpc, final test loss",
             "\\begin{tabular}{l cc cc}", "\\toprule",
             " & \\multicolumn{2}{c}{RS (Adam, protocol)} & \\multicolumn{2}{c}{RS direct jump} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Task & best (acc / bpc) & final & best (acc / bpc) & final \\\\", "\\midrule"]
    for task in TASKS:
        cells = []
        for o in ("rs", "rsj"):
            if o == "rs":
                runs = list(chosen[(task, "rs")][0].values()) if (task, "rs") in chosen else []
            else:
                runs = [r for (t_, o_, e_, s_, tg) , r in v2.items() if t_ == task and o_ == "rsj"]
            if not runs:
                cells += ["--", "--"]; continue
            evs = [best_eval(r, "val_loss") for r in runs]
            sec = ms([100 * e["test_acc"] for e in evs], 1) if task in IMAGE else ms([e["test_loss"] / LN2 for e in evs], 2)
            cells += [f"{ms([e['test_loss'] for e in evs])} ({sec})", ms([final(r)["test_loss"] for r in runs])]
            summary["rsj"][f"{task}/{o}"] = {"best": mean([e["test_loss"] for e in evs]), "final": mean([final(r)["test_loss"] for r in runs]), "n": len(runs)}
        lines.append(f"{TASK_LABEL[task]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "rsj_table.tex"), "w").write("\n".join(lines) + "\n")

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
