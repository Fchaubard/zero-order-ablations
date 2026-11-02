#!/usr/bin/env python3
"""Turn results/*.json into the paper's CSVs and LaTeX tables (and a plain-text grid).

    data/curve_{task}.csv            time(min),bp,spsa,rs,np,anp,inp,danp,ff   test loss of the
                                     selected eps (lowest final validation loss) per optimizer
    data/curve_{task}_lr{eps}.csv    same columns, one file per eps
    data/lr_{task}.csv               eps,bp,...,ff  final test loss per eps (nan if missing)
    tables/stage1_main_table.tex     Table: test loss (test acc % or bpc), best probe method bold (stage 1, seed 0)
    tables/stage1_throughput_table.tex  steps/h and forward passes/h (thousands) (stage 1)
    tables/full_grid.tex             val / test loss for every eps, selected eps bold, dagger = diverged
    results/summary.json             everything the tables contain, machine readable

usage: python scripts/collect_results.py [--results results] [--paper paper]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

TASKS = ["mlp_mnist", "conv_cifar10", "lstm_ptb", "transformer_ptb"]
OPTS = ["bp", "spsa", "rs", "np", "anp", "inp", "danp", "ff"]
PROBE = ["spsa", "rs", "np", "anp", "inp", "danp"]
EPS = ["0.1", "0.01", "0.001"]
TASK_LABEL = {"mlp_mnist": "MNIST / MLP", "conv_cifar10": "CIFAR-10 / ConvNet",
              "lstm_ptb": "PTB / char-LSTM", "transformer_ptb": "PTB / char-Transformer"}
OPT_LABEL = {o: o.upper() for o in OPTS}
GRID_MIN = list(range(0, 61, 5))
LN2 = math.log(2)


def load(results_dir):
    runs = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        if os.path.basename(f) == "summary.json":
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        c = r.get("config", {})
        task, opt, lr = c.get("task"), c.get("opt"), c.get("lr")
        if task is None or opt is None or lr is None or not r.get("history"):
            continue
        eps = f"{float(lr):g}"
        runs[(task, opt, eps)] = r
    return runs


def final(r):
    return r.get("final") or r["history"][-1]


def curve_value(r, t_min):
    """Test loss of the evaluation closest to t_min minutes of train time, provided it lies
    within half an evaluation interval (150 s) of it; nan otherwise (e.g. after a diverged run
    stopped).  Evaluations trigger at the end of a 20-step chunk, so they land a few seconds
    after each 5-minute mark (several seconds for the slow LSTM probe methods)."""
    hist = r["history"]
    target = t_min * 60.0
    best = min(hist, key=lambda h: abs(h["train_time"] - target))
    if abs(best["train_time"] - target) > 150.0:
        return float("nan")
    return best["test_loss"]


def fmt_loss(v):
    """LaTeX loss cell: 3 decimals, or mantissa x 10^e for the huge values of diverged runs."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if abs(v) >= 1000:
        e = int(math.floor(math.log10(abs(v))))
        return f"${v / 10 ** e:.1f}{{\\times}}10^{{{e}}}$"
    return f"{v:.3f}"


def fmt(v, nd=3):
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join("nan" if (isinstance(v, float) and math.isnan(v)) else (f"{v:.5f}" if isinstance(v, float) else str(v)) for v in row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    a = ap.parse_args()
    runs = load(a.results)
    data_dir, tab_dir = os.path.join(a.paper, "data"), os.path.join(a.paper, "tables")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # ---- selection by validation loss ------------------------------------------------
    sel = {}
    for task in TASKS:
        for opt in OPTS:
            cands = [(final(runs[(task, opt, e)])["val_loss"], e) for e in EPS if (task, opt, e) in runs]
            cands = [(v, e) for v, e in cands if v is not None and not math.isnan(v)]
            sel[(task, opt)] = min(cands)[1] if cands else None

    # ---- curves --------------------------------------------------------------------
    for task in TASKS:
        times = [t for t in GRID_MIN if any((task, o, e) in runs and any(h["train_time"] <= t * 60 + 1 for h in runs[(task, o, e)]["history"]) for o in OPTS for e in EPS)]
        for eps in EPS:
            rows = [[t, *[curve_value(runs[(task, o, eps)], t) if (task, o, eps) in runs else float("nan") for o in OPTS]] for t in times]
            write_csv(os.path.join(data_dir, f"curve_{task}_lr{eps}.csv"), ["time", *OPTS], rows)
        rows = [[t, *[curve_value(runs[(task, o, sel[(task, o)])], t) if sel[(task, o)] else float("nan") for o in OPTS]] for t in times]
        write_csv(os.path.join(data_dir, f"curve_{task}.csv"), ["time", *OPTS], rows)
        rows = [[float(e), *[final(runs[(task, o, e)])["test_loss"] if (task, o, e) in runs else float("nan") for o in OPTS]] for e in EPS]
        write_csv(os.path.join(data_dir, f"lr_{task}.csv"), ["eps", *OPTS], rows)

    # ---- main table: two rows per task (test loss; test accuracy % or bits per char) ----
    summary = {"selected_eps": {f"{t}/{o}": sel[(t, o)] for t in TASKS for o in OPTS}, "main": {}, "throughput": {}, "grid": {}}
    lines = ["% generated by scripts/collect_results.py -- rows: test loss, then test accuracy (%) or bits per character; best probe method per task in bold; dagger = diverged",
             "\\begin{tabular}{ll c cccccc c}", "\\toprule",
             " & & BP & \\multicolumn{6}{c}{Zero-order probe methods} & FF \\\\", "\\cmidrule(lr){4-9}",
             "Task & Metric & & SPSA & RS & NP & ANP & INP & DANP & \\\\", "\\midrule"]
    grid_txt = [f"{'task':18s} " + " ".join(f"{o:>8s}" for o in OPTS)]
    for ti, task in enumerate(TASKS):
        is_cls = task in ("mlp_mnist", "conv_cifar10")
        loss_cells, sec_cells, vals = [], [], {}
        for o in OPTS:
            e = sel[(task, o)]
            if e is None:
                loss_cells.append("--"); sec_cells.append("--"); vals[o] = float("nan"); continue
            r = runs[(task, o, e)]; fr = final(r)
            loss = fr["test_loss"]; vals[o] = loss
            dag = "$^\\dagger$" if r.get("diverged") else ""
            loss_cells.append(f"{fmt_loss(loss)}{dag}")
            sec = f"{100 * fr['test_acc']:.1f}" if is_cls else f"{loss / LN2:.2f}"
            if is_cls and fr.get("test_acc_goodness") is not None:
                sec += f" ({100 * fr['test_acc_goodness']:.1f})"
            sec_cells.append(sec)
            summary["main"][f"{task}/{o}"] = {"eps": e, "test_loss": loss, "test_acc": fr.get("test_acc"), "test_bpc": loss / LN2,
                                              "test_acc_goodness": fr.get("test_acc_goodness"),
                                              "val_loss": fr["val_loss"], "diverged": r.get("diverged", False)}
        probe_vals = {o: vals[o] for o in PROBE if not math.isnan(vals[o])}
        if probe_vals:
            i = OPTS.index(min(probe_vals, key=probe_vals.get))
            loss_cells[i] = f"\\textbf{{{loss_cells[i]}}}"
            sec_cells[i] = f"\\textbf{{{sec_cells[i]}}}"
        lines.append(f"{TASK_LABEL[task]} & test loss & " + " & ".join(loss_cells) + " \\\\")
        lines.append(f" & {'acc.\\ (\\%)' if is_cls else 'bits/char'} & " + " & ".join(sec_cells) + " \\\\")
        if ti < len(TASKS) - 1:
            lines.append("\\addlinespace[2pt]")
        grid_txt.append(f"{task:18s} " + " ".join(f"{fmt(vals[o]):>8s}" for o in OPTS))
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "stage1_main_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- throughput table -----------------------------------------------------------
    lines = ["% generated by scripts/collect_results.py -- steps/h and forward passes/h (thousands) of the selected eps",
             "\\begin{tabular}{ll cccccccc}", "\\toprule",
             "Task & Metric & BP & SPSA & RS & NP & ANP & INP & DANP & FF \\\\", "\\midrule"]
    for ti, task in enumerate(TASKS):
        st, fw = [], []
        for o in OPTS:
            e = sel[(task, o)]
            if e is None:
                st.append("--"); fw.append("--"); continue
            fr = final(runs[(task, o, e)])
            tt = max(fr["train_time"], 1e-9)
            sph = fr["step"] / tt * 3600 / 1000
            fph = fr.get("forward_passes", fr["step"] * runs[(task, o, e)]["config"].get("forward_passes_per_step", 1)) / tt * 3600 / 1000
            st.append(f"{sph:.1f}"); fw.append(f"{fph:.1f}")
            summary["throughput"][f"{task}/{o}"] = {"steps_per_hour": sph * 1000, "forward_passes_per_hour": fph * 1000, "steps": fr["step"], "train_time": tt}
        lines.append(f"{TASK_LABEL[task]} & steps/h (k) & " + " & ".join(st) + " \\\\")
        lines.append(" & fwd passes/h (k) & " + " & ".join(fw) + " \\\\")
        if ti < len(TASKS) - 1:
            lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "stage1_throughput_table.tex"), "w").write("\n".join(lines) + "\n")

    # ---- full grid ------------------------------------------------------------------
    lines = ["% generated by scripts/collect_results.py -- validation / test loss for every eps; selected eps in bold; $\\dagger$ = diverged (stopped early)",
             "\\begin{tabular}{ll ccc}", "\\toprule", " & & \\multicolumn{3}{c}{validation / test loss} \\\\", "\\cmidrule(lr){3-5}",
             "Task & Optimizer & $\\varepsilon=0.1$ & $\\varepsilon=0.01$ & $\\varepsilon=0.001$ \\\\", "\\midrule"]
    for ti, task in enumerate(TASKS):
        for oi, o in enumerate(OPTS):
            cells = []
            for e in EPS:
                r = runs.get((task, o, e))
                if r is None:
                    cells.append("--"); continue
                fr = final(r)
                c = f"{fmt_loss(fr['val_loss'])} / {fmt_loss(fr['test_loss'])}" + ("$^\\dagger$" if r.get("diverged") else "")
                if sel[(task, o)] == e:
                    c = f"\\textbf{{{c}}}"
                cells.append(c)
                summary["grid"][f"{task}/{o}/{e}"] = {"val_loss": fr["val_loss"], "test_loss": fr["test_loss"], "test_acc": fr.get("test_acc"),
                                                      "steps": fr["step"], "train_time": fr["train_time"], "diverged": r.get("diverged", False), "status": r.get("status")}
            lines.append(f"{TASK_LABEL[task] if oi == 0 else ''} & {OPT_LABEL[o]} & " + " & ".join(cells) + " \\\\")
        if ti < len(TASKS) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "full_grid.tex"), "w").write("\n".join(lines) + "\n")

    # ---- early-stopping oracle: best validation-loss evaluation over all eps and times ----
    summary["early_stop"] = {}
    lines = ["% generated by scripts/collect_results.py -- test loss / accuracy at the evaluation with the lowest validation loss over all eps and all 5-minute evaluations (early-stopping oracle); best probe method per task in bold",
             "\\begin{tabular}{ll c cccccc c}", "\\toprule",
             " & & BP & \\multicolumn{6}{c}{Zero-order probe methods} & FF \\\\", "\\cmidrule(lr){4-9}",
             "Task & Metric & & SPSA & RS & NP & ANP & INP & DANP & \\\\", "\\midrule"]
    grid_es = [f"{'task':18s} " + " ".join(f"{o:>8s}" for o in OPTS)]
    for ti, task in enumerate(TASKS):
        is_cls = task in ("mlp_mnist", "conv_cifar10")
        loss_cells, sec_cells, at_cells, vals = [], [], [], {}
        for o in OPTS:
            best = None
            for e in EPS:
                r = runs.get((task, o, e))
                if r is None:
                    continue
                for h in r["history"]:
                    v = h.get("val_loss")
                    if v is None or math.isnan(v) or h.get("test_loss") is None or math.isnan(h["test_loss"]):
                        continue
                    if best is None or v < best[0]:
                        best = (v, e, h)
            if best is None:
                loss_cells.append("--"); sec_cells.append("--"); at_cells.append("--"); vals[o] = float("nan"); continue
            v, e, h = best
            vals[o] = h["test_loss"]
            loss_cells.append(fmt_loss(h["test_loss"]))
            sec = f"{100 * h['test_acc']:.1f}" if is_cls else f"{h['test_loss'] / LN2:.2f}"
            if is_cls and h.get("test_acc_goodness") is not None:
                sec += f" ({100 * h['test_acc_goodness']:.1f})"
            sec_cells.append(sec)
            at_cells.append(f"{e}/{h['train_time'] / 60:.0f}")
            summary["early_stop"][f"{task}/{o}"] = {"eps": e, "minute": h["train_time"] / 60, "step": h["step"], "val_loss": v,
                                                    "test_loss": h["test_loss"], "test_acc": h.get("test_acc"), "test_bpc": h["test_loss"] / LN2,
                                                    "test_acc_goodness": h.get("test_acc_goodness")}
        probe_vals = {o: vals[o] for o in PROBE if not math.isnan(vals[o])}
        if probe_vals:
            i = OPTS.index(min(probe_vals, key=probe_vals.get))
            loss_cells[i] = f"\\textbf{{{loss_cells[i]}}}"
            sec_cells[i] = f"\\textbf{{{sec_cells[i]}}}"
        lines.append(f"{TASK_LABEL[task]} & test loss & " + " & ".join(loss_cells) + " \\\\")
        lines.append(f" & {'acc.\\ (\\%)' if is_cls else 'bits/char'} & " + " & ".join(sec_cells) + " \\\\")
        lines.append(" & $\\varepsilon$/min & " + " & ".join(at_cells) + " \\\\")
        if ti < len(TASKS) - 1:
            lines.append("\\addlinespace[2pt]")
        grid_es.append(f"{task:18s} " + " ".join(f"{fmt(vals[o]):>8s}" for o in OPTS))
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "early_stop_table.tex"), "w").write("\n".join(lines) + "\n")

    json.dump(summary, open(os.path.join(a.results, "summary.json"), "w"), indent=1)
    print(f"{len(runs)} runs loaded; selected eps per (task, opt):")
    for task in TASKS:
        print(f"  {task:18s} " + " ".join(f"{o}={sel[(task, o)] or '-':>5s}" for o in OPTS))
    print("\nfinal test loss (selected eps):")
    print("\n".join(grid_txt))
    print("\nearly-stopping oracle test loss (best val-loss evaluation over all eps and times):")
    print("\n".join(grid_es))


if __name__ == "__main__":
    main()
