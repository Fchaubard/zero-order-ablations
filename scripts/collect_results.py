#!/usr/bin/env python3
"""Turn results/*.json into the paper's CSVs and LaTeX tables (and a plain-text grid).

    data/curve_{task}.csv            time(min),bp,spsa,rs,np,anp,inp,danp,ff   test loss of the
                                     selected eps (lowest final validation loss) per optimizer
    data/curve_{task}_lr{eps}.csv    same columns, one file per eps
    data/lr_{task}.csv               eps,bp,...,ff  final test loss per eps (nan if missing)
    tables/main_table.tex            Table: test loss (test acc % or bpc), best probe method bold
    tables/throughput_table.tex      steps/h and forward passes/h (thousands)
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
    """Test loss at the last evaluation with train_time <= t_min minutes (+1 s slack);
    nan if the run had not reached t_min (or stopped before it: diverged runs end early)."""
    hist = r["history"]
    t = t_min * 60 + 1.0
    done = r.get("status") in ("done", "diverged") or r.get("final") is not None
    vals = [h for h in hist if h["train_time"] <= t]
    if not vals:
        return float("nan")
    last = vals[-1]
    if last is not hist[-1] or not done:
        return last["test_loss"]
    # the run's last eval: hold its value only at its own time, nan afterwards if it stopped early
    return last["test_loss"] if last["train_time"] >= t_min * 60 - 1.0 or not r.get("diverged") else float("nan")


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

    # ---- main table -----------------------------------------------------------------
    summary = {"selected_eps": {f"{t}/{o}": sel[(t, o)] for t in TASKS for o in OPTS}, "main": {}, "throughput": {}, "grid": {}}
    lines = ["% generated by scripts/collect_results.py -- test loss (test accuracy % | bits per char); best probe method per row in bold",
             "\\begin{tabular}{l c cccccc c}", "\\toprule",
             " & BP & \\multicolumn{6}{c}{Zero-order probe methods} & FF \\\\", "\\cmidrule(lr){3-8}",
             "Task & & SPSA & RS & NP & ANP & INP & DANP & \\\\", "\\midrule"]
    grid_txt = [f"{'task':18s} " + " ".join(f"{o:>8s}" for o in OPTS)]
    for task in TASKS:
        cells, vals = [], {}
        for o in OPTS:
            e = sel[(task, o)]
            if e is None:
                cells.append("--"); vals[o] = float("nan"); continue
            fr = final(runs[(task, o, e)])
            loss = fr["test_loss"]; vals[o] = loss
            second = f"{100 * fr['test_acc']:.1f}\\%" if fr.get("test_acc") is not None and task in ("mlp_mnist", "conv_cifar10") else f"{loss / LN2:.2f}"
            cells.append(f"{loss:.3f} ({second})")
            summary["main"][f"{task}/{o}"] = {"eps": e, "test_loss": loss, "test_acc": fr.get("test_acc"), "test_bpc": loss / LN2,
                                              "val_loss": fr["val_loss"], "diverged": runs[(task, o, e)].get("diverged", False)}
        probe_vals = {o: vals[o] for o in PROBE if not math.isnan(vals[o])}
        if probe_vals:
            best = min(probe_vals, key=probe_vals.get)
            i = OPTS.index(best)
            cells[i] = f"\\textbf{{{cells[i]}}}"
        lines.append(f"{TASK_LABEL[task]} & " + " & ".join(cells) + " \\\\")
        grid_txt.append(f"{task:18s} " + " ".join(f"{fmt(vals[o]):>8s}" for o in OPTS))
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(tab_dir, "main_table.tex"), "w").write("\n".join(lines) + "\n")

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
    open(os.path.join(tab_dir, "throughput_table.tex"), "w").write("\n".join(lines) + "\n")

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
                c = f"{fr['val_loss']:.3f} / {fr['test_loss']:.3f}" + ("$^\\dagger$" if r.get("diverged") else "")
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

    json.dump(summary, open(os.path.join(a.results, "summary.json"), "w"), indent=1)
    print(f"{len(runs)} runs loaded; selected eps per (task, opt):")
    for task in TASKS:
        print(f"  {task:18s} " + " ".join(f"{o}={sel[(task, o)] or '-':>5s}" for o in OPTS))
    print("\nfinal test loss (selected eps):")
    print("\n".join(grid_txt))


if __name__ == "__main__":
    main()
