#!/usr/bin/env python3
"""Write placeholder CSVs and LaTeX tables so that the paper skeleton compiles.

Everything written here is SYNTHETIC.  The real script (to be written later) will read
results/{task}-{opt}-lr{eps}.json and overwrite the same files with real numbers:

    data/curve_{task}.csv            time,bp,spsa,rs,np,anp,inp,danp,ff   (selected eps)
    data/curve_{task}_lr{eps}.csv    same columns, one file per eps       (Appendix C)
    data/lr_{task}.csv               eps,bp,spsa,...,ff  (final test loss per eps)
    tables/main_table.tex            Table 1 body (tabular)
    tables/throughput_table.tex      Table 2 body (tabular)
    tables/full_grid.tex             Table 3 body (tabular)

Run from anywhere:  python3 paper/make_dummy_data.py
"""
from __future__ import annotations

import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TABLES = os.path.join(HERE, "tables")

TASKS = ["mlp_mnist", "conv_cifar10", "lstm_ptb", "transformer_ptb"]
OPTS = ["bp", "spsa", "rs", "np", "anp", "inp", "danp", "ff"]
EPS = ["0.1", "0.01", "0.001"]

TASK_LABEL = {
    "mlp_mnist": "MNIST / MLP",
    "conv_cifar10": "CIFAR-10 / ConvNet",
    "lstm_ptb": "PTB / char-LSTM",
    "transformer_ptb": "PTB / char-Transformer",
}
OPT_LABEL = {o: o.upper() for o in OPTS}
TIMES = list(range(0, 61, 5))          # evaluation every 5 min, plus t = 0 and t = 60

# synthetic decay rates (per minute) so that the dummy curves are distinguishable
RATE = {"bp": 0.12, "spsa": 0.02, "rs": 0.015, "np": 0.03, "anp": 0.035,
        "inp": 0.025, "danp": 0.045, "ff": 0.06}
FLOOR = {"bp": 0.10, "spsa": 0.9, "rs": 1.1, "np": 0.7, "anp": 0.6,
         "inp": 0.8, "danp": 0.5, "ff": 0.4}
L0 = {"mlp_mnist": 2.30, "conv_cifar10": 2.30, "lstm_ptb": 3.90, "transformer_ptb": 3.90}


def dummy_curve(task: str, opt: str, eps: str, rng: random.Random) -> list[float]:
    scale = {"0.1": 1.3, "0.01": 1.0, "0.001": 0.5}[eps]
    rate = RATE[opt] * scale
    floor = FLOOR[opt] * (L0[task] / 2.3)
    return [floor + (L0[task] - floor) * math.exp(-rate * t) * (1 + 0.02 * rng.random())
            for t in TIMES]


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    rng = random.Random(0)
    header = ["time", *OPTS]

    for task in TASKS:
        curves = {eps: {o: dummy_curve(task, o, eps, rng) for o in OPTS} for eps in EPS}
        # one file per eps (Appendix C)
        for eps in EPS:
            rows = [[t, *[curves[eps][o][i] for o in OPTS]] for i, t in enumerate(TIMES)]
            write_csv(os.path.join(DATA, f"curve_{task}_lr{eps}.csv"), header, rows)
        # "selected" eps = 0.01 for the dummy main figure
        rows = [[t, *[curves["0.01"][o][i] for o in OPTS]] for i, t in enumerate(TIMES)]
        write_csv(os.path.join(DATA, f"curve_{task}.csv"), header, rows)
        # final test loss vs eps
        rows = [[float(eps), *[curves[eps][o][-1] for o in OPTS]] for eps in EPS]
        write_csv(os.path.join(DATA, f"lr_{task}.csv"), ["eps", *OPTS], rows)

    # ---- Table 1: main table ------------------------------------------------------
    with open(os.path.join(TABLES, "main_table.tex"), "w") as f:
        f.write("% PLACEHOLDER written by make_dummy_data.py -- entries are loss (acc or bpc).\n")
        f.write("% Best zero-order probe method per row should be wrapped in \\textbf{}.\n")
        f.write("\\begin{tabular}{l c cccccc c}\n\\toprule\n")
        f.write(" & BP & \\multicolumn{6}{c}{Zero-order probe methods} & FF \\\\\n")
        f.write("\\cmidrule(lr){3-8}\n")
        f.write("Task & & SPSA & RS & NP & ANP & INP & DANP & \\\\\n\\midrule\n")
        for task in TASKS:
            cells = " & ".join("-- (--)" for _ in OPTS)
            f.write(f"{TASK_LABEL[task]} & {cells} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Table 2: throughput ------------------------------------------------------
    with open(os.path.join(TABLES, "throughput_table.tex"), "w") as f:
        f.write("% PLACEHOLDER written by make_dummy_data.py -- steps/h and forward passes/h (thousands).\n")
        f.write("\\begin{tabular}{ll cccccccc}\n\\toprule\n")
        f.write("Task & Metric & " + " & ".join(OPT_LABEL[o] for o in OPTS) + " \\\\\n\\midrule\n")
        for i, task in enumerate(TASKS):
            cells = " & ".join("--" for _ in OPTS)
            f.write(f"{TASK_LABEL[task]} & steps/h (k) & {cells} \\\\\n")
            f.write(f" & fwd passes/h (k) & {cells} \\\\\n")
            if i < len(TASKS) - 1:
                f.write("\\addlinespace[2pt]\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- Table 3: full 96-run grid -------------------------------------------------
    with open(os.path.join(TABLES, "full_grid.tex"), "w") as f:
        f.write("% PLACEHOLDER written by make_dummy_data.py -- val / test loss for every eps.\n")
        f.write("% The selected eps per row (lowest val loss) should be wrapped in \\textbf{}.\n")
        f.write("\\begin{tabular}{ll ccc}\n\\toprule\n")
        f.write(" & & \\multicolumn{3}{c}{validation / test loss} \\\\\n")
        f.write("\\cmidrule(lr){3-5}\n")
        f.write("Task & Optimizer & $\\varepsilon=0.1$ & $\\varepsilon=0.01$ & $\\varepsilon=0.001$ \\\\\n\\midrule\n")
        for i, task in enumerate(TASKS):
            for j, o in enumerate(OPTS):
                lead = TASK_LABEL[task] if j == 0 else ""
                cells = " & ".join("-- / --" for _ in EPS)
                f.write(f"{lead} & {OPT_LABEL[o]} & {cells} \\\\\n")
            if i < len(TASKS) - 1:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"wrote dummy CSVs to {DATA} and placeholder tables to {TABLES}")


if __name__ == "__main__":
    main()
