"""Summarise the results JSONs written by zoo.train.

    python -m zoo.status --out results            # table of every run + best epsilon per (task, opt)
    python -m zoo.status --out results --grid     # 4x8 grid of the selected runs' test loss
    python -m zoo.status --out results --grid --csv grid.csv --metric test_acc

Model selection follows PROTOCOL.md: per (task, optimizer) the epsilon with the lowest final
*validation* loss among completed runs (partial runs are only used when nothing is complete,
and are marked with '*'; diverged runs are marked '!').
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

from .common import TASK_ORDER, fmt_lr, is_complete, load_json
from .optimizers import ALL_OPTIMIZERS

METRICS = ("test_loss", "test_acc", "test_bpc", "val_loss", "val_acc", "val_bpc", "step", "train_loss")


def load_results(out_dir: str) -> List[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        r = load_json(path)
        if not r or "config" not in r:
            continue
        cfg = r["config"]
        rec = r.get("final") or (r["history"][-1] if r.get("history") else {})
        complete = is_complete(r)
        if r.get("error"):
            status = "error"
        elif r.get("diverged"):
            status = "diverged"
        elif complete:
            status = "done"
        else:
            status = r.get("status", "partial")
        rows.append({
            "path": path, "task": cfg.get("task"), "opt": cfg.get("opt"), "lr": float(cfg.get("lr", float("nan"))),
            "complete": complete, "diverged": bool(r.get("diverged")), "error": r.get("error"), "status": status,
            "train_time": r.get("train_time", rec.get("train_time")), "n_evals": len(r.get("history", [])),
            **{k: rec.get(k) for k in METRICS},
        })
    return rows


def _order_key(r: dict) -> Tuple[int, int, float]:
    t = TASK_ORDER.index(r["task"]) if r["task"] in TASK_ORDER else len(TASK_ORDER)
    o = ALL_OPTIMIZERS.index(r["opt"]) if r["opt"] in ALL_OPTIMIZERS else len(ALL_OPTIMIZERS)
    return (t, o, -r["lr"])


def _f(v, fmt: str = "%.4f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    try:
        return fmt % v
    except TypeError:
        return str(v)


def table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    lines = [fmt.format(*headers), "  ".join("-" * w for w in widths)]
    lines += [fmt.format(*r) for r in rows]
    return "\n".join(lines)


def runs_table(rows: List[dict]) -> str:
    out = []
    for r in sorted(rows, key=_order_key):
        err = (str(r["error"])[:40] + "...") if r["error"] and len(str(r["error"])) > 40 else (r["error"] or "")
        out.append([r["task"], r["opt"], fmt_lr(r["lr"]), _f(r["step"], "%d"), _f(r["train_time"], "%.0f"),
                    _f(r["val_loss"]), _f(r["val_acc"]), _f(r["test_loss"]), _f(r["test_acc"]),
                    _f(r["test_bpc"], "%.3f"), "yes" if r["diverged"] else "", r["status"], err])
    return table(["task", "opt", "lr", "steps", "t_train", "val_loss", "val_acc", "test_loss", "test_acc",
                  "bpc", "div", "status", "error"], out)


def select_best(rows: List[dict]) -> Dict[Tuple[str, str], dict]:
    """Lowest final validation loss per (task, opt); completed runs beat partial ones."""
    best: Dict[Tuple[str, str], dict] = {}
    for r in rows:
        if r["val_loss"] is None or r["error"]:
            continue
        v = r["val_loss"]
        if isinstance(v, float) and math.isnan(v):
            continue
        key = (r["task"], r["opt"])
        cur = best.get(key)
        rank = (0 if r["complete"] else 1, v)
        if cur is None or rank < (0 if cur["complete"] else 1, cur["val_loss"]):
            best[key] = r
    return best


def best_table(best: Dict[Tuple[str, str], dict]) -> str:
    out = []
    for (task, opt), r in sorted(best.items(), key=lambda kv: _order_key(kv[1])):
        mark = "" if r["complete"] else "*"
        mark += "!" if r["diverged"] else ""
        out.append([task, opt, fmt_lr(r["lr"]) + mark, _f(r["val_loss"]), _f(r["test_loss"]),
                    _f(r["test_acc"]), _f(r["test_bpc"], "%.3f"), _f(r["step"], "%d")])
    return table(["task", "opt", "best_lr", "val_loss", "test_loss", "test_acc", "test_bpc", "steps"], out)


def grid(best: Dict[Tuple[str, str], dict], metric: str, tasks=TASK_ORDER, opts=ALL_OPTIMIZERS) -> List[List[str]]:
    """rows = tasks, columns = optimizers; cell = metric of the selected run (* partial, ! diverged)."""
    fmt = "%d" if metric == "step" else ("%.3f" if metric.endswith("bpc") else "%.4f")
    out = [["task", *opts]]
    for t in tasks:
        row = [t]
        for o in opts:
            r = best.get((t, o))
            if r is None or r.get(metric) is None:
                row.append("-")
            else:
                row.append(_f(r[metric], fmt) + ("" if r["complete"] else "*") + ("!" if r["diverged"] else ""))
        out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--grid", action="store_true", help="print the tasks x optimizers grid")
    ap.add_argument("--metric", default="test_loss", choices=METRICS, help="grid cell metric")
    ap.add_argument("--csv", default=None, help="write the grid to this CSV file")
    ap.add_argument("--quiet", action="store_true", help="skip the per-run table")
    args = ap.parse_args(argv)

    rows = load_results(args.out)
    if not rows:
        print(f"no results in {args.out}/")
        return 1
    n_done = sum(r["complete"] for r in rows)
    n_err = sum(1 for r in rows if r["error"])
    n_div = sum(r["diverged"] for r in rows)
    if not args.quiet:
        print(runs_table(rows))
        print()
    print(f"{len(rows)} result files: {n_done} complete, {n_div} diverged, {n_err} errors, "
          f"{len(rows) - n_done - n_err} partial/running\n")
    best = select_best(rows)
    print("best epsilon per (task, optimizer) by final validation loss  (* partial run, ! diverged):")
    print(best_table(best))
    if args.grid or args.csv:
        g = grid(best, args.metric)
        print(f"\n{args.metric} grid (rows: tasks, cols: optimizers; best-by-val epsilon):")
        print(table(g[0], g[1:]))
        if args.csv:
            with open(args.csv, "w", newline="") as f:
                csv.writer(f).writerows(g)
            print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
