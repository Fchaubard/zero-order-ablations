"""Sweep orchestration: tasks x optimizers x epsilons over a pool of GPUs.

    python -m zoo.sweep --gpus 0,1,2,3,4,5,6,7 --tasks all --opts all --lrs 0.1,0.01,0.001 \
        --budget 3600 --eval-every 300 --out results --logdir runs [--dry-run] [--order interleaved]

* One worker thread per GPU pulls configurations from a shared queue and runs each as a
  ``python -m zoo.train ...`` subprocess with ``CUDA_VISIBLE_DEVICES`` set to that GPU;
  stdout+stderr go to ``runs/{name}.log``.
* Resumable: a config whose results JSON already has a ``final`` block and no ``error``
  is skipped (``--force`` reruns everything).
* ``--gpus cpu,cpu`` runs two CPU workers (CUDA_VISIBLE_DEVICES="") - handy for tests.
"""
from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple

from .common import DEFAULT_LRS, TASK_ORDER, fmt_lr, fmt_seconds, is_complete, load_json, result_path, run_name
from .optimizers import ALL_OPTIMIZERS, EXTRA_OPTIMIZERS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDERS = ("interleaved", "task-major", "opt-major")


def _csv(s: str, allowed: List[str], what: str) -> List[str]:
    if s.strip().lower() == "all":
        return list(allowed)
    items = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise SystemExit(f"unknown {what}: {bad} (choose from {allowed})")
    return items


def enumerate_configs(tasks: List[str], opts: List[str], lrs: List[float], order: str,
                      seeds: List[int] = (0,)) -> List[Tuple[str, str, float, int]]:
    """``interleaved`` = sort by (seed, lr, opt, task): consecutive configs round-robin over the
    tasks, so every batch of 8 covers all 4 tasks and 2 optimizers at the same epsilon."""
    if order == "task-major":
        return [(t, o, lr, s) for s in seeds for t in tasks for o in opts for lr in lrs]
    if order == "opt-major":
        return [(t, o, lr, s) for s in seeds for o in opts for t in tasks for lr in lrs]
    if order == "interleaved":
        return [(t, o, lr, s) for s in seeds for lr in lrs for o in opts for t in tasks]
    raise ValueError(order)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", default="0", help="comma-separated GPU ids, one worker each ('cpu' = CPU worker)")
    p.add_argument("--tasks", default="all", help=f"'all' or comma list of {TASK_ORDER}")
    p.add_argument("--opts", default="all", help=f"'all' or comma list of {ALL_OPTIMIZERS}")
    p.add_argument("--lrs", default=",".join(fmt_lr(x) for x in DEFAULT_LRS))
    p.add_argument("--budget", type=float, default=3600.0)
    p.add_argument("--eval-every", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0, help="single seed (ignored when --seeds is given)")
    p.add_argument("--seeds", default=None, help="comma list of seeds, e.g. 0,1,2 (seed 0 keeps the plain run name)")
    p.add_argument("--out", default="results")
    p.add_argument("--logdir", default="runs")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--order", choices=ORDERS, default="interleaved")
    p.add_argument("--wandb-project", default="zoo-survey")
    p.add_argument("--no-wandb", action="store_true", help="pass --no-wandb to every run")
    p.add_argument("--force", action="store_true", help="rerun configs that already have a complete result")
    p.add_argument("--dry-run", action="store_true", help="print the configs / commands and exit")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--synthetic", action="store_true", help="pass --synthetic to every run (smoke test)")
    p.add_argument("--train-arg", action="append", default=[], help="extra argument forwarded to zoo.train (repeatable)")
    p.add_argument("--extra-jobs", default=None,
                   help="job-list mode only: a second file that is re-read whenever a worker needs a job; its "
                        "not-yet-started lines take precedence over the main queue (lets jobs be added to a running sweep)")
    p.add_argument("--jobs", default=None,
                   help="file with one zoo.train argument line per run (e.g. '--task mlp_mnist --opt np --lr 0.001 "
                        "--seed 1 --sigma 0.01 --out results_decouple'); replaces the tasks x opts x lrs grid. "
                        "--budget/--eval-every/--out/--data-root/--wandb-project are defaults a line may override.")
    return p.parse_args(argv)


def build_cmd(args, task: str, opt: str, lr: float, seed: int = 0) -> List[str]:
    cmd = [args.python, "-m", "zoo.train", "--task", task, "--opt", opt, "--lr", fmt_lr(lr),
           "--budget", f"{args.budget:g}", "--eval-every", f"{args.eval_every:g}", "--seed", str(seed),
           "--out", args.out, "--data-root", args.data_root, "--wandb-project", args.wandb_project]
    if args.no_wandb:
        cmd.append("--no-wandb")
    if args.synthetic:
        cmd.append("--synthetic")
    cmd.extend(args.train_arg)
    return cmd


def _job_argv(args, line: str) -> List[str]:
    """Sweep defaults first, then the job line (argparse keeps the last occurrence)."""
    import shlex
    base = ["--budget", f"{args.budget:g}", "--eval-every", f"{args.eval_every:g}", "--out", args.out,
            "--data-root", args.data_root, "--wandb-project", args.wandb_project]
    if args.no_wandb:
        base.append("--no-wandb")
    if args.synthetic:
        base.append("--synthetic")
    return base + args.train_arg + shlex.split(line)


def read_jobs(args, path: str):
    """Returns [(line_no, argv, result_path, run_name)] for the non-empty, non-comment lines."""
    from .train import name_from_args, parse_args as train_parse_args
    jobs = []
    with open(path) as f:
        for i, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            argv = _job_argv(args, line)
            targs = train_parse_args(argv)
            name = name_from_args(targs)
            jobs.append((i, argv, os.path.join(targs.out, name + ".json"), name))
    return jobs


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _final_summary(path: str) -> Tuple[str, dict]:
    r = load_json(path)
    if not r:
        return "no result", {}
    rec = r.get("final") or (r["history"][-1] if r.get("history") else {})
    bits = []
    if "test_loss" in rec:
        bits.append(f"test_loss={rec['test_loss']:.4f}")
        if "test_bpc" in rec:
            bits.append(f"bpc={rec['test_bpc']:.3f}")
        elif "test_acc" in rec:
            bits.append(f"acc={rec['test_acc']:.4f}")
        bits.append(f"steps={rec.get('step', '?')}")
    if r.get("diverged"):
        bits.append("DIVERGED")
    if r.get("error"):
        bits.append(f"ERROR: {str(r['error'])[:80]}")
    if "final" not in r and not r.get("error"):
        bits.append("(partial)")
    return " ".join(bits) if bits else r.get("status", "?"), rec


class Sweep:
    def __init__(self, args, configs):
        self.args = args
        self.q: "queue.Queue[Tuple[int, str, str, float, int]]" = queue.Queue()
        for i, c in enumerate(configs):
            self.q.put((i, *c))
        self.total = len(configs)
        self.lock = threading.Lock()
        self.procs: List[subprocess.Popen] = []
        self.rows: List[dict] = []
        self.stop = threading.Event()
        self.started = set()          # names of jobs taken from --extra-jobs

    def say(self, msg: str):
        with self.lock:
            print(f"[{_now()}] {msg}", flush=True)

    def run_one(self, gpu: str, idx: int, task: str, opt: str, lr: float, seed: int = 0):
        name = run_name(task, opt, lr, seed)
        cmd = build_cmd(self.args, task, opt, lr, seed)
        self._run_cmd(gpu, idx, name, cmd, result_path(self.args.out, task, opt, lr, seed))

    def run_job(self, gpu: str, idx: int, argv: List[str], path: str, name: str):
        self._run_cmd(gpu, idx, name, [self.args.python, "-m", "zoo.train", *argv], path)

    def _run_cmd(self, gpu: str, idx: int, name: str, cmd: List[str], path: str):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else gpu
        env.setdefault("PYTHONUNBUFFERED", "1")
        log_path = os.path.join(self.args.logdir, name + ".log")
        self.say(f"START  gpu{gpu:<3} {name:34s} ({idx + 1}/{self.total}) -> {log_path}")
        t0 = time.time()
        with open(log_path, "a") as log:
            log.write(f"# {_now()} {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            with self.lock:
                self.procs.append(proc)
            rc = proc.wait()
            with self.lock:
                self.procs.remove(proc)
        elapsed = time.time() - t0
        summary, rec = _final_summary(path)
        self.say(f"DONE   gpu{gpu:<3} {name:34s} exit={rc} {fmt_seconds(elapsed):>9s} {summary}")
        with self.lock:
            self.rows.append({"idx": idx, "name": name, "gpu": gpu, "exit": rc, "elapsed": elapsed,
                              "summary": summary, **{k: rec.get(k) for k in ("step", "val_loss", "test_loss", "test_acc", "test_bpc")}})

    def _extra_job(self):
        """Next not-yet-started job of --extra-jobs (re-read on every call), or None."""
        path = getattr(self.args, "extra_jobs", None)
        if not path or not os.path.exists(path):
            return None
        with self.lock:
            try:
                jobs = read_jobs(self.args, path)
            except Exception as e:  # noqa: BLE001
                self.say(f"could not read {path}: {e!r}")
                return None
            for i, argv, rpath, name in jobs:
                if name in self.started or is_complete(load_json(rpath)):
                    continue
                self.started.add(name)
                self.total += 1
                return (self.total - 1, argv, rpath, name)
        return None

    def worker(self, gpu: str):
        while not self.stop.is_set():
            item = self._extra_job()
            if item is None:
                try:
                    item = self.q.get_nowait()
                except queue.Empty:
                    return
            try:
                if len(item) == 4 and isinstance(item[1], list):      # job-list mode: (idx, argv, path, name)
                    self.run_job(gpu, *item)
                else:
                    idx, task, opt, lr, seed = item
                    self.run_one(gpu, idx, task, opt, lr, seed)
            except Exception as e:  # noqa: BLE001
                self.say(f"worker gpu{gpu} crashed on {item}: {e!r}")

    def terminate(self):
        self.stop.set()
        with self.lock:
            procs = list(self.procs)
        for p in procs:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(2)
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass

    def summary_table(self) -> str:
        rows = sorted(self.rows, key=lambda r: r["idx"])
        head = f"{'name':34s} {'gpu':>4s} {'exit':>4s} {'elapsed':>9s} {'steps':>8s} {'val_loss':>9s} {'test_loss':>9s} {'acc/bpc':>8s}  status"
        lines = [head, "-" * len(head)]
        for r in rows:
            m = r.get("test_bpc") if r.get("test_bpc") is not None else r.get("test_acc")
            fm = lambda v, f: (f % v) if isinstance(v, (int, float)) else "-"  # noqa: E731
            status = "ok" if r["exit"] == 0 else "FAILED"
            if "DIVERGED" in r["summary"]:
                status = "diverged"
            lines.append(f"{r['name']:34s} {r['gpu']:>4s} {r['exit']:>4d} {fmt_seconds(r['elapsed']):>9s} "
                         f"{fm(r.get('step'), '%8d'):>8s} {fm(r.get('val_loss'), '%9.4f'):>9s} "
                         f"{fm(r.get('test_loss'), '%9.4f'):>9s} {fm(m, '%8.4f'):>8s}  {status}")
        n_fail = sum(1 for r in rows if r["exit"] != 0)
        lines.append(f"{len(rows)} runs finished, {n_fail} failed")
        return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = _csv(args.tasks, TASK_ORDER, "task")
    # "all" = the protocol's 8 optimizers; ablation optimizers (EXTRA_OPTIMIZERS) only when named
    opts = ALL_OPTIMIZERS if args.opts.strip().lower() == "all" else _csv(args.opts, ALL_OPTIMIZERS + EXTRA_OPTIMIZERS, "optimizer")
    try:
        lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"bad --lrs {args.lrs!r}")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--gpus must list at least one GPU id (or 'cpu')")

    if args.jobs:
        jobs = read_jobs(args, args.jobs)
        todo = [(i, argv, path, name) for i, argv, path, name in jobs if args.force or not is_complete(load_json(path))]
        done = {name for _, _, _, name in jobs} - {name for _, _, _, name in todo}
        print(f"{len(jobs)} jobs in {args.jobs}; {len(done)} already complete, {len(todo)} to run on {len(gpus)} worker(s): {gpus}", flush=True)
        for i, argv, path, name in jobs:
            print(f"[{i:3d}] {'run ' if name not in done else 'skip'} {name}  ->  {path}")
        if args.dry_run:
            if todo:
                print("example command:", " ".join([args.python, "-m", "zoo.train", *todo[0][1]]))
            print(f"dry run: {len(todo)} jobs would run, {len(done)} skipped")
            return 0
        if not todo:
            print("nothing to do")
            return 0
        os.makedirs(args.logdir, exist_ok=True)
        sweep = Sweep(args, [])
        for j, (i, argv, path, name) in enumerate(todo):
            sweep.q.put((j, argv, path, name))
        sweep.total = len(todo)
        return _run_sweep(args, sweep, gpus)

    try:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds else [args.seed]
    except ValueError:
        raise SystemExit(f"bad --seeds {args.seeds!r}")
    all_configs = enumerate_configs(tasks, opts, lrs, args.order, seeds)
    todo, skipped = [], []
    for c in all_configs:
        path = result_path(args.out, *c)
        if not args.force and is_complete(load_json(path)):
            skipped.append(c)
        else:
            todo.append(c)
    print(f"{len(all_configs)} configs ({len(tasks)} tasks x {len(opts)} opts x {len(lrs)} lrs x {len(seeds)} seeds, order={args.order}); "
          f"{len(skipped)} already complete in {args.out}/, {len(todo)} to run on {len(gpus)} worker(s): {gpus}",
          flush=True)
    for i, c in enumerate(all_configs):
        tag = "skip" if c in skipped else "run "
        print(f"[{i + 1:3d}/{len(all_configs)}] {tag} {run_name(*c)}")
    if args.dry_run:
        if todo:
            print("example command:", " ".join(build_cmd(args, *todo[0])))
        print(f"dry run: {len(todo)} configs would run, {len(skipped)} skipped")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    sweep = Sweep(args, todo)
    return _run_sweep(args, sweep, gpus)


def _run_sweep(args, sweep: "Sweep", gpus: List[str]) -> int:
    threads = [threading.Thread(target=sweep.worker, args=(g,), daemon=True, name=f"gpu{g}") for g in gpus]
    t0 = time.time()
    for t in threads:
        t.start()
        time.sleep(0.2)   # stagger start-up a little
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        sweep.say("interrupted: terminating running jobs ...")
        sweep.terminate()
        for t in threads:
            t.join(timeout=5)
        print(sweep.summary_table())
        return 130
    print(f"\nsweep finished in {fmt_seconds(time.time() - t0)}")
    print(sweep.summary_table())
    print(f"status table: {args.python} -m zoo.status --out {args.out} --grid")
    return 1 if any(r["exit"] != 0 for r in sweep.rows) else 0


if __name__ == "__main__":
    sys.exit(main())
