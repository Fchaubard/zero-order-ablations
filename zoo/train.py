"""Train one (task, optimizer, epsilon) configuration under the PROTOCOL.md budget.

    python -m zoo.train --task mlp_mnist --opt bp --lr 0.01 --budget 3600 --eval-every 300 \
        --seed 0 --out results --wandb --wandb-project zoo-survey --data-root ./data

* ``train_time`` counts ONLY the seconds spent inside ``opt.step`` (measured in chunks of
  20 steps, with ``torch.cuda.synchronize()`` before reading the clock on CUDA), so data
  sampling, evaluation and logging are excluded from the budget.
* Every ``--eval-every`` seconds of train time (and once at the end) the model is evaluated
  on the full validation and test splits through ``opt.predict``.
* Divergence watchdog: a non-finite loss, or a training loss > 2x its initial value, or a
  validation loss > 2x its step-0 value, at two consecutive evaluations, stops the run and
  marks it ``diverged`` (the step-0 validation loss is the same clean quantity for every
  optimizer; the training loss of SPSA is a probe mean and is inflated at large epsilon).
* ``results/{task}-{opt}-lr{lr}.json`` is rewritten atomically after every evaluation
  (``history`` grows; ``final`` is only added when the run completes), so a partial result
  survives a crash.  On an exception the JSON carries ``error`` + ``traceback`` and the
  process exits with status 1.
* wandb: run ``{task}-{opt}-lr{lr}`` in group ``{task}``; every metric uses ``train_time``
  as its x-axis.  The API key comes from ``WANDB_API_KEY``; without it (or with
  ``--no-wandb``) the run goes offline/disabled without crashing.
"""
from __future__ import annotations

import argparse
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
import traceback
from typing import Optional

import numpy as np
import torch

from .common import atomic_write_json, fmt_lr, run_name
from .data import Data
from .models import TASKS, build_model
from .optimizers import ALL_OPTIMIZERS, get_optimizer, per_sample_loss
from .optimizers.base import ADAM_BETAS
from .perturb import count_params, ctx, players

LN2 = math.log(2.0)
DIVERGE_FACTOR = 2.0   # PROTOCOL.md: train loss > 2x its initial value at two consecutive evaluations -> diverged
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- utilities
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--opt", required=True, choices=ALL_OPTIMIZERS)
    p.add_argument("--lr", type=float, required=True, help="learning rate == epsilon (probe scale)")
    p.add_argument("--budget", type=float, default=3600.0, help="seconds of train time (opt.step only)")
    p.add_argument("--eval-every", type=float, default=300.0, help="seconds of train time between evaluations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results", help="directory for the results JSON")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", default="zoo-survey")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--device", default=None, help="override (default: cuda if available else cpu)")
    p.add_argument("--chunk", type=int, default=20, help="steps per timing chunk")
    p.add_argument("--log-every", type=int, default=100, help="steps between train-stat log lines")
    p.add_argument("--synthetic", action="store_true", help="random data of the right shapes (smoke tests)")
    p.add_argument("--quiet", action="store_true")
    # ---- ablation knobs (PROTOCOL.md defaults when omitted; each one is recorded in the run name) ----
    p.add_argument("--sigma", type=float, default=None,
                   help="probe scale decoupled from the learning rate (default: sigma = lr, the protocol's tie)")
    p.add_argument("--passes", type=int, default=None,
                   help="forward passes per step for the probe methods (default 16; SPSA uses passes/2 pairs, "
                        "RS / NP / ANP / DANP passes-1 probes, INP max(1, (passes-1)//L) draws per layer)")
    p.add_argument("--weight-decay", type=float, default=0.0, help="decoupled weight decay (AdamW); 0 = protocol")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False,
                   help="allow TF32 for cuDNN convolutions / matmuls (default: full fp32 arithmetic)")
    return p.parse_args(argv)


def run_tag(args) -> str:
    """Name suffix for non-protocol settings, e.g. ``-sig0.001``, ``-K64``, ``-wd0.0001``, ``-tf32``."""
    tag = ""
    if getattr(args, "sigma", None) is not None:
        tag += f"-sig{fmt_lr(args.sigma)}"
    if getattr(args, "passes", None) is not None:
        tag += f"-K{int(args.passes)}"
    if getattr(args, "weight_decay", 0.0):
        tag += f"-wd{fmt_lr(args.weight_decay)}"
    if getattr(args, "tf32", False):
        tag += "-tf32"
    return tag


def name_from_args(args) -> str:
    return run_name(args.task, args.opt, args.lr, args.seed, run_tag(args))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def _float(v) -> float:
    try:
        return float(v.item()) if hasattr(v, "item") else float(v)
    except Exception:  # noqa: BLE001
        return float("nan")


@torch.no_grad()
def evaluate(opt, data: Data, split: str) -> dict:
    """Mean per-sample loss over the FULL split via ``opt.predict`` (never ``model(x)``),
    plus accuracy (per image, or per next-character) and bits-per-char for LM."""
    loss_sum = torch.zeros((), dtype=torch.float64, device=data.device)
    correct = torch.zeros((), dtype=torch.float64, device=data.device)
    n_rows = 0
    n_tok = 0
    for x, y in data.eval_batches(split):
        logits = opt.predict(x)
        loss_sum += per_sample_loss(logits, y).double().sum()
        correct += (logits.argmax(-1) == y).double().sum()
        n_rows += y.shape[0]
        n_tok += y.numel()
    loss = loss_sum.item() / max(n_rows, 1)
    out = {"loss": loss, "acc": correct.item() / max(n_tok, 1), "n": n_rows}
    if data.kind == "lm":
        out["bpc"] = loss / LN2
    if data.kind == "cls" and hasattr(opt, "predict_goodness"):
        # forward-forward: Hinton's label-accumulation inference (one pass per candidate label)
        corr_g = torch.zeros((), dtype=torch.float64, device=data.device)
        for x, y in data.eval_batches(split):
            corr_g += (opt.predict_goodness(x).argmax(1) == y).double().sum()
        out["acc_goodness"] = corr_g.item() / max(n_tok, 1)
    return out


class _Window:
    """Running sums over a window of steps (reset at each log / eval)."""

    def __init__(self):
        self.loss_sum, self.n, self.t0, self.step0 = 0.0, 0, 0.0, 0

    def add(self, loss_sum: float, n: int):
        self.loss_sum += loss_sum
        self.n += n

    def mean(self) -> float:
        return self.loss_sum / self.n if self.n else float("nan")

    def rate(self, train_time: float, step: int) -> float:
        dt = train_time - self.t0
        return (step - self.step0) / dt if dt > 0 else float("nan")

    def reset(self, train_time: float, step: int):
        self.loss_sum, self.n, self.t0, self.step0 = 0.0, 0, train_time, step


class WandbLogger:
    """Thin wrapper that never raises: wandb problems must not kill a training run."""

    def __init__(self, enabled: bool, project: str, name: str, group: str, config: dict):
        self.run = None
        self.wandb = None
        if not enabled:
            return
        try:
            os.environ.setdefault("WANDB_SILENT", "true")
            import wandb  # type: ignore

            key = os.environ.get("WANDB_API_KEY", "").strip()
            mode = "online" if key else "offline"
            if not key:
                print("[train] WANDB_API_KEY not set: running wandb in offline mode", flush=True)
            kw = {}
            try:
                kw["settings"] = wandb.Settings(init_timeout=120)
            except Exception:  # noqa: BLE001
                pass
            self.run = wandb.init(project=project, name=name, group=group, config=config, mode=mode,
                                  dir=os.path.join(REPO_ROOT, "wandb") if mode == "offline" else None, **kw)
            wandb.define_metric("train_time")
            wandb.define_metric("*", step_metric="train_time")
            self.wandb = wandb
        except Exception as e:  # noqa: BLE001
            print(f"[train] wandb disabled: {e!r}", flush=True)
            self.run = None

    def log(self, d: dict):
        if self.run is not None:
            try:
                self.wandb.log({k: v for k, v in d.items() if v is not None})
            except Exception as e:  # noqa: BLE001
                print(f"[train] wandb.log failed: {e!r}", flush=True)

    def summary(self, d: dict):
        if self.run is not None:
            try:
                self.run.summary.update(d)
            except Exception as e:  # noqa: BLE001
                print(f"[train] wandb summary failed: {e!r}", flush=True)

    def finish(self, exit_code: int = 0):
        if self.run is not None:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception:  # noqa: BLE001
                pass
            self.run = None


# ----------------------------------------------------------------------------- training loop
def train_loop(args, data: Data, opt, result: dict, logger: WandbLogger, out_path: str, wall0: float,
               quiet: bool = False) -> None:
    device = data.device
    cuda = device.type == "cuda"
    bs = data.batch_size
    fp = int(getattr(opt, "forward_passes_per_step", 1))
    is_lm = data.kind == "lm"
    budget, eval_every = float(args.budget), float(args.eval_every)
    chunk, log_every = max(1, args.chunk), max(1, args.log_every)
    name = result["config"]["name"]
    history = result["history"]

    def sync():
        if cuda:
            torch.cuda.synchronize(device)

    train_time, step = 0.0, 0
    next_eval, next_log = eval_every, log_every
    initial_loss: Optional[float] = None
    initial_val = [None]                      # clean validation loss at step 0 (optimizer-independent reference)
    bad_evals, diverged, last_eval_step = 0, False, -1
    stop_reason = None
    ev, lg = _Window(), _Window()

    def do_eval():
        nonlocal bad_evals, diverged, last_eval_step, stop_reason
        mode = ctx.mode
        val, test = evaluate(opt, data, "val"), evaluate(opt, data, "test")
        ctx.mode = mode
        train_loss = ev.mean() if ev.n else None      # None at the step-0 evaluation
        rec = {
            "train_time": train_time, "step": step, "samples": step * bs, "forward_passes": step * fp,
            "train_loss": train_loss, "val_loss": val["loss"], "val_acc": val["acc"],
            "test_loss": test["loss"], "test_acc": test["acc"],
            "steps_per_sec": ev.rate(train_time, step), "wall_time": time.time() - wall0,
        }
        if is_lm:
            rec["val_bpc"], rec["test_bpc"] = val["bpc"], test["bpc"]
        if "acc_goodness" in test:
            rec["val_acc_goodness"], rec["test_acc_goodness"] = val["acc_goodness"], test["acc_goodness"]
        history.append(rec)
        # ---- divergence watchdog (PROTOCOL.md); the step-0 evaluation is the reference
        if train_loss is None:
            initial_val[0] = val["loss"]
        elif not math.isfinite(train_loss) or not math.isfinite(val["loss"]):
            diverged, stop_reason = True, "non-finite loss at evaluation"
        elif (initial_loss is not None and train_loss > DIVERGE_FACTOR * initial_loss) or \
                (initial_val[0] is not None and val["loss"] > DIVERGE_FACTOR * initial_val[0]):
            bad_evals += 1
            if bad_evals >= 2:
                which = "train" if (initial_loss is not None and train_loss > DIVERGE_FACTOR * initial_loss) else "validation"
                diverged, stop_reason = True, f"{which} loss > {DIVERGE_FACTOR:g}x its initial value at two consecutive evals"
        else:
            bad_evals = 0
        result.update(diverged=diverged, train_time=train_time, steps=step, status="running",
                      wall_time_total=time.time() - wall0, stop_reason=stop_reason)
        atomic_write_json(out_path, result)
        logger.log(rec)
        if not quiet:
            train_loss = float("nan") if train_loss is None else train_loss
            if is_lm:
                m = (f"train {train_loss:.4f} | val {val['loss']:.4f} ({val['bpc']:.3f} bpc) "
                     f"| test {test['loss']:.4f} ({test['bpc']:.3f} bpc)")
            else:
                m = (f"train {train_loss:.4f} | val {val['loss']:.4f} acc {val['acc']:.4f} "
                     f"| test {test['loss']:.4f} acc {test['acc']:.4f}")
            print(f"[{name}] t={train_time:.1f}s step={step} ({rec['steps_per_sec']:.1f} st/s) {m}"
                  + (" DIVERGED" if diverged else ""), flush=True)
        ev.reset(train_time, step)
        last_eval_step = step

    do_eval()                                    # step-0 evaluation: curves start at the initial loss
    while train_time < budget and not diverged:
        n = 1 if step == 0 else chunk            # first step alone -> exact initial loss
        batches = [data.train_batch() for _ in range(n)]   # sampled outside the clock
        acc = torch.zeros((), device=device)
        sync()
        t0 = time.perf_counter()
        for x, y in batches:
            loss = opt.step(x, y)
            acc += loss.detach() if torch.is_tensor(loss) else float(loss)
        sync()
        train_time += time.perf_counter() - t0
        step += n
        chunk_sum = acc.item()                   # one host sync per chunk, outside the clock
        if initial_loss is None:
            initial_loss = chunk_sum / n
            result["initial_train_loss"] = initial_loss
        ev.add(chunk_sum, n)
        lg.add(chunk_sum, n)
        if not math.isfinite(chunk_sum):
            diverged, stop_reason = True, "non-finite train loss"
        if step >= next_log:
            d = {"train_time": train_time, "step": step, "samples": step * bs, "forward_passes": step * fp,
                 "train_loss": lg.mean(), "steps_per_sec": lg.rate(train_time, step),
                 "wall_time": time.time() - wall0}
            try:
                d.update({f"opt/{k}": _float(v) for k, v in (opt.extra_log() or {}).items()})
            except Exception as e:  # noqa: BLE001
                print(f"[train] extra_log failed: {e!r}", flush=True)
            logger.log(d)
            lg.reset(train_time, step)
            while next_log <= step:
                next_log += log_every
        if diverged or train_time >= next_eval or train_time >= budget:
            do_eval()
            while next_eval <= train_time:
                next_eval += eval_every
    if last_eval_step != step:
        do_eval()
    result.update(train_time=train_time, steps=step, diverged=diverged, stop_reason=stop_reason)


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    args = parse_args(argv)
    name = name_from_args(args)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, name + ".json")
    seed_everything(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # Precision: the probe methods correlate per-example loss differences of order sigma with
        # the injected noise, and at sigma = 1e-3 those differences are of the same order as TF32's
        # ~1e-3 relative rounding.  PyTorch leaves TF32 ON for cuDNN convolutions by default (and
        # off for matmul), so the ConvNet runs would otherwise mix precisions; make it explicit.
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)

    config = {**vars(args), "lr": args.lr, "lr_str": fmt_lr(args.lr), "name": name, "device": str(device),
              "hostname": socket.gethostname(),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "git_commit": git_commit(), "torch": torch.__version__, "python": platform.python_version(),
              "betas": list(ADAM_BETAS), "argv": sys.argv[1:], "tag": run_tag(args),
              "sigma_eff": args.lr if args.sigma is None else args.sigma,
              "passes_eff": None, "precision": "tf32" if args.tf32 else "fp32",
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    result = {"config": config, "history": [], "diverged": False, "status": "starting", "wall_time_total": 0.0}
    wall0 = time.time()
    logger: Optional[WandbLogger] = None
    try:
        data = Data(args.task, device, root=args.data_root, _synthetic=args.synthetic)
        model = build_model(args.task, vocab=data.n_classes).to(device)
        ctx.set_layers(players(model))
        opt = get_optimizer(args.opt, model, args.lr, data.kind, data.n_classes,
                            sigma=args.sigma, n_passes=args.passes, weight_decay=args.weight_decay)
        config.update(n_params=count_params(model), batch_size=data.batch_size, seq_len=data.seq_len,
                      kind=data.kind, n_classes=data.n_classes, n_train=data.n_train, data_source=data.source,
                      forward_passes_per_step=int(getattr(opt, "forward_passes_per_step", 1)),
                      passes_eff=int(getattr(opt, "forward_passes_per_step", 1)), eps_eff=float(getattr(opt, "eps", args.lr)))
        logger = WandbLogger(args.wandb, args.wandb_project, name, args.task, config)
        if not args.quiet:
            print(f"[{name}] {data.summary()}", flush=True)
            print(f"[{name}] params={config['n_params']} fp/step={config['forward_passes_per_step']} "
                  f"device={device} ({config['gpu'] or 'cpu'}) budget={args.budget:g}s eval_every={args.eval_every:g}s "
                  f"seed={args.seed} lr={args.lr:g} sigma={config['eps_eff']:g} wd={args.weight_decay:g} "
                  f"precision={config['precision']} wandb={'on' if logger.run else 'off'}", flush=True)
        train_loop(args, data, opt, result, logger, out_path, wall0, quiet=args.quiet)
        final = dict(result["history"][-1])
        result.update(final=final, status="diverged" if result["diverged"] else "done",
                      wall_time_total=time.time() - wall0)
        atomic_write_json(out_path, result)
        logger.summary({f"final/{k}": v for k, v in final.items()} | {"diverged": result["diverged"]})
        logger.finish(0)
        if not args.quiet:
            extra = f" test_acc={final['test_acc']:.4f}" if data.kind == "cls" else f" test_bpc={final['test_bpc']:.4f}"
            print(f"[{name}] {result['status']}: steps={final['step']} train_time={final['train_time']:.1f}s "
                  f"wall={result['wall_time_total']:.1f}s val_loss={final['val_loss']:.4f} "
                  f"test_loss={final['test_loss']:.4f}{extra} -> {out_path}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001
        tb = traceback.format_exc()
        result.update(error=f"{type(e).__name__}: {e}", traceback=tb, status="error",
                      wall_time_total=time.time() - wall0)
        try:
            atomic_write_json(out_path, result)
        except Exception as e2:  # noqa: BLE001
            print(f"[train] could not write {out_path}: {e2!r}", file=sys.stderr)
        print(tb, file=sys.stderr, flush=True)
        print(f"[{name}] FAILED: {result['error']} (partial result in {out_path})", file=sys.stderr, flush=True)
        if logger is not None:
            logger.finish(1)
        return 1


if __name__ == "__main__":
    sys.exit(main())
