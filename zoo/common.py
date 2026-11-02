"""Small helpers shared by train / sweep / status (no torch import here)."""
from __future__ import annotations

import json
import os
from typing import Optional

# Canonical orderings used by the sweep, the status table and the paper grid.
TASK_ORDER = ["mlp_mnist", "conv_cifar10", "lstm_ptb", "transformer_ptb"]
DEFAULT_LRS = [0.1, 0.01, 0.001]


def fmt_lr(lr) -> str:
    """0.1 -> '0.1', 0.01 -> '0.01', 0.001 -> '0.001', 1e-5 -> '1e-05' (stable across CLI/JSON)."""
    return f"{float(lr):g}"


def run_name(task: str, opt: str, lr, seed: int = 0, tag: str = "") -> str:
    """Seed 0 keeps the historical name ``{task}-{opt}-lr{eps}``; other seeds append ``-s{seed}``;
    ``tag`` (e.g. ``-sig0.001-K64``) records non-protocol settings, see ``zoo.train.run_tag``."""
    return f"{task}-{opt}-lr{fmt_lr(lr)}" + (tag or "") + (f"-s{int(seed)}" if int(seed) != 0 else "")


def result_path(out_dir: str, task: str, opt: str, lr, seed: int = 0, tag: str = "") -> str:
    return os.path.join(out_dir, run_name(task, opt, lr, seed, tag) + ".json")


def atomic_write_json(path: str, obj) -> None:
    """Write JSON to a temp file in the same directory and rename it into place."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o):
    # tensors / numpy scalars / anything with .item(); otherwise fall back to str()
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_complete(result: Optional[dict]) -> bool:
    """A run counts as finished when it has a `final` block and no `error`."""
    return bool(result) and "final" in result and not result.get("error")


def fmt_seconds(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"
