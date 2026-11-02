"""CPU-only, network-free tests for zoo.data, zoo.train, zoo.sweep and zoo.status (< 90 s)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from zoo.optimizers import ALL_OPTIMIZERS
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from zoo.data import Data, build_char_corpus  # noqa: E402
from zoo.models import TASKS, build_model  # noqa: E402
from zoo.optimizers.base import per_sample_loss  # noqa: E402

PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": ROOT, "WANDB_MODE": "disabled", "CUDA_VISIBLE_DEVICES": ""}


def run(module_and_args, timeout=120):
    return subprocess.run([PY, "-m", *module_and_args], cwd=ROOT, env=ENV,
                          capture_output=True, text=True, timeout=timeout)


# ----------------------------------------------------------------------------- data
@pytest.mark.parametrize("task", list(TASKS))
def test_synthetic_data_shapes_and_model_forward(task):
    d = Data(task, "cpu", _synthetic=True)
    assert d.kind == TASKS[task][1]
    x, y = d.train_batch()
    if d.kind == "cls":
        assert (d.n_classes, d.batch_size, d.seq_len) == (10, 128, None)
        assert x.shape == (128, *((1, 28, 28) if task == "mlp_mnist" else (3, 32, 32)))
        assert y.shape == (128,) and x.dtype == torch.float32 and y.dtype == torch.long
        assert int(y.max()) < 10
    else:
        assert (d.n_classes, d.batch_size, d.seq_len) == (50, 64, 64)
        assert x.shape == (64, 64) == y.shape and x.dtype == torch.long
        assert x[:, 1:].equal(y[:, :-1])  # y is x shifted by one character
    assert d.n_train > 0
    for split in ("val", "test"):
        n, nb = 0, 0
        for xb, yb in d.eval_batches(split):
            n += xb.shape[0]
            nb += 1
            assert xb.shape[1:] == x.shape[1:] and yb.shape[0] == xb.shape[0]
        assert nb >= 1 and n == d.n_eval(split)
    model = build_model(task, vocab=d.n_classes)
    loss = per_sample_loss(model(x), y)
    assert loss.shape == (d.batch_size,) and torch.isfinite(loss).all()


def test_ptb_text_override():
    text = {
        "train": " the cat sat on the mat \n the dog ate \n" * 50,
        "valid": " the cat ate \n" * 20,
        "test": " a dog sat zzz\n" * 20,
    }
    d = Data("lstm_ptb", "cpu", text=text)
    joined_train = "\n".join(text["train"].splitlines())
    vocab = "".join(sorted(set(joined_train)))
    assert d.vocab == vocab and d.n_classes == len(vocab)
    assert "z" not in d.vocab                    # unseen in training text -> mapped to space
    assert d.n_train == len(joined_train)
    assert d.decode(d.encode(" the cat ")) == " the cat "
    x, y = d.train_batch()
    assert x.shape == (64, 64) and x[:, 1:].equal(y[:, :-1]) and int(x.max()) < d.n_classes
    (xb, yb), = list(d.eval_batches("test"))     # small split -> a single eval batch
    n_win = (d.split_size("test") - 1) // 64
    ids = d._ids["test"]
    assert xb.shape == (n_win, 64)
    assert xb.reshape(-1).equal(ids[: n_win * 64]) and yb.reshape(-1).equal(ids[1: n_win * 64 + 1])
    c = build_char_corpus(text)
    assert c["train"].equal(d._ids["train"]) and c["n_unk_test"] > 0 and c["n_unk_train"] == 0
    # the transformer shares the same data
    d2 = Data("transformer_ptb", "cpu", text=text)
    assert d2.vocab == d.vocab and d2._ids["train"].equal(d._ids["train"])


# ----------------------------------------------------------------------------- train
def test_train_smoke_writes_json(tmp_path):
    out = tmp_path / "results"
    r = run(["zoo.train", "--task", "mlp_mnist", "--opt", "bp", "--lr", "0.01", "--budget", "5",
             "--eval-every", "2", "--seed", "0", "--no-wandb", "--synthetic", "--out", str(out)], timeout=150)
    assert r.returncode == 0, r.stdout + r.stderr
    path = out / "mlp_mnist-bp-lr0.01.json"
    assert path.exists()
    j = json.loads(path.read_text())
    assert j["status"] == "done" and j["diverged"] is False and "error" not in j
    cfg = j["config"]
    assert (cfg["task"], cfg["opt"], cfg["lr"]) == ("mlp_mnist", "bp", 0.01)
    assert cfg["n_params"] == 39578 and cfg["batch_size"] == 128
    assert cfg["forward_passes_per_step"] == 1 and cfg["betas"] == [0.9, 0.99]
    assert "hostname" in cfg and "device" in cfg
    h = j["history"]
    assert len(h) >= 2                               # evals at ~2 s, ~4 s and at the end
    for k in ("train_time", "step", "samples", "forward_passes", "train_loss",
              "val_loss", "val_acc", "test_loss", "test_acc", "steps_per_sec"):
        assert k in h[-1], k
    f = j["final"]
    assert f == h[-1]
    assert f["train_time"] >= 5.0 and f["step"] > 0 and f["samples"] == f["step"] * 128
    assert 0.0 <= f["test_acc"] <= 1.0 and f["val_loss"] < j["initial_train_loss"]
    assert j["wall_time_total"] >= f["train_time"]
    assert [e["train_time"] for e in h] == sorted(e["train_time"] for e in h)
    assert "[mlp_mnist-bp-lr0.01] t=" in r.stdout    # the one-line progress message


def test_train_error_writes_partial_json(tmp_path, monkeypatch):
    import zoo.train as T

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(T, "get_optimizer", boom)
    rc = T.main(["--task", "mlp_mnist", "--opt", "bp", "--lr", "0.01", "--budget", "1", "--eval-every", "1",
                 "--no-wandb", "--synthetic", "--quiet", "--out", str(tmp_path)])
    assert rc == 1
    j = json.loads((tmp_path / "mlp_mnist-bp-lr0.01.json").read_text())
    assert "boom" in j["error"] and "RuntimeError" in j["traceback"]
    assert "final" not in j and j["status"] == "error"


# ----------------------------------------------------------------------------- sweep / status
def test_sweep_dry_run_lists_96_configs(tmp_path):
    out = tmp_path / "results"
    r = run(["zoo.sweep", "--dry-run", "--gpus", "0,1,2,3,4,5,6,7", "--tasks", "all", "--opts", "all",
             "--lrs", "0.1,0.01,0.001", "--out", str(out), "--logdir", str(tmp_path / "runs")])
    assert r.returncode == 0, r.stderr
    n = 4 * len(ALL_OPTIMIZERS) * 3
    lines = [l for l in r.stdout.splitlines() if l.startswith("[") and f"/{n}]" in l]
    assert len(lines) == n
    assert f"dry run: {n} configs would run, 0 skipped" in r.stdout
    names = [l.split()[-1] for l in lines]
    assert len(set(names)) == n
    first8 = names[:8]
    assert {n.split("-")[0] for n in first8} == set(TASKS)     # interleaved: 4 tasks x 2 opts
    assert {n.split("-")[1] for n in first8} == {"bp", "spsa"}
    # a completed result is skipped, an errored one is rerun
    out.mkdir()
    (out / "mlp_mnist-bp-lr0.1.json").write_text(json.dumps({"config": {}, "final": {"test_loss": 1.0}}))
    (out / "mlp_mnist-bp-lr0.01.json").write_text(json.dumps({"config": {}, "final": {}, "error": "x"}))
    r = run(["zoo.sweep", "--dry-run", "--gpus", "0", "--out", str(out), "--logdir", str(tmp_path / "runs")])
    assert f"dry run: {n - 1} configs would run, 1 skipped" in r.stdout, r.stdout
    assert "skip mlp_mnist-bp-lr0.1" in r.stdout and "run  mlp_mnist-bp-lr0.01" in r.stdout


def test_sweep_runs_on_cpu_and_status_reads_it(tmp_path):
    out, logdir = tmp_path / "results", tmp_path / "runs"
    r = run(["zoo.sweep", "--gpus", "cpu", "--tasks", "mlp_mnist", "--opts", "bp", "--lrs", "0.01",
             "--budget", "1", "--eval-every", "1", "--no-wandb", "--synthetic",
             "--out", str(out), "--logdir", str(logdir)], timeout=150)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "START" in r.stdout and "DONE" in r.stdout and "exit=0" in r.stdout
    assert (logdir / "mlp_mnist-bp-lr0.01.log").exists()
    j = json.loads((out / "mlp_mnist-bp-lr0.01.json").read_text())
    assert "final" in j and "error" not in j
    s = run(["zoo.status", "--out", str(out), "--grid", "--csv", str(tmp_path / "grid.csv")])
    assert s.returncode == 0, s.stdout + s.stderr
    assert "1 result files: 1 complete" in s.stdout
    rows = (tmp_path / "grid.csv").read_text().splitlines()
    assert rows[0] == "task," + ",".join(ALL_OPTIMIZERS) and rows[1].startswith("mlp_mnist,")
    assert rows[1].split(",")[1] != "-"
