"""Tests for the forward-forward optimizer (CPU, fast)."""
import math

import pytest
import torch

from zoo.models import TASKS, build_model
from zoo.optimizers import get_optimizer


def _synthetic_cls(n=512, d=784, n_classes=10, seed=0):
    """Well separated 10-class problem: class means + isotropic noise, (N,1,28,28)."""
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n_classes, d, generator=g) * 1.5
    y = torch.arange(n) % n_classes
    x = means[y] + torch.randn(n, d, generator=g)
    return x.view(n, 1, 28, 28), y


def test_mlp_learns_synthetic_problem():
    torch.manual_seed(0)
    x, y = _synthetic_cls()
    model = build_model("mlp_mnist")
    opt = get_optimizer("ff", model, lr=0.01, task_kind="cls", n_classes=10)
    assert opt.forward_passes_per_step == 2

    loss0 = None
    for step in range(150):
        loss = opt.step(x, y)
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        loss0 = loss if loss0 is None else loss0

    logits = opt.predict(x)
    assert logits.shape == (512, 10)
    acc = (logits.argmax(1) == y).float().mean().item()
    log = opt.extra_log()
    print(f"\nreadout acc={acc:.3f}  loss {float(loss0):.3f} -> {float(loss):.3f}  "
          f"g_pos={log['ff_goodness_pos']:.3f} g_neg={log['ff_goodness_neg']:.3f}")
    assert acc > 0.6, f"readout accuracy {acc:.3f} <= 0.6"
    assert log["ff_goodness_pos"] > log["ff_goodness_neg"]
    assert float(loss) < float(loss0)

    # Hinton's label-accumulation inference
    gm = opt.predict_goodness(x)
    assert gm.shape == (512, 10)
    g_acc = (gm.argmax(1) == y).float().mean().item()
    print(f"goodness-inference acc={g_acc:.3f}")
    assert g_acc > 0.6, f"goodness inference accuracy {g_acc:.3f} <= 0.6"


def _batch(task, kind, n=8, vocab=50):
    g = torch.Generator().manual_seed(1)
    if kind == "lm":
        x = torch.randint(0, vocab, (n, 64), generator=g)
        y = torch.randint(0, vocab, (n, 64), generator=g)
        return x, y, vocab
    shape = (n, 1, 28, 28) if task == "mlp_mnist" else (n, 3, 32, 32)
    return torch.randn(*shape, generator=g), torch.randint(0, 10, (n,), generator=g), 10


@pytest.mark.parametrize("task", list(TASKS))
def test_smoke_all_models(task):
    torch.manual_seed(0)
    kind = TASKS[task][1]
    x, y, n_classes = _batch(task, kind)
    model = build_model(task, vocab=50)
    opt = get_optimizer("ff", model, lr=0.01, task_kind=kind, n_classes=n_classes)

    for _ in range(2):
        loss = opt.step(x, y)
        assert torch.isfinite(loss), f"{task}: non-finite train loss"

    # Locality: in one step every parameter must receive gradient from exactly ONE local
    # loss (its own block's goodness, the embedding from block 0, the readout from CE).
    # A leaf-tensor hook fires once per backward() that reaches it, so a leak across a
    # block boundary would make a count exceed 1.
    counts = {name: 0 for name, _ in model.named_parameters()}
    handles = []
    for name, p in model.named_parameters():
        def _hook(grad, name=name):
            counts[name] += 1
        handles.append(p.register_hook(_hook))
    loss = opt.step(x, y)
    for h in handles:
        h.remove()
    assert torch.isfinite(loss), f"{task}: non-finite train loss"
    bad = {k: v for k, v in counts.items() if v != 1}
    assert not bad, f"{task}: parameters not trained by exactly one local loss: {bad}"
    log = opt.extra_log()
    assert all(math.isfinite(v) for v in log.values())

    # every parameter received a (non-trivial) gradient from exactly one local loss
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{task}: {name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{task}: {name} grad not finite"
        assert p.grad.abs().sum() > 0, f"{task}: {name} grad is all zero"
    if task == "transformer_ptb":
        assert model.tok.weight.grad.abs().sum() > 0 and model.pos.weight.grad.abs().sum() > 0
    if task == "lstm_ptb":
        assert model.blocks[0].gates.weight.grad.abs().sum() > 0
        assert model.blocks[0].gates.bias.grad.abs().sum() > 0
    for p in model.parameters():
        assert torch.isfinite(p).all(), f"{task}: parameters not finite after update"

    logits = opt.predict(x)
    expected = (x.shape[0], 64, 50) if kind == "lm" else (x.shape[0], 10)
    assert logits.shape == expected, f"{task}: predict shape {tuple(logits.shape)} != {expected}"
    assert torch.isfinite(logits).all()
    assert not logits.requires_grad

    if kind == "cls":
        gm = opt.predict_goodness(x)
        assert gm.shape == (x.shape[0], 10) and torch.isfinite(gm).all()
    else:
        with pytest.raises(ValueError):
            opt.predict_goodness(x)
