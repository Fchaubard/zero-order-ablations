"""Tests for identity-path credit (ipc): it computes exactly the identity-path rule (every branch's parameters
take the gradient of <branch output, top-stream credit>, inputs detached), the head and the top block's MLP match
the true gradient, it refuses non-residual models, and it learns (CPU, fast)."""
import pytest
import torch
import torch.nn.functional as F

from zoo.models import build_model
from zoo.optimizers import get_optimizer, per_sample_loss
from zoo.perturb import ctx


def _batch(seed=0, n=4, t=64, v=50):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, v, (n, t), generator=g), torch.randint(0, v, (n, t), generator=g)


def _identity_path_reference(model, x, y):
    """The identity-path rule by autograd (float64 model)."""
    ctx.mode = "bp"
    h = model.embed(x)
    hs = [h]
    for b in model.blocks:
        h = b(h)
        hs.append(h)
    xL = hs[-1].detach().requires_grad_(True)
    lt = F.cross_entropy(model.readout(model.ln_f(xL)).transpose(1, 2), y, reduction="none")
    g = torch.autograd.grad(lt.sum(), xL)[0]
    n = lt.numel()
    model.zero_grad()
    F.cross_entropy(model.readout(model.ln_f(xL.detach())).transpose(1, 2), y, reduction="none").mean().backward()
    (model.embed(x) * g).sum().div(n).backward()
    hin = hs[0].detach()
    for b in model.blocks:
        a = b.attn(b.ln1(hin))
        (a * g).sum().div(n).backward()
        mid = (hin + a).detach()
        f = b.ff2(F.relu(b.ff1(b.ln2(mid))))
        (f * g).sum().div(n).backward()
        hin = (mid + f).detach()
    ref = {k: p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad()
    return ref


def _ipc_estimate(model, x, y, eps=1e-4):
    opt = get_optimizer("ipc", model, eps, "lm", 50, loss64=True)
    opt.adam.step = lambda: None
    opt.step(x, y)
    return {k: p.grad.clone() for k, p in model.named_parameters()}


def test_matches_identity_path_rule():
    torch.manual_seed(0)
    m = build_model("transformer_ptb", vocab=50).double()
    x, y = _batch()
    ref = _identity_path_reference(m, x, y)
    est = _ipc_estimate(m, x, y)
    for k in ref:
        if k.endswith("k.bias"):        # zero in both (softmax is invariant to a per-query shift)
            continue
        c = F.cosine_similarity(ref[k].flatten(), est[k].flatten(), 0).item()
        assert c > 0.9995, (k, c)
        r = (est[k].norm() / ref[k].norm()).item()
        assert abs(r - 1) < 0.01, (k, r)


def test_top_of_network_is_exact():
    torch.manual_seed(1)
    m = build_model("transformer_ptb", vocab=50).double()
    x, y = _batch(1)
    ctx.mode = "bp"
    m.zero_grad()
    per_sample_loss(m(x), y).mean().backward()
    true = {k: p.grad.clone() for k, p in m.named_parameters()}
    est = _ipc_estimate(m, x, y)
    top = len(m.blocks) - 1
    for k in ("readout.weight", "readout.bias", f"blocks.{top}.ff1.weight", f"blocks.{top}.ff2.weight"):
        assert F.cosine_similarity(true[k].flatten(), est[k].flatten(), 0).item() > 0.9999, k


def test_refuses_non_residual_models():
    for task in ("mlp_mnist", "conv_cifar10", "lstm_ptb"):
        m = build_model(task, vocab=50) if task == "lstm_ptb" else build_model(task)
        kind = "lm" if task == "lstm_ptb" else "cls"
        with pytest.raises(ValueError):
            get_optimizer("ipc", m, 1e-3, kind, 50 if kind == "lm" else 10)


def test_learns():
    torch.manual_seed(2)
    m = build_model("transformer_ptb", vocab=50)
    x = torch.arange(64).remainder(7).repeat(8, 1)          # a periodic sequence: next token is predictable
    y = torch.roll(x, -1, 1)
    opt = get_optimizer("ipc", m, 1e-2, "lm", 50)
    first = float(opt.step(x, y))
    for _ in range(60):
        last = float(opt.step(x, y))
    assert last < 0.5 * first, (first, last)
    assert opt.forward_passes_per_step == 1
