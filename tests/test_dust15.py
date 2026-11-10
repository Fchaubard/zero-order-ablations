"""Tests for 1.5-DUST (dust15): the DUST step still runs on every model, the antithetic pairs converge to the
exact gradient where DUST's per-unit credit is exact (the MLP's hidden layers), the curvature factor is the only
difference between alpha = 0.5 and alpha = 0, and it learns (CPU, fast)."""
import torch

from zoo.models import build_model
from zoo.optimizers import get_optimizer, per_sample_loss
from zoo.perturb import ctx, players

SHAPES = {"mlp_mnist": ("cls", 10), "conv_cifar10": ("cls", 10), "lstm_ptb": ("lm", 50), "transformer_ptb": ("lm", 50)}


def _batch(task, seed=0, n=8):
    g = torch.Generator().manual_seed(seed)
    if task == "mlp_mnist":
        return torch.randn(n, 1, 28, 28, generator=g), torch.randint(0, 10, (n,), generator=g)
    if task == "conv_cifar10":
        return torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)
    return torch.randint(0, 50, (n, 32), generator=g), torch.randint(0, 50, (n, 32), generator=g)


def _estimate(name, model, task, x, y, eps=1e-3, seed=1, **kw):
    kind, nc = SHAPES[task]
    ctx.set_layers(players(model))
    attrs = {k: kw.pop(k) for k in ("alpha", "lam") if k in kw}
    opt = get_optimizer(name, model, eps, kind, nc, loss64=True, **kw)
    for k, v in attrs.items():
        setattr(opt, k, v)
    opt.adam.step = lambda: None
    torch.manual_seed(seed)
    opt.step(x, y)
    return {k: p.grad.clone() for k, p in model.named_parameters()}


def test_runs_on_every_model():
    for task in SHAPES:
        torch.manual_seed(0)
        m = build_model(task, vocab=50)
        x, y = _batch(task)
        g = _estimate("dust15", m, task, x, y)
        assert all(torch.isfinite(v).all() for v in g.values()), task
        assert sum(float(v.abs().sum()) for v in g.values()) > 0, task


def test_pairs_converge_to_exact_gradient_on_mlp():
    """On the MLP a jitter at one example changes only that example's loss, so DUST's per-unit credit is exact and
    the antithetic estimate of the hidden layers' gradient converges to backprop's as the pairs grow."""
    torch.manual_seed(0)
    m = build_model("mlp_mnist")
    x, y = _batch("mlp_mnist", n=16)
    ctx.mode = "bp"
    m.zero_grad()
    per_sample_loss(m(x), y).mean().backward()
    ref = {k: p.grad.clone() for k, p in m.named_parameters()}
    m.zero_grad()
    est = _estimate("dust15", m, "mlp_mnist", x, y, eps=1e-3, n_passes=1 + 2 * 2000, alpha=0.0)
    for k in ref:
        if k.startswith("readout"):
            continue                       # the head is DUST's cached-logit estimate, not the pairs
        cos = torch.nn.functional.cosine_similarity(est[k].flatten(), ref[k].flatten(), dim=0)
        assert cos > 0.95, (k, float(cos))
        ratio = float(est[k].norm() / ref[k].norm())
        assert 0.85 < ratio < 1.15, (k, ratio)


def test_curvature_factor_is_the_only_difference():
    """With a damping floor far above any curvature, (c + lambda)^alpha is the constant lambda^alpha, so alpha = 0.5
    gives the alpha = 0 estimate divided by sqrt(lambda) at the direct sites (same draws)."""
    torch.manual_seed(0)
    m = build_model("transformer_ptb", vocab=50)
    x, y = _batch("transformer_ptb")
    g0 = _estimate("dust15", m, "transformer_ptb", x, y, alpha=0.0)
    g5 = _estimate("dust15", m, "transformer_ptb", x, y, alpha=0.5, lam=1e8)
    for k in g0:
        if k.startswith("readout"):
            continue                       # the head is DUST's, untouched by alpha
        assert float((g5[k] * 1e4 - g0[k]).norm() / g0[k].norm()) < 1e-4, k


def test_learns():
    torch.manual_seed(0)
    m = build_model("mlp_mnist")
    ctx.set_layers(players(m))
    opt = get_optimizer("dust15", m, 1e-2, "cls", 10, loss64=True)
    x, y = _batch("mlp_mnist", n=64)
    first = float(opt.step(x, y))
    for _ in range(150):
        last = float(opt.step(x, y))
    assert last < first - 0.3, (first, last)
