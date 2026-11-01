"""Tests for the node-perturbation family (np, anp, inp, danp).  CPU only, < 60 s.

Gradient-quality test (seed 0, MLP 784-48-32-10 from zoo.models, N = 64 random
MNIST-shaped inputs with random labels, sigma = lr = 1e-3, 400 noisy passes
[400 per layer for inp]).  Measured cosine similarity between the averaged estimate
and the true backprop gradient (weight + bias of the layer):

    optimizer   readout (layer 3)   first layer (layer 1)
    np          0.993               0.805         
    anp         0.922               0.794          
    inp         0.997               0.948          

Thresholds asserted: > 0.9 for the readout, > 0.5 for the first layer.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from zoo.models import TASKS, build_model
from zoo.optimizers import get_optimizer, per_sample_loss
from zoo.perturb import ctx, layer_params, players

torch.set_num_threads(max(1, torch.get_num_threads()))


def _flat_grad(layer) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in layer_params(layer)])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def _true_grads(model, x, y):
    ctx.mode = "bp"
    model.zero_grad(set_to_none=True)
    per_sample_loss(model(x), y).mean().backward()
    g = [_flat_grad(l).clone() for l in players(model)]
    model.zero_grad(set_to_none=True)
    return g


@pytest.fixture(scope="module")
def mlp_batch():
    torch.manual_seed(0)
    model = build_model("mlp_mnist")
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    return model, x, y, _true_grads(model, x, y)


@pytest.mark.parametrize("name", ["np", "anp", "inp"])
def test_estimate_aligns_with_backprop(mlp_batch, name):
    model, x, y, true = mlp_batch
    torch.manual_seed(1)
    m = copy.deepcopy(model)
    opt = get_optimizer(name, m, lr=1e-3, task_kind="cls", n_classes=10)
    loss = opt.estimate_grad(x, y, n_passes=400)
    assert loss.dim() == 0 and not loss.requires_grad and torch.isfinite(loss)
    est = [_flat_grad(l) for l in players(m)]
    cos_ro, cos_l1 = _cos(est[-1], true[-1]), _cos(est[0], true[0])
    print(f"\n{name}: cos(readout) = {cos_ro:.4f}   cos(first layer) = {cos_l1:.4f}")
    assert cos_ro > 0.9, cos_ro
    assert cos_l1 > 0.5, cos_l1


def _offdiag_corr_sq(model, x) -> float:
    """Mean squared off-diagonal entry of the (uncentred) correlation matrix of the first
    layer's decorrelated inputs x* = R x on a clean pass."""
    first = players(model)[0]
    ctx.set_layers(players(model))
    ctx.mode = "clean"
    ctx.reset()
    with torch.no_grad():
        model(x)
        xd = torch.cat(first.hd, 0)
        C = xd.T @ xd / xd.shape[0]
        d = C.diagonal().clamp_min(1e-12).sqrt()
        corr = C / (d[:, None] * d[None, :])
        off = corr - torch.diag(corr.diagonal())
        n = corr.shape[0]
        val = off.pow(2).sum() / (n * (n - 1))
    ctx.reset()
    return float(val)


def test_danp_decorrelation_reduces_input_correlation():
    torch.manual_seed(0)
    model = build_model("mlp_mnist")
    opt = get_optimizer("danp", model, lr=1e-3, task_kind="cls", n_classes=10)
    first = players(model)[0]
    assert first.R is not None and first.R.shape == (784, 784) and first.R.device == first.weight.device
    # strongly correlated inputs: 16 latent factors mixed into 784 dims + a little noise
    z = torch.randn(256, 16)
    A = torch.randn(16, 784)
    x = z @ A / 4 + 0.5 * torch.randn(256, 784)
    y = torch.randint(0, 10, (256,))
    before = _offdiag_corr_sq(model, x)
    for _ in range(20):
        loss = opt.step(x, y)                  # every step ends with a decorrelation update
        assert torch.isfinite(loss)
    after = _offdiag_corr_sq(model, x)
    print(f"\ndanp: mean squared off-diagonal correlation {before:.4f} -> {after:.4f}")
    assert after < before
    assert torch.isfinite(first.R).all()


def _batch(task, N=8, vocab=50):
    kind = TASKS[task][1]
    if kind == "lm":
        return torch.randint(0, vocab, (N, 64)), torch.randint(0, vocab, (N, 64)), vocab
    if task == "mlp_mnist":
        return torch.randn(N, 1, 28, 28), torch.randint(0, 10, (N,)), 10
    return torch.randn(N, 3, 32, 32), torch.randint(0, 10, (N,)), 10


@pytest.mark.parametrize("name", ["np", "anp", "inp", "danp"])
@pytest.mark.parametrize("task", list(TASKS))
def test_smoke_all_models(task, name):
    torch.manual_seed(0)
    model = build_model(task, vocab=50)
    x, y, n_classes = _batch(task)
    opt = get_optimizer(name, model, lr=1e-3, task_kind=TASKS[task][1], n_classes=n_classes)
    for _ in range(3):
        loss = opt.step(x, y)
        assert loss.dim() == 0 and not loss.requires_grad
        assert torch.isfinite(loss), loss
    for n, p in model.named_parameters():
        assert p.grad is not None, n
        assert torch.isfinite(p.grad).all(), n
        assert torch.isfinite(p).all(), n
    log = opt.extra_log()
    assert "mean_abs_dloss" in log and log["mean_abs_dloss"] >= 0.0


@pytest.mark.parametrize("name", ["np", "anp", "inp", "danp"])
@pytest.mark.parametrize("task", list(TASKS))
def test_forward_pass_budget(task, name):
    """Exactly 16 model forward calls per step (after the one-off batch-axis probe)."""
    torch.manual_seed(0)
    model = build_model(task, vocab=50)
    x, y, n_classes = _batch(task, N=4)
    opt = get_optimizer(name, model, lr=1e-3, task_kind=TASKS[task][1], n_classes=n_classes)
    opt.step(x, y)                                # first step includes the tiny probe passes
    calls = [0]
    h = model.register_forward_hook(lambda *a: calls.__setitem__(0, calls[0] + 1))
    try:
        opt.step(x, y)
    finally:
        h.remove()
    assert calls[0] == 16, calls[0]


def test_transformer_position_embedding_is_shared():
    """The learned position embedding is called on arange(T) (no batch axis) and must be
    detected as shared even when T == N (the protocol's 64 x 64 LM batches)."""
    torch.manual_seed(0)
    model = build_model("transformer_ptb", vocab=50)
    x = torch.randint(0, 50, (64, 64))
    y = torch.randint(0, 50, (64, 64))
    opt = get_optimizer("np", model, lr=1e-3, task_kind="lm", n_classes=50)
    opt.estimate_grad(x, y, n_passes=1)
    shared = dict(zip(players(model), opt._shared))
    assert shared[model.pos] is True
    assert shared[model.tok] is False
    assert sum(shared.values()) == 1


@pytest.mark.parametrize("label", ["tiny_lstm", "tiny_transformer"])
def test_inp_matches_backprop_on_tiny_lm_models(label):
    """INP is unbiased (exact directional derivative as sigma -> 0), so on tiny instances of
    the LM model classes (few perturbed nodes) 1000 single-layer passes per layer must give
    cos > 0.9 and |est| / |true| in (0.75, 1.33) for EVERY layer.  This exercises the
    multi-call LSTM gate layer (same delta_L for all time steps), the embedding backward
    and the batch-shared position embedding.  Measured: LSTM cos >= 0.96, ratio <= 1.11;
    transformer cos >= 0.97, ratio <= 1.10 (seed 0)."""
    from zoo.models import CharLSTM, CharTransformer
    torch.manual_seed(0)
    V, T, N = 10, 8, 16
    if label == "tiny_lstm":
        model = CharLSTM(vocab=V, d_emb=6, hidden=6)
    else:
        model = CharTransformer(vocab=V, d=8, n_heads=2, d_ff=8, n_layers=1, max_len=T)
    x = torch.randint(0, V, (N, T))
    y = torch.randint(0, V, (N, T))
    true = _true_grads(model, x, y)
    opt = get_optimizer("inp", model, lr=1e-3, task_kind="lm", n_classes=V)
    opt.estimate_grad(x, y, n_passes=1000)
    names = [n for n, mod in model.named_modules() if any(mod is l for l in players(model))]
    for name, layer, t in zip(names, players(model), true):
        e = _flat_grad(layer)
        cos, ratio = _cos(e, t), float(e.norm() / t.norm())
        print(f"\n{label} inp {name}: cos={cos:.3f} ratio={ratio:.2f}")
        assert cos > 0.9, (name, cos)
        assert 0.75 < ratio < 1.33, (name, ratio)
