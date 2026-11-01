"""CPU tests for the two parameter-space zero-order optimizers: SPSA ("spsa") and
random search ("rs").  Runs in well under a minute on CPU.

Measured values (torch 2.15, CPU -- see the tests for the exact setup):
* SPSA vs backprop on the 39,578-parameter MLP, 2400 antithetic pairs (300 steps'
  worth of probes), eps = 1e-3:  cosine 0.243 with seed 0, range 0.233-0.243 over
  seeds 0-7.  Theory: ghat = (1/N) sum_k (z_k . g) z_k has
  cos(ghat, g) ~ sqrt(N / (N + n)) = sqrt(2400 / 41978) = 0.239 for N = 2400 probes in
  n = 39,578 dimensions, so the measurement sits right on the prediction.  The
  threshold below (0.18) is set a bit under the measurement and far above 0.1.
* SPSA on the 10-dim quadratic ||theta - c||^2, 2400 pairs: cosine 0.999
  (central differences are exact on a quadratic; sqrt(2400 / 2410) = 0.998).
* Random search on the quadratic, eps = 0.1: loss 7.09 -> 0.023 in 200 steps (seed 3);
  > 99 % decrease on seeds 0-7; acceptance rate over the last 100 steps 0.56-0.76 at
  step 100 and 0.01-0.11 at step 200 (converged, improvements become rare).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from zoo.models import TASKS, build_model
from zoo.optimizers import OPTIMIZERS, get_optimizer, per_sample_loss
from zoo.optimizers.random_search import RandomSearch
from zoo.optimizers.spsa import SPSA
from zoo.perturb import ctx

VOCAB = 50
SPSA_MLP_COS_THRESHOLD = 0.18   # measured 0.24 (see module docstring); must be clearly > 0.1


# ----------------------------------------------------------------------------- helpers
def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def true_grad(model: nn.Module, x, y) -> torch.Tensor:
    """Flat backprop gradient of the batch-mean per_sample_loss."""
    ctx.mode = "bp"
    model.zero_grad(set_to_none=True)
    loss = per_sample_loss(model(x), y).mean()
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return torch.cat([g.reshape(-1) for g in grads])


def make_batch(task: str, n: int):
    kind = TASKS[task][1]
    if kind == "lm":
        x = torch.randint(0, VOCAB, (n, 64))
        y = torch.randint(0, VOCAB, (n, 64))
        return x, y, kind, VOCAB
    shape = (n, 1, 28, 28) if task == "mlp_mnist" else (n, 3, 32, 32)
    x = torch.randn(*shape)
    y = torch.randint(0, 10, (n,))
    return x, y, kind, 10


class Quadratic(nn.Module):
    """f(theta) = ||theta - c||^2 with a single 10-dim parameter (no P-layers needed:
    the optimizers only touch model.parameters())."""

    def __init__(self, c: torch.Tensor):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros_like(c))
        self.register_buffer("c", c.clone())

    def forward(self, x=None):
        return ((self.theta - self.c) ** 2).sum()


def quad_loss(model, x, y):
    return model(x)


@torch.no_grad()
def quad_value(model: Quadratic) -> float:
    return float(model())


# ----------------------------------------------------------------------------- (1) SPSA estimator
def test_spsa_cosine_with_backprop_on_mlp():
    torch.manual_seed(0)
    model = build_model("mlp_mnist")
    x, y, kind, n_classes = make_batch("mlp_mnist", 64)
    g_true = true_grad(model, x, y)
    assert g_true.numel() == 39578

    opt = SPSA(model, lr=1e-3, task_kind=kind, n_classes=n_classes)
    theta_before = flat_params(model)
    g_hat = opt.estimate_grad(x, y, n_pairs=300 * SPSA.n_pairs)   # 300 steps' worth = 2400 pairs
    cos = cosine(g_hat, g_true)
    print(f"\nSPSA/MLP cosine over 2400 pairs: {cos:.4f} (theory sqrt(N/(N+n)) = "
          f"{math.sqrt(2400 / (2400 + g_true.numel())):.4f})")

    assert torch.isfinite(g_hat).all()
    assert torch.equal(flat_params(model), theta_before), "estimate_grad must not move the parameters"
    assert SPSA_MLP_COS_THRESHOLD > 0.1
    assert cos > SPSA_MLP_COS_THRESHOLD, f"cosine {cos:.4f} <= {SPSA_MLP_COS_THRESHOLD}"


def test_spsa_cosine_on_quadratic():
    torch.manual_seed(1)
    c = torch.randn(10)
    model = Quadratic(c)
    opt = SPSA(model, lr=1e-3, task_kind="cls", n_classes=0, loss_fn=quad_loss)
    g_true = 2 * (model.theta.detach() - c)
    g_hat = opt.estimate_grad(None, None, n_pairs=300 * SPSA.n_pairs)
    cos = cosine(g_hat, g_true)
    print(f"\nSPSA/quadratic cosine over 2400 pairs: {cos:.4f}")
    assert cos > 0.95
    # magnitude is also right (unbiased estimator, exact on a quadratic)
    assert abs(float(g_hat.norm() / g_true.norm()) - 1.0) < 0.1


def test_spsa_step_uses_16_passes_and_logs_projected_grad_std():
    torch.manual_seed(2)
    model = build_model("mlp_mnist")
    x, y, kind, n_classes = make_batch("mlp_mnist", 16)
    opt = SPSA(model, lr=1e-3, task_kind=kind, n_classes=n_classes)
    calls = []
    orig = opt.loss_fn
    opt.loss_fn = lambda m, a, b: (calls.append(1), orig(m, a, b))[1]
    assert opt.extra_log() == {}
    loss = opt.step(x, y)
    assert len(calls) == 16
    assert loss.dim() == 0 and not loss.requires_grad and torch.isfinite(loss)
    log = opt.extra_log()
    assert set(log) == {"proj_grad_std"} and math.isfinite(log["proj_grad_std"])


# ----------------------------------------------------------------------------- (2) random search
def test_random_search_decreases_quadratic():
    torch.manual_seed(3)
    c = torch.randn(10)
    model = Quadratic(c)
    opt = RandomSearch(model, lr=0.1, task_kind="cls", n_classes=0, loss_fn=quad_loss)
    assert opt.extra_log() == {}
    l_init = quad_value(model)
    first = opt.step(None, None)
    assert torch.isclose(first, torch.tensor(l_init)), "step must return L0 = L(theta)"
    for _ in range(99):
        opt.step(None, None)
    # after 100 steps the search is still finding improvements (measured 0.56-0.76 over 8 seeds)
    acc_100 = opt.extra_log()["acceptance_rate"]
    for _ in range(100):
        opt.step(None, None)
    l_final = quad_value(model)
    log = opt.extra_log()
    print(f"\nRS/quadratic: loss {l_init:.3f} -> {l_final:.3f}, acceptance @100 {acc_100:.2f}, "
          f"@200 {log['acceptance_rate']:.2f}, last best improvement {log['best_improvement']:.4f}")
    assert l_final < 0.5 * l_init                      # measured: > 99 % decrease on every seed tried
    assert 0.0 < acc_100 <= 1.0
    assert 0.0 <= log["acceptance_rate"] <= 1.0        # near convergence improvements become rare
    assert math.isfinite(log["best_improvement"])


def test_random_search_uses_16_passes_and_returns_clean_loss():
    torch.manual_seed(4)
    model = build_model("mlp_mnist")
    x, y, kind, n_classes = make_batch("mlp_mnist", 16)
    opt = RandomSearch(model, lr=1e-2, task_kind=kind, n_classes=n_classes)
    calls = []
    orig = opt.loss_fn
    opt.loss_fn = lambda m, a, b: (calls.append(1), orig(m, a, b))[1]
    with torch.no_grad():
        ctx.mode = "bp"
        l0 = per_sample_loss(model(x), y).mean()
    loss = opt.step(x, y)
    assert len(calls) == 16
    assert torch.isclose(loss, l0)
    assert not loss.requires_grad


# ----------------------------------------------------------------------------- (3) smoke on all 4 models
@pytest.mark.parametrize("opt_name", ["spsa", "rs"])
@pytest.mark.parametrize("task", list(TASKS))
def test_smoke_all_models(task, opt_name):
    torch.manual_seed(5)
    model = build_model(task, vocab=VOCAB)
    x, y, kind, n_classes = make_batch(task, 8)
    opt = get_optimizer(opt_name, model, lr=1e-2, task_kind=kind, n_classes=n_classes)
    assert OPTIMIZERS[opt_name] is type(opt)
    assert opt.forward_passes_per_step == 16
    theta0 = flat_params(model)
    for _ in range(3):
        loss = opt.step(x, y)
        assert loss.dim() == 0 and torch.isfinite(loss), f"{task}/{opt_name}: non-finite loss"
    theta1 = flat_params(model)
    assert torch.isfinite(theta1).all()
    assert not torch.equal(theta0, theta1), f"{task}/{opt_name}: parameters did not change"
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    for v in opt.extra_log().values():
        assert math.isfinite(v)
    # evaluation path still works afterwards
    assert torch.isfinite(opt.predict(x)).all()
