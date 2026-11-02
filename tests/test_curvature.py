"""Tests for the 1.5-order rules (spsa15, anp15): pass budget, estimator alignment with the
true gradient on every model family, and learning on a synthetic problem (CPU, fast)."""
import torch

from zoo.models import CharLSTM, CharTransformer, build_model
from zoo.optimizers import get_optimizer, per_sample_loss
from zoo.perturb import ctx, players


def _true_grad(model, x, y):
    ctx.mode = "bp"
    for p in model.parameters():
        p.grad = None
    per_sample_loss(model(x), y).mean().backward()
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    for p in model.parameters():
        p.grad = None
    return g


def _flat_grad(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()])


def _cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


class _Counter:
    """Counts model forward calls."""

    def __init__(self, model):
        self.n = 0
        self.h = model.register_forward_hook(lambda *a: self._inc())

    def _inc(self):
        self.n += 1


def test_forward_pass_budget_17():
    torch.manual_seed(0)
    for task, x, y in [("mlp_mnist", torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))),
                       ("lstm_ptb", torch.randint(0, 50, (4, 6)), torch.randint(0, 50, (4, 6)))]:
        for name in ("spsa15", "anp15"):
            model = build_model(task, vocab=50)
            opt = get_optimizer(name, model, 0.001, "cls" if task == "mlp_mnist" else "lm", 10 if task == "mlp_mnist" else 50)
            assert opt.forward_passes_per_step == 17
            opt.step(x, y)                    # first step: NP-family one-off shape probing (2 tiny clean passes)
            cnt = _Counter(model)
            loss = opt.step(x, y)
            assert torch.isfinite(loss)
            assert cnt.n == 17, (task, name, cnt.n)


def test_anp15_aligns_with_gradient_on_mlp():
    torch.manual_seed(0)
    model = build_model("mlp_mnist")
    x, y = torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))
    g = _true_grad(model, x, y)
    opt = get_optimizer("anp15", model, 0.01, "cls", 10)
    opt.estimate_grad(x, y, n_passes=200)          # 200 antithetic pairs
    c = _cos(_flat_grad(model), g)
    print(f"\nanp15 MLP cosine (200 pairs, sigma 0.01) = {c:.3f}")
    assert c > 0.5                                  # measured 0.69 (the curvature weights lower ANP's 0.79)
    # with the curvature exponent off the estimator is central-difference ANP: cosine ~0.75
    opt = get_optimizer("anp15", model, 0.001, "cls", 10)
    opt.alpha = 0.0
    opt.estimate_grad(x, y, n_passes=200)
    c0 = _cos(_flat_grad(model), g)
    print(f"anp15 MLP cosine with alpha = 0 (sigma 0.001) = {c0:.3f}")
    assert c0 > 0.65


def test_spsa15_aligns_with_gradient_on_mlp():
    torch.manual_seed(0)
    model = build_model("mlp_mnist")
    x, y = torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))
    g = _true_grad(model, x, y)
    opt = get_optimizer("spsa15", model, 0.001, "cls", 10)
    ghat = opt.estimate_grad(x, y, n_pairs=200)
    c = _cos(ghat, g)
    print(f"spsa15 MLP cosine (200 pairs) = {c:.3f}")
    assert c > 0.05                                 # weight-space: ~sqrt(pairs / n_params)


def test_anp15_token_local_on_sequence_models():
    torch.manual_seed(0)
    N, T, V = 16, 8, 20
    x, y = torch.randint(0, V, (N, T)), torch.randint(0, V, (N, T))
    for model in (CharLSTM(vocab=V, d_emb=8, hidden=16), CharTransformer(vocab=V, d=16, n_heads=2, d_ff=16, n_layers=1, max_len=T)):
        ctx.set_layers(players(model))
        g = _true_grad(model, x, y)
        opt = get_optimizer("anp15", model, 0.001, "lm", V)
        opt.estimate_grad(x, y, n_passes=300)
        c = _cos(_flat_grad(model), g)
        print(f"anp15 {type(model).__name__} cosine (300 pairs, token-local) = {c:.3f}  layouts={opt._layout}")
        assert c > 0.2
        assert "tok" in opt._layout or "call" in opt._layout


def test_anp15_learns_synthetic_problem():
    torch.manual_seed(0)
    means = torch.randn(10, 784) * 1.5
    y = torch.arange(256) % 10
    x = (means[y] + torch.randn(256, 784)).view(256, 1, 28, 28)
    model = build_model("mlp_mnist")
    opt = get_optimizer("anp15", model, 0.01, "cls", 10)
    l0 = float(opt.step(x, y))
    for _ in range(60):
        l = float(opt.step(x, y))
    acc = (opt.predict(x).argmax(1) == y).float().mean().item()
    print(f"anp15 synthetic: loss {l0:.3f} -> {l:.3f}, acc {acc:.3f}")
    assert l < 0.5 * l0 and acc > 0.6
