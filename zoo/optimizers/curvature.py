"""1.5-order (curvature-preconditioned) zero-order rules: ``spsa15`` and ``anp15``.

Budget: **17 forward passes per step = 1 clean + 8 antithetic pairs** (the clean pass is
needed for the three-point curvature estimate), as specified for the survey.

``spsa15`` -- 1.5-SPSA, copied from https://github.com/Fchaubard/1.5-SPSA-LLM
(``scripts/train.py``, ``SPSATrainer.step`` with ``use_curvature=True``)::

    L0 = L(theta)                                        clean pass
    for k = 1..K:   z_k ~ N(0, I)
        L+ = L(theta + eps z_k),  L- = L(theta - eps z_k)
        curv_k  = |L+ - 2 L0 + L-| / eps^2               three-point directional curvature
        c_k     = max(curv_k ** alpha, lambda)           alpha = saturating_alpha = 0.1, lambda = lambda_reg = 1.0
        coeff_k = (L+ - L-) / (2 eps K c_k)
    ghat = sum_k coeff_k z_k

Differences from the repository: it draws Rademacher (+-1, bit-packed) directions and applies
``theta -= lr * ghat`` (plain SGD); the survey uses Gaussian directions like its SPSA and hands
``ghat`` to Adam(lr = eps) as every other method does (PROTOCOL.md).  The returned training
loss is the clean loss L0.

``anp15`` -- 1.5-ANP (central-difference activity perturbation with a three-point curvature
preconditioner), from the specification "1.5-ANP: Curvature-Preconditioned Activity
Perturbation for Constant-Memory Recurrent and Token-Local Pretraining" (F. Chaubard, 2025,
unpublished; reference PyTorch loops for an LSTM and a Transformer block).  Per step one clean
pass records every layer's input h and pre-activation a; each of the K antithetic pairs injects
+sigma xi and -sigma xi (the same xi) into every layer.  For every attribution unit u -- an
example on the image tasks, a (sequence, position) token on the LM tasks ("token-local"
attribution; the LSTM gate layer is called once per position) -- with N_u perturbed nodes::

    d_u^(k)      = a_u^+ - a_u^-                      realized central activity difference, all layers
    ||d_u||^2    = sum over the N_u nodes of u
    c_u^(k)      = 4 N_u |l_u^+ - 2 l_u^0 + l_u^-| / (||d_u||^2 + eps_mach)
    beta_u^(k)   = N_u (l_u^+ - l_u^-) / ((||d_u||^2 + eps_mach) (c_u^(k) + lambda)^alpha)
    ghat_{a_u}   = (1/K) sum_k beta_u^(k) d_u^(k)
    ghat_W       = mean_u ghat_{a_u} h_u^T              (the one-layer local backward of zoo.perturb)

with lambda = 1e-3, alpha = 0.5, eps_mach = 1e-12 (the spec's Transformer values; its LSTM loop
uses lambda = 1e-4).  The transformer's learned position embedding is perturbed once per batch
(shape (T, d)): its nodes count once per position and its weight is the sum over sequences of
beta.  Deviations from the spec's reference loops: the noisy passes run the perturbed
recurrent state of the LSTM without the spec's 0.5 damping of the state deviation (the
perturbable-layer framework runs every noisy pass as an independent forward), and the update
is handed to Adam (protocol) instead of SGD.  The returned training loss is the clean loss.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from ..perturb import ctx
from . import register
from .base import per_sample_loss
from .node_perturbation import _NodePerturbation
from .spsa import SPSA


# ----------------------------------------------------------------------------- 1.5-SPSA
def loss64_fn(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batch-mean loss with the logits cast to fp64 before the cross-entropy: the three-point
    second differences of the curvature rules at sigma = 1e-3 are otherwise at the fp32 rounding
    floor (the forward pass itself stays in fp32)."""
    return per_sample_loss(model(x).double(), y).mean()


@register("spsa15")
class CurvatureSPSA(SPSA):
    forward_passes_per_step = 17          # 1 clean + 8 antithetic pairs
    saturating_alpha = 0.1                # repo default --saturating_alpha
    lambda_reg = 1.0                      # repo default --lambda_reg
    loss64 = False                        # evaluate the losses in fp64 (set by zoo.train --loss64)

    def __init__(self, model, lr, task_kind, n_classes, loss64: bool = False, **kw):
        super().__init__(model, lr, task_kind, n_classes, **kw)
        self.n_pairs = max(1, (self.forward_passes_per_step - 1) // 2)
        self._last_curv: Optional[torch.Tensor] = None
        self.loss64 = bool(loss64)
        if self.loss64:
            self.loss_fn = loss64_fn

    @torch.no_grad()
    def _estimate(self, x, y, n_pairs: int):
        """1 clean pass + n_pairs antithetic pairs; returns (ghat, losses, gs) with
        losses = [L0, L+_1, L-_1, ...] and gs the curvature-scaled coefficients."""
        eps = self.eps
        theta0 = self.flat.get()
        l0 = self._loss(x, y)                                      # clean pass
        ghat = torch.zeros_like(theta0)
        losses, gs, curvs = [l0], [], []
        lam = torch.full_like(l0, self.lambda_reg)
        try:
            for _ in range(n_pairs):
                z = torch.randn_like(theta0)
                self.flat.set(torch.add(theta0, z, alpha=eps))
                l_plus = self._loss(x, y)
                self.flat.set(torch.add(theta0, z, alpha=-eps))
                l_minus = self._loss(x, y)
                curv = (l_plus - 2.0 * l0 + l_minus).abs() / (eps * eps)          # |L+ - 2L0 + L-| / eps^2
                c = torch.maximum(curv.pow(self.saturating_alpha), lam)          # max(curv^alpha, lambda)
                g = (l_plus - l_minus) / (2.0 * eps * c)                          # (L+ - L-) / (2 eps c)
                ghat.addcmul_(z, g)
                losses += [l_plus, l_minus]
                gs.append(g)
                curvs.append(curv)
        finally:
            self.flat.set(theta0)
        ghat.div_(n_pairs)                                                        # the 1/n_perts of the repo
        self._last_curv = torch.stack(curvs).mean()
        return ghat, torch.stack(losses), torch.stack(gs)

    def step(self, x, y):
        ghat, losses, gs = self._estimate(x, y, self.n_pairs)                     # 17 forward passes
        self._last_g = gs
        self.flat.set_grad(ghat)
        self.adam.step()
        return losses[0]                                                          # clean loss

    def extra_log(self) -> dict:
        d = super().extra_log()
        if self._last_curv is not None:
            d["curvature"] = float(self._last_curv)
        return d


# ----------------------------------------------------------------------------- 1.5-ANP
def per_unit_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-example loss (N,) for classification; per-token loss (N, T) for language modelling."""
    if logits.dim() == 3:
        return F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
    return per_sample_loss(logits, y)


@register("anp15")
class CurvatureANP(_NodePerturbation):
    variant = "anp15"
    forward_passes_per_step = 17          # 1 clean + 8 antithetic pairs
    lam = 1e-3                            # curvature damping floor (spec: lambda)
    alpha = 0.5                           # sub-linear curvature exponent (spec: alpha_c)
    eps_mach = 1e-12

    def __init__(self, model, lr, task_kind, n_classes, loss64: bool = False, **kw):
        super().__init__(model, lr, task_kind, n_classes, **kw)
        self.n_pairs = max(1, (self.forward_passes_per_step - 1) // 2)
        self._layout: Optional[List[str]] = None      # per layer: "ex" | "tok" | "call" | "shared"
        self._last_curv: Optional[torch.Tensor] = None
        self.loss64 = bool(loss64)                    # per-unit losses in fp64 (zoo.train --loss64)

    def _unit_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return per_unit_loss(logits.double() if self.loss64 else logits, y)

    # ------------------------------------------------------------------ layouts
    def _layouts(self, N: int, T: Optional[int]) -> List[str]:
        """How each layer's recorded activity maps onto attribution units.
        ``ex``: (N, ...) one entry per example (image tasks);  ``tok``: (N, T, ...) per token;
        ``call``: the layer is called once per position with (N, ...) entries (LSTM gates);
        ``shared``: (T, ...) without a batch dimension (transformer position embedding)."""
        out = []
        for li, layer in enumerate(self.layers):
            a = layer.a
            if self._shared[li]:
                out.append("shared")
            elif T is None:
                out.append("ex")
            elif len(a) == T and all(t.dim() == 2 for t in a):
                out.append("call")
            elif len(a) == 1 and a[0].dim() >= 3 and a[0].shape[1] == T:
                out.append("tok")
            else:
                raise RuntimeError(f"anp15: cannot map layer {li} activity shapes {[tuple(t.shape) for t in a]} onto tokens")
        return out

    def _noisy_pass15(self, x, y, sign: float, bank):
        ctx.mode, ctx.sigma, ctx.noise_layer, ctx.sign, ctx.xi_bank = "noisy", self.eps, None, sign, bank
        for layer in self.layers:
            layer.xi, layer.a_noisy = [], []
        with torch.no_grad():
            loss = self._unit_loss(self.model(x), y)
        return loss, [list(layer.a_noisy) for layer in self.layers]

    # ------------------------------------------------------------------ estimator
    def estimate_grad(self, x: torch.Tensor, y: torch.Tensor, n_passes: Optional[int] = None) -> torch.Tensor:
        """Write the 1.5-ANP estimate into ``p.grad`` and return the detached mean clean loss.
        ``n_passes`` = number of antithetic pairs (default 8)."""
        ctx.set_layers(self.layers)
        if self._shared is None:
            self._shared = self._probe_shared(x)
        self.adam.zero_grad(set_to_none=True)
        N = x.shape[0]
        K = n_passes if n_passes is not None else self.n_pairs

        # ---- clean pass (grad enabled: builds each layer's local graph) ----
        ctx.mode, ctx.noise_layer, ctx.sign, ctx.xi_bank = "clean", None, 1.0, None
        ctx.reset()
        logits = self.model(x)
        l0 = self._unit_loss(logits.detach(), y)                # (N,) or (N, T)
        T = l0.shape[1] if l0.dim() == 2 else None
        if self._layout is None:
            self._layout = self._layouts(N, T)
        G = [[torch.zeros_like(a) for a in layer.a] for layer in self.layers]
        # perturbed nodes per attribution unit (N_u), from the clean activity shapes
        n_units = 0
        for li, layer in enumerate(self.layers):
            lay = self._layout[li]
            for a in layer.a:
                if lay == "ex":
                    n_units += a.numel() // N
                elif lay == "tok":
                    n_units += a.numel() // (N * T)
                elif lay == "call":
                    n_units += a.numel() // N                   # one call = one position
                else:                                            # shared (T, ...)
                    n_units += a.numel() // T
        if self._layout.count("call"):
            # the gate layer contributes once per position, but the loop above added it T times
            for li, layer in enumerate(self.layers):
                if self._layout[li] == "call":
                    n_units -= (len(layer.a) - 1) * (layer.a[0].numel() // N)
        units = float(n_units)
        n_units_t = l0.new_tensor(units)

        abs_dloss = l0.new_zeros(())
        curv_sum = l0.new_zeros(())
        inv = 1.0 / (N * (T or 1) * K)                           # batch-mean over units, mean over pairs
        try:
            for _ in range(K):
                lp, a_plus = self._noisy_pass15(x, y, +1.0, None)
                bank = [list(layer.xi) for layer in self.layers]
                lm, a_minus = self._noisy_pass15(x, y, -1.0, bank)
                D = [[ap - am for ap, am in zip(Lp, Lm)] for Lp, Lm in zip(a_plus, a_minus)]
                # ---- ||d_u||^2 per attribution unit ----
                sq = torch.zeros_like(l0)
                for li, Dl in enumerate(D):
                    lay = self._layout[li]
                    if lay == "ex":
                        for d in Dl:
                            sq = sq + d.reshape(N, -1).pow(2).sum(1)
                    elif lay == "tok":
                        for d in Dl:
                            sq = sq + d.reshape(N, T, -1).pow(2).sum(2)
                    elif lay == "call":
                        sq = sq + torch.stack([d.reshape(N, -1).pow(2).sum(1) for d in Dl], 1)
                    else:
                        for d in Dl:
                            sq = sq + d.reshape(T, -1).pow(2).sum(1)[None, :]
                sq = sq + self.eps_mach
                second = (lp - 2.0 * l0 + lm).abs()
                c = 4.0 * n_units_t * second / sq                               # three-point curvature
                beta = n_units_t * (lp - lm) / (sq * (c + self.lam).pow(self.alpha))
                w = beta * inv
                # ---- accumulate ghat_a = beta d per layer / call ----
                for li, Dl in enumerate(D):
                    lay = self._layout[li]
                    for i, d in enumerate(Dl):
                        if lay == "ex":
                            G[li][i].add_(w.view(-1, *([1] * (d.dim() - 1))) * d)
                        elif lay == "tok":
                            G[li][i].add_(w.view(N, T, *([1] * (d.dim() - 2))) * d)
                        elif lay == "call":
                            G[li][i].add_(w[:, i].view(-1, *([1] * (d.dim() - 1))) * d)
                        else:
                            G[li][i].add_(w.sum(0).view(T, *([1] * (d.dim() - 1))) * d)
                abs_dloss = abs_dloss + (lp - lm).abs().mean()
                curv_sum = curv_sum + c.mean()
        finally:
            ctx.sign, ctx.xi_bank = 1.0, None
        for layer, Gl in zip(self.layers, G):
            if layer.a:
                torch.autograd.backward(layer.a, grad_tensors=Gl)
        self._last_abs_dloss = abs_dloss / K
        self._last_curv = curv_sum / K
        ctx.mode = "bp"
        return l0.mean()

    def extra_log(self) -> dict:
        d = super().extra_log()
        if self._last_curv is not None:
            d["curvature"] = float(self._last_curv)
        return d
