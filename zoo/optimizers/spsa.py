"""SPSA / Gaussian-smoothing gradient estimator with central differences (MeZO-style).

Estimator (8 antithetic pairs = 16 forward passes per step).  With theta the flat
parameter vector, eps the probe scale (== lr) and L(.) the batch-mean loss:

    z_k ~ N(0, I_n)                                   k = 1..K   (K = 8)
    L+_k = L(theta + eps z_k),   L-_k = L(theta - eps z_k)
    g_k  = (L+_k - L-_k) / (2 eps)                    projected (directional) gradient
    ghat = (1/K) sum_k g_k z_k                        unbiased for grad of the Gaussian-
                                                      smoothed loss E_z[L(theta + eps z)]

``ghat`` is written into every ``p.grad`` and Adam(lr = eps) takes the step.

Mechanics mirror MeZO (Malladi et al. 2023, "Fine-Tuning Language Models with Just
Forward Passes", https://github.com/princeton-nlp/MeZO, file
``large_models/trainer.py``):

* ``zo_perturb_parameters``: perturb the parameters *in place*,
  ``param.data = param.data + scaling_factor * z * zo_eps`` with ``z`` drawn by a
  seeded ``torch.normal`` so the probe never has to be stored.
* ``zo_step``: perturb with scaling_factor=+1, loss1; scaling_factor=-2, loss2;
  ``projected_grad = (loss1 - loss2) / (2 * zo_eps)``; scaling_factor=+1 to reset.
* ``zo_update``: re-sample the same ``z`` from the saved seed and apply
  ``param.data -= lr * projected_grad * z``.

Differences from MeZO: (i) we average K = 8 pairs per step instead of one;
(ii) the model is only ~40k parameters, so instead of re-generating ``z`` from a seed
we hold the probe for the duration of the pair and accumulate ``g_k z_k`` on the fly;
(iii) instead of MeZO's "+1, -2, +1" in-place walk (which restores theta only up to
floating-point round-off) we keep a flat copy ``theta0`` of the parameters and write
``theta0 +/- eps z`` into the model, so the restore is exact; (iv) the update is
handed to Adam through ``p.grad`` rather than applied as plain SGD.

All forward passes run with ``ctx.mode = "bp"`` under ``torch.no_grad()`` (plain
forward, nothing recorded).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch

from ..perturb import ctx
from . import register
from .base import BaseOptimizer, per_sample_loss


def default_loss_fn(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batch-mean loss: the mean of ``per_sample_loss`` (the shared loss definition)."""
    return per_sample_loss(model(x), y).mean()


class _FlatParams:
    """Read / write all parameters of a model as one flat vector (in place; the
    parameters keep their storage, so Adam's references stay valid)."""

    def __init__(self, params: List[torch.nn.Parameter]):
        self.params = params
        self.slices = []
        off = 0
        for p in params:
            self.slices.append((off, off + p.numel()))
            off += p.numel()
        self.numel = off

    @torch.no_grad()
    def get(self) -> torch.Tensor:
        """A fresh (n,) copy of the current parameters."""
        return torch.cat([p.reshape(-1) for p in self.params])

    @torch.no_grad()
    def set(self, vec: torch.Tensor) -> None:
        """Copy the flat vector ``vec`` into the parameters (one fused foreach copy)."""
        torch._foreach_copy_(
            self.params, [vec[a:b].view_as(p) for p, (a, b) in zip(self.params, self.slices)]
        )

    def set_grad(self, g: torch.Tensor) -> None:
        """Expose slices of the flat (pseudo-)gradient ``g`` as ``p.grad`` for Adam."""
        for p, (a, b) in zip(self.params, self.slices):
            p.grad = g[a:b].view_as(p)


@register("spsa")
class SPSA(BaseOptimizer):
    forward_passes_per_step = 16
    n_pairs = 8   # antithetic pairs per step -> 2 * n_pairs forward passes

    def __init__(self, model, lr, task_kind, n_classes,
                 loss_fn: Optional[Callable] = None, **kw):
        super().__init__(model, lr, task_kind, n_classes, **kw)
        self.n_pairs = max(1, self.forward_passes_per_step // 2)   # 16 passes -> 8 antithetic pairs
        self.loss_fn = loss_fn if loss_fn is not None else default_loss_fn
        self.flat = _FlatParams([p for p in model.parameters() if p.requires_grad])
        self._last_g: Optional[torch.Tensor] = None   # (n_pairs,) projected gradients of the last step

    @torch.no_grad()
    def _loss(self, x, y) -> torch.Tensor:
        ctx.mode = "bp"
        return self.loss_fn(self.model, x, y).detach()

    @torch.no_grad()
    def _estimate(self, x, y, n_pairs: int):
        """Run ``n_pairs`` antithetic pairs.  Returns (ghat, losses, gs) with
        ghat the flat (n,) estimate, losses the 2*n_pairs probe losses and gs the
        n_pairs projected gradients.  Parameters are restored exactly on return."""
        eps = self.eps
        theta0 = self.flat.get()
        ghat = torch.zeros_like(theta0)
        losses, gs = [], []
        try:
            for _ in range(n_pairs):
                z = torch.randn_like(theta0)                       # z_k ~ N(0, I)
                self.flat.set(torch.add(theta0, z, alpha=eps))     # theta + eps z
                l_plus = self._loss(x, y)
                self.flat.set(torch.add(theta0, z, alpha=-eps))    # theta - eps z
                l_minus = self._loss(x, y)
                g = (l_plus - l_minus) / (2.0 * eps)               # projected gradient (0-dim, stays on device)
                ghat.addcmul_(z, g)                                # ghat += g_k z_k
                losses += [l_plus, l_minus]
                gs.append(g)
        finally:
            self.flat.set(theta0)                                  # exact restore
        ghat.div_(n_pairs)
        return ghat, torch.stack(losses), torch.stack(gs)

    def estimate_grad(self, x, y, n_pairs: Optional[int] = None) -> torch.Tensor:
        """Flat SPSA gradient estimate averaged over ``n_pairs`` antithetic pairs.
        Does not touch ``p.grad`` or Adam; parameters are left unchanged."""
        ghat, _, _ = self._estimate(x, y, self.n_pairs if n_pairs is None else n_pairs)
        return ghat

    def step(self, x, y):
        ghat, losses, gs = self._estimate(x, y, self.n_pairs)      # 16 forward passes
        self._last_g = gs
        self.flat.set_grad(ghat)
        self.adam.step()
        return losses.mean()   # mean of the 16 probe losses (no extra clean pass)

    def extra_log(self) -> dict:
        if self._last_g is None:
            return {}
        return {"proj_grad_std": float(self._last_g.std())}
