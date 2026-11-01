"""Random search (jumping): (1+lambda) greedy random search with lambda = 15.

Per step (16 forward passes):

    L0    = L(theta)                              1 clean pass
    z_k ~ N(0, I_n),  L_k = L(theta + eps z_k)    k = 1..15 candidates
    k*    = argmin_k L_k
    ghat  = ((L_k* - L0) / eps) * z_k*   if L_k* < L0   (negative scalar times the winning
                                                        direction, so Adam moves along +z_k*)
          = 0                            otherwise      (Adam still steps: momentum only)

``ghat`` is written into every ``p.grad`` and Adam(lr = eps) always takes a step.  The
returned training loss is L0 (the clean, unperturbed loss).  Everything stays on the
device: the winner is picked with ``torch.argmin`` / ``torch.where``; the Python floats
for logging are computed lazily in ``extra_log``.

All forward passes run with ``ctx.mode = "bp"`` under ``torch.no_grad()``.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from ..perturb import ctx
from . import register
from .base import BaseOptimizer
from .spsa import _FlatParams, default_loss_fn


@register("rs")
class RandomSearch(BaseOptimizer):
    forward_passes_per_step = 16
    n_candidates = 15        # 1 clean pass + 15 candidates
    acceptance_window = 100  # steps over which the acceptance rate is averaged

    def __init__(self, model, lr, task_kind, n_classes,
                 loss_fn: Optional[Callable] = None, **kw):
        super().__init__(model, lr, task_kind, n_classes, **kw)
        self.loss_fn = loss_fn if loss_fn is not None else default_loss_fn
        self.flat = _FlatParams([p for p in model.parameters() if p.requires_grad])
        dev = self.flat.params[0].device
        # ring buffer of accept flags (1.0 = the step found an improving candidate)
        self._accepts = torch.zeros(self.acceptance_window, device=dev)
        self._n_steps = 0
        self._last_gain: Optional[torch.Tensor] = None   # L0 - L_k* of the last step (0-dim)

    @torch.no_grad()
    def _loss(self, x, y) -> torch.Tensor:
        ctx.mode = "bp"
        return self.loss_fn(self.model, x, y).detach()

    @torch.no_grad()
    def _search(self, x, y):
        """Returns (L0, ghat, improved, gain); parameters restored exactly."""
        eps = self.eps
        theta0 = self.flat.get()
        l0 = self._loss(x, y)                                          # clean pass
        Z = torch.randn(self.n_candidates, theta0.numel(), device=theta0.device, dtype=theta0.dtype)
        losses = []
        try:
            for k in range(self.n_candidates):
                self.flat.set(torch.add(theta0, Z[k], alpha=eps))      # theta + eps z_k
                losses.append(self._loss(x, y))
        finally:
            self.flat.set(theta0)                                      # exact restore
        losses = torch.stack(losses)                                   # (15,)
        k_star = torch.argmin(losses)
        l_star = losses[k_star]
        improved = l_star < l0
        scale = torch.where(improved, (l_star - l0) / eps, torch.zeros_like(l0))
        ghat = Z[k_star] * scale                                       # 0 when nothing improved
        return l0, ghat, improved, l0 - l_star

    def step(self, x, y):
        l0, ghat, improved, gain = self._search(x, y)                  # 16 forward passes
        self._accepts[self._n_steps % self.acceptance_window] = improved.to(self._accepts.dtype)
        self._n_steps += 1
        self._last_gain = gain
        self.flat.set_grad(ghat)
        self.adam.step()                                               # always (momentum-only when ghat = 0)
        return l0

    def extra_log(self) -> dict:
        if self._n_steps == 0:
            return {}
        n = min(self._n_steps, self.acceptance_window)
        return {
            "acceptance_rate": float(self._accepts.sum() / n),         # running mean over the last <=100 steps
            "best_improvement": float(self._last_gain),                # L0 - L_k* (negative if no candidate improved)
        }
