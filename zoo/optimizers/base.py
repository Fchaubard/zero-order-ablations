"""Optimizer interface.

An optimizer owns a model and an Adam instance (lr = epsilon, betas = (0.9, 0.99)) and
implements ``step(x, y)``: do whatever forward passes it needs (16 for the zero-order
methods), write a gradient estimate into ``p.grad`` for every model parameter, call
``self.adam.step()`` and return the (detached, scalar) training loss of the clean,
unperturbed model on the batch.

``per_sample_loss(logits, y)`` is the single loss definition shared by all methods:
cross-entropy per example for classification, mean next-character cross-entropy per
sequence for language modelling.  The reported "loss" everywhere is its mean.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

ADAM_BETAS = (0.9, 0.99)


def per_sample_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:  # (N, T, V) language modelling
        return F.cross_entropy(logits.transpose(1, 2), y, reduction="none").mean(1)
    return F.cross_entropy(logits, y, reduction="none")


class BaseOptimizer:
    name = "base"
    forward_passes_per_step = 16   # the budget every zero-order method must respect

    def __init__(self, model: torch.nn.Module, lr: float, task_kind: str, n_classes: int,
                 sigma: float | None = None, n_passes: int | None = None, weight_decay: float = 0.0,
                 optim: str = "adam", loss64: bool = False, **kw):
        self.model = model
        self.lr = lr
        # PROTOCOL.md ties the probe scale to the learning rate (sigma = None -> eps = lr); the
        # decoupling ablation passes an explicit sigma.
        self.eps = lr if sigma is None else float(sigma)
        self.task_kind = task_kind   # "cls" or "lm"
        self.n_classes = n_classes
        # Forward-pass budget per step (protocol: 16); subclasses derive their probe counts from it.
        if n_passes is not None:
            self.forward_passes_per_step = int(n_passes)
        self.weight_decay = float(weight_decay)
        self.optim = optim
        # ``self.adam`` is the step taken on p.grad: Adam (protocol), AdamW (backprop decay ablation) or plain SGD
        # without momentum (theta <- theta - lr * ghat, the update of the 1.5-SPSA paper and the 1.5-ANP spec).
        if optim == "sgd":
            self.adam = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=self.weight_decay)
        elif self.weight_decay > 0:   # decoupled weight decay (AdamW); only used for the backprop reference ablation
            self.adam = torch.optim.AdamW(model.parameters(), lr=lr, betas=ADAM_BETAS, weight_decay=self.weight_decay)
        else:
            self.adam = torch.optim.Adam(model.parameters(), lr=lr, betas=ADAM_BETAS)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_log(self) -> dict:
        """Optional extra scalars to log (e.g. acceptance rate for random search)."""
        return {}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Logits used for evaluation.  Default: the plain model forward in "bp" mode.
        Forward-forward overrides this (it normalises between blocks and embeds a
        neutral label), so evaluation code must call ``opt.predict`` not ``model``."""
        from ..perturb import ctx
        ctx.mode = "bp"
        return self.model(x)
