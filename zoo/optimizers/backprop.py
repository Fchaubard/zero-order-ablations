"""Backpropagation reference: one forward + one backward per step, Adam(lr)."""
from __future__ import annotations

import torch

from ..perturb import ctx
from . import register
from .base import BaseOptimizer, per_sample_loss


@register("bp")
class Backprop(BaseOptimizer):
    forward_passes_per_step = 1

    def step(self, x, y):
        ctx.mode = "bp"
        self.adam.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = per_sample_loss(logits, y).mean()
        loss.backward()
        self.adam.step()
        return loss.detach()
