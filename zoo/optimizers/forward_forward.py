"""Hinton's forward-forward algorithm (FF) as an "optimizer": local, layer-wise
goodness losses, no backpropagation across blocks.

References (copied, not invented)
---------------------------------
* G. Hinton, "The Forward-Forward Algorithm: Some Preliminary Investigations",
  arXiv:2212.13345 (2022).  Eq. (1): p(positive) = sigmoid(sum_j y_j^2 - theta); the
  length of every hidden vector is normalised before it is fed to the next layer so
  that only the *orientation* is passed on (sec. 2.1); supervised MNIST embeds the
  label in the first 10 pixels, negatives carry an incorrect label (sec. 3.3); a
  "neutral label composed of ten entries of 0.1" is used for a single-pass readout
  through a softmax learned during training (sec. 3.3).
* M. Pezeshki, pytorch_forward_forward, https://github.com/mohammadpz/pytorch_forward_forward
  (main.py): ``overlay_y_on_x`` (zero the first 10 inputs, write ``x.max()`` at the
  label index), ``Layer.forward`` (``x / (x.norm(2, 1, keepdim=True) + 1e-4)`` then
  linear + ReLU), ``Layer.train`` (goodness ``h.pow(2).mean(1)``, ``threshold = 2.0``,
  ``loss = log(1 + exp(cat([-g_pos + threshold, g_neg - threshold]))).mean()``),
  ``Net.predict`` (overlay each candidate label, sum the goodness over *all* layers,
  argmax), negatives = images with shuffled (wrong) labels.

How it maps onto this code base
-------------------------------
Every model is ``readout(block_L(...block_1(embed(x))))``.  A *block* of the model is
one FF "layer": it is trained with its own goodness loss on the length-normalised,
detached output of the previous block, so no gradient ever crosses a block boundary.
``ctx.mode`` stays ``"bp"`` throughout (plain differentiable forward inside a block).

Images (task_kind "cls"), mohammadpz / Hinton sec. 3.3 recipe:
  positive = overlay(x, y); negative = overlay(x, wrong label); neutral = all ten label
  entries set to 0.1 (scaled by ``x.max()`` like the label itself).  The label is written
  into the first ``n_classes`` entries of ``x.flatten(1)``; for CIFAR that is the first
  10 pixels of channel 0, row 0 - deliberately simple.  The readout (softmax) is
  trained by cross-entropy on the length-normalised features of the last block from
  the neutral-label pass over the real images (Hinton: "a neutral label ... used as
  the inputs to a softmax that has been learned during training").
Sequences (task_kind "lm"), unsupervised FF with corrupted sequences as negatives:
  positive = x; negative = x with ``corrupt_frac`` (30 %) of the positions replaced by
  uniformly random tokens (re-sampled every step).  Goodness of an example is the mean
  of the squared activities over positions and features.  Normalisation between blocks
  is per position over the feature dimension.  The readout is trained by next-token
  cross-entropy at every position on the (detached, normalised) features of the real
  sequences.  No published FF recipe for LSTM / transformer language models was found
  (Hinton's own next-character experiment, sec. 6, used self-generated negatives and
  could not be replicated by him), so this is the closest faithful analogue.

Threshold.  mohammadpz's ``threshold = 2.0`` is used for every block whose activity is
unbounded (ReLU MLP / conv blocks, the transformer's residual stream).  The LSTM block
output is ``sigmoid(o) * tanh(c)`` which lies in (-1, 1), so its goodness (mean of
squares) is bounded by 1 and a threshold of 2.0 is unattainable: the positive term of
the loss then dominates and simply saturates every unit for positives *and* negatives
(measured: no separation at all).  For bounded blocks we therefore use
``threshold = 0.25``, a quarter of the attainable goodness range (measured on
synthetic text: real 0.59 vs corrupted 0.33 goodness after 300 steps).  The per-block
thresholds are stored in ``self.thresholds``.

Budget: one positive and one negative forward pass per step (plus, for classification,
the neutral-label pass that feeds the readout, batched together with the other two).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ..perturb import ctx
from . import register
from .base import BaseOptimizer, per_sample_loss


@register("ff")
class ForwardForward(BaseOptimizer):
    forward_passes_per_step = 2   # positive + negative (the 16-pass budget is for probe methods)

    def __init__(self, model, lr, task_kind, n_classes, threshold: float = 2.0,
                 corrupt_frac: float = 0.3, neutral_value: float = 0.1, **kw):
        super().__init__(model, lr, task_kind, n_classes, **kw)
        self.threshold = float(threshold)          # mohammadpz: self.threshold = 2.0
        from ..models import LSTMLayer
        self.bounded_threshold = 0.25              # for blocks whose activity lies in (-1, 1)
        self.thresholds = [self.bounded_threshold if isinstance(b, LSTMLayer) else self.threshold
                           for b in model.blocks]
        self.corrupt_frac = float(corrupt_frac)    # LM negatives: fraction of corrupted positions
        self.neutral_value = float(neutral_value)  # Hinton: neutral label = ten entries of 0.1
        self.ln_f = getattr(model, "ln_f", None)   # CharTransformer applies ln_f before the readout
        self._g_pos: Optional[torch.Tensor] = None
        self._g_neg: Optional[torch.Tensor] = None
        self._readout_loss: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ helpers
    def _normalise(self, h: torch.Tensor) -> torch.Tensor:
        """Length normalisation (Hinton sec. 2.1; mohammadpz Layer.forward:
        ``x / (x.norm(2, 1, keepdim=True) + 1e-4)``).  Images: per example over all
        non-batch dims.  Sequences (N, T, D): per position over D."""
        if self.task_kind == "lm":
            return h / (h.norm(2, dim=-1, keepdim=True) + 1e-4)
        n = h.flatten(1).norm(2, dim=1)
        return h / (n.view(-1, *([1] * (h.dim() - 1))) + 1e-4)

    @staticmethod
    def _goodness(out: torch.Tensor) -> torch.Tensor:
        """mohammadpz: ``h.pow(2).mean(1)`` - mean of squared activities over every
        non-batch dim (features; channels x H x W; positions x features)."""
        return out.pow(2).flatten(1).mean(1)

    def _ff_loss(self, g_pos: torch.Tensor, g_neg: torch.Tensor, threshold: float) -> torch.Tensor:
        """mohammadpz Layer.train:
        ``torch.log(1 + torch.exp(torch.cat([-g_pos + threshold, g_neg - threshold]))).mean()``
        written with softplus (identical function, overflow-safe)."""
        return F.softplus(torch.cat([-g_pos + threshold, g_neg - threshold])).mean()

    def _overlay(self, x: torch.Tensor, y: Optional[torch.Tensor], xmax: Optional[torch.Tensor] = None):
        """mohammadpz ``overlay_y_on_x``: zero the first ``n_classes`` entries of the
        flattened image and write ``x.max()`` at the label index.  ``y is None`` writes
        Hinton's neutral label (every entry ``neutral_value``, scaled by ``x.max()``).
        For (N,3,32,32) CIFAR the first 10 entries are channel 0, row 0, pixels 0..9."""
        N = x.shape[0]
        xmax = x.max() if xmax is None else xmax
        flat = x.flatten(1).clone()
        flat[:, : self.n_classes] = 0.0
        if y is None:
            flat[:, : self.n_classes] = self.neutral_value * xmax
        else:
            flat[torch.arange(N, device=x.device), y] = xmax
        return flat.reshape(x.shape)

    def _wrong_labels(self, y: torch.Tensor) -> torch.Tensor:
        """A label sampled uniformly from the n_classes - 1 incorrect classes."""
        return (y + torch.randint(1, self.n_classes, y.shape, device=y.device)) % self.n_classes

    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        """LM negative: replace ``corrupt_frac`` of the positions by random tokens."""
        mask = torch.rand(x.shape, device=x.device) < self.corrupt_frac
        rnd = torch.randint(0, self.n_classes, x.shape, device=x.device)
        return torch.where(mask, rnd, x)

    def _readout(self, feats: torch.Tensor) -> torch.Tensor:
        if self.ln_f is not None:
            feats = self.ln_f(feats)
        return self.model.readout(feats)

    # ------------------------------------------------------------------ training
    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ctx.mode = "bp"
        self.adam.zero_grad(set_to_none=True)
        N = x.shape[0]

        # --- build the positive / negative (/ neutral) batch -------------------------
        if self.task_kind == "cls":
            xmax = x.max()                           # once per batch, like mohammadpz
            x_all = torch.cat([self._overlay(x, y, xmax),
                               self._overlay(x, self._wrong_labels(y), xmax),
                               self._overlay(x, None, xmax)], 0)     # (3N, ...)
            readout_slice = slice(2 * N, 3 * N)      # readout learns from the neutral pass
        else:
            x_all = torch.cat([x, self._corrupt(x)], 0)              # (2N, T)
            readout_slice = slice(0, N)              # readout learns from the real sequences

        # --- local, block-wise goodness losses (one pass over the batch) --------------
        # Block 0 is NOT detached from the embedding: the token / position embeddings are
        # trained by block 0's goodness loss (for images embed is the identity).
        h = self.model.embed(x_all)
        g_pos_sum = 0.0
        g_neg_sum = 0.0
        for i, block in enumerate(self.model.blocks):
            inp = self._normalise(h)
            if i > 0:
                inp = inp.detach()                   # gradients stay inside this block
            out = block(inp)
            g = self._goodness(out)
            g_pos, g_neg = g[:N], g[N:2 * N]
            loss = self._ff_loss(g_pos, g_neg, self.thresholds[i])
            loss.backward()                          # local: reaches only this block (+ embed for i == 0)
            g_pos_sum = g_pos_sum + g_pos.detach().mean()
            g_neg_sum = g_neg_sum + g_neg.detach().mean()
            h = out.detach()

        # --- readout: cross-entropy on detached, normalised last-block features --------
        feats = self._normalise(h[readout_slice]).detach()
        logits = self._readout(feats)
        ce = per_sample_loss(logits, y).mean()
        ce.backward()

        # one Adam step: every parameter received gradient from exactly one local loss
        self.adam.step()

        nb = len(self.model.blocks)
        self._g_pos = g_pos_sum / nb          # kept as tensors: no GPU sync inside step
        self._g_neg = g_neg_sum / nb
        self._readout_loss = ce.detach()
        return ce.detach()

    def extra_log(self) -> dict:
        if self._g_pos is None:
            return {}
        return {"ff_goodness_pos": float(self._g_pos), "ff_goodness_neg": float(self._g_neg)}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _run_blocks(self, h: torch.Tensor):
        """FF forward through the blocks with length normalisation in between.
        Returns the final activity and the per-block goodness (N,) list."""
        goodness = []
        for block in self.model.blocks:
            h = block(self._normalise(h))
            goodness.append(self._goodness(h))
        return h, goodness

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Logits for evaluation: neutral-label single pass (cls) or plain pass (lm),
        normalising between blocks exactly as in training, then the trained readout."""
        ctx.mode = "bp"
        if self.task_kind == "cls":
            x = self._overlay(x, None)
        h, _ = self._run_blocks(self.model.embed(x))
        return self._readout(self._normalise(h))

    @torch.no_grad()
    def predict_goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Hinton's label-accumulation inference (mohammadpz ``Net.predict``):
        overlay each candidate label, run the blocks, sum the goodness over all blocks.
        Returns the (N, n_classes) goodness matrix; ``argmax(1)`` is the prediction.
        Classification only."""
        if self.task_kind != "cls":
            raise ValueError("predict_goodness is defined for classification only")
        ctx.mode = "bp"
        xmax = x.max()
        cols = []
        for label in range(self.n_classes):
            y = torch.full((x.shape[0],), label, dtype=torch.long, device=x.device)
            _, goodness = self._run_blocks(self.model.embed(self._overlay(x, y, xmax)))
            cols.append(sum(goodness).unsqueeze(1))
        return torch.cat(cols, 1)
