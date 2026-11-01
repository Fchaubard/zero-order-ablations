"""Perturbable layers shared by every model and every optimizer.

Every trainable layer in every model is a PLinear, PConv2d or PEmbedding.  A single
global context (``ctx``) decides what a layer does on forward:

``ctx.mode == "bp"``
    Plain differentiable forward.  Used by backprop, by the forward-forward local
    losses and by evaluation.
``ctx.mode == "clean"``
    Forward with the layer *input detached*.  Each layer appends its input ``h``
    and its pre-activation ``a`` to ``layer.h`` / ``layer.a``.  ``a`` keeps a local
    autograd graph to the layer's own parameters only, so
    ``torch.autograd.backward(layer.a, grad_tensors=G)`` computes
    ``dL/dW = G^T h`` (and the matching conv / embedding form) for *any* layer type
    without back-propagating through nonlinearities.  This is the node-perturbation
    weight update written as a one-layer backward.
``ctx.mode == "noisy"``
    Graph-free forward that adds Gaussian noise ``ctx.sigma * xi`` to the
    pre-activation of every layer (or only to layer ``ctx.noise_layer`` for
    iterative NP).  Records ``layer.xi`` (the unit-variance probe) and
    ``layer.a_noisy``.  Downstream layers see the perturbed activity, so the
    recorded ``a_noisy - a`` of later layers includes propagated perturbation
    (used by activity-based NP).

Decorrelation (DANP):  ``layer.enable_decorrelation()`` adds a square matrix ``R``
(initialised to identity, NOT an optimizer parameter) that multiplies the layer
input before the weight: ``a = W (R h) + b``.  ``R`` is trained with the local
decorrelation rule of Ahmad et al. (2022) via ``layer.decorrelation_update(lr)``,
using the inputs recorded on the last clean pass.  For PConv2d the decorrelated
vector is the unfolded ``C_in*k*k`` patch.  PEmbedding never decorrelates (its
input is a one-hot index).

A layer called several times in one forward pass (the LSTM gate layer is called
once per time step) accumulates one entry per call in its lists; ``ctx.reset()``
clears the lists before each pass.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Ctx:
    def __init__(self):
        self.mode: str = "bp"          # "bp" | "clean" | "noisy"
        self.sigma: float = 0.0        # probe std used in "noisy" mode
        self.noise_layer: Optional[int] = None  # None = perturb all layers (NP/ANP/DANP); int = only that layer (INP)
        self.layers: List["PLayer"] = []

    def set_layers(self, layers: List["PLayer"]):
        self.layers = list(layers)
        for i, l in enumerate(self.layers):
            l.index = i

    def reset(self):
        for l in self.layers:
            l.clear()


ctx = _Ctx()


def xavier_(w: torch.Tensor):
    nn.init.xavier_uniform_(w)


class PLayer(nn.Module):
    """Base class: bookkeeping shared by all perturbable layers."""

    def __init__(self):
        super().__init__()
        self.index: int = -1
        self.R: Optional[torch.Tensor] = None   # decorrelation matrix (buffer), None = off
        self.clear()

    def clear(self):
        self.h: List[torch.Tensor] = []        # inputs on the clean pass (after decorrelation)
        self.a: List[torch.Tensor] = []        # pre-activations on the clean pass (with local graph)
        self.xi: List[Optional[torch.Tensor]] = []  # unit probes on the noisy pass (None if this layer was not perturbed)
        self.a_noisy: List[torch.Tensor] = []  # pre-activations on the noisy pass
        self.hd: List[torch.Tensor] = []       # decorrelated inputs flattened to (M, d) for the R update

    # ---- decorrelation ------------------------------------------------------------
    @property
    def decor_dim(self) -> int:
        raise NotImplementedError

    def enable_decorrelation(self):
        d = self.decor_dim
        # ``__init__`` stores the placeholder ``self.R = None`` as a plain attribute and
        # ``register_buffer`` refuses to shadow an existing attribute (KeyError), so drop
        # the placeholder first.  R is created on the layer's own device / dtype so that
        # a model already on the GPU does not get a CPU buffer.
        if "R" not in self._buffers:
            self.__dict__.pop("R", None)
        self.register_buffer("R", torch.eye(d, device=self.weight.device, dtype=self.weight.dtype))

    def _decorrelate(self, x2d: torch.Tensor) -> torch.Tensor:
        """x2d: (M, d) rows -> (M, d) decorrelated rows (R x)."""
        xd = x2d @ self.R.T
        if ctx.mode == "clean":
            self.hd.append(xd.detach())
        return xd

    @torch.no_grad()
    def decorrelation_update(self, lr: float):
        """Local rule of Ahmad et al. 2022 / Dalm et al. 2023:
        R <- R - lr * (C - diag(C)) R,  C = E[x_d x_d^T] over the recorded rows."""
        if self.R is None or not self.hd:
            return
        x = torch.cat(self.hd, 0)
        C = x.T @ x / x.shape[0]
        C.diagonal().zero_()
        self.R -= lr * (C @ self.R)

    # ---- core: record / perturb a computed pre-activation ---------------------------
    def _emit(self, a: torch.Tensor) -> torch.Tensor:
        mode = ctx.mode
        if mode == "clean":
            self.a.append(a)
            return a
        if mode == "noisy":
            if ctx.noise_layer is None or ctx.noise_layer == self.index:
                xi = torch.randn_like(a)
                a = a + ctx.sigma * xi
                self.xi.append(xi)
            else:
                self.xi.append(None)
            self.a_noisy.append(a)
            return a
        return a


class PLinear(PLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        xavier_(self.weight)

    @property
    def decor_dim(self):
        return self.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = ctx.mode
        if mode == "clean":
            x = x.detach()
        if self.R is not None:
            shp = x.shape
            x = self._decorrelate(x.reshape(-1, shp[-1])).reshape(shp)
        if mode == "noisy":
            with torch.no_grad():
                a = F.linear(x, self.weight, self.bias)
            return self._emit(a)
        if mode == "clean":
            self.h.append(x)
        a = F.linear(x, self.weight, self.bias)
        return self._emit(a)


class PConv2d(PLayer):
    def __init__(self, in_ch: int, out_ch: int, k: int, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_ch, self.out_ch, self.k, self.padding = in_ch, out_ch, k, padding
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, k, k))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        xavier_(self.weight)

    @property
    def decor_dim(self):
        return self.in_ch * self.k * self.k

    def _conv(self, x):
        if self.R is None:
            return F.conv2d(x, self.weight, self.bias, padding=self.padding)
        # decorrelated conv: unfold patches, decorrelate each C*k*k patch vector, matmul with W.
        N, C, H, W = x.shape
        cols = F.unfold(x, self.k, padding=self.padding)            # (N, C*k*k, L)
        L = cols.shape[-1]
        rows = cols.transpose(1, 2).reshape(N * L, -1)              # (N*L, C*k*k)
        rows = self._decorrelate(rows)
        out = rows @ self.weight.reshape(self.out_ch, -1).T          # (N*L, out)
        if self.bias is not None:
            out = out + self.bias
        Ho = H + 2 * self.padding - self.k + 1
        Wo = W + 2 * self.padding - self.k + 1
        return out.reshape(N, L, self.out_ch).transpose(1, 2).reshape(N, self.out_ch, Ho, Wo)

    def forward(self, x):
        mode = ctx.mode
        if mode == "clean":
            x = x.detach()
            self.h.append(x)
        if mode == "noisy":
            with torch.no_grad():
                a = self._conv(x)
            return self._emit(a)
        return self._emit(self._conv(x))


class PEmbedding(PLayer):
    """Embedding = linear layer on a one-hot input; the node is the embedding output."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, dim))
        xavier_(self.weight)

    @property
    def decor_dim(self):
        return 0

    def enable_decorrelation(self):
        pass  # one-hot inputs are already decorrelated

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        mode = ctx.mode
        if mode == "clean":
            self.h.append(idx)
        if mode == "noisy":
            with torch.no_grad():
                a = F.embedding(idx, self.weight)
            return self._emit(a)
        return self._emit(F.embedding(idx, self.weight))


def players(model: nn.Module) -> List[PLayer]:
    """All perturbable layers of a model in module-registration order."""
    return [m for m in model.modules() if isinstance(m, PLayer)]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def layer_params(layer: PLayer):
    return [p for p in (layer.weight, getattr(layer, "bias", None)) if p is not None]
