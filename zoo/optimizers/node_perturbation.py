"""Node-perturbation family: ``np``, ``anp``, ``inp``, ``danp``.

Update rules are copied from

* Dalm, van Gerven, Ahmad, "Node Perturbation Can Effectively Train Multi-Layer Neural
  Networks" (arXiv:2310.00965, https://arxiv.org/html/2310.00965v7): Eq. 5 (NP),
  Eq. 3 (INP), Eq. 6 (ANP), Algorithm 1 (DANP) and Appendix D (sigma, alpha);
* the authors' reference implementation
  https://github.com/SanderDalm/DANP_open_source (algorithms.py), which fixes the
  conventions the paper leaves open: per-example loss differences, the batch MEAN of the
  per-example updates, N = total number of perturbed (pre-activation) units of the
  network (the input layer is not counted), the per-example norm ``||delta a||`` taken
  over the concatenation of every layer's activity difference, and the mean over the
  noise samples when several noisy passes are used per update.

Notation (per layer l, per example n):  ``a_l`` pre-activation, ``x_{l-1}`` layer input
(the decorrelated input ``x*_{l-1} = R_{l-1} x_{l-1}`` when decorrelation is on),
``eps_l = sigma * xi_l`` injected noise (``xi`` unit Gaussian), ``delta L = L_noisy -
L_clean`` the per-example loss difference and ``delta a_l = a~_l - a_l`` the measured
activity difference (which for downstream layers includes propagated perturbation).

    NP   (Eq. 5)  dL/dW_l ~= delta_L * eps_l / sigma^2              x_{l-1}^T   all layers perturbed
    INP  (Eq. 3)  dL/dW_l ~= N_l * delta_L * eps_l / ||eps_l||^2     x_{l-1}^T   only layer l perturbed
    ANP  (Eq. 6)  dL/dW_l ~= N   * delta_L * delta_a_l / ||delta_a||^2 x_{l-1}^T   all layers perturbed
    DANP (Alg. 1) ANP with x* = R x and, after every weight update,
                  R_l <- R_l - alpha * (x*_l x*_l^T - diag(x*_l^2)) R_l   (batch mean, alpha = 1e-3)

``N_l`` is the number of perturbed units of layer l (for a layer called several times per
forward, e.g. the LSTM gate layer, all calls / time steps count), ``N = sum_l N_l``, and
the norms are per example over the same set of units.  The weight update is the batch
mean of the per-example updates, i.e. an estimate of the gradient of the MEAN loss, and
the mean over the noisy passes of one step.

Implementation on top of ``zoo.perturb``: the clean pass records every layer's input
``h`` and pre-activation ``a`` (local graph to that layer's own weight / bias only); each
noisy pass records ``xi`` and ``a_noisy``.  The per-layer estimate ``G_i`` (shape of
``a[i]``) is accumulated over the noisy passes and written into ``p.grad`` with one local
``torch.autograd.backward(layer.a, grad_tensors=G)`` per layer, which computes
``dL/dW = sum_i G_i^T h_i`` for linear / conv / embedding alike.

A layer whose recorded activity does not carry the batch dimension (the transformer's
learned position embedding is called on ``arange(T)``) is *shared*: its noise perturbs
every example at once, so it takes the batch-mean of the per-example weights, its nodes
are counted once in ``N`` and its ``||delta a||^2`` is added to every example's norm.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from ..perturb import ctx, players
from . import register
from .base import BaseOptimizer, per_sample_loss

# Decorrelation learning rate alpha: Dalm et al. 2023, Appendix D ("alpha = 10^-3", chosen by
# manual exploration) and the ``--decor_lr`` default (1e-3) of DANP_open_source/main.py.
LR_DECOR = 1e-3


class _NodePerturbation(BaseOptimizer):
    """Shared machinery; subclasses only set ``variant`` and ``decorrelate``."""

    variant: str = "np"            # "np" | "anp" | "inp"
    decorrelate: bool = False
    forward_passes_per_step = 16
    n_noisy: int = forward_passes_per_step - 1   # 15 noisy passes after the 1 clean pass

    def __init__(self, model, lr, task_kind, n_classes, lr_decor: float = LR_DECOR, **kw):
        super().__init__(model, lr, task_kind, n_classes, **kw)
        self.n_noisy = max(1, self.forward_passes_per_step - 1)   # 16 passes -> 1 clean + 15 noisy
        self.layers = players(model)
        ctx.set_layers(self.layers)
        self.lr_decor = lr_decor
        self._shared: Optional[List[bool]] = None      # per layer, see _probe_shared
        self._last_abs_dloss: Optional[torch.Tensor] = None
        if self.decorrelate:
            dev = next(model.parameters()).device
            for layer in self.layers:
                layer.enable_decorrelation()           # no-op for PEmbedding
            model.to(dev)                              # R buffers live with the weights

    # ------------------------------------------------------------------ public API
    @property
    def inp_passes_per_layer(self) -> int:
        """R = max(1, 15 // L): single-layer noisy passes per layer, so that 1 + R*L <= 16."""
        return max(1, self.n_noisy // len(self.layers))

    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.estimate_grad(x, y)
        self.adam.step()
        if self.decorrelate:
            for layer in self.layers:                  # uses the inputs recorded on the clean pass
                layer.decorrelation_update(self.lr_decor)
        ctx.mode = "bp"
        return loss

    def extra_log(self) -> dict:
        if self._last_abs_dloss is None:
            return {}
        return {"mean_abs_dloss": float(self._last_abs_dloss)}

    def estimate_grad(self, x: torch.Tensor, y: torch.Tensor, n_passes: Optional[int] = None) -> torch.Tensor:
        """Write the gradient estimate of the batch-mean loss into ``p.grad`` of every
        parameter (after zeroing) and return the detached mean clean loss.

        ``n_passes``: np / anp / danp = number of noisy passes (default 15);
        inp = number of single-layer noisy passes per layer (default max(1, 15 // L)).
        """
        ctx.set_layers(self.layers)                    # make the global ctx point at this model
        if self._shared is None:
            self._shared = self._probe_shared(x)
        self.adam.zero_grad(set_to_none=True)
        N = x.shape[0]
        sigma = self.eps

        # ---- clean pass: the only pass with grad enabled; builds each layer's local graph ----
        ctx.mode, ctx.noise_layer = "clean", None
        ctx.reset()                                    # wipes h / a / hd / xi / a_noisy of every layer
        logits = self.model(x)
        loss_clean = per_sample_loss(logits.detach(), y)             # (N,), no graph
        a_clean = [[a.detach() for a in layer.a] for layer in self.layers]
        G = [[torch.zeros_like(a) for a in al] for al in a_clean]   # accumulated grad_tensors
        abs_dloss = loss_clean.new_zeros(())
        n_dloss = 0

        # ---- noisy passes (no grad).  They only append to xi / a_noisy, which we clear before
        #      each pass, so the clean records survive without snapshotting. ----
        if self.variant == "inp":
            R = n_passes if n_passes is not None else self.inp_passes_per_layer
            for li in range(len(self.layers)):
                for _ in range(R):
                    dl = self._noisy_pass(x, y, noise_layer=li) - loss_clean
                    self._accumulate_inp(G, li, dl, N, sigma)
                    abs_dloss = abs_dloss + dl.abs().sum()
                    n_dloss += N
            scale = 1.0 / R
        else:
            K = n_passes if n_passes is not None else self.n_noisy
            for _ in range(K):
                dl = self._noisy_pass(x, y, noise_layer=None) - loss_clean
                if self.variant == "np":
                    self._accumulate_np(G, dl, N, sigma)
                else:
                    self._accumulate_anp(G, a_clean, dl, N)
                abs_dloss = abs_dloss + dl.abs().sum()
                n_dloss += N
            scale = 1.0 / K

        # ---- one local backward per layer: dL/dW_l = sum_i G_i^T h_i ----
        for layer, Gl in zip(self.layers, G):
            if layer.a:
                torch.autograd.backward(layer.a, grad_tensors=[g.mul_(scale) for g in Gl])
        self._last_abs_dloss = abs_dloss / max(n_dloss, 1)
        ctx.mode = "bp"
        return loss_clean.mean()

    # ------------------------------------------------------------------ passes
    def _noisy_pass(self, x, y, noise_layer: Optional[int]) -> torch.Tensor:
        ctx.mode, ctx.sigma, ctx.noise_layer = "noisy", self.eps, noise_layer
        for layer in self.layers:                      # clear only the noisy records of the previous pass
            layer.xi, layer.a_noisy = [], []
        with torch.no_grad():
            return per_sample_loss(self.model(x), y)   # (N,)

    @torch.no_grad()
    def _probe_shared(self, x) -> List[bool]:
        """Which layers record activity WITHOUT a batch dimension (e.g. the transformer's
        position embedding).  Two tiny clean passes with batch sizes 2 and 3: a layer is
        per-example iff its leading dim follows the batch size.  (Shape-based detection on
        the real batch alone is ambiguous when T == N, the protocol's LM setting 64 x 64.)
        One-off cost at the first step."""
        N = x.shape[0]
        ks = (2, 3) if N >= 3 else (N,)
        ctx.mode, ctx.noise_layer = "clean", None
        per_example = [True] * len(self.layers)
        for k in ks:
            ctx.reset()
            self.model(x[:k])
            for li, layer in enumerate(self.layers):
                if not layer.a or any(a.shape[0] != k for a in layer.a):
                    per_example[li] = False
        ctx.reset()
        return [not pe for pe in per_example]

    # ------------------------------------------------------------------ estimators
    @staticmethod
    def _add(G: torch.Tensor, w: torch.Tensor, D: torch.Tensor, shared: bool):
        """G += w_n * D_n  (w already divided by the batch size so that summing over n gives
        the batch mean).  Shared layers get the batch-mean weight sum_n w_n."""
        if shared:
            G.add_(w.sum() * D)
        else:
            G.add_(w.view(-1, *([1] * (D.dim() - 1))) * D)

    def _sq_and_count(self, D: List[torch.Tensor], shared: bool, N: int, sq: torch.Tensor):
        """Accumulate per-example ||D||^2 (over all calls of one layer) into ``sq`` (N,) and
        return the number of perturbed units of that layer."""
        n = 0
        for d in D:
            if shared:
                sq = sq + d.pow(2).sum()
                n += d.numel()
            else:
                sq = sq + d.reshape(N, -1).pow(2).sum(1)
                n += d.numel() // N
        return sq, n

    def _accumulate_np(self, G, dl, N, sigma):
        # NP (Eq. 5): G = delta_L * eps / sigma^2 = delta_L * xi / sigma   (eps = sigma * xi)
        w = dl / (sigma * N)
        for li, layer in enumerate(self.layers):
            for i, xi in enumerate(layer.xi):
                self._add(G[li][i], w, xi, self._shared[li])

    def _accumulate_anp(self, G, a_clean, dl, N):
        # ANP (Eq. 6): G_l = N * delta_L * delta_a_l / ||delta_a||^2,  delta_a = a_noisy - a_clean,
        # ||delta_a||^2 per example over ALL layers (and calls), N = total number of perturbed units.
        D = [[an - ac for an, ac in zip(layer.a_noisy, acl)] for layer, acl in zip(self.layers, a_clean)]
        sq = dl.new_zeros(N)
        n_total = 0
        for li, Dl in enumerate(D):
            sq, n = self._sq_and_count(Dl, self._shared[li], N, sq)
            n_total += n
        w = n_total * dl / (sq.clamp_min(torch.finfo(sq.dtype).tiny) * N)
        for li, Dl in enumerate(D):
            for i, d in enumerate(Dl):
                self._add(G[li][i], w, d, self._shared[li])

    def _accumulate_inp(self, G, li, dl, N, sigma):
        # INP (Eq. 3): G_l = N_l * delta_L * eps_l / ||eps_l||^2 with noise eps_l = sigma * xi_l injected
        # in layer l only; ||eps_l||^2 per example over all units (and calls) of layer l.
        # (For the feed-forward models eps_l == a_noisy_l - a_l, the form used in the authors' code.)
        layer = self.layers[li]
        V = [sigma * xi for xi in layer.xi]
        sq, n_l = self._sq_and_count(V, self._shared[li], N, dl.new_zeros(N))
        w = n_l * dl / (sq.clamp_min(torch.finfo(sq.dtype).tiny) * N)
        for i, v in enumerate(V):
            self._add(G[li][i], w, v, self._shared[li])


@register("np")
class NodePerturbation(_NodePerturbation):
    """Classic node perturbation (Eq. 5): 1 clean + 15 all-layer noisy passes, averaged."""
    variant = "np"


@register("anp")
class ActivityNodePerturbation(_NodePerturbation):
    """Activity-based node perturbation (Eq. 6): uses the measured delta_a of every layer."""
    variant = "anp"


@register("inp")
class IterativeNodePerturbation(_NodePerturbation):
    """Iterative node perturbation (Eq. 3): one layer perturbed per pass, R passes per layer."""
    variant = "inp"


@register("danp")
class DecorrelatedActivityNodePerturbation(_NodePerturbation):
    """ANP with layer-wise input decorrelation (Algorithm 1): x* = R x, R updated with the
    local rule R <- R - alpha (x* x*^T - diag(x*^2)) R after every Adam step, alpha = 1e-3."""
    variant = "anp"
    decorrelate = True
