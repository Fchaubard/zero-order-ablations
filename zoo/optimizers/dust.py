"""DUST (Dahal, Mandal, Guelbahar, Vegesna, "Dust: Pretraining Transformers Without Backpropagation",
Q Labs, Oct. 2026, https://qlabs.sh/research/dust), ported from the authors' minimal reference
implementation https://github.com/qlabs-eng/dust (``dust.py``, MIT licence) onto the survey's
perturbable layers and protocol.

One step (reference ``DUST.step``):

1. A clean pass caches every layer's input and output and the per-unit losses (a unit is a token on
   the LM tasks and an example on the image tasks).
2. *Direct draws*: a group of sites has its output jittered at every unit, ``y + sigma * a`` with
   ``a ~ N(0, I)``; the forward is rerun and each unit's jitter is rewarded by the loss reduction at that
   unit, ``r = l_clean - l_perturbed``, centred across the draws of the group (reference: across each
   chunk of ``nrep`` draws).  The estimated error at the site's output is ``-(1/K) sum_k r_k a_k / sigma``.
   Groups (reference ``Recipe``): per block the two output projections together ("writers": attention
   ``o`` and MLP ``ff2``), the MLP input layer alone ("hidden"), the attention output before ``o`` alone
   ("hubs", a virtual site), and the token embedding alone.  Draws are front-loaded towards the first
   blocks as in the reference allocation (writers 38, 32, ...; hubs 6, 6, ...; embedding 10).
3. *Local draws*: the attention internals (q, k, v) are not rewarded by the loss but by the change they
   make to their block's attention output, recomputed from the cached clean q, k, v alone, scored per head
   against the hub's estimated error: ``score_s = <ghat_o(s), Delta o(s)>``; queries keep their own token's
   score, keys and values sum the scores of later tokens decayed by ``gamma = 0.98`` per token of lag.
4. *Head*: the readout is jittered directly on the cached logits (only the cross-entropy is recomputed),
   with four times the direct population; the reward is centred over the batch.
5. Every layer's weight gradient is its estimated output error times its cached input, summed over units
   and divided by the number of units (one local backward per layer, as for the node rules here).

Adaptations to the survey (stated in the paper):

* Protocol budget and optimizer: 1 clean pass + 15 direct draws = 16 forward passes per step (local and
  head draws are partial recomputations and, as in the reference, not counted as passes; their cost is in
  the wall-clock budget), Adam with lr = eps, and one probe scale tied to the step size for every site
  (the reference uses SGD with momentum and per-site scales 0.2 / 0.4 / 0.05).
* The survey's Transformer has a learned position embedding (the reference uses rotary embeddings): it is
  a direct site like the token embedding, with its noise shared across the batch.  It has no value
  embeddings, attention gates or residual mixing scalars (trained by weight-space ES in the reference).
* Non-transformer models: the MLP's and ConvNet's hidden layers are direct sites (one group each, the
  front-loaded allocation), rewarded per example; the head is jittered on the cached logits.  On the
  LSTM the token embedding and the gate layer are direct sites rewarded, like every direct draw of the
  reference, by their own token's loss reduction.  (Crediting the gate layer's jitter with the decayed
  later-token reductions, the credit DUST gives keys and values, aligned worse with the true gradient at
  16 passes -- whole-model cosine 0.14 against 0.25 at initialisation -- and is available as
  ``recurrent=True`` but not used.)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..perturb import PLayer, ctx, players
from . import register
from .base import BaseOptimizer
from .curvature import per_unit_loss

GAMMA = 0.98            # reference default ``Recipe.gamma``
HEAD_MULT = 4           # reference: head draws = 4 * population
LOCAL_SHARE = {"q": 2, "v": 20, "k": 12}   # reference local draws per block at population 256


def _alloc(total: int, weights: List[float], floor: int = 1) -> List[int]:
    """Split ``total`` draws over groups in proportion to ``weights`` (largest remainder), at least ``floor`` each."""
    n = len(weights)
    out = [floor] * n
    rest = total - floor * n
    if rest < 0:
        raise ValueError(f"dust: {total} draws cannot give {n} groups {floor} each")
    s = float(sum(weights))
    raw = [rest * w / s for w in weights]
    base = [int(math.floor(r)) for r in raw]
    out = [o + b for o, b in zip(out, base)]
    left = rest - sum(base)
    for i in sorted(range(n), key=lambda i: raw[i] - base[i], reverse=True)[:left]:
        out[i] += 1
    return out


class _Site:
    """A jitter site: a perturbable layer's output (``kind='out'``) or the input of the attention output
    projection (``kind='hub'``).  ``layout`` maps its recorded tensors onto units:
    ``ex`` (N, ...), ``tok`` (N, T, ...), ``call`` (T calls of (N, F): the LSTM gate layer), ``shared`` (T, D)."""

    def __init__(self, layer: PLayer, kind: str = "out", attn=None, recurrent: bool = False):
        self.layer, self.kind, self.attn, self.recurrent = layer, kind, attn, recurrent
        self.layout: Optional[str] = None


@register("dust")
class DUST(BaseOptimizer):
    forward_passes_per_step = 16          # 1 clean pass + 15 direct draws
    gamma = GAMMA
    head_mult = HEAD_MULT

    def __init__(self, model, lr, task_kind, n_classes, loss64: bool = False, **kw):
        super().__init__(model, lr, task_kind, n_classes, **kw)
        self.loss64 = bool(loss64)
        self.layers = players(model)
        ctx.set_layers(self.layers)
        self.n_direct = max(1, self.forward_passes_per_step - 1)
        self.head = model.readout
        self._build_groups(model)
        self._hooks = []
        self._jitter: Dict[int, List[torch.Tensor]] = {}     # id(site) -> per-call jitter tensors (output sites)
        self._hub_jitter: Dict[int, torch.Tensor] = {}       # id(attention module) -> jitter of the o input
        self._calls: Dict[int, int] = {}
        for site in {id(s): s for g, _ in self.groups for s in g}.values():
            if site.kind == "out":
                self._hooks.append(site.layer.register_forward_hook(self._out_hook(site)))
        for blk in self._blocks:
            attn = blk.attn
            self._hooks.append(attn.o.register_forward_pre_hook(self._hub_hook(attn)))
        self._last_abs_r: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ structure
    def _build_groups(self, model):
        name = type(model).__name__
        self._blocks = []
        self.local: List[Tuple[object, Dict[str, int]]] = []
        if name == "CharTransformer":
            self._blocks = list(model.blocks)
            B = len(self._blocks)
            ref_w = [38, 32, 16, 10, 6, 4, 2, 2]
            ref_h = [6, 6, 4, 2, 2, 2, 2, 2]
            wts = [ref_w[min(b, 7)] for b in range(B)]
            hub = [ref_h[min(b, 7)] for b in range(B)]
            groups = []
            for b, blk in enumerate(self._blocks):
                groups.append(([_Site(blk.attn.o), _Site(blk.ff2)], wts[b]))         # writers
                groups.append(([_Site(blk.ff1)], wts[b]))                              # hidden
                groups.append(([_Site(blk.attn.o, kind="hub", attn=blk.attn)], hub[b]))  # hubs
            groups.append(([_Site(model.tok)], 10))                                     # embedding
            groups.append(([_Site(model.pos)], 10))                                     # learned positions (adaptation)
            counts = _alloc(self.n_direct, [w for _, w in groups])
            self.groups = [(g, k) for (g, _), k in zip(groups, counts)]
            # local draws per block: the reference ratio q : v : k = 2 : 20 : 12, scaled to this population
            # (one draw per block per two direct draws of that block, at least one per kind)
            for b, blk in enumerate(self._blocks):
                per_block = sum(k for (g, k) in self.groups if any(getattr(s, "attn", None) is blk.attn or s.layer in
                                                                      (blk.attn.o, blk.ff1, blk.ff2) for s in g))
                n_loc = max(3, per_block)
                ks = _alloc(n_loc, [LOCAL_SHARE[k] for k in ("q", "k", "v")])
                self.local.append((blk.attn, dict(zip(("q", "k", "v"), ks))))
        elif name == "CharLSTM":
            lstm = model.blocks[0]
            groups = [([_Site(lstm.gates)], 38 + 38), ([_Site(model.tok)], 10)]
            counts = _alloc(self.n_direct, [w for _, w in groups])
            self.groups = [(g, k) for (g, _), k in zip(groups, counts)]
        else:   # MLP / ConvNet: every hidden layer is a direct site, front-loaded allocation
            hidden = [l for l in self.layers if l is not model.readout]
            ref_w = [38, 32, 16, 10, 6, 4, 2, 2]
            groups = [([_Site(l)], ref_w[min(i, 7)]) for i, l in enumerate(hidden)]
            counts = _alloc(self.n_direct, [w for _, w in groups])
            self.groups = [(g, k) for (g, _), k in zip(groups, counts)]
        self.n_head = self.head_mult * self.n_direct

    def allocation(self) -> dict:
        d = {}
        for g, k in self.groups:
            key = "+".join(f"{s.kind}:{type(s.layer).__name__}{s.layer.index}" for s in g)
            d[key] = k
        d["head"] = self.n_head
        for i, (attn, ks) in enumerate(self.local):
            d[f"local{i}"] = ks
        return d

    # ------------------------------------------------------------------ hooks
    def _out_hook(self, site: _Site):
        def hook(module, args, output):
            j = self._jitter.get(id(site))
            if j is None:
                return None
            i = self._calls.get(id(site), 0)
            self._calls[id(site)] = i + 1
            return output + j[i]
        return hook

    def _hub_hook(self, attn):
        def hook(module, args):
            j = self._hub_jitter.get(id(attn))
            if j is None:
                return None
            return (args[0] + j,)
        return hook

    # ------------------------------------------------------------------ helpers
    def _unit_loss(self, logits, y):
        return per_unit_loss(logits.double() if self.loss64 else logits, y).float()

    def _layout(self, site: _Site, N: int, T: Optional[int]) -> str:
        if site.kind == "hub":
            return "tok"
        a = site.layer.a
        if T is None:
            return "ex"
        if len(a) == T and all(t.dim() == 2 and t.shape[0] == N for t in a) and T > 1:
            return "call"
        if len(a) == 1 and a[0].dim() == 2:          # LM layer without a batch dimension: learned positions (T, D)
            return "shared"
        if len(a) == 1 and a[0].dim() >= 3 and a[0].shape[0] == N and a[0].shape[1] == T:
            return "tok"
        raise RuntimeError(f"dust: cannot map layer {site.layer.index} activity {[tuple(t.shape) for t in a]} onto units")

    def _site_shapes(self, site: _Site) -> List[torch.Size]:
        if site.kind == "hub":
            return [site.attn.o.h[0].shape]
        return [t.shape for t in site.layer.a]

    @staticmethod
    def _bcast(r: torch.Tensor, layout: str, t_idx: int, shape: torch.Size) -> torch.Tensor:
        """Per-unit weights r ((N,) or (N, T)) broadcast onto one recorded tensor of the site."""
        if layout == "ex":
            return r.view(-1, *([1] * (len(shape) - 1)))
        if layout == "tok":
            return r.view(r.shape[0], r.shape[1], *([1] * (len(shape) - 2)))
        if layout == "call":
            return r[:, t_idx].view(-1, *([1] * (len(shape) - 1)))
        return r.sum(0).view(-1, *([1] * (len(shape) - 1)))       # shared (T, ...): sum over the batch

    def _future_credit(self, c: torch.Tensor) -> torch.Tensor:
        """r_t = sum_{s >= t} gamma^(s - t) c_s along the token axis of (..., T)."""
        T = c.shape[-1]
        pos = torch.arange(T, device=c.device)
        lag = pos[None, :] - pos[:, None]                          # [t, s]
        C = (lag >= 0).to(c.dtype) * (self.gamma ** lag.clamp(min=0).to(c.dtype))
        return c @ C.T

    # ------------------------------------------------------------------ the step
    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ctx.set_layers(self.layers)
        self.adam.zero_grad(set_to_none=True)
        sigma = self.eps
        N = x.shape[0]

        # ---- 1. clean pass: caches inputs h and outputs a (local graphs) of every layer ----
        ctx.mode, ctx.noise_layer = "clean", None
        ctx.reset()
        logits = self.model(x)
        l0 = self._unit_loss(logits.detach(), y)                   # (N,) or (N, T)
        T = l0.shape[1] if l0.dim() == 2 else None
        n_units = N * (T or 1)
        for g, _ in self.groups:
            for s in g:
                if s.layout is None:
                    s.layout = self._layout(s, N, T)
        errors: Dict[int, List[torch.Tensor]] = {}                 # id(layer) -> error per recorded tensor (output sites)
        hub_err: Dict[int, torch.Tensor] = {}
        abs_r = l0.new_zeros(())

        # ---- 2. direct draws ----
        ctx.mode = "bp"
        with torch.no_grad():
            for g, K in self.groups:
                noises, rewards = [], []
                for _ in range(K):
                    draw = []
                    self._jitter, self._hub_jitter, self._calls = {}, {}, {}
                    for s in g:
                        shp = self._site_shapes(s)
                        if s.layout == "shared":
                            a = [torch.randn(shp[0], device=x.device)]
                        else:
                            a = [torch.randn(sh, device=x.device) for sh in shp]
                        draw.append(a)
                        if s.kind == "hub":
                            self._hub_jitter[id(s.attn)] = sigma * a[0]
                        else:
                            self._jitter[id(s)] = [sigma * t for t in a]
                    try:
                        lk = self._unit_loss(self.model(x), y)
                    finally:
                        self._jitter, self._hub_jitter, self._calls = {}, {}, {}
                    noises.append(draw)
                    rewards.append(l0 - lk)                       # loss reduction per unit
                R = torch.stack(rewards)                          # (K, N[, T])
                if K > 1:
                    R = R - R.mean(0, keepdim=True)               # centred across the draws of the group
                abs_r = abs_r + R.abs().mean()
                for si, s in enumerate(g):
                    Rs = self._future_credit(R) if (s.recurrent and T is not None) else R
                    shp = self._site_shapes(s)
                    acc = [torch.zeros(sh, device=x.device) for sh in shp]
                    for k in range(K):
                        for ti, a in enumerate(noises[k][si]):
                            acc[ti].sub_(self._bcast(Rs[k], s.layout, ti, a.shape) * a / sigma)
                    acc = [t / K for t in acc]
                    if s.kind == "hub":
                        hub_err[id(s.attn)] = acc[0]
                    else:
                        errors[id(s.layer)] = acc

            # ---- 3. local draws: attention internals scored against the hub's estimated error ----
            for attn, ks in self.local:
                tgt = hub_err.get(id(attn))
                if tgt is None:
                    continue
                for kind, err in self._local_errors(attn, ks, tgt, sigma).items():
                    errors[id(getattr(attn, kind))] = [err]

            # ---- 4. head: jitter the cached logits, recompute only the cross-entropy ----
            errors[id(self.head)] = [self._head_error(y, l0, sigma)]

        # ---- 5. reassemble: one local backward per layer (dL/dW = sum_u err_u h_u^T / n_units) ----
        for layer in self.layers:
            E = errors.get(id(layer))
            if E is None or not layer.a:
                raise RuntimeError(f"dust: no estimate for layer {layer.index} ({type(layer).__name__})")
            torch.autograd.backward(layer.a, grad_tensors=[e.to(a.dtype) / n_units for e, a in zip(E, layer.a)])
        self.adam.step()
        self._last_abs_r = abs_r / max(len(self.groups), 1)
        ctx.mode = "bp"
        return l0.mean()

    # ------------------------------------------------------------------ head and local estimates
    def _head_error(self, y, l0, sigma):
        z = self.head.a[0].detach()                                # cached logits (N, C) or (N, T, V)
        K = self.n_head
        err = torch.zeros_like(z)
        for _ in range(K):
            a = torch.randn_like(z)
            r = l0 - self._unit_loss(z + sigma * a, y)
            if r.shape[0] > 1:
                r = r - r.mean(0, keepdim=True)                    # reference: centred over the batch
            err.sub_(r.view(*r.shape, 1) * a / sigma)
        return err / K

    def _local_errors(self, attn, ks: Dict[str, int], target: torch.Tensor, sigma: float) -> Dict[str, torch.Tensor]:
        q0, k0, v0 = (t.a[0].detach() for t in (attn.q, attn.k, attn.v))   # (N, T, D) clean pre-activations
        N, T, D = q0.shape
        H = attn.nh
        hd = D // H
        mask = torch.ones(T, T, dtype=torch.bool, device=q0.device).tril()

        def heads(t):                                              # (..., N, T, D) -> (..., N, H, T, hd)
            return t.view(*t.shape[:-1], H, hd).transpose(-3, -2)

        def attend(q, k, v):
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            return att @ v                                         # (..., N, H, T, hd)

        qh, kh, vh = heads(q0), heads(k0), heads(v0)
        base = attend(qh, kh, vh)
        tg = heads(target)                                         # (N, H, T, hd)
        pos = torch.arange(T, device=q0.device)
        lag = pos[None, :] - pos[:, None]                          # [t (jitter), s (output)]
        credit = (lag >= 0).float() * (self.gamma ** lag.clamp(min=0).float())
        out = {}
        for kind in ("q", "k", "v"):
            K = ks[kind]
            est = torch.zeros(N, H, T, hd, device=q0.device)
            for _ in range(K):
                a = torch.randn(N, H, T, hd, device=q0.device)
                if kind == "q":
                    delta = attend(qh + sigma * a, kh, vh) - base
                elif kind == "k":
                    delta = attend(qh, kh + sigma * a, vh) - base
                else:
                    delta = attend(qh, kh, sigma * a)              # attention is linear in the values
                score = (tg * delta).sum(-1)                       # (N, H, T): per head, per output token
                if kind != "q":
                    score = score @ credit.T                       # jitter at t collects later outputs s >= t
                est.add_(score.unsqueeze(-1) * a / sigma)
            out[kind] = (est / K).transpose(1, 2).reshape(N, T, D)
        return out

    def extra_log(self) -> dict:
        return {} if self._last_abs_r is None else {"mean_abs_reward": float(self._last_abs_r)}
