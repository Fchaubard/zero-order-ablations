"""Identity-path credit (IPC), from the Noise speedrun (Q Labs, Oct. 2026, https://github.com/qlabs-eng/noise,
records #7 and later of the 10M speedrun track; PR #16 by Lazarus-931, MIT licence), ported onto the survey's
character Transformer.

IPC needs a residual stream, so it runs on the Transformer only.  It spends one clean forward pass per step;
every learning signal is then a measured change of a loss (or of a local output scored against a measured
credit) from cheap partial recomputations on the cached activations, as in the reference:

1. *Top-stream credit.*  Each token's credit at the last residual stream (the input of the final LayerNorm),
   g_u = d l_u / d x_L(u), is measured by central differences along a full orthonormal cycle of directions,
   each one a re-evaluation of the final LayerNorm, the readout and the token's cross-entropy at the moved
   stream (per-token functions, so every token is probed at once without interference).  With a complete
   orthonormal basis the cycle returns the credit exactly, up to the finite-difference error.
2. *Identity path.*  Every earlier residual stream receives the same credit (the reference scales it by its
   residual mixing scalars; this model has none, so the scale is one).  The Jacobians of the later blocks are
   dropped: this is the method's one approximation.
3. *Writers* (attention output projection o and MLP output layer ff2) take the stream credit as their output
   error; the token and position embeddings take it too.
4. *One local step* for the internals.  MLP hidden units: each unit's pre-activation is moved along a full
   orthonormal cycle over the hidden width; each move's change of the block output, through the ReLU and ff2, is
   scored against the stream credit.  Attention: the target at the attention output before o is the stream
   credit through o (a linear map); values take their exact credit sum_s A[s, t] target_s (attention is linear in
   its values); queries are moved along a full orthonormal cycle per head and the attention rerun, each query's
   output change scored against its own target; keys are moved one position and one coordinate at a time
   ("single-key experiments", evaluated in closed form: one attention logit per query changes), each query's
   output change scored against its target.
5. *Readout*: the cross-entropy's change along every logit, by central differences evaluated in closed form on
   the cached logits (the reference's "coordinate head").

Every layer's weight gradient is its output error times its cached input (one local backward per layer, as for
the node rules here).  Adaptations to the survey: Adam with lr = eps and every finite-difference step equal to
the same eps (the reference uses ANVIL II on the block matrices and step sizes relative to the probed vector's
RMS); no residual scalars, value embeddings, gates, rotary embeddings or QK norm (absent from this model); the
learned position embedding takes the stream credit like the token embedding; full cycles on the hidden layer
(the reference moves each token's 256 largest units).
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn.functional as F

from ..perturb import ctx, players
from . import register
from .base import BaseOptimizer
from .curvature import per_unit_loss

LN_EPS = 1e-5          # nn.LayerNorm default (the model's LayerNorms carry no affine parameters)
CHUNK = 2 ** 25        # elements per partial-recomputation chunk


def _orthonormal(d: int, device, dtype=torch.float32) -> torch.Tensor:
    """A random orthonormal basis of R^d, rows are the directions (Haar via QR with sign fix)."""
    q, r = torch.linalg.qr(torch.randn(d, d, device=device, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.T.contiguous().to(dtype)


@register("ipc")
class IdentityPathCredit(BaseOptimizer):
    forward_passes_per_step = 1           # one clean pass; all other evaluations are partial recomputations

    def __init__(self, model, lr, task_kind, n_classes, loss64: bool = False, **kw):
        super().__init__(model, lr, task_kind, n_classes, **kw)
        if type(model).__name__ != "CharTransformer":
            raise ValueError("ipc needs a residual stream: it runs on the Transformer only")
        self.loss64 = bool(loss64)
        self.layers = players(model)
        ctx.set_layers(self.layers)
        self.blocks = list(model.blocks)
        self.head = model.readout
        self._xL = None
        self._hook = model.ln_f.register_forward_pre_hook(self._capture_top)

    def _capture_top(self, module, args):
        if ctx.mode == "clean":
            self._xL = args[0].detach()

    # ------------------------------------------------------------------ measurements
    def _head_error(self, z: torch.Tensor, y: torch.Tensor, s: float) -> torch.Tensor:
        """Central differences of each token's cross-entropy along each of its own logits, evaluated in closed
        form: moving logit j by +-s changes log Z by log1p(p_j expm1(+-s)) and the target term by -+s."""
        dt = torch.float64 if self.loss64 else torch.float32
        z = z.to(dt)
        p = torch.softmax(z, -1)
        up = torch.log1p(p * math.expm1(s))
        down = torch.log1p(p * math.expm1(-s))
        change = up - down
        change = change.scatter_add(-1, y.unsqueeze(-1), torch.full_like(y.unsqueeze(-1), -2 * s, dtype=dt))
        return (change / (2 * s)).float()

    def _top_credit(self, x: torch.Tensor, y: torch.Tensor, s: float) -> torch.Tensor:
        """Each token's credit at the last residual stream, by central differences along a full orthonormal
        cycle, each a re-evaluation of the final LayerNorm, the readout and the token's cross-entropy."""
        dt = torch.float64 if self.loss64 else torch.float32
        shape = x.shape
        xf = x.reshape(-1, shape[-1]).to(dt)
        yf = y.reshape(-1)
        n, d = xf.shape
        Q = _orthonormal(d, x.device, dt)                          # (d, d), rows are directions
        W = self.head.weight.detach().to(dt)                       # (V, d)
        b = self.head.bias.detach().to(dt)
        per = max(1, CHUNK // (2 * d * (d + W.shape[0])))
        credit = torch.empty(n, d, device=x.device, dtype=dt)

        def loss_at(moved, yt):                                    # moved (c, P, d) -> per-direction loss (c, P)
            h = F.layer_norm(moved, (d,), eps=LN_EPS)
            logits = h @ W.T + b
            return torch.logsumexp(logits, -1) - logits.gather(-1, yt.view(-1, 1, 1).expand(-1, moved.shape[1], 1)).squeeze(-1)

        for i in range(0, n, per):
            xs, ys = xf[i:i + per], yf[i:i + per]
            plus = loss_at(xs.unsqueeze(1) + s * Q.unsqueeze(0), ys)
            minus = loss_at(xs.unsqueeze(1) - s * Q.unsqueeze(0), ys)
            credit[i:i + per] = ((plus - minus) / (2 * s)) @ Q     # sum_j (d l / d q_j) q_j
        return credit.view(shape).float()

    def _hidden_credit(self, u: torch.Tensor, g: torch.Tensor, w2: torch.Tensor, s: float) -> torch.Tensor:
        """MLP hidden credit, one local step from the stream credit g: every unit's pre-activation is moved along a
        full orthonormal cycle; each move's change of the block output, through the ReLU and ff2, is scored
        against g (scoring through ff2, a linear map, is done first: w = g W2)."""
        shape = u.shape
        uf = u.reshape(-1, shape[-1])
        w = (g.reshape(-1, g.shape[-1]) @ w2)                      # (n, dff): g scored through ff2
        n, k = uf.shape
        B = _orthonormal(k, u.device)                              # (k, k)
        per = max(1, CHUNK // (k * k))
        out = torch.empty_like(uf)
        for i in range(0, n, per):
            ut, wt_ = uf[i:i + per].unsqueeze(1), w[i:i + per].unsqueeze(-1)
            change = F.relu(ut + s * B) - F.relu(ut - s * B)       # (c, k dirs, k units)
            score = (change @ wt_).squeeze(-1) / (2 * s)           # (c, k dirs)
            out[i:i + per] = score @ B
        return out.view(shape)

    def _attention_credit(self, attn, target: torch.Tensor, s: float) -> Dict[str, torch.Tensor]:
        """Query, key and value errors, one local step from the target at the attention output (before o)."""
        q0, k0, v0 = (t.a[0].detach().float() for t in (attn.q, attn.k, attn.v))   # (N, T, D)
        N, T, D = q0.shape
        H = attn.nh
        hd = D // H
        scale = 1.0 / math.sqrt(hd)
        mask = torch.ones(T, T, dtype=torch.bool, device=q0.device).tril()

        def heads(t):
            return t.view(N, T, H, hd).transpose(1, 2)              # (N, H, T, hd)

        def merge(t):
            return t.transpose(1, 2).reshape(N, T, D)

        def attend(q, k, v):
            att = (q @ k.transpose(-2, -1)) * scale
            return att.masked_fill(~mask, float("-inf")).softmax(-1) @ v

        qh, kh, vh, tg = heads(q0), heads(k0), heads(v0), heads(target)
        A = ((qh @ kh.transpose(-2, -1)) * scale).masked_fill(~mask, float("-inf")).softmax(-1)  # (N, H, Ts, Tt)
        o = A @ vh

        # values: attention is linear in its values, so position t's credit is sum_s A[s, t] target_s
        err_v = A.transpose(-2, -1) @ tg

        # queries: a full orthonormal cycle per head; query s moves only output s
        Bq = _orthonormal(hd, q0.device)
        err_q = torch.zeros_like(qh)
        per = max(1, CHUNK // (2 * N * H * T * max(T, hd)))
        for j0 in range(0, hd, per):
            dirs = Bq[j0:j0 + per].view(-1, 1, 1, 1, hd)           # (J, 1, 1, 1, hd)
            plus = attend(qh.unsqueeze(0) + s * dirs, kh, vh)
            minus = attend(qh.unsqueeze(0) - s * dirs, kh, vh)
            score = ((plus - minus) * tg.unsqueeze(0)).sum(-1) / (2 * s)   # (J, N, H, T)
            err_q += torch.einsum("jnht,jd->nhtd", score, Bq[j0:j0 + per])

        # keys: single-key experiments in closed form.  Moving key t by +-s e_j moves query s's logit to t by
        # +-s q_s[j] / sqrt(hd); its output changes by (a' - a)(v_t - o_s) / (1 - a + a'), a' = a exp(dl), scored
        # against target_s: C[s, t] = target_s . (v_t - o_s).
        C = (tg @ vh.transpose(-2, -1)) - (tg * o).sum(-1, keepdim=True)   # (N, H, Ts, Tt)
        err_k = torch.zeros_like(kh)
        nb = max(1, CHUNK // (2 * H * T * T * hd))
        for i in range(0, N, nb):
            Ai, Ci, qi = A[i:i + nb], C[i:i + nb], qh[i:i + nb]
            dl = (s * scale) * qi.unsqueeze(3)                      # (n, H, Ts, 1, hd): logit move per coordinate j
            Ae = Ai.unsqueeze(-1)
            cp = Ae * torch.expm1(dl)
            cm = Ae * torch.expm1(-dl)
            f = (cp / (1 + cp) - cm / (1 + cm)) * Ci.unsqueeze(-1)  # (n, H, Ts, Tt, hd)
            err_k[i:i + nb] = f.sum(2) / (2 * s)                    # sum over queries s
        return {"q": merge(err_q), "k": merge(err_k), "v": merge(err_v)}

    # ------------------------------------------------------------------ the step
    def step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ctx.set_layers(self.layers)
        self.adam.zero_grad(set_to_none=True)
        s = self.eps
        ctx.mode, ctx.noise_layer = "clean", None
        ctx.reset()
        logits = self.model(x)
        l0 = per_unit_loss(logits.detach().double() if self.loss64 else logits.detach(), y).float()   # (N, T)
        n_units = l0.numel()
        errors: Dict[int, List[torch.Tensor]] = {}
        with torch.no_grad():
            errors[id(self.head)] = [self._head_error(self.head.a[0].detach(), y, s)]
            g = self._top_credit(self._xL, y, s)                    # (N, T, d)
            for blk in self.blocks:
                errors[id(blk.attn.o)] = [g]
                errors[id(blk.ff2)] = [g]
                errors[id(blk.ff1)] = [self._hidden_credit(blk.ff1.a[0].detach().float(), g,
                                                            blk.ff2.weight.detach().float(), s)]
                target = g @ blk.attn.o.weight.detach().float()     # credit at the attention output, through o
                for kind, err in self._attention_credit(blk.attn, target, s).items():
                    errors[id(getattr(blk.attn, kind))] = [err]
            errors[id(self.model.tok)] = [g]
            errors[id(self.model.pos)] = [g.sum(0)]                 # shared over the batch
        for layer in self.layers:
            E = errors.get(id(layer))
            if E is None or not layer.a:
                raise RuntimeError(f"ipc: no estimate for layer {layer.index} ({type(layer).__name__})")
            torch.autograd.backward(layer.a, grad_tensors=[e.to(a.dtype) / n_units for e, a in zip(E, layer.a)])
        self.adam.step()
        ctx.mode = "bp"
        return l0.mean()
