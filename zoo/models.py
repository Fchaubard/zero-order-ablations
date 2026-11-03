"""The four ~40k-parameter models.

Every model has the same skeleton so that every optimizer can drive it:

    logits = model.readout( block_L( ... block_1( model.embed(x) ) ) )

* ``embed``   : identity for images; token (+ learned position) embedding for text.
* ``blocks``  : nn.ModuleList of blocks; each maps an activity tensor to the next one.
* ``readout`` : the final PLinear that produces logits (classes, or next-char over V).

All trainable layers are PLinear / PConv2d / PEmbedding (see perturb.py), Xavier-
initialised with zero biases (LSTM forget-gate bias = 1).  LayerNorms carry no affine
parameters so that *every* parameter of every model lives in a perturbable layer.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .perturb import PConv2d, PEmbedding, PLinear, count_params, ctx

PTB_VOCAB = 50   # overwritten by data loader at construction time (see build_model)

# Model sizes: "s" = the protocol's ~40k parameters, "m" ~ 400k, "l" ~ 4M (widths tuned so that every
# model lands within a few percent of the target; depth and everything else unchanged).
SIZES = {
    "s": {"mlp": dict(hidden=(48, 32)), "conv": dict(chans=(16, 32, 32), fc=48), "lstm": dict(d_emb=32, hidden=80),
          "transformer": dict(d=48, d_ff=64)},
    "m": {"mlp": dict(hidden=(400, 240)), "conv": dict(chans=(48, 96, 96), fc=176), "lstm": dict(d_emb=96, hidden=265),
          "transformer": dict(d=152, d_ff=304)},
    "l": {"mlp": dict(hidden=(1600, 1700)), "conv": dict(chans=(128, 256, 256), fc=760), "lstm": dict(d_emb=256, hidden=870),
          "transformer": dict(d=496, d_ff=992)},
}
SIZE_TAG = {"s": "", "m": "-sz400k", "l": "-sz4m"}


class Seq(nn.Module):
    """A block = perturbable layer followed by fixed ops (ReLU, pool, flatten)."""

    def __init__(self, *mods):
        super().__init__()
        self.mods = nn.ModuleList(mods)

    def forward(self, x):
        for m in self.mods:
            x = m(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(1)


class BaseModel(nn.Module):
    def embed(self, x):
        return x

    def forward(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.readout(h)


# ------------------------------------------------------------------ MNIST MLP (39,578 params)
class MLP(BaseModel):
    def __init__(self, in_dim=784, hidden=(48, 32), n_classes=10):
        super().__init__()
        dims = [in_dim, *hidden]
        self.blocks = nn.ModuleList(
            [Seq(PLinear(dims[i], dims[i + 1]), nn.ReLU()) for i in range(len(hidden))]
        )
        self.readout = PLinear(dims[-1], n_classes)

    def embed(self, x):
        return x.flatten(1)


# ------------------------------------------------------------------ CIFAR-10 ConvNet (39,450 params)
class ConvNet(BaseModel):
    def __init__(self, in_ch=3, chans=(16, 32, 32), fc=48, n_classes=10):
        super().__init__()
        c = [in_ch, *chans]
        blocks = [Seq(PConv2d(c[i], c[i + 1], 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)) for i in range(len(chans))]
        blocks[-1].mods.append(Flatten())
        blocks.append(Seq(PLinear(chans[-1] * 4 * 4, fc), nn.ReLU()))
        self.blocks = nn.ModuleList(blocks)
        self.readout = PLinear(fc, n_classes)


# ------------------------------------------------------------------ char-LSTM on PTB (41,810 params @ V=50)
class LSTMLayer(nn.Module):
    """Single-layer LSTM written with one PLinear for all four gates so that the
    gate pre-activations are perturbable nodes.  Input (N,T,D) -> (N,T,H)."""

    def __init__(self, d_in, hidden):
        super().__init__()
        self.hidden = hidden
        self.gates = PLinear(d_in + hidden, 4 * hidden)
        with torch.no_grad():
            self.gates.bias[hidden:2 * hidden].fill_(1.0)   # forget-gate bias

    def forward(self, x):
        N, T, _ = x.shape
        H = self.hidden
        h = x.new_zeros(N, H)
        c = x.new_zeros(N, H)
        outs = []
        mode = ctx.mode
        if mode == "clean":
            ctx.clean_states = []
        damp = ctx.damp if (mode == "noisy" and ctx.damp is not None and len(ctx.clean_states) == T) else None
        for t in range(T):
            g = self.gates(torch.cat([x[:, t], h], dim=1))
            i, f, gg, o = g.split(H, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
            h = torch.sigmoid(o) * torch.tanh(c)
            outs.append(h)
            if mode == "clean":
                ctx.clean_states.append((h.detach(), c.detach()))
            elif damp is not None:
                # 1.5-ANP spec: truncate the perturbation cascade softly (0.5) so that multi-step eligibility
                # flows without variance explosion over long T -- the state that enters step t+1 is damped,
                # the recorded output of step t is not.
                hc, cc = ctx.clean_states[t]
                h = hc + damp * (h - hc)
                c = cc + damp * (c - cc)
        return torch.stack(outs, dim=1)


class CharLSTM(BaseModel):
    def __init__(self, vocab=PTB_VOCAB, d_emb=32, hidden=80):
        super().__init__()
        self.tok = PEmbedding(vocab, d_emb)
        self.blocks = nn.ModuleList([LSTMLayer(d_emb, hidden)])
        self.readout = PLinear(hidden, vocab)

    def embed(self, idx):
        return self.tok(idx)


# ------------------------------------------------------------------ char-Transformer on PTB (39,250 params @ V=50, T=64)
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.d, self.nh = d, n_heads
        self.q, self.k, self.v, self.o = PLinear(d, d), PLinear(d, d), PLinear(d, d), PLinear(d, d)

    def forward(self, x):
        N, T, D = x.shape
        hd = D // self.nh
        q = self.q(x).view(N, T, self.nh, hd).transpose(1, 2)
        k = self.k(x).view(N, T, self.nh, hd).transpose(1, 2)
        v = self.v(x).view(N, T, self.nh, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(N, T, D)
        return self.o(y)


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        self.ff1 = PLinear(d, d_ff)
        self.ff2 = PLinear(d_ff, d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff2(F.relu(self.ff1(self.ln2(x))))
        return x


class CharTransformer(BaseModel):
    def __init__(self, vocab=PTB_VOCAB, d=48, n_heads=4, d_ff=64, n_layers=2, max_len=64):
        super().__init__()
        self.tok = PEmbedding(vocab, d)
        self.pos = PEmbedding(max_len, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d, elementwise_affine=False)
        self.readout = PLinear(d, vocab)

    def embed(self, idx):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        return self.tok(idx) + self.pos(pos)[None]

    def forward(self, idx):
        h = self.embed(idx)
        for b in self.blocks:
            h = b(h)
        return self.readout(self.ln_f(h))


TASKS = {
    # task name -> (model constructor, kind)
    "mlp_mnist": (MLP, "cls"),
    "conv_cifar10": (ConvNet, "cls"),
    "lstm_ptb": (CharLSTM, "lm"),
    "transformer_ptb": (CharTransformer, "lm"),
}


def build_model(task: str, vocab: int | None = None, size: str = "s") -> nn.Module:
    ctor, kind = TASKS[task]
    key = {"mlp_mnist": "mlp", "conv_cifar10": "conv", "lstm_ptb": "lstm", "transformer_ptb": "transformer"}[task]
    kw = dict(SIZES[size][key])
    if kind == "lm":
        return ctor(vocab=vocab or PTB_VOCAB, **kw)
    return ctor(**kw)


if __name__ == "__main__":
    for sz in SIZES:
        for t in TASKS:
            m = build_model(t, size=sz)
            print(f"{sz} {t:18s} {count_params(m):9d} params")
