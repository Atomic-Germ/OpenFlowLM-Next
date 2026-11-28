"""fp64 CPU reference for Gemma 4 E2B / E4B / 12B text (recipes/gemma4.py), from the model's own GGUF.

The weights are the GGUF's, dequantized by llama.cpp's own `gguf` package, so a disagreement with the NPU
is the kernels' or the packing's, not a difference of source weights. The math is transformers'
Gemma4ForCausalLM (models/gemma4/modeling_gemma4.py) in float64, over a whole prompt at once (causal mask,
so it equals token-at-a-time decode):

  per layer: x = rms(h) * attn_norm; q = rope(rms_head(Wq x) * q_norm); own layers k = rope(rms_head(Wk x)
  * k_norm), v = rms_head(Wv x) (no weight); shared layers take K, V from their source layer; scores q.k
  UNSCALED, causal (plus the 512-row window on sliding layers); h += rms(Wo o) * post_attention_norm;
  h += rms(down(gelu_tanh(gate x') * up x')) * post_ffw_norm with x' = rms(h) * ffn_norm; then the per-layer
  input: h += rms(proj(gelu_tanh(inp_gate h) * ple_l)) * post_norm; h *= layer_output_scale.
  ple = (rms(per_layer_model_proj(e) / sqrt(hidden)) * per_layer_proj_norm + per_layer_token_embd[id] * 16)
  / sqrt(2), e = token_embd[id] * sqrt(hidden). Logits: the tied head on rms(h) * output_norm, then
  30 * tanh(logits / 30).

RoPE: sliding layers rotate the whole 256-wide head (pairs (i, i + 128), theta 1e4); global layers rotate
pairs (i, i + 256) for i < 64 only, at theta^(-2i/512) -- transformers' `proportional` type, which llama.cpp
writes as rope_freqs (1 for the 64 rotated pairs, 1e30 for the rest).

The 12B's global layers have no attn_v: V is the raw k projection, before k_norm and RoPE (attention_k_eq_v).

    python replica_gemma4.py <model.gguf> --ids 2,818,5279 [--layers 6] [--out ref.npz]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def rms(x, w=None, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    y = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)
    return y if w is None else y * w


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class Gguf:
    """Tensors of a GGUF as float64 [rows, cols] (row = output), dequantized by llama.cpp's gguf package."""

    def __init__(self, path):
        import gguf
        self._gguf = gguf
        self.r = gguf.GGUFReader(str(path))
        self.t = {t.name: t for t in self.r.tensors}
        self.md = {}
        for k, f in self.r.fields.items():
            try:
                v = f.contents()
            except Exception:
                continue
            self.md[k] = v

    def has(self, name):
        return name in self.t

    def get(self, name, rows=None):
        t = self.t[name]
        data = t.data if rows is None else t.data[np.asarray(rows)]
        a = self._gguf.quants.dequantize(data, t.tensor_type)
        return np.asarray(a, dtype=np.float64)

    def scalar(self, name):
        return float(self.get(name).reshape(-1)[0])


@dataclass
class Geo:
    hidden: int
    nh: int
    kvh: int
    hd: int
    ghd: int
    grot: int
    window: int
    theta: float
    theta_local: float
    ple: int
    eps: float
    softcap: float
    kinds: list = field(default_factory=list)         # per model layer: "sliding" | "full"
    sources: dict = field(default_factory=dict)       # shared layer -> the layer whose K / V it uses
    n_layers: int = 0
    gkvh: int = 0                                     # the global layers' kv heads (the 12B: 1); 0 = kvh
    k_eq_v: bool = False                              # the global layers have no attn_v: V is the raw k projection

    def kv_heads(self, full: bool) -> int:
        return (self.gkvh or self.kvh) if full else self.kvh

    @classmethod
    def from_gguf(cls, g: Gguf):
        a = "gemma4"
        md = g.md
        pat = list(md[f"{a}.attention.sliding_window_pattern"])
        n = int(md[f"{a}.block_count"])
        shared = int(md.get(f"{a}.attention.shared_kv_layers", 0))
        kinds = ["sliding" if p else "full" for p in pat]
        first = n - shared
        src = {}
        for l in range(first, n):
            src[l] = max(j for j in range(first) if kinds[j] == kinds[l])
        rf = g.get("rope_freqs.weight").reshape(-1) if g.has("rope_freqs.weight") else None
        ghd = int(md[f"{a}.attention.key_length"])
        grot = 2 * int((rf == 1.0).sum()) if rf is not None else ghd // 4
        kv = md[f"{a}.attention.head_count_kv"]
        kvs = [int(x) for x in kv] if isinstance(kv, (list, tuple, np.ndarray)) else [int(kv)] * n
        kvh = kvs[kinds.index("sliding")] if "sliding" in kinds else kvs[0]
        gkvh = kvs[kinds.index("full")] if "full" in kinds else kvh
        g0 = kinds.index("full") if "full" in kinds else -1
        return cls(hidden=int(md[f"{a}.embedding_length"]), nh=int(md[f"{a}.attention.head_count"]),
                   kvh=kvh, hd=int(md[f"{a}.attention.key_length_swa"]),
                   ghd=ghd, grot=grot, window=int(md[f"{a}.attention.sliding_window"]),
                   theta=float(md[f"{a}.rope.freq_base"]), theta_local=float(md[f"{a}.rope.freq_base_swa"]),
                   ple=int(md.get(f"{a}.embedding_length_per_layer_input", 0)),
                   eps=float(f"{float(md[f'{a}.attention.layer_norm_rms_epsilon']):.6g}"),
                   softcap=float(md.get(f"{a}.final_logit_softcapping", 0.0)), kinds=kinds, sources=src, n_layers=n,
                   gkvh=gkvh if gkvh != kvh else 0,
                   k_eq_v=g0 >= 0 and g0 not in src and not g.has(f"blk.{g0}.attn_v.weight"))


def rope_full(x, pos, theta):
    """NEOX pairs (i, i + hd/2) over the whole head, theta^(-2i/hd)."""
    hd = x.shape[-1]
    half = hd // 2
    inv = theta ** (-2.0 * np.arange(half) / hd)
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * c - b * s, b * c + a * s], -1)


def rope_proportional(x, pos, theta, rot):
    """Pairs (i, i + hd/2) for i < rot/2 at theta^(-2i/hd); every other dim passes through."""
    hd = x.shape[-1]
    half, r = hd // 2, rot // 2
    inv = theta ** (-2.0 * np.arange(r) / hd)
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    y = x.copy()
    a, b = x[..., :r], x[..., half:half + r]
    y[..., :r] = a * c - b * s
    y[..., half:half + r] = b * c + a * s
    return y


def attention(q, K, V, sliding, window):
    """q [T, NH, hd], K / V [T, KVH, hd]; unscaled scores, causal (+ window)."""
    T, nh, _ = q.shape
    kvh = K.shape[1]
    out = np.zeros_like(q)
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    allowed = j <= i
    if sliding:
        allowed &= (i - j) < window
    for h in range(nh):
        g = h // (nh // kvh)
        s = q[:, h, :] @ K[:, g, :].T
        s = np.where(allowed, s, -np.inf)
        s -= s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        out[:, h, :] = p @ V[:, g, :]
    return out


def ple_inputs(g: Gguf, G: Geo, ids, emb, cols):
    """[T, len(cols), ple]: the per-layer inputs for the model layers `cols` (None: the model has none)."""
    if not G.ple:
        return None
    T = len(ids)
    tok = g.get("per_layer_token_embd.weight", rows=ids).reshape(T, -1, G.ple)[:, cols, :] * np.sqrt(G.ple)
    W = g.get("per_layer_model_proj.weight")                       # [L * ple, hidden]
    proj = (emb @ W.T) * G.hidden ** -0.5
    proj = proj.reshape(T, -1, G.ple)[:, cols, :]
    proj = rms(proj, g.get("per_layer_proj_norm.weight").reshape(-1), G.eps)
    return (proj + tok) * 2.0 ** -0.5


def layer(g: Gguf, G: Geo, l, h, pos, ple_l, kv, keep):
    """Model layer l over the whole prompt. `kv` holds (K, V) of earlier layers; `keep` says to store this one's."""
    b = f"blk.{l}."
    full = G.kinds[l] == "full"
    hd = G.ghd if full else G.hd
    T = h.shape[0]
    x = rms(h, g.get(b + "attn_norm.weight").reshape(-1), G.eps)
    q = (x @ g.get(b + "attn_q.weight").T).reshape(T, G.nh, hd)
    q = rms(q, g.get(b + "attn_q_norm.weight").reshape(-1), G.eps)
    rot = (lambda t: rope_proportional(t, pos, G.theta, G.grot)) if full else (lambda t: rope_full(t, pos, G.theta_local))
    q = rot(q)
    if l in G.sources:
        K, V = kv[G.sources[l]]
    else:
        kvh = G.kv_heads(full)
        K = (x @ g.get(b + "attn_k.weight").T).reshape(T, kvh, hd)
        # K = V (the 12B's global layers): v is the raw k projection, taken before k_norm and RoPE
        V = K if (full and G.k_eq_v) else (x @ g.get(b + "attn_v.weight").T).reshape(T, kvh, hd)
        K = rot(rms(K, g.get(b + "attn_k_norm.weight").reshape(-1), G.eps))
        V = rms(V, None, G.eps)
        if keep:
            kv[l] = (K, V)
    o = attention(q, K, V, not full, G.window).reshape(T, G.nh * hd)
    a = o @ g.get(b + "attn_output.weight").T
    h = h + rms(a, g.get(b + "post_attention_norm.weight").reshape(-1), G.eps)
    x = rms(h, g.get(b + "ffn_norm.weight").reshape(-1), G.eps)
    m = (gelu_tanh(x @ g.get(b + "ffn_gate.weight").T) * (x @ g.get(b + "ffn_up.weight").T)) @ g.get(b + "ffn_down.weight").T
    h = h + rms(m, g.get(b + "post_ffw_norm.weight").reshape(-1), G.eps)
    if ple_l is not None:
        p = gelu_tanh(h @ g.get(b + "inp_gate.weight").T) * ple_l
        p = p @ g.get(b + "proj.weight").T
        h = h + rms(p, g.get(b + "post_norm.weight").reshape(-1), G.eps)
    return h * g.scalar(b + "layer_output_scale.weight") if g.has(b + "layer_output_scale.weight") else h


def forward(g: Gguf, ids, n_layers=None, G: Geo | None = None, want_logits=True, layers=None, sources=None):
    """The first n_layers layers over the prompt (or exactly `layers`, with `sources` naming each shared one's
    cache owner -- a slice stitched from chosen layers), then optionally the final norm and the soft-capped head
    on the last position. Returns (residuals per layer [T, hidden], logits of the last position or None)."""
    G = G or Geo.from_gguf(g)
    if layers is None:
        layers = list(range(n_layers or G.n_layers))
    if sources is not None:
        from dataclasses import replace
        G = replace(G, sources=dict(sources))
    ids = list(ids)
    emb = g.get("token_embd.weight", rows=ids) * np.sqrt(G.hidden)
    ple = ple_inputs(g, G, ids, emb, list(layers))
    pos = np.arange(len(ids), dtype=np.float64)
    keep = set(G.sources.values())
    kv, h, res = {}, emb, []
    for i, l in enumerate(layers):
        h = layer(g, G, l, h, pos, None if ple is None else ple[:, i, :], kv, l in keep)
        res.append(h)
    if not want_logits:
        return res, None
    return res, head(g, G, h[-1])


def requant_q4_1(w):
    """The engine's re-quantization of a non-4-bit head (pools.cpp std_perm_gguf into a 5120-byte pool):
    per 32-value block, m = min rounded toward -inf in bf16, d = (max - m) / 15 rounded toward +inf,
    code = int((v - m) / d + 0.5) clipped to [0, 15] -- all in f32 -- read back as m + code * d."""
    v = np.asarray(w, np.float32).reshape(-1, 32)

    def bf16(x, up):
        u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
        neg = (u >> 31) == 1
        trunc = (u >> 16) << 16
        bump = ((u + 0xFFFF) >> 16) << 16
        return np.where(neg != up, bump, trunc).astype(np.uint32).view(np.float32)
    m = bf16(v.min(1), up=False)                      # toward -inf
    d = bf16(((v.max(1) - m) / np.float32(15.0)).astype(np.float32), up=True)   # toward +inf
    inv = np.where(d > 0, np.float32(1.0) / np.where(d > 0, d, 1), 0).astype(np.float32)
    q = np.clip(((v - m[:, None]) * inv[:, None] + np.float32(0.5)).astype(np.int32), 0, 15)
    return (m[:, None] + q.astype(np.float32) * d[:, None]).reshape(np.shape(w)).astype(np.float64)


def head(g: Gguf, G: Geo, h, as_npu=False):
    """The final norm, the tied head and the soft cap on one residual row. `as_npu` reads the head
    through the engine's q4_1 re-quantization (a Q6_K head is not 4-bit), which is what the NPU streams."""
    hn = rms(h, g.get("output_norm.weight").reshape(-1), G.eps)
    vocab = g.t["token_embd.weight"].data.shape[0]
    rq = requant_q4_1 if as_npu else (lambda w: w)
    logits = np.concatenate([rq(g.get("token_embd.weight", rows=range(a, min(a + 16384, vocab)))) @ hn
                             for a in range(0, vocab, 16384)])    # in slabs: the head is 5 GB as one fp64 matrix
    if G.softcap:
        logits = G.softcap * np.tanh(logits / G.softcap)
    return logits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--logits-at", default="", help="also the head after these layer counts, from the same pass")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    g = Gguf(a.gguf)
    G = Geo.from_gguf(g)
    ids = [int(x) for x in a.ids.split(",")]
    res, logits = forward(g, ids, a.layers or None, G=G)
    top = np.argsort(-logits)[:5]
    print("top5", [(int(t), round(float(logits[t]), 4)) for t in top])
    extra = {}
    for k in [int(x) for x in a.logits_at.split(",") if x]:
        lk = head(g, G, res[k - 1][-1])
        extra[f"logits_l{k}"] = lk
        print(f"after {k} layers: top5", [(int(t), round(float(lk[t]), 4)) for t in np.argsort(-lk)[:5]])
    if a.out:
        np.savez(a.out, ids=np.asarray(ids), logits=logits, res=np.stack([r[-1] for r in res]), **extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
