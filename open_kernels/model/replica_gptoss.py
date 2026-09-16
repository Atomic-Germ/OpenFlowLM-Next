"""fp64 CPU reference for one GPT-OSS decode step -- the oracle the gptoss kernels will be
checked against when they exist.

GPT-OSS is a dense GQA attention layer over an MoE feed-forward, with a 128-row sliding
window on every other layer. Four things in it are new to this codebase and this module is
the reference for all four:

  * the per-head attention sink -- one learned logit joining the softmax denominator with no
    value behind it. `replica_dense.sink_softmax` is the math; `sink_attention` is the GQA
    loop around it, windowed on a sliding layer.
  * a bias on o_proj, on the router and on all three expert projections. The dense recipe's
    bias covers q, k and v only.
  * a clamped SwiGLU in the experts: `(up + 1) * gate * sigmoid(1.702 * gate)`, gate clipped
    above at 7 and up clipped both ways. `hidden_act` says silu and is wrong about it.
  * HF stores gate and up fused and INTERLEAVED down the fused projection's output rows --
    gate at the even rows, up at the odd. GGUF splits them again (`ffn_gate_exps` /
    `ffn_up_exps`), which is what `q4nx-build/configs/gpt-oss.json` maps, so everything here
    works from a split pair and `split_gate_up` is the rule that connects the two.

Every function takes plain arrays, so the whole module runs against transformers' own
modules without a container -- which is the point, because the container does not exist yet
and a reference checked only against our own dequantiser is how OPEN-PACK-Q4-0 went unnoticed
for two sessions.

Traces: OPEN-GPTOSS-FFN-REF, OPEN-ATTN-SINK, OPEN-FAMILY-GPTOSS (specs/open-engine/spec.md).
"""
from __future__ import annotations

import numpy as np

from replica_dense import rms, sink_softmax  # noqa: F401

# Family constants, not spec fields: every GPT-OSS has these and a field would move every
# spec hash for nothing. transformers keeps them the same way, on GptOssExperts.
ALPHA = 1.702
LIMIT = 7.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def clamped_swiglu(gate, up, alpha=ALPHA, limit=LIMIT):
    """GPT-OSS's expert activation. The two clamps are not the same: gate is clipped above
    only, up both ways, and the `+ 1` on up means a zeroed up row still passes the gate
    through rather than killing the channel."""
    g = np.minimum(np.asarray(gate, np.float64), limit)
    u = np.clip(np.asarray(up, np.float64), -limit, limit)
    return (u + 1.0) * (g * sigmoid(alpha * g))


def split_gate_up(gate_up, axis=-1):
    """(gate, up) from HF's fused `gate_up_proj` output axis: gate is the even entries, up
    the odd. Whoever packs a container from HF safetensors has to do this; a GGUF source has
    it done already."""
    gu = np.asarray(gate_up)
    even = [slice(None)] * gu.ndim
    odd = [slice(None)] * gu.ndim
    even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
    return gu[tuple(even)], gu[tuple(odd)]


def fuse_gate_up(gate, up, axis=-1):
    """The inverse of `split_gate_up` -- what a packer writes back when it has to hand HF's
    row order to something that expects it."""
    g, u = np.asarray(gate), np.asarray(up)
    if g.shape != u.shape:
        raise ValueError(f"gate {g.shape} and up {u.shape} must match")
    ax = axis % g.ndim
    return np.stack([g, u], ax + 1).reshape(g.shape[:ax] + (2 * g.shape[ax],) + g.shape[ax + 1:])


def expert_ffn(x, Wg, bg, Wu, bu, Wd, bd):
    """One expert, weights in the linear convention `y = W @ x + b`: Wg and Wu are
    [moe_intermediate, hidden], Wd is [hidden, moe_intermediate]."""
    x = np.asarray(x, np.float64)
    h = clamped_swiglu(np.asarray(Wg, np.float64) @ x + np.asarray(bg, np.float64),
                       np.asarray(Wu, np.float64) @ x + np.asarray(bu, np.float64))
    return np.asarray(Wd, np.float64) @ h + np.asarray(bd, np.float64)


def route(x, Wr, br, top_k):
    """(expert ids, weights) for one token. The router has a bias, which no other family's
    does. HF takes the top-k of the logits and softmaxes those; that is the same number as
    softmaxing all of them and renormalising the top-k, and the order is the same, so the
    existing router core's shape still fits -- only the bias and the widths change."""
    lg = np.asarray(Wr, np.float64) @ np.asarray(x, np.float64) + np.asarray(br, np.float64)
    idx = np.argsort(-lg, kind="stable")[:top_k]
    e = np.exp(lg[idx] - lg[idx].max())
    return idx, e / e.sum()


def moe_block(x, Wr, br, Wg, bg, Wu, bu, Wd, bd, top_k, top=None):
    """The whole MoE feed-forward for one token. The expert weights are stacked on axis 0.
    `top` overrides the routing, so a kernel comparison can hold the expert choice fixed when
    two logits are a near-tie, the way replica.moe_decode does."""
    idx, w = route(x, Wr, br, top_k)
    if top is not None:
        idx = np.asarray(top, np.int64)
        lg = np.asarray(Wr, np.float64) @ np.asarray(x, np.float64) + np.asarray(br, np.float64)
        e = np.exp(lg[idx] - lg[idx].max())
        w = e / e.sum()
    out = np.zeros(np.shape(Wd)[1], np.float64)
    for e_id, ww in zip(idx, w):
        out += ww * expert_ffn(x, Wg[e_id], bg[e_id], Wu[e_id], bu[e_id], Wd[e_id], bd[e_id])
    return out, idx, w


def window_start(pos, sliding_window):
    """The first cached row this token may see. HF's sliding mask is `kv > q - window`, so a
    128-row window keeps this token and the 127 before it."""
    if not sliding_window:
        return 0
    return max(0, pos + 1 - sliding_window)


def sink_attention(q, K, V, sinks, sliding_window=None, pos=None):
    """GQA attention for one token with a per-head sink. `q` is [nh, hd]; `K` and `V` are
    [T, kvh, hd] including this token's row; `sinks` is one scalar per query head.

    The sink is the raw stored scalar -- the scores are divided by sqrt(head_dim) before it
    joins them, so it is not scaled. Getting that backwards is the quiet way to be wrong."""
    q = np.asarray(q, np.float64)
    K, V = np.asarray(K, np.float64), np.asarray(V, np.float64)
    nh, hd = q.shape
    kvh = K.shape[1]
    grp = nh // kvh
    t0 = window_start(K.shape[0] - 1 if pos is None else pos, sliding_window)
    o = np.zeros((nh, hd), np.float64)
    for h in range(nh):
        s = (K[t0:, h // grp] @ q[h]) / np.sqrt(hd)
        o[h] = sink_softmax(s, float(np.asarray(sinks)[h])) @ V[t0:, h // grp]
    return o


def gptoss_layer_step(w, x_res, K, V, pos, *, num_heads, num_kv_heads, head_dim,
                      top_k, inv_freq, rope_theta=0.0, rope_scale=1.0, norm_eps=1e-5,
                      sliding_window=None):
    """One token through a whole GPT-OSS layer. Returns (residual, K, V).

    `w` is a mapping of plain arrays in the linear convention, so this runs against a
    transformers layer's own parameters with no container in the way:
      ln1, ln2                      [hidden]
      wq, wk, wv, wo + bq, bk, bv, bo
      sinks                         [num_heads]
      router_w [E, hidden], router_b [E]
      gate_w / up_w [E, moe_inter, hidden], gate_b / up_b [E, moe_inter]
      down_w [E, hidden, moe_inter], down_b [E, hidden]
    """
    from replica_dense import rope
    nh, kvh, hd = num_heads, num_kv_heads, head_dim
    x = (rms(x_res, norm_eps) * np.asarray(w["ln1"], np.float64)).astype(np.float64)
    q = (np.asarray(w["wq"], np.float64) @ x + w["bq"]).reshape(nh, hd)
    k = (np.asarray(w["wk"], np.float64) @ x + w["bk"]).reshape(kvh, hd)
    v = (np.asarray(w["wv"], np.float64) @ x + w["bv"]).reshape(kvh, hd)
    rot = 2 * len(inv_freq)
    q = rope(q, pos, rot, rope_theta, inv_freq, rope_scale)
    k = rope(k, pos, rot, rope_theta, inv_freq, rope_scale)
    K = np.concatenate([K, k[None]], 0) if K is not None and len(K) else k[None]
    V = np.concatenate([V, v[None]], 0) if V is not None and len(V) else v[None]
    o = sink_attention(q, K, V, w["sinks"], sliding_window, pos)
    res = x_res + (np.asarray(w["wo"], np.float64) @ o.reshape(nh * hd) + w["bo"])
    xm = rms(res, norm_eps) * np.asarray(w["ln2"], np.float64)
    ffn, _, _ = moe_block(xm, w["router_w"], w["router_b"], w["gate_w"], w["gate_b"],
                          w["up_w"], w["up_b"], w["down_w"], w["down_b"], top_k)
    return res + ffn, K, V
