"""fp64 CPU reference for one Qwen3.5 dense decode step -- the oracle the composed
kernels (designs/layer_x/lx.py, ax.py with the dense tail) are checked against.

HF-faithful math in float64 from the same `.q4nx` bytes the NPU pools are packed
from, so a disagreement is the kernels' or the packing's, not a difference of
source weights. The layer is the Qwen3.6-MoE layer with the MoE block replaced
by a silu-gated dense FFN:

  * linear-attention layer: the gated DeltaNet of replica.py `linear_decode`,
    written against the ModelSpec instead of the 27B's constants and against
    this container's names (`model.layers.N.`, the bf16 alpha / beta copies)
  * full-attention layer: replica.py `attn_decode`'s math -- fused q_proj
    [q | gate], q/k RMSNorm, partial RoPE, softmax attention, the sigmoid gate
  * both then: res = x + block_out; xm = post_attention_norm(res);
    silu(gate) * up @ down; xres = res + ffn_out
  * final model.norm, then the q8 lm_head

Every weight comes back through `Q4NX.matmul_w`, which reads a q8 tensor the way the
POOL holds it: at q8 where the kernel set streams it at q8 (`ssm_out_proj` on a Qwen3.5
container, OPEN-QUANT-Q8), re-quantised to q4_1 where the packer re-quantises it. So a
slice compare measures the kernels either way, and `make_decode.py --requant` swings the
whole run -- spec, pools and reference together -- onto the fallback for the A/B.

State is carried by the caller: `(conv_state, S)` per linear layer, `(K, V)` per
attention layer -- decode from position 0 with zeroed state is exact prefill,
since each layer's state update only ever sees one token at a time.
"""
from __future__ import annotations

import numpy as np

from recipes.spec import FULL


def rms(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def silu(x):
    return x / (1 + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rope(t, p, rot, inv_freq):
    half = rot // 2
    ang = p * np.asarray(inv_freq, np.float64)
    c, s = np.cos(ang), np.sin(ang)
    y = t.copy()
    x1, x2 = t[..., :half], t[..., half:rot]
    y[..., :half] = x1 * c - x2 * s
    y[..., half:rot] = x2 * c + x1 * s
    return y


def linear_decode(m, spec, layer, x_res, conv_state, S):
    """One token through a gated-DeltaNet layer, up to (and including) the out projection.
    Returns (attention-block output, new conv_state, S updated in place)."""
    pre = f"model.layers.{layer}."
    hid, eps = spec.hidden, spec.norm_eps
    kh, vh, kd, vd = spec.lin_key_heads, spec.lin_value_heads, spec.lin_key_dim, spec.lin_value_dim
    kw, vw, nch = kh * kd, vh * vd, spec.lin_qkv_dim
    x = (rms(x_res, eps) * m.bf16(pre + "input_layernorm.weight")).astype(np.float32)
    Wqkv = m.matmul_w(pre + "linear_attn.qkv_proj.weight", nch, hid)
    Wz = m.matmul_w(pre + "self_attn.gate_proj.weight", vw, hid)
    Wout = m.matmul_w(pre + "linear_attn.ssm_out_proj.weight", hid, vw)
    convw = m.bf16(pre + "linear_attn.ssm_conv1d.weight")
    qkv = x @ Wqkv.T
    z = silu(x @ Wz.T)
    c = silu((convw * np.vstack([conv_state, qkv])).sum(0))     # depthwise conv over the taps

    def l2n(a):
        return a / np.sqrt((a ** 2).sum(-1, keepdims=True) + 1e-6)

    q = l2n(c[:kw].reshape(kh, kd))
    k = l2n(c[kw:2 * kw].reshape(kh, kd))
    v = c[2 * kw:].reshape(vh, vd).astype(np.float64)
    # this container stores alpha / beta q8 AND as bf16 [heads, hidden]; the projection
    # wants [hidden, heads], which is the layout the packer's `transpose` op writes
    Wa = m.bf16(pre + "linear_attn.ssm_alpha_proj.bf16.weight").T
    Wb = m.bf16(pre + "linear_attn.ssm_beta_proj.bf16.weight").T
    A = m.f32(pre + "linear_attn.ssm_a")                        # the file stores -exp(A_log)
    dtb = m.f32(pre + "linear_attn.ssm_dt.bias")
    decay = np.exp(A * np.log1p(np.exp(x @ Wa + dtb)))
    beta = sigmoid(x @ Wb)
    grp = vh // kh                                              # value heads per key head
    o = np.zeros((vh, vd))
    for h in range(vh):
        kk, qq = k[h // grp], q[h // grp]
        S[h] *= decay[h]
        delta = beta[h] * (v[h] - S[h].T @ kk)
        S[h] += np.outer(kk, delta)
        o[h] = (S[h].T @ qq) / np.sqrt(vd)
    nw = m.bf16(pre + "linear_attn.ssm_norm.weight")
    og = (rms(o, eps) * nw).reshape(vw) * z
    new_conv = np.vstack([conv_state[1:], qkv[None, :]])
    return og.astype(np.float32) @ Wout.T, new_conv, S


def attn_decode(m, spec, layer, x_res, K, V, pos):
    """One token through a gated full-attention layer, up to (and including) o_proj.
    Returns (attention-block output, K, V)."""
    pre = f"model.layers.{layer}."
    hid, eps = spec.hidden, spec.norm_eps
    nh, kvh, hd = spec.num_heads, spec.num_kv_heads, spec.head_dim
    qw, kvw = nh * hd, kvh * hd
    inv = spec.rope_inv_freq()
    x = (rms(x_res, eps) * m.bf16(pre + "input_layernorm.weight")).astype(np.float32)
    Wqg = m.matmul_w(pre + "self_attn.q_proj.weight", 2 * qw, hid)   # the fused [q | gate] rows
    Wq, Wg = Wqg[:qw], Wqg[qw:]
    Wk = m.matmul_w(pre + "self_attn.k_proj.weight", kvw, hid)
    Wv = m.matmul_w(pre + "self_attn.v_proj.weight", kvw, hid)
    Wo = m.matmul_w(pre + "self_attn.o_proj.weight", hid, qw)
    qn = m.bf16(pre + "self_attn.q_norm.weight")
    kn = m.bf16(pre + "self_attn.k_norm.weight")
    q = (rms((x @ Wq.T).reshape(nh, hd), eps) * qn).astype(np.float64)
    g = (x @ Wg.T).reshape(nh, hd)
    k = (rms((x @ Wk.T).reshape(kvh, hd), eps) * kn).astype(np.float64)
    v = (x @ Wv.T).reshape(kvh, hd).astype(np.float64)
    q, k = rope(q, pos, spec.rotary_dim, inv), rope(k, pos, spec.rotary_dim, inv)
    K = np.concatenate([K, k[None]], 0)
    V = np.concatenate([V, v[None]], 0)
    grp = nh // kvh
    o = np.zeros((nh, hd))
    for h in range(nh):
        s = (K[:, h // grp] @ q[h]) / np.sqrt(hd)
        a = np.exp(s - s.max())
        o[h] = (a / a.sum()) @ V[:, h // grp]
    og = o * sigmoid(g)
    return og.reshape(qw).astype(np.float32) @ Wo.T, K, V


def ffn(m, spec, layer, res):
    """post-attention norm -> silu(gate) * up @ down. Returns the FFN's output."""
    pre = f"model.layers.{layer}."
    hid, ff, eps = spec.hidden, spec.intermediate, spec.norm_eps
    xm = (rms(res, eps) * m.bf16(pre + "post_attention_layernorm.weight")).astype(np.float32)
    Wup = m.matmul_w(pre + "mlp.up_proj.weight", ff, hid)
    Wg = m.matmul_w(pre + "mlp.gate_proj.weight", ff, hid)
    Wd = m.matmul_w(pre + "mlp.down_proj.weight", hid, ff)
    h = silu(xm @ Wg.T) * (xm @ Wup.T)
    return h.astype(np.float32) @ Wd.T


def layer_decode(m, spec, layer, x_res, conv_state, S, K, V, pos):
    """A whole layer. `conv_state` / `S` are used on a linear layer, `K` / `V` on a full
    one; the unused pair is passed through. Returns (xres, conv_state, S, K, V)."""
    if spec.layer_types[layer] == FULL:
        out, K, V = attn_decode(m, spec, layer, x_res, K, V, pos)
    else:
        out, conv_state, S = linear_decode(m, spec, layer, x_res, conv_state, S)
    res = x_res + out
    return res + ffn(m, spec, layer, res), conv_state, S, K, V


def final_logits(m, spec, x_res):
    hn = (rms(x_res, spec.norm_eps) * m.bf16("model.norm.weight")).astype(np.float32)
    m.hidden = spec.hidden
    return hn, m.lmhead_logits(hn)
