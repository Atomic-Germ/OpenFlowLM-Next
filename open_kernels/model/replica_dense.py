"""fp64 CPU reference for one Qwen3 dense decode step -- the oracle the dense
kernels (designs/dense/dx.py, lm_head_q4) are checked against.

HF-faithful math in float64 from the same `.q4nx` bytes the NPU pools are
packed from (q4nx.py dequantizes the chunks), so a disagreement is the kernels'
or the packing's, not a difference of source weights:

  * x = rms(res) * ln_w; q k v projections; q/k RMSNorm over the head with the
    stored weights; RoPE over the first rotary_dim dims (the whole head, or 96
    of 128 on Phi-3, half-split pairs, the model's theta and scaling); GQA
    softmax attention over the KV cache; o_proj; residual.
    HunYuan norms after the rotation instead (recipes.dense.QKNORM_POST_ROPE),
    which is attn.h's ATTN_QKNORM_POST
  * post-attention norm; silu(gate) * up @ down; residual
  * final norm, then the q4_1 lm_head

State (K, V) per layer is carried by the caller: decode from position 0 is
exact prefill.
"""
from __future__ import annotations

import numpy as np

from q4nx import CHUNK_Q4, dq_chunks_q4_1


def rms(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def silu(x):
    return x / (1 + np.exp(-x))


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def rope(t, p, rot, theta, inv_freq=None, scale=1.0):
    """The first `rot` dims of each head rotated in half-split pairs; the rest pass through
    (Phi-3 rotates 96 of 128). `scale` is longrope's attention factor on cos and sin."""
    half = rot // 2
    ang = p * (np.asarray(inv_freq, np.float64) if inv_freq is not None else theta ** (-np.arange(half) / half))
    c, s = scale * np.cos(ang), scale * np.sin(ang)
    y = t.copy()
    x1, x2 = t[..., :half], t[..., half:rot]
    y[..., :half] = x1 * c - x2 * s
    y[..., half:rot] = x2 * c + x1 * s
    return y


def dense_decode(m, spec, layer, x_res, K, V, pos, max_ctx=4096):
    """One token through a dense layer. Returns (residual, K, V). `max_ctx` picks longrope's
    factor list, as the kernel set's export does."""
    from recipes.dense import QKNORM_POST_ROPE
    from recipes.spec import DENSE_LOCAL
    pre = f"model.layers.{layer}."
    hid, nh, kvh, hd, ff = spec.hidden, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.intermediate
    local = spec.layer_types[layer] == DENSE_LOCAL
    eps, inv, rsc = spec.norm_eps, spec.rope_inv_freq(local=local, ctx=max_ctx), spec.rope_scale()
    act = gelu_tanh if spec.activation == "gelu_tanh" else silu
    x = (rms(x_res, eps) * m.bf16(pre + "input_layernorm.weight")).astype(np.float32)
    Wq = m.matmul_w(pre + "self_attn.q_proj.weight", nh * hd, hid)
    Wk = m.matmul_w(pre + "self_attn.k_proj.weight", kvh * hd, hid)
    Wv = m.matmul_w(pre + "self_attn.v_proj.weight", kvh * hd, hid)
    Wo = m.matmul_w(pre + "self_attn.o_proj.weight", hid, nh * hd)
    post = spec.family in QKNORM_POST_ROPE          # the norm weight multiplies after RoPE
    q = (x @ Wq.T).reshape(nh, hd).astype(np.float64)
    k = (x @ Wk.T).reshape(kvh, hd).astype(np.float64)
    if spec.qk_norm:
        qn = m.bf16(pre + "self_attn.q_norm.weight")
        kn = m.bf16(pre + "self_attn.k_norm.weight")
        q, k = rms(q, eps), rms(k, eps)
        if not post:
            q, k = q * qn, k * kn
    v = (x @ Wv.T).reshape(kvh, hd).astype(np.float64)
    q = rope(q, pos, spec.rotary_dim, spec.rope_theta, inv, rsc)
    k = rope(k, pos, spec.rotary_dim, spec.rope_theta, inv, rsc)
    if spec.qk_norm and post:
        q, k = q * qn, k * kn
    K = np.concatenate([K, k[None]], 0)
    V = np.concatenate([V, v[None]], 0)
    o = np.zeros((nh, hd))
    grp = nh // kvh
    t0 = max(0, pos + 1 - spec.sliding_window) if (local and spec.sliding_window) else 0   # the window's first row
    for h in range(nh):
        s = (K[t0:, h // grp] @ q[h]) / np.sqrt(hd)
        a = np.exp(s - s.max())
        o[h] = (a / a.sum()) @ V[t0:, h // grp]
    out = o.reshape(nh * hd).astype(np.float32) @ Wo.T
    Wup = m.matmul_w(pre + "mlp.up_proj.weight", ff, hid)
    Wg = m.matmul_w(pre + "mlp.gate_proj.weight", ff, hid)
    Wd = m.matmul_w(pre + "mlp.down_proj.weight", hid, ff)
    if spec.sandwich_norms:
        t = (rms(out, eps) * m.bf16(pre + "post_attention_layernorm.weight")).astype(np.float32)
        res = x_res + t
        xm = (rms(res, eps) * m.bf16(pre + "pre_feedforward_layernorm.weight")).astype(np.float32)
        h = act(xm @ Wg.T) * (xm @ Wup.T)
        out2 = h.astype(np.float32) @ Wd.T
        t2 = (rms(out2, eps) * m.bf16(pre + "post_feedforward_layernorm.weight")).astype(np.float32)
        return res + t2, K, V
    res = x_res + out
    xm = (rms(res, eps) * m.bf16(pre + "post_attention_layernorm.weight")).astype(np.float32)
    h = act(xm @ Wg.T) * (xm @ Wup.T)
    return res + h.astype(np.float32) @ Wd.T, K, V


def lmhead_q4_logits(m, hn, spec, block=2048):
    """logits[vocab] = W_lm[vocab, hidden] @ hn from the q4_1 chunks in the file's raster order."""
    hid = spec.hidden
    raw = np.frombuffer(m.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, CHUNK_Q4)
    ncol = hid // 256
    nch = raw.shape[0]
    hn = np.asarray(hn, np.float64)
    logits = np.zeros(nch // ncol * 32, np.float64)
    for c0 in range(0, nch, block):
        ce = min(c0 + block, nch)
        w = dq_chunks_q4_1(raw[c0:ce]).reshape(ce - c0, 32, 256).astype(np.float64)
        for c in range(c0, ce):
            r0, k0 = 32 * (c // ncol), 256 * (c % ncol)
            logits[r0:r0 + 32] += w[c - c0] @ hn[k0:k0 + 256]
    return logits


def final_logits(m, spec, x_res):
    hn = (rms(x_res, spec.norm_eps) * m.bf16("model.norm.weight")).astype(np.float32)
    return hn, lmhead_q4_logits(m, hn, spec)
