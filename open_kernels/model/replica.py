"""fp64 CPU reference for one Qwen3.6-MoE decode step -- the oracle the open
kernels are checked against.

This is the HF-faithful math, computed in float64 from the same `.q4nx` weights
the NPU pools are packed from, so a disagreement is the kernels' (or the pool
packing's), not a difference of source weights. Every element was verified
against captured NPU buffers by phlegm (tools/kernel-interp/decode_step.py,
full_forward.py), which this vendors and parameterizes:

  * linear-attention layer: qkv -> depthwise conv1d(k=4) + SiLU -> q/k L2 norm
    -> gated delta rule (decay = exp(a * softplus(x@Wa + dt_bias)),
    beta = sigmoid(x@Wb), o = S'^T q / sqrt(128)) -> RMSNormGated * silu(z)
    -> out_proj -> residual
  * MoE block: router softmax, top-8 renormalized, silu(gate)*up @ down, plus
    the shared expert gated by sigmoid(xm @ shared_expert_gate)
  * full-attention layer: fused q_proj [q | gate], q/k RMSNorm with the stored
    effective weights, partial RoPE (rotary dim 64 of 256, half-split,
    theta 1e7), softmax attention over the KV cache, sigmoid gate, o_proj
  * final model.norm, then the q8 lm_head

State is carried by the caller: `(conv_state, S)` per linear layer, `(K, V)`
per attention layer -- decode from position 0 with zeroed state is exact
prefill, since each layer's state update only ever sees one token at a time.
"""
from __future__ import annotations

import numpy as np

ROPE_THETA = 1e7
ROTARY_HALF = 32          # rotary dim 64 of head_dim 256, split in half


def rms(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def silu(x):
    return x / (1 + np.exp(-x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear_decode(m, layer, x_res, conv_state, S):
    """One token through a linear-attention (gated DeltaNet) layer.
    Returns (residual, new conv_state, S updated in place)."""
    ln = m.bf16(f"model.layer.{layer}.input_layernorm.weight")
    x = (rms(x_res) * ln).astype(np.float32)
    Wqkv = m.matmul_w(f"model.layer.{layer}.linear_attn.qkv_proj.weight", 8192, 2048)
    Wz = m.matmul_w(f"model.layer.{layer}.self_attn.gate_proj.weight", 4096, 2048)
    Wout = m.matmul_w(f"model.layer.{layer}.linear_attn.ssm_out_proj.weight", 2048, 4096)
    convw = m.bf16(f"model.layer.{layer}.linear_attn.ssm_conv1d.weight")
    qkv = x @ Wqkv.T
    z = silu(x @ Wz.T)
    c = silu((convw * np.vstack([conv_state, qkv])).sum(0))     # depthwise, this token

    def l2n(a):
        return a / np.sqrt((a ** 2).sum(-1, keepdims=True) + 1e-6)

    q = l2n(c[:2048].reshape(16, 128))
    k = l2n(c[2048:4096].reshape(16, 128))
    v = c[4096:].reshape(32, 128).astype(np.float64)
    Wa = m.bf16(f"model.layer.{layer}.linear_attn.ssm_alpha_proj.weight")
    Wb = m.bf16(f"model.layer.{layer}.linear_attn.ssm_beta_proj.weight")
    A = m.f32(f"model.layer.{layer}.linear_attn.ssm_a")          # file stores -exp(A_log)
    dtb = m.f32(f"model.layer.{layer}.linear_attn.ssm_dt.bias")
    decay = np.exp(A * np.log1p(np.exp(x @ Wa + dtb)))
    beta = sigmoid(x @ Wb)
    o = np.zeros((32, 128))
    for h in range(32):
        kk, qq = k[h // 2], q[h // 2]                            # 16 key heads, 32 value heads
        S[h] *= decay[h]
        delta = beta[h] * (v[h] - S[h].T @ kk)
        S[h] += np.outer(kk, delta)
        o[h] = (S[h].T @ qq) / np.sqrt(128)
    nw = m.bf16(f"model.layer.{layer}.linear_attn.ssm_norm.weight")
    og = (rms(o) * nw).reshape(4096) * z
    new_conv = np.vstack([conv_state[1:], qkv[None, :]])
    return x_res + og.astype(np.float32) @ Wout.T, new_conv, S


def attn_decode(m, layer, x_res, K, V, pos):
    """One token through a full-attention layer. Returns (residual, K, V)."""
    ln = m.bf16(f"model.layer.{layer}.input_layernorm.weight")
    x = (rms(x_res) * ln).astype(np.float32)
    Wqg = m.matmul_w(f"model.layer.{layer}.self_attn.q_proj.weight", 8192, 2048)
    Wq, Wg = Wqg[:4096], Wqg[4096:]
    Wk = m.matmul_w(f"model.layer.{layer}.self_attn.k_proj.weight", 512, 2048)
    Wv = m.matmul_w(f"model.layer.{layer}.self_attn.v_proj.weight", 512, 2048)
    Wo = m.matmul_w(f"model.layer.{layer}.self_attn.o_proj.weight", 2048, 4096)
    qn = m.bf16(f"model.layer.{layer}.self_attn.q_norm.weight")
    kn = m.bf16(f"model.layer.{layer}.self_attn.k_norm.weight")
    q = (rms((x @ Wq.T).reshape(16, 256)) * qn).astype(np.float64)
    g = (x @ Wg.T).reshape(16, 256)
    k = (rms((x @ Wk.T).reshape(2, 256)) * kn).astype(np.float64)
    v = (x @ Wv.T).reshape(2, 256).astype(np.float64)

    def rope(t_, p):
        ang = p * ROPE_THETA ** (-np.arange(ROTARY_HALF) / ROTARY_HALF)
        c, s = np.cos(ang), np.sin(ang)
        y = t_.copy()
        x1, x2 = t_[..., :ROTARY_HALF], t_[..., ROTARY_HALF:2 * ROTARY_HALF]
        y[..., :ROTARY_HALF] = x1 * c - x2 * s
        y[..., ROTARY_HALF:2 * ROTARY_HALF] = x2 * c + x1 * s
        return y

    q, k = rope(q, pos), rope(k, pos)
    K = np.concatenate([K, k[None]], 0)          # [pos+1, 2 kv heads, 256]
    V = np.concatenate([V, v[None]], 0)
    o = np.zeros((16, 256))
    for h in range(16):
        s = (K[:, h // 8] @ q[h]) / 16.0         # 1/sqrt(head_dim)
        a = np.exp(s - s.max())
        o[h] = (a / a.sum()) @ V[:, h // 8]
    og = o * sigmoid(g)
    return x_res + og.reshape(4096).astype(np.float32) @ Wo.T, K, V


def route(m, layer, x_res):
    """(xm, probabilities, top-8 expert ids) for the MoE block."""
    postln = m.bf16(f"model.layer.{layer}.post_attention_layernorm.weight")
    xm = (rms(x_res) * postln).astype(np.float32)
    lg = xm @ m.bf16(f"model.layer.{layer}.moe_router.weight")
    p = np.exp(lg - lg.max())
    p /= p.sum()
    return xm, p, np.argsort(-p, kind="stable")[:8]


def moe_decode(m, layer, x_res, top=None):
    """The MoE block. `top` overrides the routing (to compare like for like when
    the NPU's 8th slot is a near-tie); the weights stay the replica's."""
    xm, p, mine = route(m, layer, x_res)
    top = mine if top is None else np.asarray(top, np.int64)
    w8 = p[top] / p[top].sum()
    out = np.zeros(2048)
    for e, ww in zip(top, w8):
        h = silu(m.expert_w(layer, "gate", int(e)) @ xm) * (m.expert_w(layer, "up", int(e)) @ xm)
        out += ww * (m.expert_w(layer, "down", int(e)) @ h)
    sh = m.shared_w(layer, "down") @ (silu(m.shared_w(layer, "gate") @ xm) * (m.shared_w(layer, "up") @ xm))
    sg = sigmoid(xm @ m.bf16(f"model.layer.{layer}.shared_expert_gate.weight"))
    return x_res + out + sg * sh


def final_logits(m, x_res):
    hn = (rms(x_res) * m.bf16("model.norm.weight")).astype(np.float32)
    return hn, m.lmhead_logits(hn)
