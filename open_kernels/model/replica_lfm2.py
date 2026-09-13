"""fp64 CPU reference for one LFM2 decode step -- the oracle the short-conv kernels
will be checked against.

Ten of LFM2-1.2B's sixteen layers replace attention with a three-tap depthwise causal
convolution; the other six are plain GQA attention with q/k RMSNorm. Every layer has the
same silu-gated FFN and the same two norms, so an attention layer IS the dense recipe's
block and goes through `replica_dense.dense_decode` unchanged -- only the sequence mixer
is new.

The short-conv block, per token (transformers' `Lfm2ShortConv`):

    h = W_in @ x                       # 3 x hidden rows, in the order [B | C | u]
    Bx = B * u                         # the value gated on the way in
    conv = sum_k state[k] * w[:, k]    # state[L-1] is this token's Bx, state[0] the oldest
    y = C * conv                       # gated again on the way out
    out = W_out @ y

No bias and no normalisation inside the block; `conv_bias` is false on every LFM2 that
ships. What the cache holds is `Bx`, not the block input -- the same thing
`Cache.update_conv_state` keeps -- so a decode from a zeroed state is exact prefill.

Traces: OPEN-SHORT-CONV-REF, OPEN-FAMILY-LFM2 (specs/open-engine/spec.md).
"""
from __future__ import annotations

import numpy as np

from replica_dense import dense_decode, final_logits, rms, silu  # noqa: F401
from recipes.spec import FULL, SHORT_CONV

# An LFM2 attention layer is a dense layer: same norms, same residuals, same FFN, and the
# container uses the dense tensor names for all of it.
attn_decode = dense_decode


def layer_kind(spec, layer: int) -> str:
    """"attn" or "short_conv" -- which half of this module layer `layer` goes through."""
    t = spec.layer_types[layer]
    if t == FULL:
        return "attn"
    if t == SHORT_CONV:
        return "short_conv"
    raise ValueError(f"lfm2: layer {layer} is a {t!r} layer, which this replica has no path for")


def conv_state_shape(spec) -> tuple[int, int]:
    """[L-1, conv width]: the taps a new token still needs, oldest first. The conv width is
    the hidden size (LFM2's `conv_dim`), which is why it is not a spec field."""
    return (spec.conv_kernel - 1, spec.hidden)


def conv_state_bytes(spec) -> int:
    """One conv layer's state as the engine would carry it, f32."""
    n, w = conv_state_shape(spec)
    return n * w * 4


def short_conv_step(x, W_in, w_conv, W_out, conv_state):
    """One token through the short-conv block. Returns (out[hidden], new conv_state).

    `w_conv` is [width, taps] as the container stores it (`shortconv.conv.weight`, a
    squeezed depthwise `Conv1d` weight): tap `taps-1` is this token's own."""
    x = np.asarray(x, np.float64)
    W_in, w_conv, W_out = (np.asarray(a, np.float64) for a in (W_in, w_conv, W_out))
    hid = x.shape[-1]
    if W_in.shape != (3 * hid, hid):
        raise ValueError(f"short_conv: in_proj is {W_in.shape}, want 3 x hidden = "
                         f"({3 * hid}, {hid})")
    taps = w_conv.shape[-1]
    conv_state = np.asarray(conv_state, np.float64).reshape(-1, w_conv.shape[0])
    if conv_state.shape[0] != taps - 1:
        raise ValueError(f"short_conv: conv state has {conv_state.shape[0]} rows, a {taps}-tap "
                         f"conv needs {taps - 1}")
    h = W_in @ x
    B, C, u = h[:hid], h[hid:2 * hid], h[2 * hid:]
    window = np.vstack([conv_state, (B * u)[None, :]])      # [taps, width], newest last
    conv = (window.T * w_conv).sum(-1)
    return W_out @ (C * conv), window[1:]


def short_conv_sequence(xs, W_in, w_conv, W_out):
    """A whole prompt through the block in the padded-convolution form transformers uses
    for prefill: [T, hidden] in, [T, hidden] out, left-padded with taps-1 zeros. Here to be
    compared against `short_conv_step` -- the two must agree to the bit."""
    xs = np.atleast_2d(np.asarray(xs, np.float64))
    W_in, w_conv, W_out = (np.asarray(a, np.float64) for a in (W_in, w_conv, W_out))
    hid = xs.shape[-1]
    taps = w_conv.shape[-1]
    h = xs @ W_in.T                                          # [T, 3*hidden]
    Bx = h[:, :hid] * h[:, 2 * hid:]
    pad = np.vstack([np.zeros((taps - 1, hid)), Bx])          # causal left pad
    conv = sum(pad[k:k + xs.shape[0]] * w_conv[:, k] for k in range(taps))
    return (h[:, hid:2 * hid] * conv) @ W_out.T


def short_conv_decode(m, spec, layer, x_res, conv_state):
    """One token through a whole short-conv LAYER, weights from the `.q4nx` container:
    the entry norm, the block, the residual, then the same FFN tail an attention layer has.
    Returns (residual, new conv_state)."""
    pre = f"model.layers.{layer}."
    hid, ff, eps = spec.hidden, spec.intermediate, spec.norm_eps
    x = rms(x_res, eps) * m.bf16(pre + "input_layernorm.weight")
    W_in = m.matmul_w(pre + "shortconv.in_proj.weight", 3 * hid, hid)
    W_out = m.matmul_w(pre + "shortconv.out_proj.weight", hid, hid)
    w_conv = m.bf16(pre + "shortconv.conv.weight")           # [hidden, taps]
    out, conv_state = short_conv_step(x, W_in, w_conv, W_out, conv_state)
    res = x_res + out
    xm = rms(res, eps) * m.bf16(pre + "post_attention_layernorm.weight")
    Wg = m.matmul_w(pre + "mlp.gate_proj.weight", ff, hid)
    Wup = m.matmul_w(pre + "mlp.up_proj.weight", ff, hid)
    Wd = m.matmul_w(pre + "mlp.down_proj.weight", hid, ff)
    h = silu(xm @ Wg.T) * (xm @ Wup.T)
    return res + h @ Wd.T, conv_state
