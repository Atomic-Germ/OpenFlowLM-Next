"""Independent float64 oracle for synthetic WideDeltaNet hardware acceptance.

This is test tooling, never a runtime fallback. Inputs have already been
quantized to the same bf16/f32 values sent to the device.
"""
import numpy as np
from ml_dtypes import bfloat16


def ab_reference(xn, weights, a, dt_bias):
    alpha, logits = weights.astype(np.float64) @ xn.astype(np.float64)
    decay = np.exp(a.astype(np.float64) * np.logaddexp(0, alpha + dt_bias.astype(np.float64)))
    beta = 1 / (1 + np.exp(-logits))
    return np.stack((alpha, logits, decay, beta))


def glue_reference(qkv, state, convw, ab, key_heads=16, dim=128):
    heads = ab.shape[1]
    kw = key_heads * dim
    seq = np.vstack((state.astype(np.float64), qkv.astype(np.float64)))
    c = np.sum(convw.astype(np.float64) * seq, axis=0)
    c = c / (1 + np.exp(-c))
    q, k = c[:kw].reshape(key_heads, dim), c[kw:2 * kw].reshape(key_heads, dim)
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    records = np.zeros((heads, 512), np.float64)
    records[:, :dim] = np.repeat(k, heads // key_heads, axis=0)
    records[:, dim:2 * dim] = np.repeat(q, heads // key_heads, axis=0)
    records[:, 2 * dim:3 * dim] = c[2 * kw:].reshape(heads, dim)
    records[:, 384:386] = ab[2:].T
    nstate = np.vstack((state[1:], qkv.astype(bfloat16)))
    return nstate, records


def step_reference(state, records):
    k, q, v = (records[:, start:start + 128].astype(np.float64) for start in (0, 128, 256))
    s = state.astype(np.float64) * records[:, 384, None, None]
    delta = records[:, 385, None] * (v - np.einsum("hij,hi->hj", s, k))
    s += k[:, :, None] * delta[:, None, :]
    output = np.einsum("hij,hi->hj", s, q) / np.sqrt(128)
    return s, output


def metric(got, ref, cosine_threshold):
    if got.shape != ref.shape:
        return dict(passed=False, error=f"shape {got.shape} != {ref.shape}")
    got, ref = got.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    if not np.isfinite(got).all() or not np.isfinite(ref).all():
        return dict(passed=False, error="non-finite data")
    ref_scale, got_scale = np.max(np.abs(ref)), np.max(np.abs(got))
    error = np.max(np.abs(got - ref))
    rel = float(error / ref_scale) if ref_scale else (0.0 if error == 0 else float("inf"))
    # Exact zero is useful for state/reset tests; cosine is undefined there.
    if np.array_equal(got, ref):
        cosine = 1.0
    elif ref_scale == 0 or got_scale == 0:
        cosine = 0.0
    else:
        # Cosine is scale invariant. Normalize first instead of adding an
        # absolute epsilon that dominates small-but-nonzero tensors.
        a, b = got / got_scale, ref / ref_scale
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return dict(passed=rel < 1e-4 and cosine > cosine_threshold, maxrel=rel, cosine=cosine)
