"""Independent FP64 gated full-attention checks from captured Q/K/V/gate.

Recompute Q/K RMSNorm, partial RoPE, causal GQA and sigmoid gating without
BF16 rounding. Carry reference K/V from position zero. Score every head;
absolute max error <0.02 and cosine >0.9999 are both required. This conditions on captured
projections, not an independent layer-input reference or recurrent-state gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decode_bundle import load_bundle
from decode_reference import load_reference, sfx
from q4nx import Q4NX, bf16_to_f32, f32_to_bf16
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout
from recipes.spec import FULL

MAXABS = 0.02
COSINE_MIN = 0.9999


def attention_reference(q, k, v, gate, qnorm, knorm, eps, rotary_dim, inv_freq):
    q, k, v, gate, qnorm, knorm, inv = [np.asarray(a, np.float64)
                                      for a in (q, k, v, gate, qnorm, knorm, inv_freq)]
    if (q.ndim != 3 or not all(q.shape) or k.ndim != 3 or not all(k.shape)
            or k.shape[0] != q.shape[0] or k.shape[2] != q.shape[2]
            or q.shape[1] % k.shape[1] or v.shape != k.shape or gate.shape != q.shape
            or qnorm.shape != (q.shape[2],) or knorm.shape != qnorm.shape):
        raise ValueError('invalid attention shapes')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('invalid norm epsilon')
    if (type(rotary_dim) is not int or not 0 <= rotary_dim <= q.shape[2]
            or rotary_dim % 2 or inv.shape != (rotary_dim // 2,)):
        raise ValueError('invalid rotary geometry')
    if not all(np.isfinite(a).all() for a in (q, k, v, gate, qnorm, knorm, inv)):
        raise ValueError('non-finite attention input')

    def transform(a, w):
        x = a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + eps) * w
        half = rotary_dim // 2
        angles = np.arange(len(a))[:, None, None] * inv[None, None, :]
        c, s = np.cos(angles), np.sin(angles)
        out = x.copy()
        out[..., :half] = x[..., :half] * c - x[..., half:rotary_dim] * s
        out[..., half:rotary_dim] = x[..., :half] * s + x[..., half:rotary_dim] * c
        return out

    qr, kr = transform(q, qnorm), transform(k, knorm)
    group = q.shape[1] // k.shape[1]
    out = np.empty_like(q)
    e = np.exp(-np.abs(gate))
    sigmoid = np.where(gate >= 0, 1 / (1 + e), e / (1 + e))
    for t in range(len(q)):
        for h in range(q.shape[1]):
            kh = h // group
            scores = kr[:t + 1, kh] @ qr[t, h] / np.sqrt(q.shape[2])
            weights = np.exp(scores - scores.max())
            out[t, h] = ((weights / weights.sum()) @ v[:t + 1, kh]) * sigmoid[t, h]
    if not np.isfinite(out).all():
        raise ValueError('non-finite attention reference')
    return out


def compare_heads(got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    if got.shape != ref.shape or got.ndim != 3 or not all(got.shape):
        raise ValueError('invalid attention output shapes')
    if not np.isfinite(got).all() or not np.isfinite(ref).all():
        raise ValueError('non-finite attention output')
    rows = []
    for t in range(len(ref)):
        for h in range(ref.shape[1]):
            g, r = got[t, h], ref[t, h]
            scale, error = float(np.abs(r).max()), float(np.abs(g-r).max())
            relative = error / scale if scale else (0.0 if error == 0 else None)
            # Scaling each vector avoids overflow/underflow in cosine. Exact zero
            # pairs pass; one zero vector against a nonzero vector fails.
            gs = float(np.abs(g).max())
            if not gs or not scale:
                cosine = 1.0 if not gs and not scale else 0.0
            else:
                a, b = g / gs, r / scale
                cosine = float(np.clip(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))
            rounded = bf16_to_f32(f32_to_bf16(r.astype(np.float32))).astype(np.float64)
            if not np.isfinite(rounded).all():
                raise ValueError('non-finite rounded attention reference')
            # Diagnostics only: the unrounded primary reference always decides.
            rows.append(dict(token=t, head=h,
                             ideal_bf16_output_maxabs=float(np.abs(rounded-r).max()),
                             error_to_bf16_reference_maxabs=float(np.abs(g-rounded).max()),
                             maxabs=error, reference_maxabs=scale,
                             maxrel=relative, cosine=cosine,
                             passed=error < MAXABS and cosine > COSINE_MIN))
    return rows


def read_attention(path, offsets, heads, kv_heads, dim, output_dtype="bfloat16"):
    if output_dtype not in ("bfloat16", "float32"):
        raise ValueError("unsupported attention output dtype")
    data = path.read_bytes()
    if len(data) != offsets.AA_BYTES:
        raise ValueError(f'{path.name}: wrong act buffer bytes')
    if min(heads, kv_heads, dim) <= 0 or heads % kv_heads:
        raise ValueError('invalid head geometry')

    def region(off, count, dtype):
        size = np.dtype(dtype).itemsize
        if off < 0 or off % size or off + count * size > len(data):
            raise ValueError(f'{path.name}: region outside/alignment of act buffer')
        a = np.frombuffer(data, dtype, count, off)
        a = (bf16_to_f32(a) if dtype == np.uint16 else a).astype(np.float64)
        if not np.isfinite(a).all():
            raise ValueError(f'{path.name}: non-finite attention capture')
        return a.reshape(-1, dim)

    qw, kw = heads * dim, kv_heads * dim
    return dict(q=region(offsets.AA_QG, qw, np.float32),
                gate=region(offsets.AA_QG + qw * 4, qw, np.float32),
                k=region(offsets.AA_KVN, kw, np.float32),
                v=region(offsets.AA_KVN + kw * 4, kw, np.float32),
                og=region(offsets.AA_OG, qw, np.float32 if output_dtype == "float32" else np.uint16))


def check(model_dir, kernel_dir, out, layers=None):
    spec = spec_from_model_dir(model_dir)
    if spec.family != 'qwen35':
        raise ValueError('only qwen35 gated full attention is supported')
    meta = load_reference(out)
    selected = load_bundle(kernel_dir, manifest(spec, max_ctx=meta['max_ctx']))
    for key in ('spec_hash', 'build_key'):
        if meta[key] != selected[key]:
            raise ValueError(f'fixture {key} differs from verified export')
    if meta['hidden'] != spec.hidden or not 0 < meta['layers'] <= spec.num_layers:
        raise ValueError('fixture dimensions differ from model')
    covered = [l for l in range(meta['layers']) if spec.layer_types[l] == FULL]
    layers = covered if layers is None else layers
    if not layers or len(set(layers)) != len(layers) or any(l not in covered for l in layers):
        raise ValueError('layers must be distinct full-attention indices covered by the fixture')
    offsets = layout(spec, max_ctx=meta['max_ctx'])
    model = Q4NX(model_dir / 'model.q4nx')
    results, hashes, norm_hashes = [], {}, {}
    for layer in layers:
        norms = []
        for kind in ('q', 'k'):
            name = f'model.layers.{layer}.self_attn.{kind}_norm.weight'
            w = model.bf16(name).astype(np.float64)
            norms.append(w)
            norm_hashes[name] = hashlib.sha256(w.astype('<f8').tobytes()).hexdigest()
        captures = []
        for token in range(meta['tokens']):
            path = out / f'y_act{layer}{sfx(token)}.bin'
            captures.append(read_attention(path, offsets, spec.num_heads, spec.num_kv_heads, spec.head_dim,
                                           selected.get("layout", {}).get("attn_output_dtype", "bfloat16")))
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        a = {n: np.stack([c[n] for c in captures]) for n in captures[0]}
        ref = attention_reference(a['q'], a['k'], a['v'], a['gate'], *norms,
                                  spec.norm_eps, spec.rotary_dim, spec.rope_inv_freq())
        results.extend(dict(layer=layer, **r) for r in compare_heads(a['og'], ref))
    return dict(version=1, passed=all(r['passed'] for r in results),
                scope='Gated full-attention heads conditioned on captured Q/K/V/gate; '
                      'independent FP64 QK norm/RoPE/causal GQA, no BF16 reference rounding. '
                      'Not projection, KV-byte, DeltaNet-state or full-layer accuracy.',
                spec_hash=selected['spec_hash'], build_key=selected['build_key'],
                fixture_sha256=hashlib.sha256((out / 'decode_reference.json').read_bytes()).hexdigest(),
                maxabs_bound=MAXABS, cosine_min=COSINE_MIN, layers=layers, tokens=meta['tokens'],
                heads=spec.num_heads, capture_sha256=hashes,
                norm_sha256=norm_hashes, norm_hash_encoding='little-endian float64 effective weights',
                results=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--kernel-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='completed decode fixture')
    ap.add_argument('--layers', help='comma-separated indices; default: every covered full-attention layer')
    a = ap.parse_args()
    try:
        layers = None if a.layers is None else [int(n) for n in a.layers.split(',')]
        result = check(a.model_dir, a.kernel_dir, a.out, layers)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['passed'] else 1
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'FAIL: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
