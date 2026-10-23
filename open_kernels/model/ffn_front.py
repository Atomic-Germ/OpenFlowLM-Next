"""Independent Qwen3.5 norm + SiLU(gate)*up checks from captured pre-FFN residual.

Raw up/gate outputs are not retained, so only their combined activation is
observable. The primary reference never substitutes the captured BF16 norm.
One layer's two projection matrices are loaded at a time; no NPU is required.
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

MAXREL = 0.005


def metric(got, ref):
    if not np.isfinite(got).all() or not np.isfinite(ref).all():
        raise ValueError('non-finite reference or capture')
    scale = float(np.max(np.abs(ref)))
    error = float(np.max(np.abs(got - ref)))
    relative = error / scale if scale else (0.0 if error == 0 else None)
    return dict(maxabs=error, reference_maxabs=scale, maxrel=relative,
                passed=relative is not None and relative < MAXREL)


def activation(up, gate, x):
    u, g = up @ x, gate @ x
    # Stable sigmoid: never exponentiate a large positive number.
    e = np.exp(-np.abs(g))
    return u * g * np.where(g >= 0, 1 / (1 + e), e / (1 + e))


def compare_front(res, norm, up, gate, xm, h, eps):
    res, norm, up, gate, xm, h = [np.asarray(a, dtype=np.float64)
                                 for a in (res, norm, up, gate, xm, h)]
    if (res.ndim != 1 or not res.size or norm.shape != res.shape or xm.shape != res.shape
            or h.ndim != 1 or not h.size or up.shape != (h.size, res.size) or gate.shape != up.shape):
        raise ValueError('invalid FFN front shapes')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('invalid norm epsilon')
    if not all(np.isfinite(a).all() for a in (res, norm, up, gate, xm, h)):
        raise ValueError('non-finite FFN front input')
    xref = res / np.sqrt(np.mean(res * res) + eps) * norm
    href = activation(up, gate, xref)
    metrics = dict(norm=metric(xm, xref), front_h=metric(h, href),
                   activation_from_captured_norm=metric(h, activation(up, gate, xm)))
    rounded = bf16_to_f32(f32_to_bf16(xref.astype(np.float32))).astype(np.float64)
    hrounded = activation(up, gate, rounded)
    return dict(passed=all(m['passed'] for m in metrics.values()), metrics=metrics,
                rounded_norm_diagnostic=metric(h, hrounded),
                norm_rounding_effect=metric(hrounded, href),
                norm_bf16_rounding=dict(mismatched_values=int(np.count_nonzero(xm != rounded)),
                                        **metric(xm, rounded)))


def read_front(path, offsets, hidden, intermediate, full, norm_dtype="bfloat16"):
    if norm_dtype not in ("bfloat16", "float32"):
        raise ValueError("unsupported norm dtype")
    data = path.read_bytes()
    prefix = 'AA_' if full else 'A_'
    if len(data) != getattr(offsets, prefix + 'BYTES'):
        raise ValueError(f'{path.name}: wrong act buffer bytes')

    def region(name, count, dtype):
        off = getattr(offsets, prefix + name)
        size = np.dtype(dtype).itemsize
        if off < 0 or off % size or off + count * size > len(data):
            raise ValueError(f'{path.name}: {name} outside/alignment of act buffer')
        a = np.frombuffer(data, dtype, count=count, offset=off)
        a = (bf16_to_f32(a) if dtype == np.uint16 else a).astype(np.float64)
        if not np.isfinite(a).all():
            raise ValueError(f'{path.name}: non-finite {name}')
        return a

    return dict(res=region('RES', hidden, np.float32), xm=region('XM', hidden, np.float32 if norm_dtype == 'float32' else np.uint16),
                h=region('H', intermediate, np.float32))


def check(model_dir, kernel_dir, out, layers):
    spec = spec_from_model_dir(model_dir)
    if spec.family != 'qwen35':
        raise ValueError('only the qwen35 dense FFN is supported')
    meta = load_reference(out)
    selected = load_bundle(kernel_dir, manifest(spec, max_ctx=meta['max_ctx']))
    for key in ('spec_hash', 'build_key'):
        if meta[key] != selected[key]:
            raise ValueError(f'fixture {key} differs from verified export')
    if meta['hidden'] != spec.hidden or meta['layers'] > spec.num_layers:
        raise ValueError('fixture dimensions differ from model')
    if not layers or len(set(layers)) != len(layers) or any(l < 0 or l >= meta['layers'] for l in layers):
        raise ValueError('layers must be distinct indices covered by the fixture')
    offsets = layout(spec, max_ctx=meta['max_ctx'])
    model = Q4NX(model_dir / 'model.q4nx')
    results, hashes = [], {}
    for layer in layers:
        pre = f'model.layers.{layer}.'
        norm = model.bf16(pre + 'post_attention_layernorm.weight').astype(np.float64)
        up = model.matmul_w(pre + 'mlp.up_proj.weight', spec.intermediate, spec.hidden).astype(np.float64)
        gate = model.matmul_w(pre + 'mlp.gate_proj.weight', spec.intermediate, spec.hidden).astype(np.float64)
        for token in range(meta['tokens']):
            path = out / f'y_act{layer}{sfx(token)}.bin'
            r = read_front(path, offsets, spec.hidden, spec.intermediate, spec.layer_types[layer] == FULL,
                           norm_dtype=selected.get('layout', {}).get('ffn_norm_dtype', 'bfloat16'))
            result = compare_front(r['res'], norm, up, gate, r['xm'], r['h'], spec.norm_eps)
            results.append(dict(layer=layer, token=token, **result))
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        del up, gate
    return dict(version=1, passed=all(r['passed'] for r in results),
                scope='Norm and combined SiLU(gate)*up conditioned on captured pre-FFN residual; '
                      'no individual up/gate captures or independent attention/state reference',
                spec_hash=selected['spec_hash'], build_key=selected['build_key'],
                fixture_sha256=hashlib.sha256((out / 'decode_reference.json').read_bytes()).hexdigest(),
                maxrel_bound=MAXREL, layers=layers, tokens=meta['tokens'],
                capture_sha256=hashes, results=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--kernel-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='completed decode fixture')
    ap.add_argument('--layers', default='0,3', help='comma-separated covered layer indices')
    a = ap.parse_args()
    try:
        result = check(a.model_dir, a.kernel_dir, a.out, [int(n) for n in a.layers.split(',')])
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['passed'] else 1
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'FAIL: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
