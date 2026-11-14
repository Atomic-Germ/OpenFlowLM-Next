"""Strict Qwen3.5 two-piece FFN down check, conditioned on captured NPU h.

One layer's down weight is resident at a time. This is an independent FP64
projection from the captured input, not an end-to-end FFN or state reference.
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
from decode_reference import load_reference, read_capture, sfx
from q4nx import Q4NX, bf16_to_f32, f32_to_bf16
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout, ffn_geometry
from recipes.spec import FULL

MAXREL = 0.005


def compare_partials(h, weights, res, partials, closed, split, diagnose_bf16=False):
    """FP64 dot products, each normalized by its own reference maximum.

    Checking both pieces prevents opposite errors from cancelling in the sum.
    Zero reference vectors require exact zeros. No reference-side BF16 rounding.
    """
    h, weights, res, closed = [np.asarray(x, dtype=np.float64) for x in (h, weights, res, closed)]
    partials = [np.asarray(x, dtype=np.float64) for x in partials]
    if (len(split) != 2 or any(type(k) is not int or k <= 0 for k in split)
            or h.ndim != 1 or res.ndim != 1 or len(partials) != 2
            or sum(split) != h.size or weights.shape != (res.size, h.size)
            or closed.shape != res.shape or any(p.shape != res.shape for p in partials)
            or not h.size or not res.size):
        raise ValueError('invalid shapes or two-piece partition')
    if not all(np.isfinite(x).all() for x in [h, weights, res, closed, *partials]):
        raise ValueError('non-finite FFN input or capture')
    k = split[0]
    refs = [weights[:, :k] @ h[:k], weights[:, k:] @ h[k:]]
    pairs = dict(partial0=(partials[0], refs[0]), partial1=(partials[1], refs[1]),
                 sum=(partials[0] + partials[1], refs[0] + refs[1]),
                 closing_residual=(closed, res + refs[0] + refs[1]))
    metrics = {}
    for name, (got, ref) in pairs.items():
        if not np.isfinite(ref).all():
            raise ValueError('non-finite FP64 reference')
        scale = float(np.max(np.abs(ref)))
        error = float(np.max(np.abs(got - ref)))
        relative = error / scale if scale else (0.0 if error == 0 else None)
        metrics[name] = dict(maxabs=error, reference_maxabs=scale, maxrel=relative,
                             passed=relative is not None and relative < MAXREL)
    result = dict(passed=all(v['passed'] for v in metrics.values()), metrics=metrics)
    if diagnose_bf16:
        rounded = bf16_to_f32(f32_to_bf16(h.astype(np.float32)))
        result['bf16_input_diagnostic'] = compare_partials(rounded, weights, res, partials, closed, split)
    return result


def read_regions(path, offsets, hidden, intermediate, full):
    data = path.read_bytes()
    prefix = 'AA_' if full else 'A_'
    size = getattr(offsets, prefix + 'BYTES')
    if len(data) != size:
        raise ValueError(f'{path.name}: expected {size} bytes, got {len(data)}')

    def region(name, count):
        off = getattr(offsets, prefix + name)
        if off < 0 or off + count * 4 > size:
            raise ValueError(f'{path.name}: {name} outside act buffer')
        a = np.frombuffer(data, np.float32, count=count, offset=off).astype(np.float64)
        if not np.isfinite(a).all():
            raise ValueError(f'{path.name}: non-finite {name}')
        return a

    return dict(res=region('RES', hidden), h=region('H', intermediate),
                partials=[region('OUT2', hidden), region('OUT2B', hidden)])


def check(model_dir, kernel_dir, out, layers, diagnose_bf16=False):
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
    split = tuple(ffn_geometry(spec).DOWN_SPLIT)
    if len(split) != 2:
        raise ValueError('this model does not use a two-piece down projection')
    offsets = layout(spec, max_ctx=meta['max_ctx'])
    model = Q4NX(model_dir / 'model.q4nx')
    results = []
    captures = {}
    for layer in layers:
        # No whole-model materialization: release each down matrix before the next.
        weights = model.matmul_w(f'model.layers.{layer}.mlp.down_proj.weight',
                                 spec.hidden, spec.intermediate).astype(np.float64)
        for token in range(meta['tokens']):
            suffix = sfx(token)
            act = out / f'y_act{layer}{suffix}.bin'
            residual = out / f'y_res{layer}{suffix}.bin'
            regions = read_regions(act, offsets, spec.hidden, spec.intermediate,
                                   spec.layer_types[layer] == FULL)
            closed = read_capture(residual, spec.hidden)
            result = compare_partials(regions['h'], weights, regions['res'],
                                      regions['partials'], closed, split, diagnose_bf16=diagnose_bf16)
            results.append(dict(layer=layer, token=token, **result))
            for path in (act, residual):
                captures[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        del weights
    return dict(version=1, passed=all(r['passed'] for r in results),
                scope='FP64 down projection conditioned on captured NPU h and residual; not full FFN/state accuracy',
                spec_hash=selected['spec_hash'], build_key=selected['build_key'],
                fixture_sha256=hashlib.sha256((out / 'decode_reference.json').read_bytes()).hexdigest(),
                split=list(split), maxrel_bound=MAXREL, layers=layers, tokens=meta['tokens'],
                capture_sha256=captures, results=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--kernel-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='completed decode fixture with act/residual captures')
    ap.add_argument('--layers', default='0,3', help='comma-separated layer indices (default: linear 0 and attention 3)')
    ap.add_argument('--diagnose-bf16', action='store_true',
                    help='also measure a rounded-input diagnostic; never changes acceptance')
    args = ap.parse_args()
    try:
        result = check(args.model_dir, args.kernel_dir, args.out,
                       [int(n) for n in args.layers.split(',')], diagnose_bf16=args.diagnose_bf16)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['passed'] else 1
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f'FAIL: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
