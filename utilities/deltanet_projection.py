"""Separate DeltaNet norm and QKV projection errors in retained decode captures.

Each CPU variant carries its own unrounded conv history from zero. Diagnostic
matches never replace the primary residual-to-record gate. This is not full
DeltaNet acceptance or an emulation of AIE arithmetic. Loads one layer's Q4_1
projection weights; Q8 projection diagnostics are not implemented here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'open_kernels'), str(ROOT / 'open_kernels/model')]
import deltanet_inputs as di
from deltanet_state import finite, linear_dtypes, read_boundaries
from decode_bundle import load_bundle
from decode_reference import load_reference, sfx
from recipes.load import spec_from_model_dir
from recipes.manifest import manifest
from recipes.qwen35 import layout
from recipes.spec import LINEAR
from q4nx import Q4NX


def compare_entry(spec, weights, inputs, captures):
    di.validate_weights(spec, weights)
    if len(inputs) < 2 or len(inputs) != len(captures):
        raise ValueError('incomplete multi-token coverage')
    histories = {name: np.zeros((spec.conv_kernel-1, spec.lin_qkv_dim), np.float64)
                 for name in ('primary', 'norm_only', 'qkv_only', 'captured_entry')}
    rows = []
    for token, (x, capture) in enumerate(zip(inputs, captures)):
        x, xn = finite(x), finite(capture['xn'])
        if x.shape != (spec.hidden,) or xn.shape != x.shape:
            raise ValueError('invalid residual/norm shape')
        exact = x / np.sqrt(np.mean(x*x)+spec.norm_eps) * weights['norm']
        variants = dict(primary=(exact, weights['qkv'] @ exact),
                        norm_only=(xn, weights['qkv'] @ xn),
                        qkv_only=(exact, capture['qkv']),
                        captured_entry=(xn, capture['qkv']))
        records = {}
        for name, (norm, qkv) in variants.items():
            records[name], histories[name] = di.records_reference(
                spec, norm, qkv, histories[name], weights)
        observed = {name: di.record_metrics(capture['records'], ref)
                    for name, ref in records.items()}
        rows.append(dict(token=token,
                         primary_records_passed=all(m['passed'] for m in observed['primary'].values()),
                         observed_vs_variant=observed,
                         variant_vs_primary={name: di.record_metrics(ref, records['primary'])
                                             for name, ref in records.items() if name != 'primary'}))
    return rows


def check(model_dir, kernel_dir, out, layer):
    spec = spec_from_model_dir(model_dir)
    meta = load_reference(out)
    selected = load_bundle(kernel_dir, manifest(spec, max_ctx=meta['max_ctx']))
    if any(meta[k] != selected[k] for k in ('build_key', 'spec_hash')):
        raise ValueError('fixture/export identity mismatch')
    if spec.quant_of('linear') != 'q4_1':
        raise ValueError('projection diagnostic requires Q4_1 linear weights')
    if (spec.family != 'qwen35' or not 0 <= layer < meta['layers'] <= spec.num_layers
            or spec.layer_types[layer] != LINEAR or meta['hidden'] != spec.hidden):
        raise ValueError('invalid linear layer/fixture geometry')
    norm_dtype, conv_dtype = linear_dtypes(selected)
    offsets = layout(spec, max_ctx=meta['max_ctx'])
    weights, weight_hashes = di.load_weights(Q4NX(model_dir / 'model.q4nx'), spec, layer)
    hashes = {}

    def read(name):
        data = (out / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return data

    zero = read(f'zstate_{LINEAR}.bin')
    if len(zero) != offsets.STATE_BYTES or any(zero):
        raise ValueError('fixture must start from complete zero state')
    inputs, captures = [], []
    for t in range(meta['tokens']):
        data = read(f'y_res{layer-1}{sfx(t)}.bin' if layer else f'xres{t}.bin')
        if len(data) != spec.hidden*4:
            raise ValueError('wrong residual capture size')
        inputs.append(finite(np.frombuffer(data, '<f4')))
        act = read(f'y_act{layer}{sfx(t)}.bin')
        c = read_boundaries(act, read(f'y_state{layer}{sfx(t)}.bin'), spec, offsets, conv_dtype=conv_dtype)
        c['xn'] = di.read_entry(act, offsets, spec.hidden, norm_dtype=norm_dtype)
        captures.append(c)
    rows = compare_entry(spec, weights, inputs, captures)
    return dict(version=1, scope=__doc__.strip(),
                passed=all(r['primary_records_passed'] for r in rows),
                layer=layer, tokens=meta['tokens'], maxrel_bound=di.MAXREL,
                build_key=selected['build_key'], spec_hash=selected['spec_hash'],
                fixture_sha256=hashlib.sha256((out/'decode_reference.json').read_bytes()).hexdigest(),
                capture_sha256=hashes, weight_sha256=weight_hashes,
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), ROOT/'open_kernels/model/deltanet_inputs.py',
                                         ROOT/'open_kernels/model/deltanet_state.py',
                                         ROOT/'open_kernels/model/q4nx.py')}, results=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'kernel-dir', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--layer', type=int, required=True)
    args = parser.parse_args()
    try:
        result = check(args.model_dir, args.kernel_dir, args.out, args.layer)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['passed'] else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
