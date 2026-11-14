#!/usr/bin/env python3
"""Localize DeltaNet head-output errors on complete retained decode captures.

The primary FP64 recurrence carries its own state from zero. A separate exact
readout of captured updated state isolates the final S^T q boundary. Pair
three/four-term estimates use FP64 accumulation: they diagnose representation
loss, not AIE accumulator behavior, and never replace primary acceptance.
No weights or NPU execution are needed. Exit 1 retains unresolved head failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'open_kernels/model'))
import deltanet_state as ds
from decode_reference import sfx
from q4nx import bf16_to_f32, f32_to_bf16


def inputs(state, q):
    state, q = ds.finite(state), ds.finite(q)
    if q.ndim != 2 or q.shape[1] != ds.DIM or state.shape != (len(q), ds.DIM, ds.DIM):
        raise ValueError('invalid readout state/query shape')
    return state, q


def readout(state, q):
    state, q = inputs(state, q)
    return ds.finite(np.einsum('hij,hi->hj', state, q) / np.sqrt(ds.DIM))


def split_pair(a):
    a = ds.finite(a).astype(np.float32)
    ds.finite(a)
    high = bf16_to_f32(f32_to_bf16(a))
    low = bf16_to_f32(f32_to_bf16(a - high))
    return ds.finite(high), ds.finite(low)


def pair_dot(state, q, *, include_low_low=False):
    state, q = inputs(state, q)
    sh, sl = split_pair(state); qh, ql = split_pair(q)
    dot = lambda a, b: np.einsum('hij,hi->hj', a, b)
    out = dot(sh, qh) + dot(sh, ql) + dot(sl, qh)
    if include_low_low:
        out += dot(sl, ql)
    return ds.finite(out)


def pair_scale(a, *, include_low_low=False):
    ah, al = split_pair(a)
    ih, il = split_pair(np.array([np.float32(0.088388347648318)]))
    out = ah*ih + ah*il + al*ih
    if include_low_low:
        out += al*il
    return ds.finite(out)


def cancellation(state, q):
    state, q = inputs(state, q)
    summed = np.einsum('hij,hi->hj', state, q)
    absolute = np.einsum('hij,hi->hj', np.abs(state), np.abs(q))
    # A head-wise norm ratio, not the unstable ratio at an arbitrary zero lane.
    return [float(np.max(a)/np.max(np.abs(s))) if np.any(s) else None
            for a, s in zip(absolute, summed)]


def diagnose_sequence(captures):
    if len(captures) < 2:
        raise ValueError('multiple positions required')
    heads = captures[0]['records'].shape[0]
    reference = np.zeros((heads, ds.DIM, ds.DIM), np.float64)
    rows = []
    for token, cap in enumerate(captures):
        reference, primary = ds.step_reference(reference, cap['records'])
        state = ds.finite(cap['state'])
        got = ds.finite(cap['output'])
        if (state.ndim != 3 or state.shape[0] != heads or state.shape[1] < ds.DIM or
                state.shape[2] != ds.DIM or got.shape != primary.shape):
            raise ValueError('invalid capture state/output shape')
        state = state[:, :ds.DIM]
        q = cap['records'][:, ds.DIM:2*ds.DIM]
        conditioned = readout(state, q)
        estimates = {name: pair_scale(pair_dot(state, q, include_low_low=low), include_low_low=low)
                     for name, low in [('three_term', False), ('four_term', True)]}
        ratios = cancellation(state, q)
        for h in range(heads):
            rows.append(dict(token=token, head=h, primary_output=ds.metric(got[h], primary[h]),
                primary_state=ds.metric(state[h], reference[h]),
                captured_state_readout=ds.metric(got[h], conditioned[h]),
                state_effect=ds.metric(conditioned[h], primary[h]),
                cancellation_ratio=ratios[h],
                representation_estimates={name: dict(to_exact_readout=ds.metric(est[h], conditioned[h]),
                                                     to_observed=ds.metric(est[h], got[h]))
                                          for name, est in estimates.items()}))
    return rows



def summarize(rows):
    result = dict(heads_checked=len(rows))
    for field in ('primary_output', 'primary_state', 'captured_state_readout', 'state_effect'):
        result[field + '_failures'] = sum(not r[field]['passed'] for r in rows)
    result['representation_estimates'] = {
        name: {field + '_failures': sum(not r['representation_estimates'][name][field]['passed'] for r in rows)
               for field in ('to_exact_readout', 'to_observed')}
        for name in ('three_term', 'four_term')}
    return result


def check(model_dir, kernel_dir, out):
    baseline = ds.check(model_dir, kernel_dir, out)
    spec = ds.spec_from_model_dir(model_dir)
    meta = ds.load_reference(out)
    selected = ds.load_bundle(kernel_dir, ds.manifest(spec, max_ctx=meta['max_ctx']))
    offsets = ds.layout(spec, max_ctx=meta['max_ctx'])
    _, conv_dtype = ds.linear_dtypes(selected)

    def read(name):
        data = (out / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != baseline['capture_sha256'][name]:
            raise ValueError('capture changed during diagnostics: ' + name)
        return data

    results = []
    for layer in baseline['layers']:
        caps = [ds.read_boundaries(read(f'y_act{layer}{sfx(t)}.bin'), read(f'y_state{layer}{sfx(t)}.bin'),
                                  spec, offsets, conv_dtype=conv_dtype) for t in range(baseline['tokens'])]
        results.extend(dict(layer=layer, **row) for row in diagnose_sequence(caps))
    return dict(version=1, diagnostic_only=True, scope=__doc__.strip(),
                passed=baseline['passed'] and baseline['head_diagnostics_passed'],
                primary_whole_passed=baseline['passed'],
                primary_heads_passed=baseline['head_diagnostics_passed'],
                maxrel_bound=ds.MAXREL, cosine_min=ds.COSINE_MIN,
                layers=baseline['layers'], tokens=baseline['tokens'], heads=baseline['heads'],
                build_key=baseline['build_key'], spec_hash=baseline['spec_hash'],
                fixture_sha256=baseline['fixture_sha256'], capture_sha256=baseline['capture_sha256'],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                summary=summarize(results), results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'kernel-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = check(args.model_dir, args.kernel_dir, args.out)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report['passed'] else 1
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
