#!/usr/bin/env python3
"""Read-only local versus propagated errors at real full-model DeltaNet boundaries.

Conditional FP64 replays are diagnostics, never replacement acceptance fixtures.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('full_model', HERE / 'test-wide-full-model.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
from recipes.spec import LINEAR
from recipes.wide_deltanet_layer import state_copies
from wide_slice_reference import norm, silu, projection
from wide_deltanet_reference import ab_reference, glue_reference, step_reference


def capture(path, size, dtype, count=None):
    raw = path.read_bytes()
    if len(raw) != size + 64 or raw[-64:] != M.GUARD:
        raise ValueError(f'{path.name}: size/canary')
    a = np.frombuffer(raw[:size], dtype)[:count]
    if not np.isfinite(a.astype(np.float64)).all():
        raise ValueError(f'{path.name}: nonfinite')
    return a


def differences(got, conditional, reference):
    got = np.asarray(got).ravel()
    conditional = np.asarray(conditional).astype(got.dtype).ravel()
    reference = np.asarray(reference).astype(got.dtype).ravel()
    if got.shape != conditional.shape or got.shape != reference.shape:
        raise ValueError('boundary shapes differ')
    if not all(np.isfinite(a.astype(np.float64)).all() for a in (got, conditional, reference)):
        raise ValueError('nonfinite boundary')
    result = {}
    for key, a, b in (('device_vs_reference', got, reference),
                       ('local', got, conditional), ('propagated', conditional, reference)):
        ids = np.flatnonzero(a != b)
        result[key] = len(ids)
        result[key + '_indices'] = ids[:32].tolist()
        result[key + '_maxabs'] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    return result


def post_reference(o, z, nw):
    a = np.asarray(o).astype(np.float64).reshape(48, 128)
    gated = a / np.sqrt(np.mean(a * a, axis=1, keepdims=True) + 1e-6)
    gated *= nw.astype(np.float64)
    gated *= silu(np.asarray(z).reshape(48, 128))
    return gated.astype(np.float32).astype(bfloat16).ravel()


def diagnose(out, tag):
    meta = json.loads((out / 'slice-fixture.json').read_text())
    case = next(c for c in meta['cases'] if c['tag'] == tag)
    if case['kind'] != LINEAR:
        raise ValueError('requires a DeltaNet layer')
    i = case['layer']
    _, layout = M.S.A.geometry(meta['rows'])
    names = [f'layer{i}/pool.bin', f'layer{i}/const.bin', f'layer{i}/params.npz', f'{tag}-ref.npz']
    for name in names:
        if M.sha(out / name) != meta['fixtures'][name]:
            raise ValueError(f'fixture changed: {name}')
    for name, digest in meta['kernels'].items():
        if M.sha(name) != digest:
            raise ValueError(f'kernel changed: {name}')
    ref = dict(np.load(out / f'{tag}-ref.npz'))
    params = dict(np.load(out / f'layer{i}/params.npz'))
    pool = np.memmap(out / f'layer{i}/pool.bin', mode='r', dtype=np.uint8)
    const = np.fromfile(out / f'layer{i}/const.bin', np.uint8)
    fields = ('xn', 'qkv', 'z', 'ab', 'conv', 'vec', 'so', 'o', 'og', 'projout', 'res1', 'xm')
    got = {name: capture(out / f'{tag}-got-{name}.bin', meta['outputs']['d'][name],
                         bfloat16 if name in ('xn', 'conv', 'og', 'xm') else np.float32,
                         count=M.H if name in ('xn', 'xm') else None)
           for name in fields}
    got['xn'], got['xm'] = got['xn'][:M.H], got['xm'][:M.H]
    x = capture(out / f'{tag}-input-x.bin', M.H * 4, np.float32)
    raw = capture(out / f'{tag}-input-state.bin', layout.STATE_BYTES, np.uint8)
    conv = raw[:layout.STATE_S_OFF].view(bfloat16).reshape(3, 10240)
    state = np.stack([raw[src:src+size].view(np.float32).reshape(128, 128)
                      for _, src, size in state_copies(layout, 48, 128)])
    if not np.isfinite(conv.astype(np.float32)).all() or not np.isfinite(state).all():
        raise ValueError('nonfinite logical state')
    proj = lambda name, a: projection(pool, const, layout, LINEAR, name, a)
    conditional = dict(xn=norm(x, params['lnw']),
                       qkv=proj('qkv', got['xn']), z=proj('z', got['xn']),
                       ab=ab_reference(got['xn'], params['weights'], params['a'], params['dt']).astype(np.float32))
    conditional['conv'], conditional['vec'] = glue_reference(got['qkv'], conv, params['convw'], got['ab'].reshape(4, 48))
    conditional['so'], conditional['o'] = step_reference(state, got['vec'].reshape(48, 512))
    conditional['og'] = post_reference(got['o'], got['z'], params['nw'])
    conditional['projout'] = proj('out', got['og'])
    conditional['res1'] = (x.astype(np.float64) + got['projout'].astype(np.float64)).astype(np.float32)
    conditional['xm'] = norm(got['res1'], params['postw'])
    boundaries = {name: differences(got[name], conditional[name], ref[name]) for name in fields}
    # Single substitutions isolate how captured projection, records or post inputs
    # cross the BF16 boundary. They do not replace the full independent trajectory.
    variants = {'device_o_z': conditional['og'],
                'device_o_only': post_reference(got['o'], ref['z'], params['nw']),
                'device_z_only': post_reference(ref['o'], got['z'], params['nw']),
                'conditional_step_device_z': post_reference(conditional['o'].astype(np.float32), got['z'], params['nw'])}
    def post_from_records(records):
        _, value = step_reference(state, np.asarray(records).astype(np.float32).reshape(48, 512))
        return post_reference(value.astype(np.float32), got['z'], params['nw'])
    variants['conditional_glue_step_device_z'] = post_from_records(conditional['vec'])
    variants['reference_records_device_state_z'] = post_from_records(ref['vec'])
    for name, qkv, ab in (('device_qkv_only', got['qkv'], ref['ab']),
                          ('device_ab_only', ref['qkv'], got['ab'])):
        _, records = glue_reference(qkv, conv, params['convw'], np.asarray(ab).reshape(4, 48))
        variants[name + '_device_state_z'] = post_from_records(records)
    counterfactual = {name: differences(got['og'], value, ref['og']) for name, value in variants.items()}
    result = dict(tag=tag, diagnostic_only=True, passed=False, boundaries=boundaries,
                  post_counterfactuals=counterfactual)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tag', default='cold-0-layer10')
    args = p.parse_args()
    result = diagnose(args.out, args.tag)
    (args.out / f'{args.tag}-deltanet-boundary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
