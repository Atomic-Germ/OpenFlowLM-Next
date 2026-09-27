#!/usr/bin/env python3
"""Offline boundary diagnosis of a captured full-model attention layer.

Device inputs are used only for conditional diagnostics. This never changes
acceptance references, hardware captures, or the independent model trajectory.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('full_model', HERE/'test-wide-full-model.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
from wide_slice_reference import norm, silu, projection, evaluate
from wide_attention_reference import decode, metric
from recipes.spec import FULL


def diagnose(out, tag):
    meta = json.loads((out/'slice-fixture.json').read_text())
    case = next(c for c in meta['cases'] if c['tag'] == tag)
    if case['kind'] != FULL:
        raise ValueError('not implemented: this diagnostic requires a full attention layer')
    i, pos = case['layer'], case['pos']
    _, layout = M.S.A.geometry(meta['rows'])
    directory = out/f'layer{i}'
    for name in (f'layer{i}/pool.bin', f'layer{i}/const.bin', f'layer{i}/params.npz',
                 f'{tag}-ref.npz', f"{case['token']}-position.bin"):
        if M.sha(out/name) != meta['fixtures'][name]:
            raise ValueError(f'fixture changed: {name}')
    for name, digest in meta['kernels'].items():
        if M.sha(name) != digest:
            raise ValueError(f'kernel changed: {name}')

    def read(name, size, dtype):
        raw = (out/name).read_bytes()
        if len(raw) != size+64 or raw[-64:] != M.GUARD:
            raise ValueError(f'{name}: size/canary')
        return np.frombuffer(raw[:size], dtype)

    got = {name: read(f'{tag}-got-{name}.bin', size,
                     bfloat16 if name in M.S.A.BF16_OUTPUTS else np.float32)
           for name, size in meta['outputs']['a'].items()}
    x = read(f'{tag}-input-x.bin', M.H*4, np.float32)
    state = read(f'{tag}-input-state.bin', layout.KV_BYTES, bfloat16).reshape(meta['rows'],2,4,256)
    position = np.fromfile(out/f"{case['token']}-position.bin", np.uint8)
    cs = position[512:768].view(np.float32)
    pool = np.memmap(directory/'pool.bin', mode='r', dtype=np.uint8)
    const = np.fromfile(directory/'const.bin', np.uint8)
    params = dict(np.load(directory/'params.npz'))
    ref = dict(np.load(out/f'{tag}-ref.npz'))
    checks = []

    def measure(name, g, r):
        g, r = np.asarray(g).ravel(), np.asarray(r).ravel()
        result = metric(g, r, .005)
        if g.dtype == np.dtype(bfloat16):
            result['bf16_mismatches'] = int(np.count_nonzero(g != r.astype(bfloat16)))
        checks.append(dict(name=name, **result))
        print(name, result, flush=True)

    proj = lambda name, a: projection(pool, const, layout, FULL, name, a)
    measure('conditional_input_norm', got['xn'][:M.H], norm(x,params['lnw']))
    for name, value in (('q',got['qg'][:6144]), ('agate',got['qg'][6144:]),
                        ('k',got['kvn'][:1024]), ('v',got['kvn'][1024:])):
        measure('conditional_'+name, value, proj(name,got['xn'][:M.H]))
    new, og = decode(got['qg'].reshape(2,24,256), got['kvn'].reshape(2,4,256),
                     params['norms'], cs, state, pos)
    og = og.astype(np.float32).astype(bfloat16)
    measure('conditional_attention', got['og'], og)
    measure('conditional_attention_head11', got['og'].reshape(24,256)[11], og[11])
    measure('conditional_cache_row', got['new'], new)
    measure('conditional_out', got['projout'], proj('out',got['og']))
    measure('conditional_post_norm', got['xm'][:M.H], norm(got['res1'],params['postw']))

    def ffn(xm):
        up, gate = proj('up',xm), proj('gate',xm)
        h = (up.astype(np.float64)*silu(gate)).astype(np.float32)
        return proj('down',h)

    measure('conditional_ffn', got['fo'], ffn(got['xm'][:M.H]))
    measure('replay_device_xm_vs_reference', ffn(got['xm'][:M.H])+got['res1'], ref['y'])
    local, _ = evaluate(x,pool,const,layout,FULL,params,state.copy(),cs,pos)
    measure('layer_from_device_inputs_vs_device', got['y'], local['y'])
    measure('layer_from_device_inputs_vs_reference', local['y'], ref['y'])
    measure('device_vs_reference', got['y'], ref['y'])
    return dict(tag=tag, diagnostic_only=True, checks=checks)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=M.S.DESIGN/'wide_deltanet/build_full_model')
    p.add_argument('--tag', default='cold-1-layer63')
    args = p.parse_args()
    result = diagnose(args.out,args.tag)
    (args.out/f'{args.tag}-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
