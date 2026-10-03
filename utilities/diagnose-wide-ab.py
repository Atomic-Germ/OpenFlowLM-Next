#!/usr/bin/env python3
"""Replay captured H5120/48-head AB with guarded repeats; never model acceptance."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('full_model', HERE/'test-wide-full-model.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
from recipes.wide_deltanet import WideDeltaNet
from recipes.wide_deltanet_layer import ab_copies
from wide_deltanet_reference import ab_reference
GUARD = M.GUARD
sha = M.sha


def nonlinear(dots, a, dt):
    dots = np.asarray(dots).astype(np.float64)
    return np.stack((np.exp(a.astype(np.float64)*np.logaddexp(0, dots[0]+dt.astype(np.float64))),
                     1/(1+np.exp(-dots[1]))))


def capture(path, count, dtype=np.float32):
    raw = path.read_bytes()
    if len(raw) != count*np.dtype(dtype).itemsize+64 or raw[-64:] != GUARD:
        raise ValueError(f'{path.name}: size/canary')
    return np.frombuffer(raw[:-64], dtype)


def prepare(source, kernel, out, tag, head):
    if not 0 <= head < 48: raise ValueError('head outside 48-head geometry')
    meta = json.loads((source/'slice-fixture.json').read_text())
    case = next(c for c in meta['cases'] if c['tag'] == tag)
    if case['kind'] != M.LINEAR: raise ValueError('requires DeltaNet layer')
    i = case['layer']; _, layout = M.S.A.geometry(meta['rows'])
    names = [f'layer{i}/const.bin', f'layer{i}/params.npz']
    for name in names:
        if sha(source/name) != meta['fixtures'][name]: raise ValueError(f'fixture changed: {name}')
    for name, digest in meta['kernels'].items():
        if sha(name) != digest: raise ValueError(f'kernel changed: {name}')
    fixture = json.loads((kernel/'ab-fixture.json').read_text())
    g = WideDeltaNet(5120,16,48,128,128)
    if fixture['geometry'] != vars(g): raise ValueError('AB kernel geometry')
    for name in ('final.xclbin', 'insts.bin'):
        if sha(kernel/name) != fixture['sha256'][name]: raise ValueError(f'AB kernel changed: {name}')
    if not json.loads((kernel/'ab-results.json').read_text())['passed']:
        raise ValueError('AB primitive gate failed')
    raw = capture(source/f'{tag}-got-xn.bin', meta['outputs']['d']['xn']//2, bfloat16)
    xn = raw[:5120]
    recorded = capture(source/f'{tag}-got-ab.bin',192)
    if not np.isfinite(xn.astype(np.float32)).all() or not np.isfinite(recorded).all():
        raise ValueError('nonfinite captured input')
    const = (source/f'layer{i}/const.bin').read_bytes()
    side = bytearray(g.side_bytes)
    for dst, src, size in ab_copies(layout,5120,48): side[dst:dst+size] = const[src:src+size]
    params = dict(np.load(source/f'layer{i}/params.npz'))
    ref = ab_reference(xn,params['weights'],params['a'],params['dt'])
    out.mkdir(parents=True,exist_ok=False)
    for name in ('final.xclbin','insts.bin'): (out/name).write_bytes((kernel/name).read_bytes())
    (out/'side.bin').write_bytes(side);raw.tofile(out/'xn.bin');recorded.tofile(out/'recorded.bin')
    np.savez(out/'reference.npz',ab=ref,a=params['a'],dt=params['dt'])
    (out/'poison.bin').write_bytes(np.full(192,np.nan,np.float32).tobytes()+GUARD)
    cfg=['device','xclbin ab final.xclbin','kernelx ab ab insts.bin',f'buf side {g.side_bytes} side.bin',
         f'buf xn {g.xn_chunks*4096} xn.bin','buf result 832']
    for i in range(2): cfg+=['load result poison.bin','run ab side xn result',f'dump result got{i}.bin 832']
    (out/'ab-replay.cfg').write_text('\n'.join(cfg)+'\n')
    files=['final.xclbin','insts.bin','side.bin','xn.bin','recorded.bin','reference.npz','poison.bin','ab-replay.cfg']
    result=dict(tag=tag,head=head,source=str(source.resolve()),kernel=str(kernel.resolve()),
                sha256={name:sha(out/name) for name in files})
    (out/'ab-replay-fixture.json').write_text(json.dumps(result,indent=2)+'\n')


def compare(out, require_exact_dots=False):
    meta=json.loads((out/'ab-replay-fixture.json').read_text())
    for name,digest in meta['sha256'].items():
        if sha(out/name)!=digest: raise ValueError(f'fixture changed: {name}')
    got=capture(out/'got0.bin',192).reshape(4,48)
    again=capture(out/'got1.bin',192).reshape(4,48)
    if not np.isfinite(got).all() or not np.isfinite(again).all(): raise ValueError('nonfinite output')
    ref=dict(np.load(out/'reference.npz'));expected=ref['ab'].astype(np.float32)
    conditional=nonlinear(got[:2],ref['a'],ref['dt']).astype(np.float32)
    fields=[]
    for j,name in enumerate(('alpha','beta_logits','decay','beta')):
        ids=np.flatnonzero(got[j]!=expected[j])
        fields.append(dict(name=name,mismatches=len(ids),indices=ids.tolist(),
                           maxabs=float(np.max(np.abs(got[j].astype(np.float64)-expected[j])))))
    repeat=(out/'got0.bin').read_bytes()==(out/'got1.bin').read_bytes()
    exact=not any(r['mismatches'] for r in fields[:2]);h=meta['head']
    result=dict(diagnostic_only=True,model_passed=False,tag=meta['tag'],repeat_exact=repeat,
                exact_dots=exact,fields=fields,
                matches_recorded=bool(np.array_equal(got.ravel(),np.fromfile(out/'recorded.bin',np.float32))),
                local_nonlinear_mismatches=np.count_nonzero(got[2:]!=conditional,axis=1).tolist(),
                selected_head=dict(head=h,got=got[:,h].tolist(),reference=expected[:,h].tolist(),
                                   conditional_nonlinear=conditional[:,h].tolist()))
    (out/'ab-replay-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if repeat and (exact or not require_exact_dots) else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','compare'))
    p.add_argument('--source',type=Path);p.add_argument('--kernel',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tag',default='cold-0-layer10');p.add_argument('--head',type=int,default=4)
    p.add_argument('--require-exact-dots',action='store_true');a=p.parse_args()
    if a.stage=='prepare':
        if a.source is None or a.kernel is None:p.error('prepare requires source and kernel')
        prepare(a.source,a.kernel,a.out,a.tag,a.head)
    else:raise SystemExit(compare(a.out,a.require_exact_dots))
