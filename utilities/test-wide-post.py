#!/usr/bin/env python3
"""Guarded standalone post acceptance and conditional replay of real O/Z captures."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from ml_dtypes import bfloat16

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from wide_attention_reference import metric
GUARD=bytes([0xa5])*64
STAGES=('sum_sq','inv','normalized','weighted','silu','result')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def reference(o,z,w):
    o=o.astype(np.float64);z=z.astype(np.float64)
    return (o/np.sqrt(np.mean(o*o,axis=1,keepdims=True)+1e-6)*w.astype(np.float64)*
            z*np.exp(-np.logaddexp(0,-z))).astype(np.float32).astype(bfloat16)


def capture(path,count,dtype):
    raw=path.read_bytes()
    if len(raw)!=count*np.dtype(dtype).itemsize+64 or raw[-64:]!=GUARD:raise ValueError(f'{path.name}: size/canary')
    a=np.frombuffer(raw[:-64],dtype)
    if not np.isfinite(a.astype(np.float64)).all():raise ValueError(f'{path.name}: nonfinite')
    return a


def real_case(source,tag):
    spec=importlib.util.spec_from_file_location('post_full_model',HERE/'test-wide-full-model.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    meta=json.loads((source/'slice-fixture.json').read_text());case=next(c for c in meta['cases'] if c['tag']==tag)
    if case['kind']!=m.LINEAR:raise ValueError('requires DeltaNet layer')
    name=f"layer{case['layer']}/params.npz"
    if sha(source/name)!=meta['fixtures'][name]:raise ValueError(f'fixture changed: {name}')
    for name_,digest in meta['kernels'].items():
        if sha(name_)!=digest:raise ValueError(f'kernel changed: {name_}')
    w=np.load(source/name)['nw'].astype(bfloat16)
    o=capture(source/f'{tag}-got-o.bin',6144,np.float32).reshape(48,128)
    z=capture(source/f'{tag}-got-z.bin',6144,np.float32).reshape(48,128)
    return o,z,w


def write_fixture(out,kernel,cases,*,trace=False,diagnostic=False,source=None,tag=None):
    heads=cases[0][0].shape[0]
    if heads not in (32,48):raise ValueError('post requires 32 or 48 heads')
    out.mkdir(parents=True,exist_ok=False);n=heads*128
    for name in ('final.xclbin','insts.bin'):(out/name).write_bytes((kernel/name).read_bytes())
    cfg=['device','xclbin p final.xclbin','kernelx p p insts.bin',f'buf o {n*4}',f'buf z {n*4}',
         'buf w 4096',f'buf y {n*2+64}']
    if trace:cfg+=[f'buf t {n*6*4+64}']
    files=['final.xclbin','insts.bin','post.cfg','poison.bin']
    (out/'poison.bin').write_bytes(np.full(n,np.nan,bfloat16).tobytes()+GUARD)
    if trace:
        (out/'trace-poison.bin').write_bytes(np.full(n*6,np.nan,np.float32).tobytes()+GUARD);files+=['trace-poison.bin']
    for c,(o,z,w) in enumerate(cases):
        if o.shape!=(heads,128) or z.shape!=o.shape or w.shape!=(128,):raise ValueError('post shapes')
        if not all(np.isfinite(v.astype(np.float32)).all() for v in (o,z,w)):raise ValueError('nonfinite input')
        o.astype(np.float32).tofile(out/f'o{c}.bin');z.astype(np.float32).tofile(out/f'z{c}.bin')
        (out/f'w{c}.bin').write_bytes(w.astype(bfloat16).tobytes()+bytes(4096-256))
        reference(o,z,w).tofile(out/f'ref{c}.bin')
        files += [f'{prefix}{c}.bin' for prefix in ('o','z','w','ref')]
        for repeat in range(2):
            cfg += [f'load o o{c}.bin',f'load z z{c}.bin',f'load w w{c}.bin','load y poison.bin']
            if trace:cfg+=['load t trace-poison.bin']
            cfg += ['run p o z w y'+(' t' if trace else ''),f'dump y got{c}-{repeat}.bin {n*2+64}']
            if trace:cfg += [f'dump t trace{c}-{repeat}.bin {n*6*4+64}']
    (out/'post.cfg').write_text('\n'.join(cfg)+'\n')
    meta=dict(heads=heads,trace=trace,cases=len(cases),diagnostic_only=diagnostic,source=source,tag=tag,
              sha256={name:sha(out/name) for name in files})
    (out/'post-fixture.json').write_text(json.dumps(meta,indent=2)+'\n')


def compare(out,require_exact=False):
    meta=json.loads((out/'post-fixture.json').read_text());n=meta['heads']*128;checks=[];traces=[]
    for name,digest in meta['sha256'].items():
        if sha(out/name)!=digest:raise ValueError(f'fixture changed: {name}')
    for c in range(meta['cases']):
        ref=np.fromfile(out/f'ref{c}.bin',bfloat16)
        got=capture(out/f'got{c}-0.bin',n,bfloat16);again=capture(out/f'got{c}-1.bin',n,bfloat16)
        r=metric(got,ref,8e-3);ids=np.flatnonzero(got!=ref)
        r['passed']=bool(r['passed'] and r['cosine']>.999999 and len(ids)<n//20 and (not require_exact or len(ids)==0))
        checks.append(dict(case=c,field='output',bf16_mismatches=len(ids),indices=ids[:32].tolist(),**r))
        checks.append(dict(case=c,field='repeat',passed=(out/f'got{c}-0.bin').read_bytes()==(out/f'got{c}-1.bin').read_bytes()))
        if meta['trace']:
            a=capture(out/f'trace{c}-0.bin',n*6,np.float32).reshape(-1,6,1024)
            capture(out/f'trace{c}-1.bin',n*6,np.float32)
            checks.append(dict(case=c,field='trace_repeat',passed=(out/f'trace{c}-0.bin').read_bytes()==(out/f'trace{c}-1.bin').read_bytes()))
            checks.append(dict(case=c,field='trace_output',passed=bool(np.array_equal(a[:,5].ravel().astype(bfloat16),got))))
            if n>604:traces.append(dict(case=c,index=604,stages={name:float(a[0,j,604]) for j,name in enumerate(STAGES)}))
    result=dict(passed=all(v['passed'] for v in checks),diagnostic_only=meta['diagnostic_only'],
                checks=checks,selected_trace=traces)
    (out/'post-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','compare'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--kernel',type=Path);p.add_argument('--source',type=Path)
    p.add_argument('--tag',default='cold-0-layer10');p.add_argument('--heads',type=int,default=48,choices=(32,48))
    p.add_argument('--trace',action='store_true');p.add_argument('--require-exact',action='store_true');a=p.parse_args()
    if a.stage=='compare':raise SystemExit(compare(a.out,a.require_exact))
    if a.kernel is None:p.error('prepare requires --kernel')
    if a.source:cases=[real_case(a.source,a.tag)]
    else:
        rng=np.random.default_rng(38428);w=rng.uniform(.8,1.2,128).astype(bfloat16);cases=[]
        # Generate at full width then slice, so 32/48-head fixtures share a prefix.
        for c in range(3):
            o=rng.normal(0,.6,(48,128)).astype(np.float32);z=rng.normal(0,.6,(48,128)).astype(np.float32)
            if c==1:o.fill(0)
            if c==2:z[:,::2]=20;z[:,1::2]=-20
            cases.append((o[:a.heads],z[:a.heads],w))
    write_fixture(a.out,a.kernel,cases,trace=a.trace,diagnostic=a.source is not None,
                  source=str(a.source.resolve()) if a.source else None,tag=a.tag if a.source else None)
