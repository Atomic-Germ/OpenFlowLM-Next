#!/usr/bin/env python3
"""Guarded real RMSNorm replay; conditional diagnostics never replace model gates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

GUARD=bytes([0xA5])*64
N=5120


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def reference(x,w):
    x=x.astype(np.float64)
    return (x/np.sqrt(np.mean(x*x)+1e-6)*w.astype(np.float64)).astype(np.float32).astype(bfloat16)


def capture(path,dtype,n):
    raw=path.read_bytes();size=n*np.dtype(dtype).itemsize
    if len(raw)!=size+64 or raw[-64:]!=GUARD:raise ValueError(f'{path}: size/canary')
    data=np.frombuffer(raw[:size],dtype)
    if not np.isfinite(data.astype(np.float32)).all():raise ValueError(f'{path}: nonfinite')
    return data


def write_fixture(out,x,w,model,artifacts,trace=False,index=786):
    if x.shape!=(N,) or w.shape!=(N,) or not 0<=index<N:raise ValueError('geometry/index')
    if not np.isfinite(x).all() or not np.isfinite(w.astype(np.float32)).all():raise ValueError('nonfinite inputs')
    for name,data in artifacts.items():(out/name).write_bytes(data)
    x.astype(np.float32).tofile(out/'x.bin');np.zeros(N,np.float32).tofile(out/'add.bin')
    w.astype(bfloat16).tofile(out/'w.bin');reference(x,w).tofile(out/'reference.bin')
    model.astype(bfloat16).tofile(out/'model.bin')
    (out/'poison-y.bin').write_bytes(np.full(N,np.nan,np.float32).tobytes()+GUARD)
    (out/'poison-xn.bin').write_bytes(np.full(N,np.nan,bfloat16).tobytes()+GUARD)
    cfg=['device','xclbin ln final.xclbin','kernelx ln ln insts.bin',
         f'buf x {N*4} x.bin',f'buf add {N*4} add.bin',f'buf w {N*2} w.bin',
         f'buf y {N*4+64}',f'buf xn {N*2+64}']
    for i in range(2):cfg+=['load y poison-y.bin','load xn poison-xn.bin','run ln x add w y xn',
                            f'dump y y{i}.bin {N*4+64}',f'dump xn xn{i}.bin {N*2+64}']
    (out/'replay.cfg').write_text('\n'.join(cfg)+'\n')
    names=[*artifacts,'x.bin','add.bin','w.bin','reference.bin','model.bin','poison-y.bin','poison-xn.bin','replay.cfg']
    (out/'replay-fixture.json').write_text(json.dumps(dict(diagnostic_only=True,trace=trace,index=index,
        sha256={n:sha(out/n) for n in names}),indent=2)+'\n')


def prepare(source,kernel,out,tag='cold-0-layer4',field='xn',trace=False,index=786):
    meta=json.loads((source/'slice-fixture.json').read_text())
    case=next(c for c in meta['cases'] if c['tag']==tag)
    for name,digest in meta['kernels'].items():
        if sha(name)!=digest:raise ValueError(f'source kernel changed: {name}')
    def checked(name):
        if sha(source/name)!=meta['fixtures'][name]:raise ValueError(f'model fixture changed: {name}')
        return source/name
    weights=np.load(checked(f"layer{case['layer']}/params.npz"))['lnw' if field=='xn' else 'postw']
    model=np.load(checked(f'{tag}-ref.npz'))[field].astype(bfloat16)
    x=capture(source/(f'{tag}-input-x.bin' if field=='xn' else f'{tag}-got-res1.bin'),np.float32,N)
    if not trace:
        fixture=json.loads((kernel/'ln-fixture.json').read_text())
        if fixture['n']!=N or fixture['eps']!=1e-6:raise ValueError('LN geometry/epsilon')
        if not json.loads((kernel/'ln-results.json').read_text())['passed']:raise ValueError('LN primitive failed')
        for name,digest in fixture['sha256'].items():
            if sha(kernel/name)!=digest:raise ValueError(f'LN fixture changed: {name}')
    out.mkdir(parents=True,exist_ok=False)
    write_fixture(out,x,weights,model,{n:(kernel/n).read_bytes() for n in ('final.xclbin','insts.bin')},trace,index)
    (out/'source.json').write_text(json.dumps(dict(source=str(source.resolve()),kernel=str(kernel.resolve()),tag=tag,field=field),indent=2)+'\n')
    print(out/'replay.cfg',flush=True)


def compare(out):
    f=json.loads((out/'replay-fixture.json').read_text())
    for name,digest in f['sha256'].items():
        if sha(out/name)!=digest:raise ValueError(f'replay fixture changed: {name}')
    x=np.fromfile(out/'x.bin',np.float32);w=np.fromfile(out/'w.bin',bfloat16).astype(np.float64)
    y=capture(out/'y0.bin',np.float32,N)
    repeat=all((out/f'{n}0.bin').read_bytes()==(out/f'{n}1.bin').read_bytes() for n in ('y','xn'))
    result=dict(diagnostic_only=True,repeat_exact=repeat,residual_exact=bool(np.array_equal(y.view(np.uint32),x.view(np.uint32))))
    if f['trace']:
        v=capture(out/'xn0.bin',np.float32,N//2)
        ss=float(np.sum(x.astype(np.float64)**2));mean=ss/N+1e-6;inv=1/np.sqrt(mean);i=f['index']
        result['trace']=dict(lanes=v[:32].tolist(),correction=v[32:64].tolist(),
            sum=float(v[64]),mean=float(v[65]),inv=float(v[66]),weighted=float(v[67]),output=float(v[68]),
            reference_sum=ss,reference_mean=mean,reference_inv=inv,
            reference_weighted=float(x[i]*w[i]),reference_output=float(x[i]*w[i]*inv))
        # Trace is diagnostic only and cannot establish norm acceptance.
        result['passed']=False
    else:
        got=capture(out/'xn0.bin',bfloat16,N).view(np.uint16)
        ref=np.fromfile(out/'reference.bin',np.uint16);model=np.fromfile(out/'model.bin',np.uint16)
        ids=np.flatnonzero(got!=ref)
        result.update(local=len(ids),propagated=int(np.count_nonzero(ref!=model)),
                      device_vs_model=int(np.count_nonzero(got!=model)),first_indices=ids[:32].tolist())
        result['passed']=bool(not len(ids) and repeat and result['residual_exact'])
    (out/'replay-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','compare'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--kernel',type=Path)
    p.add_argument('--tag',default='cold-0-layer4');p.add_argument('--field',choices=('xn','xm'),default='xn')
    p.add_argument('--trace',action='store_true');p.add_argument('--index',type=int,default=786)
    a=p.parse_args()
    if a.stage=='prepare':
        if a.source is None or a.kernel is None:p.error('prepare requires --source and --kernel')
        prepare(a.source,a.kernel,a.out,a.tag,a.field,a.trace,a.index)
    else:raise SystemExit(compare(a.out))
