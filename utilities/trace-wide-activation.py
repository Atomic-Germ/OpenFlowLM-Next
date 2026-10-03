#!/usr/bin/env python3
"""Trace production SiLU stages on a guarded real FFN band; never model acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

GUARD=bytes([0xA5])*64
STAGES=('exp','denominator','inv0','inv1','inv2','silu','h','up_gate','reassociated_h','production_h')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def reference(g,u):
    g=g.astype(np.float64);u=u.astype(np.float64)
    return (u*g/(1+np.exp(-g))).astype(np.float32)


def capture(path,count):
    raw=path.read_bytes()
    if len(raw)!=count*4+64 or raw[-64:]!=GUARD:raise ValueError('size/canary')
    v=np.frombuffer(raw[:-64],np.float32)
    if not np.isfinite(v).all():raise ValueError('nonfinite trace')
    return v


def prepare(source,kernel,out,index):
    meta=json.loads((source/'trace-fixture.json').read_text())
    for name,digest in meta['sha256'].items():
        if sha(source/name)!=digest:raise ValueError(f'changed source {name}')
    width=17408
    if not 0<=index<width:raise ValueError('index')
    raw=capture(source/'trace0.bin',width*2).reshape(-1,2,64)
    u,g=raw[:,0].ravel(),raw[:,1].ravel()
    band=index//32*32
    u,g=u[band:band+32],g[band:band+32]
    # Only logical h is finite in the poisoned activation arena.
    data=(source/'got0.bin').read_bytes()
    if len(data)!=meta['act_bytes']+64 or data[-64:]!=GUARD:raise ValueError('activation canary')
    h=np.frombuffer(data,np.float32,count=width,offset=meta['h_offset'])[band:band+32]
    if not np.isfinite(h).all():raise ValueError('nonfinite recorded activation')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('final.xclbin','insts.bin'):(out/name).write_bytes((kernel/name).read_bytes())
    np.r_[g,u].astype(np.float32).tofile(out/'input.bin');h.tofile(out/'recorded.bin')
    reference(g,u).tofile(out/'reference.bin')
    (out/'poison.bin').write_bytes(np.full(320,np.nan,np.float32).tobytes()+GUARD)
    cfg=['device','xclbin p final.xclbin','kernelx p p insts.bin','buf x 256 input.bin','buf y 1344']
    for i in range(2):cfg+=['load y poison.bin','run p x y',f'dump y got{i}.bin 1344']
    (out/'trace.cfg').write_text('\n'.join(cfg)+'\n')
    files=['final.xclbin','insts.bin','input.bin','recorded.bin','reference.bin','poison.bin','trace.cfg']
    (out/'fixture.json').write_text(json.dumps(dict(index=index,lane=index%32,source=str(source.resolve()),kernel=str(kernel.resolve()),
        sha256={name:sha(out/name) for name in files}),indent=2)+'\n')
    print(out/'trace.cfg')


def compare(out):
    meta=json.loads((out/'fixture.json').read_text())
    for name,digest in meta['sha256'].items():
        if sha(out/name)!=digest:raise ValueError(f'changed fixture {name}')
    y=capture(out/'got0.bin',320).reshape(10,32);capture(out/'got1.bin',320)
    g,u=np.fromfile(out/'input.bin',np.float32).reshape(2,32);i=meta['lane']
    ref=np.fromfile(out/'reference.bin',np.float32);e=np.exp(-g.astype(np.float64));r=1/(1+e)
    report=dict(diagnostic_only=True,passed=False,index=meta['index'],gate=float(g[i]),up=float(u[i]),
                stages={name:float(y[n,i]) for n,name in enumerate(STAGES)},
                reference=dict(exp=float(e[i]),inv=float(r[i]),h=float(ref[i])),
                repeat_exact=(out/'got0.bin').read_bytes()==(out/'got1.bin').read_bytes(),
                production_matches_recorded=bool(np.array_equal(y[9],np.fromfile(out/'recorded.bin',np.float32))),
                bf16_differences=int(np.count_nonzero(y[9].astype(bfloat16)!=ref.astype(bfloat16))))
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','compare'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--kernel',type=Path)
    p.add_argument('--index',type=int,default=749);a=p.parse_args()
    if a.stage=='prepare':
        if a.source is None or a.kernel is None:p.error('prepare requires source and kernel')
        prepare(a.source,a.kernel,a.out,a.index)
    else:raise SystemExit(compare(a.out))
