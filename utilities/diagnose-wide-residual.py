#!/usr/bin/env python3
"""Offline residual counterfactuals; never change independent model acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from ml_dtypes import bfloat16

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'open_kernels'))
from wide_slice_reference import norm
from recipes.spec import LINEAR
GUARD=bytes([0xa5])*64


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def capture(path,count,dtype=np.float32,*,logical=None):
    raw=Path(path).read_bytes();size=count*np.dtype(dtype).itemsize
    if len(raw)!=size+64 or raw[-64:]!=GUARD:raise ValueError(f'{path}: size/canary')
    a=np.frombuffer(raw[:size],dtype)[:logical]
    if not np.isfinite(a.astype(np.float64)).all():raise ValueError(f'{path}: nonfinite')
    return a


def addition_error(left,right,got,ref_left,ref_right):
    a,b,g,c,d=map(float,(left,right,got,ref_left,ref_right))
    rounded=float(np.float32(a+b));ref=float(np.float32(c+d))
    return dict(left_error=a-c,right_error=b-d,local_addition=g-rounded,
                rounding_delta=(rounded-(a+b))-(ref-(c+d)),output_error=g-ref,
                got=g,reference=ref)


def analyze(x,p,ref_x,ref_p,w,got,channel):
    expected=norm((ref_x+ref_p).astype(np.float32),w);variants={}
    only=ref_x.copy();only[channel]=x[channel]
    repaired=x.copy();repaired[channel]=ref_x[channel]
    for name,a,b in [('device',x,p),('reference',ref_x,ref_p),
                     ('input_only',x,ref_p),('projection_only',ref_x,p),
                     ('input_channel_only',only,ref_p),('repaired_input_channel',repaired,p)]:
        residual=(a.astype(np.float64)+b.astype(np.float64)).astype(np.float32)
        v=norm(residual,w);ids=np.flatnonzero(v!=expected)
        variants[name]=dict(vs_reference=len(ids),indices=ids[:32].tolist(),
                            vs_device=int(np.count_nonzero(v!=got)),value=float(v[channel]),
                            residual=float(residual[channel]))
    return dict(diagnostic_only=True,model_passed=False,channel=channel,
                local_norm=variants['device']['vs_device'],counterfactuals=variants)


def diagnose(out,tag,channel):
    meta=json.loads((out/'slice-fixture.json').read_text())
    case=next(c for c in meta['cases'] if c['tag']==tag)
    if not 0<=channel<5120:raise ValueError('channel out of range')
    for name,digest in meta['kernels'].items():
        if sha(name)!=digest:raise ValueError(f'kernel changed: {name}')
    def checked(name):
        if sha(out/name)!=meta['fixtures'][name]:raise ValueError(f'fixture changed: {name}')
        with np.load(out/name) as data:return {k:data[k] for k in data.files}
    ref=checked(f'{tag}-ref.npz');w=checked(f"layer{case['layer']}/params.npz")['postw']
    x=capture(out/f'{tag}-input-x.bin',5120);p=capture(out/f'{tag}-got-projout.bin',5120)
    kind='d' if case['kind']==LINEAR else 'a'
    got=capture(out/f'{tag}-got-xm.bin',meta['outputs'][kind]['xm']//2,bfloat16,logical=5120)
    r=analyze(x,p,ref['x'],ref['projout'],w,got,channel);r['tag']=tag
    if not np.array_equal(norm((ref['x']+ref['projout']).astype(np.float32),w),ref['xm'].astype(bfloat16)):
        raise ValueError('reference norm does not reconstruct')
    history=[]
    for c in meta['cases']:
        if c['token']!=case['token'] or c['layer']>case['layer']:continue
        t=c['tag'];rr=checked(f'{t}-ref.npz')
        xx=capture(out/f'{t}-input-x.bin',5120)
        for field,left,right,rl,rright in [
          ('res1',xx,capture(out/f'{t}-got-projout.bin',5120),rr['x'],rr['projout']),
          ('y',capture(out/f'{t}-got-res1.bin',5120),capture(out/f'{t}-got-fo.bin',5120),rr['res1'],rr['fo'])]:
            yy=capture(out/f'{t}-got-{field}.bin',5120)
            history.append(dict(tag=t,field=field,**addition_error(left[channel],right[channel],yy[channel],rl[channel],rright[channel])))
    r['channel_history']=history
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tag',default='cold-0-layer11');p.add_argument('--channel',type=int,default=4872);a=p.parse_args()
    r=diagnose(a.out,a.tag,a.channel)
    (a.out/f'{a.tag}-residual{a.channel}.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
