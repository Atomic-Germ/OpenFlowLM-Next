#!/usr/bin/env python3
"""Read-only conditional/counterfactual diagnosis of a captured wide layer.

Never edits fixtures or hardware outputs. Device intermediates are used only
in offline replay to locate amplification, not as an acceptance reference.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('layer_probe',HERE/'test-wide-deltanet-layer.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)


def diagnose(out, tag):
    s,l=P.geometry()
    meta=json.loads((out/'layer-fixture.json').read_text())
    for name,digest in meta['fixtures'].items():
        if P.sha(out/name)!=digest: raise ValueError(f'fixture changed: {name}')
    for name,digest in meta['kernels'].items():
        if P.sha(name)!=digest: raise ValueError(f'kernel changed: {name}')
    ref=dict(np.load(out/f'{tag}-ref.npz'))
    got={}
    for name,size in meta['outputs'].items():
        data=(out/f'{tag}-got-{name}.bin').read_bytes()
        if len(data)!=size+64 or data[-64:]!=P.GUARD:raise ValueError(f'{name}: size/canary')
        dtype=bfloat16 if name in ('xn','conv','og','xm','discard') else np.float32
        got[name]=np.frombuffer(data[:size],dtype)
    case,t=tag.rsplit('-',1);t=int(t)
    if t:
        previous=(out/f'{case}-{t-1}-got-state.bin').read_bytes()[:-64]
    else:previous=(out/f'{case}-init.bin').read_bytes()[:-64]
    cs=np.frombuffer(previous[:l.STATE_S_OFF],bfloat16).reshape(3,P.NCH)
    state=np.stack([np.frombuffer(previous[src:src+size],np.float32).reshape(128,128)
                    for _,src,size in P.state_copies(l,48,128)])
    const=np.fromfile(out/'const.bin',np.uint8)
    convw=const[l.C_SIDE+l.SIDE_CONV:l.C_SIDE+l.SIDE_CONV+4*P.NCH*2].view(bfloat16).reshape(10,4,1024).transpose(1,0,2).reshape(4,P.NCH)
    nw=const[l.C_NW:l.C_NW+256].view(bfloat16)
    postw=const[l.C_POSTLN:l.C_POSTLN+P.H*2].view(bfloat16)
    pool=np.memmap(out/'pool.bin',mode='r',dtype=np.uint8)
    x=np.fromfile(out/f'{tag}-x.bin',np.float32)
    checks=[]
    def measure(name,g,r):
        m=P.metric(np.asarray(g).ravel(),np.asarray(r).ravel(),.005)
        if np.asarray(g).dtype==np.dtype(bfloat16):
            m['bf16_mismatches']=int(np.count_nonzero(g.ravel()!=np.asarray(r).astype(bfloat16).ravel()))
        checks.append(dict(name=name,**m));print(name,m,flush=True)
    c,v=P.glue_reference(got['qkv'],cs,convw,got['ab'].reshape(4,48))
    measure('conditional_glue_records',got['vec'],v)
    measure('conditional_conv_bits',got['conv'],c)
    st,o=P.step_reference(state,got['vec'].reshape(48,512))
    measure('conditional_step_state',got['so'],st)
    measure('conditional_step_output',got['o'],o)
    def post(o,z):
        o=o.astype(np.float64).reshape(48,128)
        return (o/np.sqrt(np.mean(o*o,axis=1,keepdims=True)+1e-6)*nw.astype(np.float64)*P.silu(z.reshape(48,128))).astype(np.float32).astype(bfloat16).ravel()
    measure('conditional_post',got['og'],post(got['o'],got['z']))
    def projection(name,x):
        storage,offset,n,k={'out':(const,l.C_WOUT,P.H,P.VW),
            'up':(pool,l.POOL_FFN_UP,P.F,P.H),'gate':(pool,l.POOL_FFN_GATE,P.F,P.H),
            'down':(pool,l.POOL_FFN_DOWN,P.H,P.F)}[name]
        return P.pool_reference(storage[offset:offset+n*k//8192*5120],x.astype(bfloat16),n,k,rs=2)
    def ffn(xm):
        u,g=projection('up',xm),projection('gate',xm)
        h=(u.astype(np.float64)*P.silu(g)).astype(np.float32)
        return projection('down',h)
    def suffix(og):
        res=(projection('out',og)+x).astype(np.float32)
        return (ffn(P.norm(res,postw))+res).astype(np.float32)
    ideal_previous=dict(np.load(out/f'{case}-{t-1}-ref.npz')) if t else dict(conv=cs,so=state)
    ideals=ideal_previous['so']; idealcs=ideal_previous['conv'].astype(bfloat16)
    def from_vec(v, st): return P.step_reference(st,v.astype(np.float32))[1].astype(np.float32)
    variants=[('ideal_og',ref['og']),('device_o_z',post(got['o'],got['z'])),('device_og',got['og']),
              ('device_z_only',post(ref['o'],got['z'])),
              ('device_previous_state_only',post(from_vec(ref['vec'],state),ref['z'])),
              ('device_records_only',post(from_vec(got['vec'].reshape(48,512),ideals),ref['z']))]
    for name, qkv, conv, ab in (
        ('device_qkv_only',got['qkv'],idealcs,ref['ab']),
        ('device_conv_only',ref['qkv'],cs,ref['ab']),
        ('device_ab_only',ref['qkv'],idealcs,got['ab'].reshape(4,48))):
        _,vec=P.glue_reference(qkv,conv,convw,ab)
        variants.append((name,post(from_vec(vec,ideals),ref['z'])))
    for name,og in variants:
        measure('replay_'+name,suffix(og),ref['y'])
    measure('replay_device_residual',ffn(P.norm(got['res1'],postw))+got['res1'],ref['y'])
    measure('replay_device_xm',ffn(got['xm'][:P.H])+got['res1'],ref['y'])
    measure('device_final',got['y'],ref['y'])
    return dict(tag=tag,diagnostic_only=True,checks=checks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=P.DESIGN/'wide_deltanet/build_layer/acceptance')
    p.add_argument('--tag',default='warm-1')
    a=p.parse_args()
    result=diagnose(a.out,a.tag)
    (a.out/f'{a.tag}-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
