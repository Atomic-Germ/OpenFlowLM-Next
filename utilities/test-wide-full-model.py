#!/usr/bin/env python3
"""Pack and validate the real 64-layer Qwen38 model with standalone open kernels.

The independent reference follows its own logits; the device follows its own
logits. No reference tensors or token IDs enter the generated harness program.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'open_kernels'),str(ROOT/'open_kernels/model'),str(ROOT/'utilities')]
from q4nx import Q4NX
from recipes import pack,qwen35
from recipes.spec import ModelSpec,LINEAR
from recipes.wide_model import read_header,validate_header
from recipes.wide_slice import layer_types
from recipes.wide_decode import embedding_commands,head_commands
from wide_slice_reference import evaluate,norm,H

spec=importlib.util.spec_from_file_location('model_decode',ROOT/'utilities/test-wide-decode.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
S,GUARD,sha,V=D.S,D.GUARD,D.sha,D.V


def parameters(q,i,kind):
    prefix=f'model.layers.{i}.'
    def bf(name): return np.frombuffer(q.raw(prefix+name),bfloat16).copy()
    p=dict(lnw=bf('input_layernorm.weight'),postw=bf('post_attention_layernorm.weight'))
    if kind==LINEAR:
        p.update(nw=bf('linear_attn.ssm_norm.weight'),
                 weights=np.stack([bf(f'linear_attn.ssm_{name}_proj.bf16.weight').reshape(48,H) for name in ('alpha','beta')]),
                 a=q.f32(prefix+'linear_attn.ssm_a'),dt=q.f32(prefix+'linear_attn.ssm_dt.bias'),
                 convw=bf('linear_attn.ssm_conv1d.weight').reshape(4,10240))
    else: p['norms']=np.stack([bf(f'self_attn.{name}_norm.weight') for name in ('q','k')])
    return p


def extract(q,name,path):
    start,end=q.tensors[name]['data_offsets']
    q.f.seek(q.data_base+start)
    with path.open('wb') as out:
        remaining=end-start
        while remaining:
            data=q.f.read(min(4*1024**2,remaining))
            if not data: raise ValueError('truncated Q4NX tensor')
            out.write(data);remaining-=len(data)


def prepare(out,model_dir,tokens):
    if not 3<=tokens<=8: raise ValueError('full-model gate requires 3..8 tokens')
    model_dir=model_dir.resolve()
    s=ModelSpec.from_hf_config(json.loads((model_dir/'config.json').read_text()))
    header,size=read_header(model_dir/'model.q4nx')
    preflight=validate_header(s,header,size)
    kernels=S.artifacts()
    lm_dir=S.DESIGN/'lm_head_q8/build_wide_5120'
    lm=json.loads((lm_dir/'lm-fixture.json').read_text())
    if not json.loads((lm_dir/'lm-results.json').read_text())['passed'] or (lm['k'],lm['n'],lm['cores'])!=(H,V,8):
        raise ValueError('accepted full-vocabulary LM head required')
    for name in ('final.xclbin','insts.bin'):
        if sha(lm_dir/name)!=lm['sha256'][name]: raise ValueError('LM kernel changed')
    out.mkdir(parents=True,exist_ok=False)
    kinds=layer_types(s,layers=64);l=qwen35.layout(s,max_ctx=257)
    plan=qwen35.pack_plan(s)
    q=Q4NX(model_dir/'model.q4nx')
    files=[];pools=[];consts=[];params=[]
    for i,kind in enumerate(kinds):
        directory=out/f'layer{i}';directory.mkdir()
        pool=np.memmap(directory/'pool.bin',mode='w+',dtype=np.uint8,shape=(l.POOL_BYTES,))
        pack.build_layer_pool(plan,kind,q,i,out=pool);pool.flush()
        const=pack.build_consts(plan,kind,q,i,l.C_BYTES if kind==LINEAR else l.CA_BYTES)
        const.tofile(directory/'const.bin')
        p=parameters(q,i,kind)
        np.savez(directory/'params.npz',**{k:v.astype(np.float32) for k,v in p.items()})
        params.append(p);pools.append(pool);consts.append(const)
        files += [directory/name for name in ('pool.bin','const.bin','params.npz')]
        print('Packed layer',i,flush=True)
    extract(q,plan['embed']['tensor'],out/'embedding.bin')
    extract(q,plan['norm']['tensor'],out/'finalw.bin')
    lmop=plan['lm_head']['ops'][0]
    start,end=q.tensors['lm_head.weight']['data_offsets']
    chunk=1024*H//8192*8704
    class Tensor:
        def raw(self,name): return raw
    with (out/'lm-weights.bin').open('wb') as f:
        for offset in range(start,end,chunk):
            raw=q.mm[q.data_base+offset:q.data_base+min(offset+chunk,end)]
            block=np.empty(len(raw),np.uint8)
            pack.apply_op(lmop,Tensor(),0,block)
            f.write(block.tobytes())
    provenance=dict(model=str(model_dir),q4nx_sha256=sha(model_dir/'model.q4nx'),
                    config_sha256=sha(model_dir/'config.json'),preflight=preflight)
    (out/'model-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    q.mm.close();q.f.close()
    files += [out/name for name in ('embedding.bin','finalw.bin','lm-weights.bin','model-provenance.json')]
    kernels['lm']=lm_dir
    cfg,hashes,setup_files,sizes=S.harness_setup(out,s,l,kinds,kernels)
    files+=setup_files
    for name,n,dtype in (('finalres',H,np.float32),('finalxn',H,bfloat16),('logits',V,np.float32),('x',H,np.float32)):
        path=out/f'poison-{name}.bin';path.write_bytes(np.full(n,np.nan,dtype).tobytes()+GUARD);files.append(path)
        if name!='x': cfg.append(f'buf {name} {n*np.dtype(dtype).itemsize+64}')
    np.array([248045],np.uint32).tofile(out/'seed.bin');files.append(out/'seed.bin')
    cfg += [f'buf lmw {(out/"lm-weights.bin").stat().st_size} lm-weights.bin',f'buf finalw {H*2} finalw.bin','buf token 4']
    rng=np.random.default_rng(0);states=[]
    for i,kind in enumerate(kinds):
        state,raw=S.initial_state(l,kind,rng,False,0);states.append(state)
        path=out/f'cold-layer{i}-init.bin';path.write_bytes(raw+GUARD);files.append(path)
        cfg.append(f"load {'state' if kind==LINEAR else 'cache'}{i} {path.name}")
    cfg += ['load token seed.bin','load x poison-x.bin']
    embedding=np.memmap(out/'embedding.bin',dtype=bfloat16,mode='r',shape=(V,H))
    finalw=np.fromfile(out/'finalw.bin',bfloat16)
    weights=np.memmap(out/'lm-weights.bin',dtype=np.uint8,mode='r')
    cases=[];decode_cases=[];selected=248045
    for t in range(tokens):
        tag=f'cold-{t}'
        cs=np.r_[np.cos(t*s.rope_theta**(-np.arange(32)/32)),np.sin(t*s.rope_theta**(-np.arange(32)/32))].astype(np.float32)
        position=np.zeros(l.E_A,np.uint8);position[:8]=np.array([t,257],np.int32).view(np.uint8);position[512:768]=cs.view(np.uint8)
        path=out/f'{tag}-position.bin';position.tofile(path);files.append(path)
        cfg += [f'load position {path.name}']+embedding_commands(H,V,tag)
        x=embedding[selected].astype(np.float32);input_token=selected
        path=out/f'{tag}-x.bin';path.write_bytes(x.tobytes()+GUARD);files.append(path)
        for i,kind in enumerate(kinds):
            layer=f'{tag}-layer{i}'
            refs,states[i]=evaluate(x,pools[i],consts[i],l,kind,params[i],states[i],cs,t);x=refs['y']
            path=out/f'{layer}-ref.npz';np.savez(path,**{k:v.astype(np.float32) for k,v in refs.items()});files.append(path)
            cases.append(dict(tag=layer,token=tag,sequence='cold',layer=i,kind=kind,pos=t))
            S.append_layer(cfg,s,l,kinds,sizes,layer,i,t)
            print('Reference',layer,flush=True)
        xn=norm(x,finalw);logits=D.LM.ORACLE.reference(weights,xn,V)
        selected=int(logits.argmax())
        path=out/f'{tag}-head-ref.npz';np.savez(path,xn=xn.astype(np.float32),logits=logits);files.append(path)
        decode_cases.append(dict(tag=tag,pos=t,input_token=input_token,next_token=selected))
        cfg += head_commands(H,V,tag)
        print('Reference token',input_token,'->',selected,flush=True)
    for i,kind in enumerate(kinds): cfg.append(f"load {'state' if kind==LINEAR else 'cache'}{i} cold-layer{i}-init.bin")
    cfg += ['load token seed.bin','load x poison-x.bin']
    for t in range(tokens):
        tag=f'repeat-{t}';cfg += [f'load position cold-{t}-position.bin']+embedding_commands(H,V,tag)
        for i in range(64): S.append_layer(cfg,s,l,kinds,sizes,f'repeat-layer{i}' if t==0 else f'{tag}-layer{i}',i,t)
        cfg += head_commands(H,V,tag)
    (out/'decode.cfg').write_text('\n'.join(cfg)+'\n');files.append(out/'decode.cfg')
    meta=dict(source='real-q4nx',rows=257,tokens=tokens,layer_types=kinds,cases=cases,outputs=sizes,
              kernels=hashes,decode_cases=decode_cases,fixtures={p.relative_to(out).as_posix():sha(p) for p in files})
    (out/'slice-fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(out/'decode.cfg',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('preflight','prepare','compare'))
    p.add_argument('--model-dir',type=Path,default=ROOT/'Models/qwen38-27b/converted')
    p.add_argument('--out',type=Path,default=S.DESIGN/'wide_deltanet/build_full_model')
    p.add_argument('--tokens',type=int,default=3)
    a=p.parse_args()
    if a.stage=='preflight':
        s=ModelSpec.from_hf_config(json.loads((a.model_dir/'config.json').read_text()))
        print(json.dumps(validate_header(s,*read_header(a.model_dir/'model.q4nx')),indent=2));return 0
    if a.stage=='prepare': prepare(a.out,a.model_dir,a.tokens);return 0
    return D.compare(a.out)


if __name__=='__main__': raise SystemExit(main())
