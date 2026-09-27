#!/usr/bin/env python3
"""Synthetic eight-layer autoregressive decode, full vocabulary and persistent state.

Prepare the independent reference, execute decode.cfg in the open harness,
then compare. CPU runtime work is token selection, embedding row lookup and
byte copies; all neural operators execute on the NPU.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'open_kernels'),str(ROOT/'utilities')]
from recipes.spec import LINEAR
from recipes.wide_decode import head_commands, embedding_commands
from wide_slice_reference import evaluate, norm, H
from wide_attention_reference import metric
from wide_deltanet_reference import metric as strict_metric


def module(name,file):
    spec = importlib.util.spec_from_file_location(name,ROOT/'utilities'/file)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


S = module('decode_slice','test-wide-slice.py')
LM = module('decode_lm','test-wide-lm-head.py')
GUARD,sha,DESIGN = S.GUARD,S.sha,S.DESIGN
V = 248320


def logits_metric(got,ref):
    g,r = np.asarray(got,np.float64),np.asarray(ref,np.float64)
    if g.ndim!=1 or g.shape!=r.shape or g.size<2 or not np.isfinite(g).all() or not np.isfinite(r).all():
        return dict(passed=False,reason='shape or nonfinite logits')
    # Preserve the existing compare_decode.py gate: correlation and exact argmax.
    gc,rc = g-g.mean(),r-r.mean()
    denom = np.linalg.norm(gc)*np.linalg.norm(rc)
    corr = float(gc@rc/denom) if denom else 0.
    return dict(passed=corr>.9999 and int(g.argmax())==int(r.argmax()),
                correlation=corr,got_token=int(g.argmax()),ref_token=int(r.argmax()),
                maxrel=float(np.max(np.abs(g-r))/max(np.max(np.abs(r)),1e-30)))


def link_asset(source,destination):
    destination.symlink_to(source.resolve())


def prepare(out,slice_dir,lm_dir,tokens):
    if not 3<=tokens<=8: raise ValueError('decode gate requires 3..8 autoregressive tokens')
    # A new directory prevents stale dumps or accidentally overwriting accepted evidence.
    out.mkdir(parents=True,exist_ok=False)
    kernels = S.artifacts()
    for directory,stem in ((slice_dir,'slice'),(lm_dir,'lm')):
        if not json.loads((directory/f'{stem}-results.json').read_text())['passed']:
            raise ValueError(f'prerequisite failed: {directory}')
    sm = json.loads((slice_dir/'slice-fixture.json').read_text())
    lm = json.loads((lm_dir/'lm-fixture.json').read_text())
    if (lm['k'],lm['n'],lm['cores'])!=(H,V,8): raise ValueError('wrong LM head geometry')
    for name in ('weights.bin','final.xclbin','insts.bin'):
        if sha(lm_dir/name)!=lm['sha256'][name]: raise ValueError(f'changed LM artifact: {name}')
    for name,digest in sm['kernels'].items():
        if sha(name)!=digest: raise ValueError(f'changed slice kernel: {name}')
    s,l = S.A.geometry(257)
    kinds = S.layer_types(s)
    files,params,pools,consts = [],[],[],[]
    for i in range(8):
        (out/f'layer{i}').mkdir()
        for name in ('pool.bin','const.bin','params.npz'):
            relative = f'layer{i}/{name}'
            if sha(slice_dir/relative)!=sm['fixtures'][relative]: raise ValueError(f'changed layer fixture: {relative}')
            link_asset(slice_dir/relative,out/relative);files.append(out/relative)
        pools.append(np.memmap(out/f'layer{i}/pool.bin',dtype=np.uint8,mode='r'))
        consts.append(np.fromfile(out/f'layer{i}/const.bin',np.uint8))
        with np.load(out/f'layer{i}/params.npz') as p: params.append(dict(p))
    kernels['lm'] = lm_dir.resolve()
    cfg,hashes,setup_files,sizes = S.harness_setup(out,s,l,kinds,kernels)
    files += setup_files
    link_asset(lm_dir/'weights.bin',out/'lm-weights.bin');files.append(out/'lm-weights.bin')
    rng = np.random.default_rng(38430)
    with (out/'embedding.bin').open('wb') as f:
        for start in range(0,V,1024):
            f.write(rng.normal(0,.6,(min(1024,V-start),H)).astype(bfloat16).tobytes())
    embedding = np.memmap(out/'embedding.bin',dtype=bfloat16,mode='r',shape=(V,H))
    finalw = rng.uniform(.8,1.2,H).astype(bfloat16)
    finalw.tofile(out/'finalw.bin')
    files += [out/'embedding.bin',out/'finalw.bin']
    for name,n,dtype in (('finalres',H,np.float32),('finalxn',H,bfloat16),('logits',V,np.float32),('x',H,np.float32)):
        path = out/f'poison-{name}.bin'
        path.write_bytes(np.full(n,np.nan,dtype).tobytes()+GUARD);files.append(path)
        if name!='x': cfg.append(f'buf {name} {n*np.dtype(dtype).itemsize+64}')
    np.array([248045],np.uint32).tofile(out/'seed.bin');files.append(out/'seed.bin')
    cfg += [f'buf lmw {(out/"lm-weights.bin").stat().st_size} lm-weights.bin',
            f'buf finalw {H*2} finalw.bin','buf token 4']
    weights = np.memmap(out/'lm-weights.bin',dtype=np.uint8,mode='r')
    cases,decode_cases = [],[]
    for sequence,start in (('cold',0),('warm',257-tokens)):
        states = []
        for i,kind in enumerate(kinds):
            state,raw = S.initial_state(l,kind,rng,sequence=='warm',start)
            states.append(state)
            path = out/f'{sequence}-layer{i}-init.bin'
            path.write_bytes(raw+GUARD);files.append(path)
            cfg.append(f"load {'state' if kind==LINEAR else 'cache'}{i} {path.name}")
        cfg += ['load token seed.bin','load x poison-x.bin']
        selected = 248045
        for t in range(tokens):
            tag = f'{sequence}-{t}';pos=start+t
            cs = np.r_[np.cos(pos*s.rope_theta**(-np.arange(32)/32)),np.sin(pos*s.rope_theta**(-np.arange(32)/32))].astype(np.float32)
            position = np.zeros(l.E_A,np.uint8)
            position[:8] = np.array([pos,257],np.int32).view(np.uint8)
            position[512:768] = cs.view(np.uint8)
            path = out/f'{tag}-position.bin';position.tofile(path);files.append(path)
            cfg += [f'load position {path.name}']+embedding_commands(H,V,tag)
            x = embedding[selected].astype(np.float32)
            path = out/f'{tag}-x.bin';path.write_bytes(x.tobytes()+GUARD);files.append(path)
            input_token = selected
            for i,kind in enumerate(kinds):
                layer = f'{tag}-layer{i}'
                refs,states[i] = evaluate(x,pools[i],consts[i],l,kind,params[i],states[i],cs,pos)
                x = refs['y']
                path = out/f'{layer}-ref.npz'
                np.savez(path,**{k:v.astype(np.float32) for k,v in refs.items()});files.append(path)
                cases.append(dict(tag=layer,token=tag,sequence=sequence,layer=i,kind=kind,pos=pos))
                S.append_layer(cfg,s,l,kinds,sizes,layer,i,pos)
                print('Reference:',layer,flush=True)
            xn = norm(x,finalw)
            logits = LM.ORACLE.reference(weights,xn,V,batch=256)
            selected = int(logits.argmax())
            order = np.partition(logits,-2)[-2:]
            path = out/f'{tag}-head-ref.npz'
            np.savez(path,xn=xn.astype(np.float32),logits=logits);files.append(path)
            decode_cases.append(dict(tag=tag,pos=pos,input_token=input_token,next_token=selected,
                                     top_margin=float(order[-1]-order[-2])))
            cfg += head_commands(H,V,tag)
            print('Reference token:',tag,input_token,'->',selected,flush=True)
    # Reset all persistent state and replay the entire cold autoregressive sequence.
    for i,kind in enumerate(kinds): cfg.append(f"load {'state' if kind==LINEAR else 'cache'}{i} cold-layer{i}-init.bin")
    cfg += ['load token seed.bin','load x poison-x.bin']
    for t in range(tokens):
        tag = f'repeat-{t}'
        cfg += [f'load position cold-{t}-position.bin']+embedding_commands(H,V,tag)
        for i in range(8):
            layer = f'repeat-layer{i}' if t==0 else f'{tag}-layer{i}'
            S.append_layer(cfg,s,l,kinds,sizes,layer,i,t)
        cfg += head_commands(H,V,tag)
    (out/'decode.cfg').write_text('\n'.join(cfg)+'\n');files.append(out/'decode.cfg')
    meta = dict(seed=38430,rows=257,tokens=tokens,layer_types=kinds,cases=cases,outputs=sizes,kernels=hashes,
                decode_cases=decode_cases,fixtures={p.relative_to(out).as_posix():sha(p) for p in files})
    # Reuse every inherited slice gate without weakening its bounds.
    (out/'slice-fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(out/'decode.cfg',flush=True)


def compare(out):
    slice_status = S.compare(out)
    meta = json.loads((out/'slice-fixture.json').read_text())
    checks = []
    def read(name,size,guard=True):
        raw = (out/name).read_bytes()
        if len(raw)!=size+(64 if guard else 0) or (guard and raw[size:]!=GUARD):
            raise ValueError(f'{name}: size/canary')
        return raw[:size]
    def exact(tag,field,ok): checks.append(dict(tag=tag,field=field,passed=bool(ok)))
    embeddings = np.memmap(out/'embedding.bin',dtype=bfloat16,mode='r',shape=(V,H))
    weights = np.memmap(out/'lm-weights.bin',dtype=np.uint8,mode='r')
    xn_inputs = [np.frombuffer(read(c['tag']+'-finalxn.bin',H*2),bfloat16) for c in meta['decode_cases']]
    # Local head check from device norm, supplementary to independent end-to-end logits.
    local = LM.ORACLE.references(weights,np.stack(xn_inputs),V)
    previous = {}
    for c,xn,conditional in zip(meta['decode_cases'],xn_inputs,local):
        tag = c['tag'];sequence=tag.split('-')[0]
        token = int(np.frombuffer(read(tag+'-token.bin',4,False),np.uint32)[0])
        nxt = int(np.frombuffer(read(tag+'-next.bin',4,False),np.uint32)[0])
        exact(tag,'autoregressive_feedback',token==previous.get(sequence,248045))
        exact(tag,'independent_reference_input',token==c['input_token'])
        exact(tag,'embedding_row',token<V and read(tag+'-layer0-input-x.bin',H*4)==embeddings[token].astype(np.float32).tobytes())
        previous[sequence] = nxt
        last=len(meta['layer_types'])-1
        exact(tag,'final_residual_copy',read(tag+'-finalres.bin',H*4)==read(f'{tag}-layer{last}-got-y.bin',H*4))
        ref = np.load(out/f'{tag}-head-ref.npz')
        checks.append(dict(tag=tag,field='final_norm',**metric(xn,ref['xn'],8e-3)))
        got = np.frombuffer(read(tag+'-logits.bin',V*4),np.float32)
        result = logits_metric(got,ref['logits'])
        checks.append(dict(tag=tag,field='logits',**result))
        exact(tag,'selected_device_argmax',nxt==int(got.argmax()))
        exact(tag,'selected_reference_argmax',nxt==c['next_token'])
        checks.append(dict(tag=tag,field='conditional_lm',**strict_metric(got,conditional,.9999999)))
        print('PASS' if result['passed'] else 'FAIL',tag,result,flush=True)
    # Every capture, including inputs, scratch and states, repeats byte-for-byte.
    for t in range(meta['tokens']):
        for file in out.glob(f'cold-{t}-layer*-*.bin'):
            suffix = file.name[len(f'cold-{t}-'):]
            repeat = f'repeat-{suffix}' if t==0 else f'repeat-{t}-{suffix}'
            exact(f'repeat-{t}',suffix,file.read_bytes()==(out/repeat).read_bytes())
        for field in ('token','next','finalres','finalxn','logits'):
            exact(f'repeat-{t}',field,(out/f'cold-{t}-{field}.bin').read_bytes()==(out/f'repeat-{t}-{field}.bin').read_bytes())
    passed = slice_status==0 and all(c['passed'] for c in checks)
    (out/'decode-results.json').write_text(json.dumps(dict(passed=passed,slice_passed=slice_status==0,checks=checks),indent=2)+'\n')
    print('PASS' if passed else 'FAIL',len(checks),'decode checks',flush=True)
    return 0 if passed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','compare'))
    p.add_argument('--out',type=Path,default=DESIGN/'wide_deltanet/build_decode')
    p.add_argument('--slice-dir',type=Path,default=DESIGN/'wide_deltanet/build_slice')
    p.add_argument('--lm-dir',type=Path,default=DESIGN/'lm_head_q8/build_wide_5120')
    p.add_argument('--tokens',type=int,default=3)
    a=p.parse_args()
    if a.stage=='prepare': prepare(a.out,a.slice_dir,a.lm_dir,a.tokens);return 0
    return compare(a.out)


if __name__=='__main__': raise SystemExit(main())
