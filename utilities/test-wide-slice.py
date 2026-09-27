#!/usr/bin/env python3
"""Prepare/compare eight synthetic Qwen38 layers with independent persistent state.

Run the generated slice.cfg with the open XRT harness between stages. All
inter-layer activations and recurrent/KV state remain device-produced bytes.
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
from recipes.wide_slice import layer_types, layer_commands, buffer_name
from recipes.wide_deltanet_layer import state_copies
from wide_attention_reference import metric
from wide_deltanet_reference import metric as strict_metric
from wide_slice_reference import pack_linear, evaluate, H, F, QW, KVW, NCH, VW


def module(name,file):
    spec = importlib.util.spec_from_file_location(name,ROOT/'utilities'/file)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


D = module('delta_layer_probe','test-wide-deltanet-layer.py')
A = module('attention_layer_probe','test-wide-attention-layer.py')
DESIGN,GUARD,sha = D.DESIGN,D.GUARD,D.sha


def artifacts():
    # Require the accepted one-layer artifacts; do not silently regress to the
    # older arithmetic with a failing numerical gate.
    for directory in (DESIGN/'wide_deltanet/build_layer_precise/acceptance',DESIGN/'attn/build_layer/acceptance'):
        if not json.loads((directory/'layer-results.json').read_text())['passed']:
            raise ValueError(f'prerequisite layer gate failed: {directory}')
        manifest = json.loads((directory/'layer-fixture.json').read_text())
        for path,digest in manifest['kernels'].items():
            if sha(path)!=digest: raise ValueError(f'prerequisite artifact changed: {path}')
    A.validate_attention_build(DESIGN/'attn/build_wide_257',257)
    A.validate_projection_build(DESIGN/'attn/build_layer/projection_k5120')
    return dict(ln=DESIGN/'wide_deltanet/build_layer/ln_precise',
                out=DESIGN/'wide_deltanet/build_layer/projection_k6144',
                ffn=DESIGN/'layer_x/build_segmented_precise/ffn',
                d_qz=DESIGN/'wide_deltanet/build_layer_compensated/projection_k5120',
                d_ab=DESIGN/'wide_deltanet/build_ab_h5120',
                d_glue=DESIGN/'wide_deltanet/build_layer_compensated/glue_precise',
                d_step=DESIGN/'wide_deltanet/build_layer_compensated/step_precise',
                d_post=DESIGN/'wide_deltanet/build_layer/post',
                a_qkvg=DESIGN/'attn/build_layer/projection_k5120',a_attn=DESIGN/'attn/build_wide_257')


def initial_state(l,kind,rng,warm,start):
    if kind==LINEAR:
        conv = np.zeros((3,NCH),bfloat16)
        so = np.zeros((48,128,128),np.float32)
        if warm:
            conv[:] = rng.normal(0,.2,conv.shape).astype(bfloat16)
            so[:] = rng.normal(0,.05,so.shape)
        raw = bytearray(l.STATE_BYTES)
        raw[:l.STATE_S_OFF] = conv.tobytes()
        compact = so.tobytes()
        for dst,src,size in state_copies(l,48,128,restore=True): raw[dst:dst+size] = compact[src:src+size]
        return dict(conv=conv,so=so),bytes(raw)
    cache = np.full((l.MAX_CTX,2,4,256),np.nan,bfloat16)
    cache[:start] = rng.normal(0,.6,(start,2,4,256)).astype(bfloat16)
    return cache,cache.tobytes()


def harness_setup(out,s,l,kinds,kernels):
    files = []
    cfg,hashes = ['device'],{}
    for name,directory in kernels.items():
        cfg += [f'xclbin {name} {directory}/final.xclbin',f'kernelx {name} {name} {directory}/insts.bin']
        for file in ('final.xclbin','insts.bin'): hashes[str(directory/file)] = sha(directory/file)
    np.ones(H,bfloat16).tofile(out/'ones.bin');np.zeros(H,np.float32).tofile(out/'zero.bin')
    files += [out/'ones.bin',out/'zero.bin']
    cfg += [f'buf pool {l.POOL_BYTES}',f'buf const {max(l.C_BYTES,l.CA_BYTES)}',
            f'buf x {H*4+64}',f'buf y {H*4+64}',f'buf position {l.E_A}',
            f'buf ones {H*2} ones.bin',f'buf zero {H*4} zero.bin']
    sizes = dict(d=D.sizes(l),a=A.outputs(l))
    extras = dict(d=dict(cs=l.STATE_S_OFF,si=48*128*128*4,qzw=(NCH+VW)*H//8192*5120,
                         ow=H*VW//8192*5120,lnw=H*2,postw=H*2,nw=4096,
                         side=4096+4*NCH*2,abs=4*H*32*2+8192),
                  a=dict(meta=l.E_A*2,qkvgw=14336*H//8192*5120,ow=H*QW//8192*5120,lnw=H*2,postw=H*2))
    for prefix in ('d','a'):
        for name,size in extras[prefix].items(): cfg.append(f'buf {prefix}_{name} {size}')
        bf16 = ('xn','conv','og','xm','discard') if prefix=='d' else A.BF16_OUTPUTS
        for name,size in sizes[prefix].items():
            if name!='y': cfg.append(f'buf {prefix}_{name} {size+64}')
            dtype = bfloat16 if name in bf16 else np.float32
            path = out/f'poison-{prefix}-{name}.bin'
            path.write_bytes(np.full(size//np.dtype(dtype).itemsize,np.nan,dtype).tobytes()+GUARD)
            files.append(path)
    for i,kind in enumerate(kinds):
        name,size = (f'state{i}',l.STATE_BYTES) if kind==LINEAR else (f'cache{i}',l.KV_BYTES)
        cfg.append(f'buf {name} {size+64}')
    return cfg,hashes,files,sizes


def append_layer(cfg,s,l,kinds,sizes,tag,i,pos):
    prefix = 'd' if kinds[i]==LINEAR else 'a'
    state_name,state_size = (f'state{i}',l.STATE_BYTES) if prefix=='d' else (f'cache{i}',l.KV_BYTES)
    cfg.extend([f'dump x {tag}-input-x.bin {H*4+64}',f'dump {state_name} {tag}-input-state.bin {state_size+64}'])
    cfg.extend(f'load {buffer_name(name,prefix,i)} poison-{prefix}-{name}.bin' for name in sizes[prefix])
    cfg.extend(layer_commands(s,l,i,pos,257))
    cfg.extend(f'dump {buffer_name(name,prefix,i)} {tag}-got-{name}.bin {size+64}' for name,size in sizes[prefix].items())
    cfg.append(f'dump {state_name} {tag}-got-state.bin {state_size+64}')
    cfg.append(f'copy x 0 y 0 {H*4}')


def prepare(out,tokens):
    if not 2<=tokens<=8: raise ValueError('the slice gate requires 2..8 tokens per sequence')
    kernels = artifacts()
    s,l = A.geometry(257)
    kinds = layer_types(s)
    out.mkdir(parents=True,exist_ok=True)
    for pattern in ('*-got-*.bin','*-input-*.bin','repeat-*.bin'):
        for p in out.glob(pattern): p.unlink()
    files,params,pools,consts = [],[],[],[]
    for i,kind in enumerate(kinds):
        directory = out/f'layer{i}'
        directory.mkdir(exist_ok=True)
        rng = np.random.default_rng(np.random.SeedSequence([38429,i]))
        pool = np.memmap(directory/'pool.bin',mode='w+',dtype=np.uint8,shape=(l.POOL_BYTES,))
        if kind==LINEAR:
            const,p = pack_linear(s,l,rng,pool)
        else:
            A.pack_weights(s,l,rng,pool)
            const = np.zeros(l.CA_BYTES,np.uint8)
            p = dict(lnw=rng.uniform(.8,1.2,H).astype(bfloat16),postw=rng.uniform(.8,1.2,H).astype(bfloat16),
                     norms=rng.uniform(.8,1.2,(2,256)).astype(bfloat16))
            for offset,a in ((l.CA_LNW,p['lnw']),(l.CA_POSTLN,p['postw']),(l.CA_META,p['norms'])):
                const[offset:offset+a.nbytes] = a.view(np.uint8).ravel()
        pool.flush()
        const.tofile(directory/'const.bin')
        np.savez(directory/'params.npz',**{k:v.astype(np.float32) for k,v in p.items()})
        params.append(p);pools.append(pool);consts.append(const)
        files += [directory/n for n in ('pool.bin','const.bin','params.npz')]
        print('Packed layer',i,kind,flush=True)
    cfg,hashes,setup_files,sizes = harness_setup(out,s,l,kinds,kernels)
    files += setup_files
    cases = []
    rng = np.random.default_rng(38429)
    for sequence,start in (('cold',0),('warm',257-tokens)):
        states = []
        for i,kind in enumerate(kinds):
            st,raw = initial_state(l,kind,rng,sequence=='warm',start)
            states.append(st)
            path = out/f'{sequence}-layer{i}-init.bin'
            path.write_bytes(raw+GUARD);files.append(path)
            cfg.append(f"load {'state' if kind==LINEAR else 'cache'}{i} {path.name}")
        for t in range(tokens):
            token = f'{sequence}-{t}';pos=start+t
            x = rng.normal(0,.6,H).astype(np.float32)
            path = out/f'{token}-x.bin';path.write_bytes(x.tobytes()+GUARD);files.append(path)
            cs = np.r_[np.cos(pos*s.rope_theta**(-np.arange(32)/32)),np.sin(pos*s.rope_theta**(-np.arange(32)/32))].astype(np.float32)
            position = np.zeros(l.E_A,np.uint8)
            position[:8] = np.array([pos,257],np.int32).view(np.uint8)
            position[512:768] = cs.view(np.uint8)
            path = out/f'{token}-position.bin';position.tofile(path);files.append(path)
            cfg += [f'load x {token}-x.bin',f'load position {token}-position.bin']
            for i,kind in enumerate(kinds):
                tag = f'{token}-layer{i}'
                refs,states[i] = evaluate(x,pools[i],consts[i],l,kind,params[i],states[i],cs,pos)
                x = refs['y']
                path = out/f'{tag}-ref.npz'
                np.savez(path,**{k:v.astype(np.float32) for k,v in refs.items()});files.append(path)
                cases.append(dict(tag=tag,token=token,sequence=sequence,layer=i,kind=kind,pos=pos))
                append_layer(cfg,s,l,kinds,sizes,tag,i,pos)
                print('Reference:',tag,flush=True)
    for i,kind in enumerate(kinds): cfg.append(f"load {'state' if kind==LINEAR else 'cache'}{i} cold-layer{i}-init.bin")
    cfg += ['load x cold-0-x.bin','load position cold-0-position.bin']
    for i in range(8): append_layer(cfg,s,l,kinds,sizes,f'repeat-layer{i}',i,0)
    (out/'slice.cfg').write_text('\n'.join(cfg)+'\n');files.append(out/'slice.cfg')
    metadata = dict(seed=38429,rows=257,tokens=tokens,layer_types=kinds,cases=cases,outputs=sizes,kernels=hashes,
                    fixtures={p.relative_to(out).as_posix():sha(p) for p in files})
    (out/'slice-fixture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'slice-results.json').unlink(missing_ok=True)
    print(out/'slice.cfg')


def compare(out):
    meta = json.loads((out/'slice-fixture.json').read_text())
    _,l = A.geometry(meta['rows'])
    for path,digest in meta['kernels'].items():
        if sha(path)!=digest: raise ValueError(f'kernel changed: {path}')
    for path,digest in meta['fixtures'].items():
        if sha(out/path)!=digest: raise ValueError(f'fixture changed: {path}')
    checks,diagnostics = [],[]
    def read(name,size):
        data = (out/name).read_bytes()
        if len(data)!=size+64 or data[size:]!=GUARD: raise ValueError(f'{name}: size/canary')
        return data[:size]
    def exact(tag,field,passed): checks.append(dict(tag=tag,field=field,passed=bool(passed)))
    def check(tag,name,g,r,tol):
        result = metric(np.asarray(g).ravel(),np.asarray(r).ravel(),tol)
        checks.append(dict(tag=tag,field=name,tolerance=tol,**result))
        if name=='y' or not result['passed']: print('PASS' if result['passed'] else 'FAIL',tag,name,result,flush=True)
    previous,sequence,token = {},None,None
    for case in meta['cases']:
        tag,i,kind = case['tag'],case['layer'],case['kind']
        prefix = 'd' if kind==LINEAR else 'a'
        state_size = l.STATE_BYTES if kind==LINEAR else l.KV_BYTES
        if sequence != case['sequence']:
            sequence = case['sequence']
            previous = {j:read(f'{sequence}-layer{j}-init.bin',l.STATE_BYTES if k==LINEAR else l.KV_BYTES)
                        for j,k in enumerate(meta['layer_types'])}
        if token != case['token']:
            token = case['token']
            expected_x = read(f'{token}-x.bin',H*4)
        x = read(f'{tag}-input-x.bin',H*4)
        exact(tag,'device_activation_chain',x==expected_x)
        exact(tag,'isolated_state_input',read(f'{tag}-input-state.bin',state_size)==previous[i])
        bf16 = ('xn','conv','og','xm','discard') if kind==LINEAR else A.BF16_OUTPUTS
        got = {name:np.frombuffer(read(f'{tag}-got-{name}.bin',size),bfloat16 if name in bf16 else np.float32)
               for name,size in meta['outputs'][prefix].items()}
        ref = np.load(out/f'{tag}-ref.npz')
        for name,tol in (('xn',8e-3),('res1',2e-2 if kind==LINEAR else 1e-2),('xm',2e-2),('y',5e-3)):
            check(tag,name,got[name][:H],ref[name],tol)
        state = read(f'{tag}-got-state.bin',state_size)
        if kind==LINEAR:
            check(tag,'conv',got['conv'],ref['conv'],2e-2)
            check(tag,'so',got['so'],ref['so'],2e-2)
            exact(tag,'conv_copy',state[:l.STATE_S_OFF]==got['conv'].tobytes())
            exact(tag,'conv_current_token',np.array_equal(got['conv'][-NCH:].view(np.uint16),got['qkv'].astype(bfloat16).view(np.uint16)))
            for h,(dst,src,size) in enumerate(state_copies(l,48,128)):
                exact(tag,f'state_head{h}_copy',state[src:src+size]==got['so'].tobytes()[dst:dst+size])
                exact(tag,f'state_head{h}_padding',not any(state[src+size:src+l.S_HEAD_BYTES]))
            heads = [strict_metric(g,r,.9999999) for g,r in zip(got['so'].reshape(48,128,128),ref['so'])]
            diagnostics.append(dict(tag=tag,field='state_head_local_1e-4',heads=heads,passed=all(h['passed'] for h in heads)))
        else:
            check(tag,'og',got['og'],ref['og'],2e-2)
            check(tag,'new',got['new'],ref['new'],1e-2)
            for h,(g,r) in enumerate(zip(got['og'].reshape(24,256),ref['og'].reshape(24,256))):
                check(tag,f'head{h}_og',g,r,2e-2)
            for n in ('k','v'):
                start=0 if n=='k' else KVW
                for h in range(4):
                    check(tag,f'head{h}_{n}',got['new'][start+h*256:start+(h+1)*256],ref['new'].ravel()[start+h*256:start+(h+1)*256],1e-2)
            expected = bytearray(previous[i]);pos=case['pos']
            expected[pos*l.KV_ROW:(pos+1)*l.KV_ROW] = got['new'].tobytes()
            exact(tag,'cache_copy_and_untouched_rows',state==expected)
            exact(tag,'v_exact_from_projection',np.array_equal(got['new'][KVW:].view(np.uint16),got['kvn'][KVW:].astype(bfloat16).view(np.uint16)))
            for name,offset,size in (('xn',l.AA_XN,H*2),('qg',l.AA_QG,QW*8),('kvn',l.AA_KVN,KVW*8),
                    ('og',l.AA_OG,QW*2),('projout',l.AA_OUT,H*4),('res1',l.AA_RES,H*4),
                    ('xm',l.AA_XM,H*2),('fo',l.AA_OUT2,H*4)):
                exact(tag,f'{name}_activation_copy',got['act'].view(np.uint8)[offset:offset+size].tobytes()==got[name].tobytes()[:size])
        previous[i] = state
        expected_x = got['y'].tobytes()
    for i,kind in enumerate(meta['layer_types']):
        state_size = l.STATE_BYTES if kind==LINEAR else l.KV_BYTES
        sizes = dict(meta['outputs']['d' if kind==LINEAR else 'a'],state=state_size)
        for name,size in sizes.items():
            exact(f'repeat-layer{i}',name,read(f'repeat-layer{i}-got-{name}.bin',size)==read(f'cold-0-layer{i}-got-{name}.bin',size))
    passed = all(c['passed'] for c in checks)
    (out/'slice-results.json').write_text(json.dumps(dict(passed=passed,checks=checks,diagnostics=diagnostics),indent=2)+'\n')
    print('PASS' if passed else 'FAIL',len(checks),'checks')
    return 0 if passed else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','compare'))
    p.add_argument('--out',type=Path,default=DESIGN/'wide_deltanet/build_slice')
    p.add_argument('--tokens',type=int,default=2)
    a=p.parse_args()
    if a.stage=='prepare': prepare(a.out,a.tokens);return 0
    return compare(a.out)


if __name__=='__main__': raise SystemExit(main())
