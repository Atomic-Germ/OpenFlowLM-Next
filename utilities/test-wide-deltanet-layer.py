#!/usr/bin/env python3
"""Synthetic complete H5120/FF17408 DeltaNet layer; byte-only NPU composition.

prepare -> run open XRT harness on layer.cfg -> compare. Does not enable model
export or the fused runtime. Uses existing production pool/constant/state layouts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'open_kernels'))
from recipes.spec import ModelSpec
from recipes import qwen35 as Q, pack
from recipes.wide_deltanet_layer import setup_commands, token_commands, state_copies
from q4_1_pack import random_q4_1_blocks, pack_q4_1_pool, pool_reference
from wide_deltanet_reference import ab_reference, glue_reference, step_reference, metric as primitive_metric
from wide_attention_reference import metric

DESIGN = ROOT / 'open_kernels/designs'
GUARD = bytes([0xA5]) * 64
H, F, NCH, VW = 5120, 17408, 10240, 6144


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def geometry():
    os.environ['OPEN_KERNELS_UNVALIDATED'] = '1'
    s = ModelSpec.from_hf_config(json.loads((ROOT / 'specs/open-engine/tests/fixtures/config_qwen38_27b.json').read_text()))
    return s, Q.layout(s)


def norm(x, w):
    x = x.astype(np.float64)
    return (x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-6) * w.astype(np.float64)).astype(np.float32).astype(bfloat16)


def silu(x):
    x = x.astype(np.float64)
    return x * np.exp(-np.logaddexp(0, -x))


class RawTensor:
    def __init__(self, data): self.data = data
    def raw(self, name): return self.data


def sizes(l):
    return dict(res0=H*4, xn=6144*2, qzout=(NCH+VW)*4, qkv=NCH*4, z=VW*4,
                ab=768, conv=l.STATE_S_OFF, vec=48*512*4, so=48*128*128*4, o=VW*4,
                og=VW*2, projout=H*4, res1=H*4, xm=6144*2, fo=H*4, y=H*4,
                discard=H*2, act=max(l.A_BYTES, l.A_OUT2+H*4*3), trace=4)


def prepare(out, tokens):
    if tokens < 2: raise ValueError('requires multiple persistent tokens')
    out.mkdir(parents=True, exist_ok=True)
    s, l = geometry()
    rng = np.random.default_rng(38427)
    pool = np.memmap(out/'pool.bin', mode='w+', dtype=np.uint8, shape=(l.POOL_BYTES,))
    const = np.zeros(l.C_BYTES, np.uint8)
    matrices = {}
    for name, n, k, dst, offset in (
        ('qkv', NCH, H, pool, l.POOL_QKV), ('z', VW, H, pool, l.POOL_Z),
        ('out', H, VW, const, l.C_WOUT), ('up', F, H, pool, l.POOL_FFN_UP),
        ('gate', F, H, pool, l.POOL_FFN_GATE), ('down', H, F, pool, l.POOL_FFN_DOWN)):
        w = pack_q4_1_pool(random_q4_1_blocks(n, k, rng, scale=.002), 2)
        dst[offset:offset+len(w)] = w
        matrices[name] = (w, n, k)
    pool.flush()
    def put(offset, array):
        raw = np.frombuffer(array.tobytes(), np.uint8)
        const[offset:offset+len(raw)] = raw
    lnw = rng.uniform(.8, 1.2, H).astype(bfloat16)
    postw = rng.uniform(.8, 1.2, H).astype(bfloat16)
    nw = rng.uniform(.8, 1.2, 128).astype(bfloat16)
    put(l.C_LNW, lnw); put(l.C_POSTLN, postw); put(l.C_NW, nw)
    weights = rng.normal(0, .015, (2, 48, H)).astype(bfloat16)
    for w, off in zip(weights, (l.SIDE_ALPHA, l.SIDE_BETA)):
        pack.apply_op(dict(op='transpose_banked', tensor='w', rows=48, cols=H,
                           elem=2, dst=l.C_SIDE+off), RawTensor(w.tobytes()), 0, const)
    small = np.zeros(1024, np.float32)
    small[:48] = -rng.uniform(.01, .08, 48)
    small[48:96] = rng.uniform(-2.5, -1.5, 48)
    put(l.C_SIDE+l.SIDE_SMALL, small)
    convw = rng.normal(0, .25, (4, NCH)).astype(bfloat16)
    put(l.C_SIDE+l.SIDE_CONV, convw.reshape(4, 10, 1024).transpose(1, 0, 2))
    const.tofile(out/'const.bin')
    np.ones(H, bfloat16).tofile(out/'ones.bin')
    np.zeros(H, np.float32).tofile(out/'zero.bin')
    cfg = ['device']
    kernels = dict(ln=DESIGN/'wide_deltanet/build_layer/ln_precise', qz=DESIGN/'wide_deltanet/build_layer/projection_k5120',
                   ab=DESIGN/'wide_deltanet/build_ab_h5120', glue=DESIGN/'wide_deltanet/build_glue',
                   step=DESIGN/'wide_deltanet/build_step', post=DESIGN/'wide_deltanet/build_layer/post',
                   out=DESIGN/'wide_deltanet/build_layer/projection_k6144', ffn=DESIGN/'layer_x/build_segmented_precise/ffn')
    artifacts = {}
    for name, directory in kernels.items():
        for file in ('final.xclbin', 'insts.bin'): artifacts[str(directory/file)] = sha(directory/file)
        cfg += [f'xclbin {name} {directory}/final.xclbin', f'kernelx {name} {name} {directory}/insts.bin']
    cfg += [f'buf pool {l.POOL_BYTES} pool.bin', f'buf const {l.C_BYTES} const.bin',
            f'buf ones {H*2} ones.bin', f'buf zero {H*4} zero.bin']
    extra = dict(x=H*4, state=l.STATE_BYTES+64, cs=l.STATE_S_OFF, si=48*128*128*4,
                 qzw=(NCH+VW)*H//8192*5120, ow=H*VW//8192*5120,
                 lnw=H*2, postw=H*2, nw=4096, side=4096+4*NCH*2, abs=4*H*32*2+8192)
    cfg += [f'buf {name} {size}' for name, size in extra.items()]
    outputs = sizes(l)
    bf16_names = ('xn', 'conv', 'og', 'xm', 'discard')
    for name, size in outputs.items():
        dtype = bfloat16 if name in bf16_names else np.float32
        (out/f'poison-{name}.bin').write_bytes(np.full(size//np.dtype(dtype).itemsize, np.nan, dtype).tobytes()+GUARD)
        cfg += [f'buf {name} {size+64}']
    cfg += setup_commands(s, l)
    tags = []
    def project(name, x):
        w, n, k = matrices[name]
        return pool_reference(w, x.astype(bfloat16), n, k, rs=2)
    def reference(x, cs, state):
        xn = norm(x, lnw)
        qkv, z = project('qkv', xn), project('z', xn)
        ab = ab_reference(xn, weights, small[:48], small[48:96]).astype(np.float32)
        cs, vec = glue_reference(qkv, cs, convw, ab)
        state, o = step_reference(state, vec.astype(np.float32))
        state, o = state.astype(np.float32), o.astype(np.float32)
        # Round only at the physical f32/BF16 boundaries, never use device output.
        a = o.astype(np.float64).reshape(48, 128)
        og = (a / np.sqrt(np.mean(a*a, axis=1, keepdims=True)+1e-6) * nw.astype(np.float64) * silu(z.reshape(48,128))).astype(np.float32).astype(bfloat16).ravel()
        projout = project('out', og)
        res1 = (projout+x).astype(np.float32)
        xm = norm(res1, postw)
        u, g = project('up', xm), project('gate', xm)
        h = (u.astype(np.float64)*silu(g)).astype(np.float32)
        fo = project('down', h)
        refs = dict(xn=xn, qkv=qkv, z=z, ab=ab, conv=cs, vec=vec, so=state, o=o,
                    og=og, projout=projout, res1=res1, xm=xm, h=h, fo=fo, y=(fo+res1).astype(np.float32))
        return cs, state, refs
    for case in ('cold', 'warm'):
        cs = np.zeros((3, NCH), bfloat16)
        state = np.zeros((48,128,128), np.float32)
        if case == 'warm':
            cs[:] = rng.normal(0,.2,cs.shape).astype(bfloat16)
            state[:] = rng.normal(0,.05,state.shape)
        raw = bytearray(l.STATE_BYTES)
        raw[:l.STATE_S_OFF] = cs.tobytes()
        compact = state.tobytes()
        for dst, src, size in state_copies(l,48,128,restore=True): raw[dst:dst+size] = compact[src:src+size]
        (out/f'{case}-init.bin').write_bytes(raw+GUARD)
        cfg += [f'load state {case}-init.bin']
        for t in range(tokens):
            tag = f'{case}-{t}'
            tags.append(tag)
            x = rng.normal(0,.6,H).astype(np.float32)
            x.tofile(out/f'{tag}-x.bin')
            cs, state, refs = reference(x,cs,state)
            np.savez(out/f'{tag}-ref.npz', **{k:v.astype(np.float32) for k,v in refs.items()})
            cfg += [f'load x {tag}-x.bin']
            cfg += [f'load {name} poison-{name}.bin' for name in outputs]
            cfg += token_commands(s,l)
            cfg += [f'dump {name} {tag}-got-{name}.bin {size+64}' for name,size in outputs.items()]
            cfg += [f'dump state {tag}-got-state.bin {l.STATE_BYTES+64}']
            print('Reference:',tag,flush=True)
    # Reset after both sequences: detect stale worker/BO state.
    cfg += ['load state cold-init.bin', 'load x cold-0-x.bin']
    cfg += [f'load {name} poison-{name}.bin' for name in outputs]
    cfg += token_commands(s,l)
    cfg += [f'dump y repeat-y.bin {H*4+64}', f'dump state repeat-state.bin {l.STATE_BYTES+64}']
    for pattern in ('*-got-*.bin', 'repeat-*.bin'):
        for path in out.glob(pattern): path.unlink()
    (out/'layer.cfg').write_text('\n'.join(cfg)+'\n')
    files = [out/name for name in ('pool.bin','const.bin','ones.bin','zero.bin','cold-init.bin','warm-init.bin','layer.cfg')]
    files += [out/f'poison-{name}.bin' for name in outputs]
    files += [out/f'{tag}-{suffix}' for tag in tags for suffix in ('x.bin','ref.npz')]
    meta = dict(seed=38427, tokens=tokens, tags=tags, kernels=artifacts,
                fixtures={p.name:sha(p) for p in files}, outputs=outputs,
                contract='designs/layer_x/compare.py full-layer gates; primitive tolerances unchanged')
    (out/'layer-fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'layer-results.json').unlink(missing_ok=True)
    print(out/'layer.cfg')


def compare(out):
    s,l = geometry()
    meta = json.loads((out/'layer-fixture.json').read_text())
    for path,digest in meta['kernels'].items():
        if sha(path)!=digest: raise ValueError(f'kernel changed: {path}')
    for path,digest in meta['fixtures'].items():
        if sha(out/path)!=digest: raise ValueError(f'fixture changed: {path}')
    checks=[]
    diagnostics=[]
    def check(tag,field,got,ref,tol):
        m=metric(got.ravel(),ref.ravel(),tol)
        checks.append(dict(tag=tag,field=field,tolerance=tol,**m))
        print(f"{'PASS' if m['passed'] else 'FAIL'} {tag} {field} {m}",flush=True)
    def exact(tag,field,passed):
        checks.append(dict(tag=tag,field=field,passed=bool(passed)))
        if not passed: print('FAIL',tag,field,flush=True)
    pool=np.memmap(out/'pool.bin',mode='r',dtype=np.uint8)
    const=np.memmap(out/'const.bin',mode='r',dtype=np.uint8)
    for tag in meta['tags']:
        got={}
        for name,size in dict(meta['outputs'],state=l.STATE_BYTES).items():
            data=(out/f'{tag}-got-{name}.bin').read_bytes()
            if len(data)!=size+64 or data[size:]!=GUARD: raise ValueError(f'{tag} {name}: size/canary')
            dtype=bfloat16 if name in ('xn','conv','og','xm','discard') else np.float32
            got[name]=np.frombuffer(data[:size],dtype)
        ref=np.load(out/f'{tag}-ref.npz')
        # Full independent intermediates are diagnostic, without inventing
        # new acceptance limits for tensors absent from the legacy layer gate.
        for name in ('qkv','z','ab','vec','o','og','projout','fo'):
            m=metric(got[name].ravel(),ref[name].ravel(),2e-2)
            m.pop('passed')
            diagnostics.append(dict(tag=tag,field=name,**m))
        head_metrics=[primitive_metric(g,r,.9999999) for g,r in zip(got['so'].reshape(48,128,128),ref['so'])]
        diagnostics.append(dict(tag=tag,field='state_head_local_1e-4',heads=head_metrics,
                                passed=all(m['passed'] for m in head_metrics)))
        for name,tol in (('xn',8e-3),('res1',2e-2),('xm',2e-2),('so',2e-2),('conv',2e-2),('y',5e-3)):
            g=got[name][:H] if name in ('xn','xm') else got[name]
            check(tag,name,g,ref[name],tol)
        state=got['state'].view(np.uint8)
        exact(tag,'conv_state_bytes',np.array_equal(state[:l.STATE_S_OFF],got['conv'].view(np.uint8)))
        for h,(dst,src,size) in enumerate(state_copies(l,48,128)):
            exact(tag,f'head{h}_state_copy',np.array_equal(state[src:src+size],got['so'].view(np.uint8)[dst:dst+size]))
            exact(tag,f'head{h}_padding_zero',not state[src+size:src+l.S_HEAD_BYTES].any())
        # Isolate each GEMV from accumulated BF16 differences: device inputs are
        # read only after inference, never fed back to the hardware program.
        for name,offset,n,k,storage,x in (
            ('qkv',l.POOL_QKV,NCH,H,pool,got['xn'][:H]),
            ('z',l.POOL_Z,VW,H,pool,got['xn'][:H]),
            ('projout',l.C_WOUT,H,VW,const,got['og']),
            ('fo',l.POOL_FFN_DOWN,H,F,pool,got['act'].view(np.uint8)[l.A_H:l.A_H+F*4].view(np.float32))):
            r=pool_reference(storage[offset:offset+n*k//8192*5120],x.astype(bfloat16),n,k,rs=2)
            m=primitive_metric(got[name],r,.9999999)
            checks.append(dict(tag=tag,field=f'conditional_{name}',**m))
            print(f"{'PASS' if m['passed'] else 'FAIL'} {tag} conditional_{name} {m}",flush=True)
        exact(tag,'conv_current_token',np.array_equal(got['conv'][-NCH:].view(np.uint16),got['qkv'].astype(bfloat16).view(np.uint16)))
        # Strict full FFN gate conditional on actual post-LN input.
        def proj(offset,n,k,x):
            return pool_reference(pool[offset:offset+n*k//8192*5120],x.astype(bfloat16),n,k,rs=2)
        u=proj(l.POOL_FFN_UP,F,H,got['xm'][:H]); g=proj(l.POOL_FFN_GATE,F,H,got['xm'][:H])
        h=(u.astype(np.float64)*silu(g)).astype(np.float32)
        r=proj(l.POOL_FFN_DOWN,H,F,h)
        m=primitive_metric(got['fo'],r,.9999999)
        checks.append(dict(tag=tag,field='conditional_ffn',**m))
        print(f"{'PASS' if m['passed'] else 'FAIL'} {tag} conditional_ffn {m}",flush=True)
    for name in ('y','state'):
        exact('repeat',name,(out/f'repeat-{name}.bin').read_bytes()==(out/f'cold-0-got-{name}.bin').read_bytes())
    passed=all(c['passed'] for c in checks)
    (out/'layer-results.json').write_text(json.dumps(dict(passed=passed,checks=checks,diagnostics=diagnostics),indent=2)+'\n')
    print('PASS' if passed else 'FAIL',len(checks),'checks')
    return 0 if passed else 1


def post_prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(38428)
    cfg=['device']
    artifacts={}
    for heads,subdir in ((32,'post32'),(48,'post')):
        directory=DESIGN/'wide_deltanet/build_layer'/subdir
        for file in ('final.xclbin','insts.bin'): artifacts[str(directory/file)]=sha(directory/file)
        cfg += [f'xclbin p{heads} {directory}/final.xclbin',f'kernelx p{heads} p{heads} {directory}/insts.bin']
    nw=rng.uniform(.8,1.2,128).astype(bfloat16)
    (out/'post-nw.bin').write_bytes(nw.tobytes()+bytes(4096-256))
    cfg += ['buf pnw 4096 post-nw.bin',f'buf po {VW*4}',f'buf pz {VW*4}',f'buf pg {VW*2+64}']
    for case in range(3):
        o=rng.normal(0,.6,(48,128)).astype(np.float32)
        z=rng.normal(0,.6,(48,128)).astype(np.float32)
        if case==1: o.fill(0)
        if case==2: z[:,::2]=20; z[:,1::2]=-20
        a=o.astype(np.float64)
        ref=(a/np.sqrt(np.mean(a*a,axis=1,keepdims=True)+1e-6)*nw.astype(np.float64)*silu(z)).astype(np.float32).astype(bfloat16)
        o.tofile(out/f'post-{case}-o.bin'); z.tofile(out/f'post-{case}-z.bin'); ref.tofile(out/f'post-{case}-ref.bin')
        cfg += [f'load po post-{case}-o.bin',f'load pz post-{case}-z.bin']
        for heads in (32,48):
            (out/f'post-poison-{heads}.bin').write_bytes(np.full(heads*128,np.nan,bfloat16).tobytes()+GUARD)
            cfg += [f'load pg post-poison-{heads}.bin',f'run p{heads} po pz pnw pg',f'dump pg post-{case}-{heads}-got.bin {heads*128*2+64}']
            (out/f'post-{case}-{heads}-got.bin').unlink(missing_ok=True)
    (out/'post.cfg').write_text('\n'.join(cfg)+'\n')
    files=[p for p in out.glob('post-*') if p.suffix=='.bin']+[out/'post.cfg']
    (out/'post-fixture.json').write_text(json.dumps(dict(kernels=artifacts,fixtures={p.name:sha(p) for p in files}),indent=2)+'\n')


def post_metric(g, r):
    m=metric(g,r,8e-3)
    # +0 and -0 represent the same result; keep the existing mismatch limit
    # for all nonzero values. Raw sign bits are not an accuracy difference.
    ndiff=int(np.count_nonzero(g!=r)) if g.shape==r.shape else max(g.size,r.size)
    m['passed']=m['passed'] and m['cosine']>.999999 and ndiff<len(r)//20
    return dict(bf16_mismatches=ndiff,**m)


def post_compare(out):
    meta=json.loads((out/'post-fixture.json').read_text())
    for p,digest in meta['kernels'].items():
        if sha(p)!=digest: raise ValueError(f'kernel changed: {p}')
    for p,digest in meta['fixtures'].items():
        if sha(out/p)!=digest: raise ValueError(f'fixture changed: {p}')
    checks=[]
    for case in range(3):
        ref=np.fromfile(out/f'post-{case}-ref.bin',bfloat16)
        for heads in (32,48):
            raw=(out/f'post-{case}-{heads}-got.bin').read_bytes()
            if len(raw)!=heads*128*2+64 or raw[-64:]!=GUARD: raise ValueError('post size/canary')
            g=np.frombuffer(raw[:-64],bfloat16); r=ref[:heads*128]
            m=post_metric(g,r)
            checks.append(dict(case=case,heads=heads,**m))
            print(checks[-1])
        a=(out/f'post-{case}-32-got.bin').read_bytes()[:-64]
        b=(out/f'post-{case}-48-got.bin').read_bytes()[:len(a)]
        checks.append(dict(case=case,field='legacy_prefix_exact',passed=a==b))
    passed=all(c['passed'] for c in checks)
    (out/'post-results.json').write_text(json.dumps(dict(passed=passed,checks=checks),indent=2)+'\n')
    return 0 if passed else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','compare','post-prepare','post-compare'))
    p.add_argument('--out',type=Path,default=DESIGN/'wide_deltanet/build_layer/acceptance')
    p.add_argument('--tokens',type=int,default=4)
    a=p.parse_args()
    out=a.out.resolve()
    if a.stage=='prepare': result=prepare(out,a.tokens)
    else: result={'compare':compare,'post-prepare':post_prepare,'post-compare':post_compare}[a.stage](out)
    sys.exit(result)
