#!/usr/bin/env python3
"""Synthetic complete H5120/FF17408 gated attention layer, with device KV state.

prepare -> open XRT harness on layer.cfg -> compare. No host neural math in
the hardware sequence, no catalogue promotion or alternate model format.
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
sys.path.insert(0, str(ROOT/'open_kernels'))
sys.path.insert(0, str(ROOT/'utilities'))
from recipes import qwen35 as Q, pack
from recipes.spec import ModelSpec, FULL
from recipes.wide_attention_layer import setup_commands, token_commands
from q4_1_pack import random_q4_1_blocks, pack_q4_1_pool, pool_reference
from wide_attention_reference import decode, norm_rope, metric

DESIGN = ROOT/'open_kernels/designs'
GUARD = bytes([0xA5])*64
H, F, QW, KVW = 5120, 17408, 6144, 1024
NPROJ = 2*(QW+KVW)
BF16_OUTPUTS = ('xn', 'new', 'og', 'xm', 'discard')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def geometry(rows):
    os.environ['OPEN_KERNELS_UNVALIDATED'] = '1'
    s = ModelSpec.from_hf_config(json.loads((ROOT/'specs/open-engine/tests/fixtures/config_qwen38_27b.json').read_text()))
    return s, Q.layout(s, max_ctx=rows)


def validate_attention_build(directory, rows):
    """The primitive fixture pins the otherwise static DMA row count to artifacts."""
    meta = json.loads((directory/'attention-fixture.json').read_text())
    if tuple(meta[k] for k in ('rows','nh','kvh','hd','rot')) != (rows,24,4,256,64):
        raise ValueError('attention build geometry does not match the layer fixture')
    for name in ('final.xclbin','insts.bin'):
        if sha(directory/name) != meta['sha256'][name]:
            raise ValueError(f'attention artifact changed: {name}; revalidate its primitive fixture')


def validate_projection_build(directory):
    meta = json.loads((directory/'probe-toolchain.json').read_text())
    if tuple(meta[k] for k in ('scope','ffn','projection_k','projection_n','weight_format')) != (
            'projection',F,H,NPROJ,'q4_1'):
        raise ValueError('projection build geometry does not match the layer fixture')


def norm(x, w):
    return norm_rope(x, w, np.empty(0)).astype(np.float32).astype(bfloat16)


def silu(x):
    x = x.astype(np.float64)
    return x*np.exp(-np.logaddexp(0, -x))


def matrices(l):
    return dict(q=(l.POOL_Q,QW,H), k=(l.POOL_K,KVW,H), v=(l.POOL_V,KVW,H),
                gate=(l.POOL_GATE,QW,H), out=(l.POOL_O,H,QW),
                up=(l.POOL_FFN_UP,F,H), fgate=(l.POOL_FFN_GATE,F,H), down=(l.POOL_FFN_DOWN,H,F))


def project(pool, layout, name, x):
    off, n, k = matrices(layout)[name]
    return pool_reference(pool[off:off+n*k//8192*5120], x.astype(bfloat16), n, k, rs=2)


def outputs(l):
    return dict(res0=H*4, xn=6144*2, projected=NPROJ*4, qg=QW*8, kvn=KVW*8,
                new=l.KV_ROW, og=QW*2, projout=H*4, res1=H*4, xm=6144*2,
                fo=H*4, y=H*4, discard=H*2, act=l.AA_BYTES,
                ffnact=max(l.A_BYTES,l.A_OUT2+H*4*3), trace=4)


def pack_weights(s, l, rng, pool):
    """Exercise the actual Qwen35 pack plan from synthetic native Q4 chunks.

    Verify each packed matrix against the independent GGUF->pool packer,
    including both halves of the fused source q_proj tensor.
    """
    ops = Q.pack_plan(s)['layer_types'][FULL]['pool']
    class Tensor:
        def __init__(self, raw): self.data = raw
        def raw(self, name): return self.data
    for suffix, n, k in [('mlp.up_proj.weight',F,H), ('mlp.gate_proj.weight',F,H),
                         ('mlp.down_proj.weight',H,F), ('self_attn.q_proj.weight',2*QW,H),
                         ('self_attn.k_proj.weight',KVW,H), ('self_attn.v_proj.weight',KVW,H),
                         ('self_attn.o_proj.weight',H,QW)]:
        blocks = random_q4_1_blocks(n, k, rng, scale=.002)
        source = Tensor(pack_q4_1_pool(blocks, rs=1).tobytes())
        for op in (op for op in ops if op['tensor'].endswith(suffix)):
            pack.apply_op(op, source, 0, pool)
            start = op.get('chunk0',0)*8192//k
            count = op['nch']*8192//k
            expected = pack_q4_1_pool(blocks[start:start+count], rs=2)
            if not np.array_equal(pool[op['dst']:op['dst']+len(expected)], expected):
                raise ValueError(f'production packing mismatch: {suffix}, row {start}')


def prepare(out, rows, tokens, attn_build, projection_build):
    if tokens < 2 or rows < 2*tokens:
        raise ValueError('requires two multi-token sequences within the cache')
    validate_attention_build(attn_build,rows)
    validate_projection_build(projection_build)
    out.mkdir(parents=True, exist_ok=True)
    s, l = geometry(rows)
    rng = np.random.default_rng(38428)
    pool = np.memmap(out/'pool.bin', mode='w+', dtype=np.uint8, shape=(l.POOL_BYTES,))
    pack_weights(s, l, rng, pool)
    pool.flush()
    const = np.zeros(l.CA_BYTES, np.uint8)
    lnw = rng.uniform(.8,1.2,H).astype(bfloat16)
    postw = rng.uniform(.8,1.2,H).astype(bfloat16)
    norms = rng.uniform(.8,1.2,(2,256)).astype(bfloat16)
    for off, a in ((l.CA_LNW,lnw), (l.CA_POSTLN,postw), (l.CA_META,norms)):
        const[off:off+a.nbytes] = a.view(np.uint8).ravel()
    const.tofile(out/'const.bin')
    np.ones(H,bfloat16).tofile(out/'ones.bin')
    np.zeros(H,np.float32).tofile(out/'zero.bin')
    kernels = dict(ln=DESIGN/'wide_deltanet/build_layer/ln_precise',
                   qkvg=projection_build.resolve(), attn=attn_build.resolve(),
                   out=DESIGN/'wide_deltanet/build_layer/projection_k6144',
                   ffn=DESIGN/'layer_x/build_segmented_precise/ffn')
    cfg, artifacts = ['device'], {}
    for name, directory in kernels.items():
        for file in ('final.xclbin','insts.bin'): artifacts[str(directory/file)] = sha(directory/file)
        cfg += [f'xclbin {name} {directory}/final.xclbin', f'kernelx {name} {name} {directory}/insts.bin']
    cfg += [f'buf pool {l.POOL_BYTES} pool.bin', f'buf const {l.CA_BYTES} const.bin',
            f'buf ones {H*2} ones.bin', f'buf zero {H*4} zero.bin']
    for name, size in dict(x=H*4, cache=l.KV_BYTES+64, meta=l.E_A*2,
                           qkvgw=NPROJ*H//8192*5120, ow=H*QW//8192*5120,
                           lnw=H*2, postw=H*2).items():
        cfg.append(f'buf {name} {size}')
    sizes = outputs(l)
    for name, size in sizes.items():
        dtype = bfloat16 if name in BF16_OUTPUTS else np.float32
        (out/f'poison-{name}.bin').write_bytes(np.full(size//np.dtype(dtype).itemsize,np.nan,dtype).tobytes()+GUARD)
        cfg.append(f'buf {name} {size+64}')
    cfg += setup_commands(s,l)
    cases = []
    proj = lambda name,x: project(pool,l,name,x)
    for sequence, start in (('cold',0), ('warm',rows-tokens)):
        cache = np.full((rows,2,4,256),np.nan,bfloat16)
        cache[:start] = rng.normal(0,.6,(start,2,4,256)).astype(bfloat16)
        (out/f'{sequence}-init.bin').write_bytes(cache.tobytes()+GUARD)
        cfg.append(f'load cache {sequence}-init.bin')
        for t in range(tokens):
            tag, pos = f'{sequence}-{t}', start+t
            cases.append(dict(tag=tag,sequence=sequence,pos=pos))
            x = rng.normal(0,.6,H).astype(np.float32)
            x.tofile(out/f'{tag}-x.bin')
            angle = pos*s.rope_theta**(-np.arange(32)/32)
            cs = np.r_[np.cos(angle),np.sin(angle)].astype(np.float32)
            meta = np.zeros(l.E_A*2,np.uint8)
            meta[:norms.nbytes] = norms.view(np.uint8).ravel()
            meta[l.E_A:l.E_A+8] = np.array([pos,rows],np.int32).view(np.uint8)
            meta[l.E_A+512:l.E_A+512+cs.nbytes] = cs.view(np.uint8)
            meta.tofile(out/f'{tag}-meta.bin')
            xn = norm(x,lnw)
            qg = np.stack([proj('q',xn),proj('gate',xn)]).reshape(2,24,256)
            kvn = np.stack([proj('k',xn),proj('v',xn)]).reshape(2,4,256)
            new, og_float = decode(qg,kvn,norms,cs,cache,pos)
            og = og_float.astype(np.float32).astype(bfloat16)
            cache[pos] = new  # Oracle state only; never sent back to the device.
            projout = proj('out',og.ravel())
            res1 = (projout+x).astype(np.float32)
            xm = norm(res1,postw)
            up, gate = proj('up',xm),proj('fgate',xm)
            h = (up.astype(np.float64)*silu(gate)).astype(np.float32)
            fo = proj('down',h)
            refs = dict(xn=xn,qg=qg,kvn=kvn,new=new,og=og,projout=projout,
                        res1=res1,xm=xm,fo=fo,y=(fo+res1).astype(np.float32))
            np.savez(out/f'{tag}-ref.npz', **{k:v.astype(np.float32) for k,v in refs.items()})
            cfg += [f'load x {tag}-x.bin', f'load meta {tag}-meta.bin']
            cfg += [f'load {name} poison-{name}.bin' for name in sizes]
            cfg += token_commands(s,l,pos,rows)
            cfg += [f'dump {name} {tag}-got-{name}.bin {size+64}' for name,size in sizes.items()]
            cfg += [f'dump cache {tag}-got-cache.bin {l.KV_BYTES+64}']
            print('Reference:',tag,'position',pos,flush=True)
    cfg += ['load cache cold-init.bin','load x cold-0-x.bin','load meta cold-0-meta.bin']
    cfg += [f'load {name} poison-{name}.bin' for name in sizes]
    cfg += token_commands(s,l,0,rows)
    cfg += [f'dump y repeat-y.bin {H*4+64}', f'dump cache repeat-cache.bin {l.KV_BYTES+64}']
    for pattern in ('*-got-*.bin','repeat-*.bin'):
        for path in out.glob(pattern): path.unlink()
    (out/'layer.cfg').write_text('\n'.join(cfg)+'\n')
    files = [out/name for name in ('pool.bin','const.bin','ones.bin','zero.bin','cold-init.bin','warm-init.bin','layer.cfg')]
    files += [out/f'poison-{name}.bin' for name in sizes]
    files += [out/f"{c['tag']}-{suffix}" for c in cases for suffix in ('x.bin','meta.bin','ref.npz')]
    manifest = dict(seed=38428,rows=rows,tokens=tokens,cases=cases,kernels=artifacts,
                    outputs=sizes,fixtures={p.name:sha(p) for p in files},
                    contract='compare_ax.py intermediate gates; compare.py final 5e-3 gate; strict conditional GEMV/FFN')
    (out/'layer-fixture.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'layer-results.json').unlink(missing_ok=True)
    print(out/'layer.cfg')


def compare(out):
    meta = json.loads((out/'layer-fixture.json').read_text())
    s,l = geometry(meta['rows'])
    for name,digest in meta['kernels'].items():
        if sha(name)!=digest: raise ValueError(f'kernel changed: {name}')
    for name,digest in meta['fixtures'].items():
        if sha(out/name)!=digest: raise ValueError(f'fixture changed: {name}')
    checks = []
    def check(tag,name,g,r,tol,cos=.9999):
        result = metric(np.asarray(g).ravel(),np.asarray(r).ravel(),tol)
        result['passed'] = result['passed'] and result['cosine']>cos
        checks.append(dict(tag=tag,field=name,tolerance=tol,**result))
        if not name.startswith('head'):
            print('PASS' if result['passed'] else 'FAIL',tag,name,result,flush=True)
    def exact(tag,name,passed):
        checks.append(dict(tag=tag,field=name,passed=bool(passed)))
    def read(name,size):
        data = (out/name).read_bytes()
        if len(data)!=size+64 or data[size:]!=GUARD: raise ValueError(f'{name}: size/canary')
        return data[:size]
    pool = np.memmap(out/'pool.bin',mode='r',dtype=np.uint8)
    const = np.fromfile(out/'const.bin',np.uint8)
    proj = lambda name,x: project(pool,l,name,x)
    sequence, expected_cache = None, None
    for case in meta['cases']:
        tag,pos = case['tag'],case['pos']
        if sequence != case['sequence']:
            sequence = case['sequence']
            expected_cache = bytearray(read(f'{sequence}-init.bin',l.KV_BYTES))
        previous_cache = np.frombuffer(expected_cache,bfloat16).reshape(meta['rows'],2,4,256).copy()
        got = {name:np.frombuffer(read(f'{tag}-got-{name}.bin',size),
                bfloat16 if name in BF16_OUTPUTS else np.float32) for name,size in meta['outputs'].items()}
        ref = np.load(out/f'{tag}-ref.npz')
        for name,tol in (('xn',8e-3),('res1',1e-2),('xm',2e-2),('og',2e-2),('new',1e-2),('y',5e-3)):
            g = got[name][:H] if name in ('xn','xm') else got[name]
            check(tag,name,g,ref[name],tol)
        for h,(g,r) in enumerate(zip(got['og'].reshape(24,256),ref['og'])):
            check(tag,f'head{h}_og',g,r,2e-2)
        for h in range(4):
            check(tag,f'head{h}_k',got['new'].reshape(2,4,256)[0,h],ref['new'][0,h],1e-2)
            check(tag,f'head{h}_v',got['new'].reshape(2,4,256)[1,h],ref['new'][1,h],1e-2)
        expected_cache[pos*l.KV_ROW:(pos+1)*l.KV_ROW] = got['new'].tobytes()
        exact(tag,'cache_device_copy_and_untouched_rows',read(f'{tag}-got-cache.bin',l.KV_BYTES)==expected_cache)
        exact(tag,'v_exact_from_device_projection',np.array_equal(got['new'][KVW:].view(np.uint16),got['kvn'][KVW:].astype(bfloat16).view(np.uint16)))
        for bo,offset,size in (('xn',l.AA_XN,H*2),('qg',l.AA_QG,QW*8),('kvn',l.AA_KVN,KVW*8),
                ('og',l.AA_OG,QW*2),('projout',l.AA_OUT,H*4),('res1',l.AA_RES,H*4),
                ('xm',l.AA_XM,H*2),('fo',l.AA_OUT2,H*4)):
            exact(tag,f'{bo}_activation_copy',got['act'].view(np.uint8)[offset:offset+size].tobytes()==got[bo].tobytes()[:size])
        # Conditional checks isolate primitives after inference, without changing
        # the independent whole-layer reference or providing device feedback.
        for name,g in (('q',got['qg'][:QW]),('gate',got['qg'][QW:]),
                       ('k',got['kvn'][:KVW]),('v',got['kvn'][KVW:])):
            check(tag,'conditional_'+name,g,proj(name,got['xn'][:H]),1e-4,.9999999)
        metadata = np.fromfile(out/f'{tag}-meta.bin',np.uint8)
        norms = const[l.CA_META:l.CA_META+1024].view(bfloat16).reshape(2,256)
        cs = metadata[l.E_A+512:l.E_A+768].view(np.float32)
        new,og = decode(got['qg'].reshape(2,24,256),got['kvn'].reshape(2,4,256),norms,cs,previous_cache,pos)
        check(tag,'conditional_new',got['new'],new,1e-2)
        check(tag,'conditional_og',got['og'],og,2e-2)
        check(tag,'conditional_out',got['projout'],proj('out',got['og']),1e-4,.9999999)
        up,gate = proj('up',got['xm'][:H]),proj('fgate',got['xm'][:H])
        h = (up.astype(np.float64)*silu(gate)).astype(np.float32)
        check(tag,'conditional_ffn',got['fo'],proj('down',h),1e-4,.9999999)
        # The down input is retained in the FFN activation BO in f32.
        check(tag,'conditional_down',got['fo'],proj('down',got['ffnact'][l.A_H//4:l.A_H//4+F]),1e-4,.9999999)
    for name,size in (('y',H*4),('cache',l.KV_BYTES)):
        exact('repeat',name,read(f'repeat-{name}.bin',size)==read(f'cold-0-got-{name}.bin',size))
    passed = all(c['passed'] for c in checks)
    (out/'layer-results.json').write_text(json.dumps(dict(passed=passed,checks=checks),indent=2)+'\n')
    print('PASS' if passed else 'FAIL',len(checks),'checks')
    return 0 if passed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','compare'))
    p.add_argument('--out',type=Path,default=DESIGN/'attn/build_layer/acceptance')
    p.add_argument('--rows',type=int,default=257)
    p.add_argument('--tokens',type=int,default=4)
    p.add_argument('--attn-build',type=Path,default=DESIGN/'attn/build_wide_257')
    p.add_argument('--projection-build',type=Path,default=DESIGN/'attn/build_layer/projection_k5120')
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare(a.out,a.rows,a.tokens,a.attn_build,a.projection_build)
        return 0
    return compare(a.out)


if __name__ == '__main__':
    raise SystemExit(main())
