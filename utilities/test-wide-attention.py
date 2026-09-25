#!/usr/bin/env python3
"""Prepare/compare exact-geometry gated attention, including device-carried KV state."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'open_kernels'))
sys.path.insert(0,str(ROOT/'utilities'))
from recipes.spec import ModelSpec
from recipes.qwen36moe import attn
from wide_attention_reference import decode, metric

GUARD = bytes([0xA5])*64


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def prepare(out, fixture, rows):
    if rows < 8:
        raise ValueError('at least eight cache rows are needed for the warm chain')
    spec = ModelSpec.from_hf_config(json.loads((ROOT/'specs/open-engine/tests/fixtures'/fixture).read_text()))
    d = attn(spec)
    nh,kvh,hd = d.NH,d.KVH,d.HD
    rng = np.random.default_rng(2964)
    norms = rng.uniform(.8,1.2,(2,hd)).astype(bfloat16)
    new_bytes, og_bytes, cache_bytes = 2*d.E_A, 2*d.QW, rows*2*d.E_A
    (out/'poison-new.bin').write_bytes(np.full(2*d.KVW,np.nan,bfloat16).tobytes()+GUARD)
    (out/'poison-og.bin').write_bytes(np.full(d.QW,np.nan,bfloat16).tobytes()+GUARD)
    cfg = ['device','xclbin a final.xclbin','kernelx a a insts.bin',
           f'buf meta {2*d.E_A}',f'buf qg {8*d.QW}',f'buf kvn {8*d.KVW}',
           f'buf cache {cache_bytes+len(GUARD)}',f'buf new {new_bytes+len(GUARD)}',
           f'buf og {og_bytes+len(GUARD)}']
    cases, files = [], ['final.xclbin','insts.bin','poison-new.bin','poison-og.bin']

    def case(name,pos,qg,kvn,cache,load_cache=True):
        i = len(cases)
        angle = pos*spec.rope_theta**(-np.arange(d.ROT//2)/(d.ROT//2))
        cs = np.r_[np.cos(angle),np.sin(angle)].astype(np.float32)
        meta = np.zeros(2*d.E_A,np.uint8)
        meta[:norms.nbytes] = norms.view(np.uint8).reshape(-1)
        meta[d.E_A:d.E_A+8] = np.array([pos,rows],np.int32).view(np.uint8)
        meta[d.E_A+512:d.E_A+512+cs.nbytes] = cs.view(np.uint8)
        new,og = decode(qg,kvn,norms,cs,cache,pos)
        for prefix,data in [('meta',meta),('qg',qg),('kvn',kvn),('ref-new',new),('ref-og',og.astype(np.float32))]:
            file=f'{prefix}{i}.bin'
            data.tofile(out/file)
            files.append(file)
        if load_cache:
            file=f'cache{i}.bin'
            (out/file).write_bytes(cache.tobytes()+GUARD)
            files.append(file)
            cfg.append(f'load cache {file}')
        cfg.extend([f'load meta meta{i}.bin',f'load qg qg{i}.bin',f'load kvn kvn{i}.bin',
                    'load new poison-new.bin','load og poison-og.bin', 'run a meta qg kvn cache new og',
                    f'dump new new{i}.bin {new_bytes+len(GUARD)}',f'dump og og{i}.bin {og_bytes+len(GUARD)}'])
        cases.append(dict(name=name,pos=pos))
        for prefix in ('new','og'): (out/f'{prefix}{i}.bin').unlink(missing_ok=True)
        return new

    def inputs():
        return (rng.normal(0,.6,(2,nh,hd)).astype(np.float32),
                rng.normal(0,.6,(2,kvh,hd)).astype(np.float32))
    first = None
    for pos in sorted({0,1,2,15,16,17,63,64,255,256,1023,1024,2047,rows}):
        if pos>rows: continue
        qg,kvn = inputs()
        cache = rng.normal(0,.6,(rows,2,kvh,hd)).astype(bfloat16)
        cache[pos:] = np.nan
        case(f'cold{pos}',pos,qg,kvn,cache)
        if first is None: first = (pos,qg.copy(),kvn.copy(),cache.copy())
    # Per-KV values distinguish the four groups, including heads crossing core boundaries.
    qg,kvn = inputs()
    qg[0] = 0
    qg[1] = 0
    kvn[0] = 0
    kvn[1] = np.arange(1,kvh+1)[:,None]
    cache = np.full((rows,2,kvh,hd),np.nan,bfloat16)
    case('groups',0,qg,kvn,cache)
    qg,kvn = inputs()
    qg[1,::2] = -20
    qg[1,1::2] = 20
    case('saturated_gate',0,qg,kvn,cache)
    qg,kvn = inputs()
    kvn[1] = 0
    case('zero_value',0,qg,kvn,cache)
    warm_start = len(cases)
    (out/'cache-warm.bin').write_bytes(cache.tobytes()+GUARD)
    files.append('cache-warm.bin')
    cfg.append('load cache cache-warm.bin')
    for pos in range(8):
        qg,kvn = inputs()
        new = case(f'warm{pos}',pos,qg,kvn,cache,False)
        cache[pos] = new  # Independent reference only; never fed to the device.
        cfg.append(f'copy cache {pos*new_bytes} new 0 {new_bytes}')
    cfg.append(f'dump cache warm-cache.bin {cache_bytes+len(GUARD)}')
    (out/'warm-cache.bin').unlink(missing_ok=True)
    case('repeat_first',*first)
    (out/'attention.cfg').write_text('\n'.join(cfg)+'\n')
    files.append('attention.cfg')
    meta = dict(fixture=fixture,rows=rows,nh=nh,kvh=kvh,hd=hd,rot=d.ROT,cores=d.ACORES,
                seed=2964,cases=cases,warm_start=warm_start,
                sha256={f:digest(out/f) for f in files})
    (out/'attention-fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'attention-results.json').unlink(missing_ok=True)
    print(f'Prepared {len(cases)} cases, Q{nh}/KV{kvh}/HD{hd}/ROT{d.ROT}, nf={rows}')


def compare(out):
    meta=json.loads((out/'attention-fixture.json').read_text())
    for f,sha in meta['sha256'].items():
        if digest(out/f)!=sha: raise ValueError(f'{f} changed since fixture generation')
    nh,kvh,hd=meta['nh'],meta['kvh'],meta['hd']
    new_bytes=4*kvh*hd
    results=[]
    for i,case in enumerate(meta['cases']):
        got={}
        for name,size,shape in [('new',new_bytes,(2,kvh,hd)),('og',2*nh*hd,(nh,hd))]:
            raw=(out/f'{name}{i}.bin').read_bytes()
            if len(raw)!=size+len(GUARD) or raw[size:]!=GUARD:
                raise ValueError(f'{case["name"]}/{name}: size or canary')
            got[name]=np.frombuffer(raw[:size],bfloat16).reshape(shape)
        refnew=np.fromfile(out/f'ref-new{i}.bin',bfloat16).reshape(2,kvh,hd)
        refog=np.fromfile(out/f'ref-og{i}.bin',np.float32).reshape(nh,hd)
        checks={name:metric(g,r,tol) for name,g,r,tol in
                [('k',got['new'][0],refnew[0],1e-2),('v',got['new'][1],refnew[1],1e-2),
                 ('og',got['og'],refog,2e-2)]}
        heads=[metric(g,r,2e-2) for g,r in zip(got['og'],refog)]
        # V is a direct fp32->bf16 conversion, so additionally require exact bytes.
        v_exact=bool(np.array_equal(got['new'][1].view(np.uint16),refnew[1].view(np.uint16)))
        passed=v_exact and all(c['passed'] for c in checks.values()) and all(h['passed'] for h in heads)
        results.append(dict(**case,passed=passed,v_exact=v_exact,checks=checks,heads=heads))
        print(f"{'PASS' if passed else 'FAIL'} {case['name']}: {checks['og']}")
    warm=(out/'warm-cache.bin').read_bytes()
    expected=b''.join((out/f'new{i}.bin').read_bytes()[:new_bytes]
                      for i in range(meta['warm_start'],meta['warm_start']+8))
    initial=(out/'cache-warm.bin').read_bytes()
    cache_exact=warm==expected+initial[8*new_bytes:]
    repeat=all((out/f'{name}0.bin').read_bytes()==(out/f'{name}{len(results)-1}.bin').read_bytes() for name in ('new','og'))
    passed=repeat and cache_exact and all(r['passed'] for r in results)
    (out/'attention-results.json').write_text(json.dumps(dict(passed=passed,repeat_exact=repeat,
                                                             cache_copy_exact=cache_exact,results=results),indent=2)+'\n')
    return 0 if passed else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','compare'))
    p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--fixture',default='config_qwen38_27b.json')
    p.add_argument('--rows',type=int,default=257)
    a=p.parse_args()
    if a.stage=='prepare':
        prepare(a.build_dir,a.fixture,a.rows)
        return 0
    return compare(a.build_dir)


if __name__=='__main__':
    raise SystemExit(main())
