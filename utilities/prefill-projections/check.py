"""Real-weight, captured-input GEMM prefill diagnostics (not full-model prefill)."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'open_kernels'))
sys.path.insert(0,str(ROOT/'open_kernels/model'))
from q4nx import Q4NX, f32_to_bf16, bf16_to_f32
from q4_1_pack import dequant_pool
from recipes.load import load_spec
from recipes.qwen35 import layout
from recipes.manifest import manifest
from recipes.pack import apply_op
from decode_bundle import load_bundle
from decode_reference import load_reference, sfx

SLOTS=((0,127,255),(31,128,254))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def tile_input(x):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or x.shape[0]!=256 or x.shape[1]%256 or not np.isfinite(x).all():
        raise ValueError('finite [256,K] input with K divisible by 256 required')
    k=x.shape[1]
    b=f32_to_bf16(x).T.reshape(k//64,64,8,32).transpose(0,2,1,3)
    return np.ascontiguousarray(b.reshape(k//64,8,8,8,4,8).transpose(0,1,2,4,3,5)).reshape(-1)


def metric(got,ref):
    error=got-ref
    norm=float(np.linalg.norm(ref)); peak=float(np.max(np.abs(ref)))
    fro=float(np.linalg.norm(error))/max(norm,1e-30)
    rel=float(np.max(np.abs(error)))/max(peak,1e-30)
    return dict(rel_fro=fro,maxrel=rel,passed=fro<=.005 and rel<.005)


def score(got,raw,rounded,slots,roles):
    got,raw,rounded=[np.asarray(a,dtype=np.float64) for a in (got,raw,rounded)]
    slots=list(slots)
    if (raw.ndim!=2 or not raw.size or got.shape!=(raw.shape[0],256) or rounded.shape!=raw.shape
            or len(slots)!=raw.shape[1] or len(set(slots))!=len(slots)
            or any(type(s)!=int or not 0<=s<256 for s in slots)
            or not all(np.isfinite(a).all() for a in (got,raw,rounded))):
        raise ValueError('complete finite arrays and distinct valid token slots required')
    coverage=np.zeros(raw.shape[0],np.int32)
    for lo,hi in roles.values():
        if not 0<=lo<hi<=raw.shape[0]:raise ValueError('invalid role range')
        coverage[lo:hi]+=1
    if not np.all(coverage==1):raise ValueError('roles must partition all rows')
    zero=bool(np.all(np.delete(got,slots,axis=1)==0))
    results={}
    for name,(lo,hi) in roles.items():
        results[name]={kind:[metric(got[lo:hi,s],ref[lo:hi,t]) for t,s in enumerate(slots)]
                       for kind,ref in [('fp32_input',raw),('bf16_input',rounded)]}
    return dict(passed=zero and all(v['passed'] for r in results.values() for group in r.values() for v in group),
                zero_columns=zero,roles=results)


def verify(hashes):
    for p,h in hashes.items():
        if digest(p)!=h:raise ValueError(f'provenance mismatch: {p}')


def activation_region(data,offsets,prefix,region,count):
    # Linear gated output is still BF16; attention gated output is FP32.
    dtype='<u2' if prefix=='A_' and region=='OG' else '<f4'
    off=getattr(offsets,prefix+region);size=np.dtype(dtype).itemsize
    if (len(data)!=getattr(offsets,prefix+'BYTES') or off%size or count<=0
            or off+size*count>len(data)
            or (region=='OG' and off+size*count>getattr(offsets,prefix+'OUT'))):
        raise ValueError('bad capture bounds')
    values=np.frombuffer(data,dtype,count=count,offset=off)
    return (bf16_to_f32(values) if size==2 else values).astype(np.float64)


def cases(m):
    result=[]
    for layer,lt in [(0,'linear_attention'),(3,'full_attention')]:
        typ=m['layer_types'][lt];gb=typ['gemm_block']
        steps=gb['program']+gb['ffn_program']
        weights=gb['weights']|gb['ffn_weights']
        if gb['t']!=256 or len(steps)!=4:raise ValueError('requires wide dense 256-token route')
        for label,region,step in zip(('entry','out','up_gate','down'),('XN','OG','XM','H'),steps):
            shape=re.fullmatch(r'gemm_n(\d+)_k(\d+)',step['kernel'])
            if not shape or step.get('split'):raise ValueError('requires unsplit Q4 GEMM')
            n,k=map(int,shape.groups());w=weights[step['args'][0]]
            ops=[typ['pack'][w['from']][i] for i in w['ops']]
            if any(o['op']!='std_perm' or o.get('split') or o['in_dim']!=k for o in ops):
                raise ValueError('unsupported weight packing')
            if sum(o['nch'] for o in ops)*8192!=n*k:raise ValueError('wrong weight geometry')
            names=(['qkv','z'] if lt=='linear_attention' else ['q','k','v','gate']) if label=='entry' else (['up','gate'] if label=='up_gate' else [label])
            roles={};row=0
            for name,op in zip(names,ops):
                rows=op['nch']*8192//k;roles[name]=[row,row+rows];row+=rows
            if label=='entry' and lt=='linear_attention':
                kw=gb['key_heads']*gb['head_dim'];qkv=gb['qkv_dim']
                if roles['qkv']!=[0,qkv]:raise ValueError('wrong linear QKV rows')
                roles={'q':[0,kw],'k':[kw,2*kw],'v':[2*kw,qkv],'z':roles['z']}
            result.append(dict(tag=f'L{layer}_{label}',layer=layer,lt=lt,region=region,
                               kernel=step['kernel'],n=n,k=k,roles=roles,ops=ops,storage=w['from']))
    return result


def prepare(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    for name in ('audit.json','report.json'):(out/name).unlink(missing_ok=True)
    spec=load_spec(a.kernels/'spec.json');meta=load_reference(a.fixture)
    m=load_bundle(a.kernels,manifest(spec,max_ctx=meta['max_ctx']))
    if (meta['tokens']!=3 or meta['layers']!=64 or any(meta[k]!=m[k] for k in ('build_key','spec_hash'))
            or any(m['layout'].get(k)!='float32' for k in ('linear_entry_dtype','attn_entry_dtype','ffn_norm_dtype','attn_output_dtype'))):
        raise ValueError('requires complete FP32 production fixture')
    inputs={}
    original=json.loads(a.original.read_text())['sha256']
    def pin(p,prior=False):
        p=p.resolve();h=digest(p)
        if prior and original.get(str(p))!=h:raise ValueError(f'original provenance mismatch: {p}')
        inputs[str(p)]=h
    pin(a.model/'model.q4nx',True)
    for l in (0,3):
        pin(a.pools/f'pool_L{l}.bin',True)
        for t in range(3):pin(a.fixture/f'y_act{l}{sfx(t)}.bin',True)
    for p in [a.original,a.fixture/'decode_reference.json',a.kernels/'spec.json',a.kernels/'manifest.json',a.kernels/'toolchain.json']:
        pin(p)
    offsets=layout(spec,max_ctx=meta['max_ctx']);model=Q4NX(a.model/'model.q4nx')
    entries=cases(m);artifacts={};commands=[]
    def save(name,arr):
        path=out/name;path.write_bytes(np.ascontiguousarray(arr).tobytes());artifacts[str(path)]=digest(path)
        return path
    for c in entries:
        tag=c['tag'];n,k=c['n'],c['k'];prefix='AA_' if c['lt']=='full_attention' else 'A_'
        acts=[]
        for t in range(3):
            p=a.fixture/f"y_act{c['layer']}{sfx(t)}.bin";data=p.read_bytes()
            acts.append(activation_region(data,offsets,prefix,c['region'],k))
        x=np.stack(acts)
        if not np.isfinite(x).all():raise ValueError('non-finite activation')
        pool=np.zeros(sum(o['nch']*5120 for o in c['ops']),np.uint8);dst=0
        source=a.pools/f"pool_L{c['layer']}.bin" if c['storage']=='pool' else a.fixture/f"consts_{c['layer']}.bin"
        pin(source,c['storage']=='pool')
        with source.open('rb') as f:
            for op in c['ops']:
                size=op['nch']*5120
                apply_op(dict(op,dst=dst),model,c['layer'],pool)
                f.seek(op['dst'])
                if pool[dst:dst+size].tobytes()!=f.read(size):raise ValueError('packed bytes differ from production')
                dst+=size
        save(tag+'-w.bin',pool)
        w=dequant_pool(pool,n,k,rs=2)
        raw=np.empty((n,3),np.float64);rounded=np.empty_like(raw)
        xb=bf16_to_f32(f32_to_bf16(x.astype(np.float32))).astype(np.float64)
        for lo in range(0,n,256):
            ww=w[lo:lo+256].astype(np.float64)
            raw[lo:lo+256]=ww@x.T;rounded[lo:lo+256]=ww@xb.T
        del w,ww,pool
        save(tag+'-fp32-ref.bin',raw);save(tag+'-bf16-ref.bin',rounded);save(tag+'-input.bin',x.astype('<f4'))
        for variant,slots in enumerate(SLOTS):
            full=np.zeros((256,k),np.float32);full[list(slots)]=x
            save(f'{tag}-x{variant}.bin',tile_input(full))
            build=c['kernel']
            for filename in ('final.xclbin','insts.bin'):pin(a.kernels/build/filename)
            # One process per shape: only one weight, input and output BO is resident.
            lines=['device',f'xclbin G {a.kernels.resolve()}/{build}/final.xclbin',
                   f'kernelx k G {a.kernels.resolve()}/{build}/insts.bin',
                   f'buf w {n*k*5//8} {out}/{tag}-w.bin',f'buf x {k*256*2} {out}/{tag}-x{variant}.bin',
                   f'buf y {n*256*4}','run k w x y',f'dump y {out}/{tag}-y{variant}.bin {n*256*4}']
            cfg=out/f'{tag}-{variant}.cfg';cfg.write_text('\n'.join(lines)+'\n');artifacts[str(cfg)]=digest(cfg)
            commands.append(str(cfg))
        print(f'prepared {tag}: {n}x{k}, roles={list(c["roles"])}',flush=True)
    sources=[Path(__file__),ROOT/'open_kernels/q4_1_pack.py',ROOT/'open_kernels/model/q4nx.py',ROOT/'open_kernels/model/decode_bundle.py',ROOT/'open_kernels/model/decode_reference.py']
    sources+=list((ROOT/'open_kernels/recipes').glob('*.py'))
    for p in sources:pin(p)
    audit=dict(scope='conditioned projections, layers 0/3, three captured decode inputs in two slot layouts',
               build_key=m['build_key'],spec_hash=m['spec_hash'],cases=entries,slots=SLOTS,
               commands=commands,inputs=inputs,artifacts=artifacts)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')


def run(a):
    out=a.out.resolve();report_path=out/'report.json';report_path.unlink(missing_ok=True)
    report=dict(passed=False,cases=[])
    try:
        audit=json.loads((out/'audit.json').read_text());verify(audit['inputs']);verify(audit['artifacts'])
        report.update(scope=audit['scope'],audit_sha256=digest(out/'audit.json'),harness_sha256=digest(a.harness))
        # Delete all expected captures before any dispatch; a failed run cannot score old data.
        for c in audit['cases']:
            for i in range(len(SLOTS)):(out/f'{c["tag"]}-y{i}.bin').unlink(missing_ok=True)
        for cfg in audit['commands']:
            log=Path(cfg).with_suffix('.log')
            with log.open('w') as f:
                subprocess.run([str(a.harness.resolve()),cfg],stdout=f,stderr=subprocess.STDOUT,check=True,
                               timeout=120,env=os.environ|{'HARNESS_TIMEOUT_MS':'30000'})
        for c in audit['cases']:
            tag,n=c['tag'],c['n']
            raw=np.fromfile(out/f'{tag}-fp32-ref.bin','<f8').reshape(n,3)
            rounded=np.fromfile(out/f'{tag}-bf16-ref.bin','<f8').reshape(n,3)
            ys=[]
            for i,slots in enumerate(SLOTS):
                path=out/f'{tag}-y{i}.bin'
                if path.stat().st_size!=n*256*4:raise ValueError('partial output dump')
                y=np.fromfile(path,'<f4').reshape(n,256);ys.append(y[:,list(slots)])
                result=score(y,raw,rounded,list(slots),c['roles'])
                report['cases'].append(dict(tag=tag,variant=i,output_sha256=digest(path),**result))
                print(f"{tag}/{i}: passed={result['passed']} worst maxrel="
                      f"{max(v['maxrel'] for r in result['roles'].values() for group in r.values() for v in group):.8g}",flush=True)
            identical=ys[0].tobytes()==ys[1].tobytes()
            for item in report['cases'][-2:]:item['slot_identity']=identical;item['passed'] &= identical
        verify(audit['artifacts'])
        report['passed']=len(report['cases'])==16 and all(c['passed'] for c in report['cases'])
    except Exception as e:
        report['error']=str(e)
        raise
    finally:report_path.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 1


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--out',type=Path,required=True)
    for name in ('model','kernels','fixture','pools','original'):p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--out',type=Path,required=True);p.add_argument('--harness',type=Path,required=True)
    a=ap.parse_args()
    return prepare(a) if a.command=='prepare' else run(a)


if __name__=='__main__':raise SystemExit(main())
