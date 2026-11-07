"""Private full-lx post-output replay with exact causal controls and independent gates."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

builder=load('_fp32_post_build',HERE/'build.py')
boundary=load('_fp32_post_boundary',HERE.parent/'prefill-sequential-drift/l0_boundaries.py')
native=boundary.native;base=native.base;digest=base.digest
COUNT=native.COUNT


def program(out,capture,build,sizes):
    lines=['device',f'xclbin lx {build}/final.xclbin',f'kernelx lx lx {build}/insts.bin',
           f'buf pool {sizes["pool"]} {capture}/pool.bin',f'buf consts {sizes["consts"]} {capture}/consts.bin',
           f'buf xres {sizes["hidden"]*4}',f'buf act {sizes["act"]}',f'buf state {sizes["state"]}']
    for p in range(COUNT):
        lines += [f'load xres {capture}/input_{p}.bin','run lx pool xres consts state act']
        lines += [f'dump {k} {out}/{k}_{p}.bin {n}' for k,n in
                  (('xres',sizes['hidden']*4),('act',sizes['act']),('state',sizes['state']))]
    return '\n'.join(lines)+'\n'


def equal_region(a,b,offset,size):
    if offset<0 or size<=0 or offset+size>min(len(a),len(b)):raise ValueError('incomplete boundary')
    return a[offset:offset+size]==b[offset:offset+size]


def accept(mode,rows):
    builder.mode_check(mode)
    if len(rows)!=COUNT or [r['position'] for r in rows]!=list(range(COUNT)):raise ValueError('incomplete/reordered causal coverage')
    return all(r['control_exact'] for r in rows) if mode=='control' else all(r['upstream_exact'] and r['residual']['passed'] for r in rows)


def sources():
    return boundary.sources() | {str(p):digest(p) for p in (Path(__file__).resolve(),HERE/'build.py')}


def validate(capture,build):
    prep=native.verified(capture);r=json.loads((capture/'report.json').read_text());a=json.loads((capture/'audit.json').read_text())
    boundary.require_control(r,a,digest(capture/'report.json'))
    identity,hashes=native.score(capture,prep)
    if identity!=r['identity'] or hashes!=r['artifacts']:raise ValueError('retained native control changed')
    b=builder.verify(build)
    resource=json.loads((build/'resources.json').read_text())
    if not resource['passed'] or resource['build']!=json.loads((build/'build.json').read_text()):raise ValueError('missing resource acceptance')
    base.verify(resource['evidence_sha256'])
    if b['production_build_key']!=prep['meta']['build_key']:raise ValueError('build and capture export mismatch')
    inputs=prep['inputs'] | prep['artifacts'] | {str(capture/n):h for n,h in hashes.items()}
    for p in (capture/'preparation.json',capture/'report.json',capture/'audit.json',build/'build.json',build/'resources.json'):
        inputs[str(p)]=digest(p)
    return prep,b,inputs


def controls(directory):
    r=json.loads((directory/'report.json').read_text());a=json.loads((directory/'audit.json').read_text())
    if (not r['complete'] or not r['passed'] or r['mode']!='control' or not a['complete'] or not a['passed']
            or a['report_sha256']!=digest(directory/'report.json')):raise ValueError('fresh audited exact control required')
    base.verify(r['sources']);base.verify(r['inputs'])
    base.verify({str(directory/n):h for n,h in r['artifacts'].items()})
    return r


def measurements(out,prep,b,capture):
    sizes=dict(prep['sizes'],act=b['geometry']['act_bytes'])
    got,hashes=native.read_outputs(out,sizes,COUNT)
    paths=[Path(p) for p in prep['inputs'] if p.endswith('/257/ref_res0.bin')]
    if len(paths)!=1:raise ValueError('ambiguous reference')
    ref=native.drift.trace.read_trace(paths[0],257,sizes['hidden'])[:COUNT]
    spec,_,_=native.context(prep['meta']);off=boundary.layout(spec,max_ctx=prep['meta']['max_ctx'])
    regions=[(off.A_XN,spec.hidden*4),(off.A_QKV,spec.lin_qkv_dim*4),
             (off.A_Z,spec.lin_value_width*4),(off.A_VEC,spec.lin_value_heads*boundary.ds.RECORD_FLOATS*4),
             (off.A_O,spec.lin_value_width*4)]
    rows=[]
    for p in range(COUNT):
        a=(out/f'act_{p}.bin').read_bytes();old=(capture/f'act_{p}.bin').read_bytes()
        state_exact=digest(out/f'state_{p}.bin')==digest(capture/f'state_{p}.bin')
        exact=state_exact and a==old and digest(out/f'xres_{p}.bin')==digest(capture/f'xres_{p}.bin')
        rows.append(dict(position=p,upstream_exact=state_exact and all(equal_region(a,old,o,n) for o,n in regions),
                         control_exact=exact,residual=boundary.front.metric(got[p],ref[p])))
    selected=[]
    if b['mode']=='fp32':
        model=base.Q4NX(Path(prep['meta']['model'])/'model.q4nx');model.native_q8=lambda n:False
        w=boundary.Weights(model);prefix='model.layers.0.'
        try:
            for p in boundary.TARGETS:
                data=(out/f'act_{p}.bin').read_bytes();legacy=data[:b['geometry']['legacy_act_bytes']]
                proxy=SimpleNamespace(read_bytes=lambda:legacy,name=f'act_{p}.bin')
                c=boundary.ds.read_boundaries(legacy,(out/f'state_{p}.bin').read_bytes(),spec,off,conv_dtype='fp32')
                og=np.frombuffer(data,'<f4',count=spec.lin_value_width,offset=b['geometry']['post_offset']).astype(np.float64)
                z=np.frombuffer(legacy,'<f4',count=spec.lin_value_width,offset=off.A_Z).astype(np.float64)
                projected=np.frombuffer(legacy,'<f4',count=spec.hidden,offset=off.A_OUT).astype(np.float64)
                post=(base.oracle.rms(c['output'].astype(np.float64),spec.norm_eps)*w.bf16(prefix+'linear_attn.ssm_norm.weight')).reshape(-1)*base.oracle.silu(z)
                f=boundary.front.read_front(proxy,off,spec.hidden,spec.intermediate,False,norm_dtype='float32')
                ff=boundary.front.compare_front(f['res'],w.bf16(prefix+'post_attention_layernorm.weight'),
                    w.matmul_w(prefix+'mlp.up_proj.weight',spec.intermediate,spec.hidden),
                    w.matmul_w(prefix+'mlp.gate_proj.weight',spec.intermediate,spec.hidden),f['xm'],f['h'],spec.norm_eps)
                d=boundary.down.read_regions(proxy,off,spec.hidden,spec.intermediate,False)
                dd=boundary.down.compare_partials(d['h'],w.matmul_w(prefix+'mlp.down_proj.weight',spec.hidden,spec.intermediate),
                    d['res'],d['partials'],got[p],tuple(boundary.ffn_geometry(spec).DOWN_SPLIT))
                row=dict(position=p,post=boundary.front.metric(og,post),
                         out=boundary.front.metric(projected,w.matmul_w(prefix+'linear_attn.ssm_out_proj.weight',spec.hidden,spec.lin_value_width).astype(np.float64)@og),front=ff,down=dd)
                row['passed']=all(row[k]['passed'] for k in ('post','out','front','down'));selected.append(row)
        finally:w.cache.clear();model.mm.close();model.f.close()
    return dict(passed=accept(b['mode'],rows) and all(r['passed'] for r in selected),cases=rows,selected=selected,artifacts=hashes)


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);r=dict(complete=False,model_acceptance=False,sources=sources())
    try:
        capture=a.capture.resolve();build=a.build.resolve();prep,b,inputs=validate(capture,build)
        if b['mode']=='fp32':
            if a.control is None:raise ValueError('--control required for candidate')
            c=controls(a.control.resolve())
            if c['capture']!=str(capture):raise ValueError('control capture differs')
            for n in ('report.json','audit.json'):inputs[str(a.control.resolve()/n)]=digest(a.control/n)
            r['control']=str(a.control.resolve())
        harness=a.harness.resolve();inputs[str(harness)]=digest(harness)
        cfg=out/'run.cfg';cfg.write_text(program(out,capture,build,dict(prep['sizes'],act=b['geometry']['act_bytes'])))
        inputs[str(cfg)]=digest(cfg);r.update(inputs=inputs,mode=b['mode'],capture=str(capture),build=str(build),command=[str(harness),str(cfg)])
        with native.drift.boundary.runner.device_lock(),(out/'native.log').open('w') as log:
            subprocess.run(r['command'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,HARNESS_TIMEOUT_MS='30000'),check=True,timeout=1800)
        measured=measurements(out,prep,b,capture);base.verify(inputs);base.verify(r['sources']);builder.verify(build)
        r.update(complete=True,**measured,log_sha256=digest(out/'native.log'))
    except Exception as error:r['error']=str(error);raise
    finally:native.write_report(out/'report.json',r)
    return 0 if r['passed'] else 1


def audit(a):
    out=a.out.resolve();result=dict(complete=False,model_acceptance=False)
    try:
        r=json.loads((out/'report.json').read_text())
        if not r['complete'] or r['model_acceptance'] is not False or r['sources']!=sources():raise ValueError('incomplete/changed experiment')
        base.verify(r['inputs']);base.verify(r['sources'])
        if digest(out/'native.log')!=r['log_sha256']:raise ValueError('native log changed')
        prep,b,_=validate(Path(r['capture']),Path(r['build']))
        if b['mode']!=r['mode']:raise ValueError('mode mismatch')
        if b['mode']=='fp32':controls(Path(r['control']))
        measured=measurements(out,prep,b,Path(r['capture']))
        if any(r[k]!=v for k,v in measured.items()):raise ValueError('experiment re-audit differs')
        base.verify(r['inputs']);base.verify(r['sources'])
        result.update(complete=True,passed=measured['passed'],report_sha256=digest(out/'report.json'))
    except Exception as error:result['error']=str(error);raise
    finally:native.write_report(out/'audit.json',result)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','audit']);p.add_argument('--out',type=Path,required=True)
    for name in ('capture','build','harness','control'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.command=='run' and any(getattr(a,k) is None for k in ('capture','build','harness')):p.error('--capture, --build and --harness required')
    raise SystemExit(globals()[a.command](a))
