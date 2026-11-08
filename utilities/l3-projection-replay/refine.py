"""Two NPU projection passes with explicit host FP32 addition and CPU causal attention."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np

s=importlib.util.spec_from_file_location('_l3_refinement_attribution',Path(__file__).with_name('attribute.py'))
attr=importlib.util.module_from_spec(s);s.loader.exec_module(attr)
replay=attr.replay
base,digest,native=replay.base,replay.digest,replay.native
heads=replay.boundary.attention
metric=replay.boundary.front.metric
COUNT=240


def sources():
    return attr.sources()|{str(Path(__file__).resolve()):digest(__file__)}


def read(path,n):
    if path.stat().st_size!=n*4:raise ValueError('invalid FP32 capture size: '+path.name)
    a=np.fromfile(path,'<f4')
    if not np.isfinite(a).all():raise ValueError('non-finite FP32 capture: '+path.name)
    return a


def residual(x,data):
    if not replay.table_identity(x,data):raise ValueError('residual requires exact input table')
    high,_=attr.table(data,5120);exact=x.astype(np.float64)-high;low=exact.astype('<f4')
    if not np.isfinite(low).all() or not np.array_equal(low.astype(np.float64),exact):
        raise ValueError('quantization residual is not exactly representable in FP32')
    return low


def combine(high,low):
    high,low=np.asarray(high),np.asarray(low)
    if (high.shape!=low.shape or high.ndim!=1 or not high.size or high.dtype!=np.dtype('<f4') or
            low.dtype!=high.dtype or not np.isfinite(high).all() or not np.isfinite(low).all()):
        raise ValueError('equal finite FP32 projection vectors required')
    result=np.add(high,low,dtype=np.float32)
    if not np.isfinite(result).all():raise ValueError('non-finite composed projection')
    return result


def compare(control,candidate,reference):
    old,new=metric(control,reference),metric(candidate,reference)
    return dict(control=old,candidate=new,improved=new['maxabs']<old['maxabs'])


def attention(projections,qn,kn,spec):
    q,g,k,v=[np.asarray(projections[n],np.float64) for n in replay.ROLES]
    return heads.attention_reference(q.reshape(-1,spec.num_heads,spec.head_dim),
        k.reshape(-1,spec.num_kv_heads,spec.head_dim),v.reshape(-1,spec.num_kv_heads,spec.head_dim),
        g.reshape(-1,spec.num_heads,spec.head_dim),qn,kn,spec.norm_eps,spec.rotary_dim,spec.rope_inv_freq())


def program(out,source,kernels):
    lines=['device',f'xclbin P {kernels}/final.xclbin',f'kernelx proj P {kernels}/insts.bin',
           f'buf w 45875200 {source}/weights.bin','buf x 20480','buf y 57344','buf tab 11520']
    for p in range(COUNT):
        for kind,path in (('hi',source/f'x{p}.bin'),('lo',out/f'low{p}.bin')):
            lines += [f'load x {path}',f'load y {source}/poison-y.bin',f'load tab {source}/poison-tab.bin',
                      'run proj w x y tab',f'dump y {out}/{kind}-y{p}.bin 57344',
                      f'dump tab {out}/{kind}-table{p}.bin 11520']
    return '\n'.join(lines)+'\n'


def names():
    return {f'{kind}-{field}{p}.bin' for kind in ('hi','lo') for field in ('y','table') for p in range(COUNT)}


def require_controls(exact_roles,exact_tables):
    if exact_roles!=960 or exact_tables!=480:raise ValueError('failed first-pass or table controls; no candidate interpretation')


def prepare(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,model_acceptance=False,sources=sources())
    try:
        source=a.source.resolve();inputs=attr.validate(source)
        old=json.loads((source/'preparation.json').read_text());build=Path(old['build'])
        for p in range(COUNT):
            residual(read(source/f'x{p}.bin',5120),(source/f'table{p}.bin').read_bytes()).tofile(out/f'low{p}.bin')
        (out/'run.cfg').write_text(program(out,source,build))
        artifacts={str(p):digest(p) for p in out.iterdir() if p.is_file()}
        base.verify(inputs);base.verify(r['sources'])
        r.update(complete=True,source=str(source),build=str(build),inputs=inputs,artifacts=artifacts,
                 scope='reuse audited binary for two passes; exact FP32 residual; no new build or production changes')
    except Exception as error:r['error']=str(error);raise
    finally:native.write_report(out/'preparation.json',r)


def verified(out):
    r=json.loads((out/'preparation.json').read_text())
    if r.get('complete') is not True or r.get('model_acceptance') is not False or r['sources']!=sources():
        raise ValueError('incomplete or changed refinement preparation')
    for key in ('sources','inputs','artifacts'):base.verify(r[key])
    replay.build.verify(Path(r['build']))
    source=Path(r['source'])
    if (out/'run.cfg').read_text()!=program(out,source,Path(r['build'])):raise ValueError('changed refinement program')
    for p in range(COUNT):
        wanted=residual(read(source/f'x{p}.bin',5120),(source/f'table{p}.bin').read_bytes())
        if read(out/f'low{p}.bin',5120).tobytes()!=wanted.tobytes():raise ValueError('changed refinement input')
    return r


def captures(out,source):
    found={p.name for pattern in ('hi-*.bin','lo-*.bin') for p in out.glob(pattern)}
    if found!=names():raise ValueError('incomplete or extra refinement capture inventory')
    high=[];low=[];combined=[];roles=0;tables=0;hashes={}
    for p in range(COUNT):
        h,l=[read(out/f'{kind}-y{p}.bin',14336) for kind in ('hi','lo')]
        roles+=sum(replay.identity(h,read(source/f'y{p}.bin',14336)).values())
        ht=(out/f'hi-table{p}.bin').read_bytes();lt=(out/f'lo-table{p}.bin').read_bytes()
        tables+=int(ht==(source/f'table{p}.bin').read_bytes())
        tables+=int(replay.table_identity(read(out/f'low{p}.bin',5120),lt))
        high.append(h);low.append(l);combined.append(combine(h,l))
    require_controls(roles,tables)
    for name in sorted(names()):hashes[name]=digest(out/name)
    return np.array(high),np.array(low),np.array(combined),hashes


def evaluate(out,prep,write):
    source=Path(prep['source']);high,low,combined,artifacts=captures(out,source)
    old=json.loads((source/'preparation.json').read_text());capture=Path(old['capture'])
    original=json.loads((capture/'preparation.json').read_text());spec,_,_=native.context(original['meta'])
    model=base.Q4NX(Path(original['meta']['model'])/'model.q4nx')
    try:
        norm=model.bf16('model.layers.3.input_layernorm.weight').astype(float)
        qn,kn=[model.bf16(f'model.layers.3.self_attn.{k}_norm.weight').astype(float) for k in ('q','k')]
    finally:model.mm.close();model.f.close()
    x=np.array([read(source/f'x{p}.bin',5120) for p in range(COUNT)],dtype=np.float64)
    residuals=np.array([read(capture/f'input_{p}.bin',5120) for p in range(COUNT)],dtype=np.float64)
    primary_x=residuals/np.sqrt(np.mean(residuals*residuals,axis=1,keepdims=True)+spec.norm_eps)*norm
    lx=np.array([read(out/f'low{p}.bin',5120) for p in range(COUNT)],dtype=np.float64)
    trajectories={k:{} for k in ('control','candidate','conditioned','primary')}
    arrays={};rows=[];start=0
    with (source/'weights.bin').open('rb') as pool:
        for role,n in zip(replay.ROLES,replay.ROWS):
            print('Scoring two-pass',role,flush=True)
            w=attr.weights(pool.read(n//32*20*5120),n,5120)
            ref=x@w.T;primary=primary_x@w.T;correction=lx@w.T
            h,l,c=high[:,start:start+n],low[:,start:start+n],combined[:,start:start+n]
            trajectories['control'][role]=h;trajectories['candidate'][role]=c
            trajectories['conditioned'][role]=ref;trajectories['primary'][role]=primary
            arrays.update({f'{role}_{k}':v for k,v in dict(control=h,correction=l,candidate=c,
                conditioned=ref,primary=primary,correction_reference=correction).items()})
            for p in range(COUNT):
                rows.append(dict(role=role,position=p,conditioned=compare(h[p],c[p],ref[p]),
                                 primary=compare(h[p],c[p],primary[p]),correction=metric(l[p],correction[p])))
            start+=n;del w
        if pool.read(1):raise ValueError('extra weights')
    attr.coverage(rows)
    print('Scoring independent causal CPU attention trajectories',flush=True)
    attn={k:attention(v,qn,kn,spec) for k,v in trajectories.items()}
    checks={f'{candidate}_vs_{ref}':heads.compare_heads(attn[candidate],attn[ref])
            for candidate in ('control','candidate') for ref in ('conditioned','primary')}
    arrays.update({'attention_'+k:v for k,v in attn.items()})
    path=out/'numerics.npz'
    if write:np.savez(path,**arrays)
    else:
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays):raise ValueError('changed refinement numeric inventory')
            for k,v in arrays.items():
                if saved[k].shape!=v.shape or saved[k].dtype!=v.dtype or saved[k].tobytes()!=v.tobytes():
                    raise ValueError('refinement numeric replay differs: '+k)
    artifacts[path.name]=digest(path)
    improved=sum(r['conditioned']['improved'] for r in rows)
    primary_passed=sum(r['primary']['candidate']['passed'] for r in rows)
    correction_passed=sum(r['correction']['passed'] for r in rows)
    attention_summary={k:dict(total=len(v),passed=sum(x['passed'] for x in v),
        maxabs=max(x['maxabs'] for x in v),min_cosine=min(x['cosine'] for x in v)) for k,v in checks.items()}
    passed=(improved==960 and primary_passed==960 and correction_passed==960 and
            all(v['passed']==5760 for k,v in attention_summary.items() if k.startswith('candidate')))
    return dict(candidate_passed=passed,control_roles=960,exact_tables=480,conditioned_improved=improved,
        primary_projection_passed=primary_passed,correction_passed=correction_passed,projection_cases=960,
        attention_summary=attention_summary,projections=rows,attention_heads=checks,artifacts=artifacts,
        selected=[r for r in rows if r['position'] in (125,205,239)],
        scope='two NPU projections; one host FP32 addition; CPU FP64 causal attention, not native attention or full-layer/model acceptance')


def run(a):
    out=a.out.resolve()
    if (out/'report.json').exists() or any((out/n).exists() for n in names()):
        raise ValueError('use fresh preparation; never overwrite refinement captures')
    r=dict(complete=False,model_acceptance=False)
    try:
        prep=verified(out);h=a.harness.resolve()
        r.update(preparation_sha256=digest(out/'preparation.json'),harness=str(h),harness_sha256=digest(h),
                 harness_evidence_sha256=native.verify_harness(h))
        env=os.environ.copy();env['HARNESS_TIMEOUT_MS']='30000'
        with native.drift.drift.boundary.runner.device_lock(),(out/'native.log').open('w') as log:
            subprocess.run([str(h),str(out/'run.cfg')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=1800)
        result=evaluate(out,prep,True);verified(out)
        if digest(h)!=r['harness_sha256'] or native.verify_harness(h)!=r['harness_evidence_sha256']:
            raise ValueError('harness changed during execution')
        r.update(complete=True,result=result,log_sha256=digest(out/'native.log'))
    except Exception as error:r['error']=str(error);raise
    finally:native.write_report(out/'report.json',r)
    return 0 if result['candidate_passed'] else 1


def audit(a):
    out=a.out.resolve();proof=dict(complete=False,model_acceptance=False)
    try:
        prep=verified(out);r=json.loads((out/'report.json').read_text())
        if (r.get('complete') is not True or r.get('model_acceptance') is not False or
                r['preparation_sha256']!=digest(out/'preparation.json') or r['harness_sha256']!=digest(r['harness']) or
                r['harness_evidence_sha256']!=native.verify_harness(r['harness']) or
                r['log_sha256']!=digest(out/'native.log')):raise ValueError('unbound refinement run')
        result=evaluate(out,prep,False)
        if result!=r['result']:raise ValueError('refinement measurements changed')
        verified(out)
        proof.update(complete=True,report_sha256=digest(out/'report.json'),candidate_passed=result['candidate_passed'],
                     control_roles=960,exact_tables=480,projection_cases=960,artifacts=len(result['artifacts']))
    except Exception as error:proof['error']=str(error);raise
    finally:native.write_report(out/'audit.json',proof)
    return 0 if proof['candidate_passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','audit'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--harness',type=Path)
    a=p.parse_args()
    if a.command=='prepare' and a.source is None:p.error('--source required')
    if a.command=='run' and a.harness is None:p.error('--harness required')
    raise SystemExit(globals()[a.command](a))
