"""Validate private integrated L3 refinement against exact staged 240-position evidence."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np
HERE=Path(__file__).resolve().parent


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


tail=load('_integrated_tail',HERE.parent/'l3-tail-replay/sweep.py')
build=load('_integrated_build',HERE/'build.py')
base,native,digest=tail.base,tail.native,tail.digest
ROLES=tail.ROLES


def sources():
    return tail.sources()|{str(p.resolve()):digest(p) for p in [Path(__file__),HERE/'build.py',HERE/'worker.py.txt',*HERE.glob('*.cc'),*HERE.glob('*.h')]}


def baseline(directory):
    prep=tail.verified(directory)
    r=json.loads((directory/'report.json').read_text());a=json.loads((directory/'audit.json').read_text())
    if (r.get('complete') is not True or a.get('complete') is not True or
        r.get('model_acceptance') is not False or a.get('model_acceptance') is not False or
        a['report_sha256']!=digest(directory/'report.json') or r['preparation_sha256']!=digest(directory/'preparation.json') or
        not a['candidate_passed'] or not r['result']['candidate_passed'] or
        (a['exact_controls'],a['artifacts'],a['overlap_arrays'])!=(720,1680,39) or
        r['harness_sha256']!=digest(r['harness']) or r['harness_evidence_sha256']!=native.verify_harness(r['harness']) or
        r['log_sha256']!=digest(directory/'native.log') or a['summary']!=r['result']['summary']):
        raise ValueError('requires complete bound full-prefix baseline')
    tail.inventory(directory)
    if set(r['result']['artifacts'])!=tail.names()|{f'P{p}.npz' for p in tail.TARGETS}:
        raise ValueError('incomplete baseline artifacts')
    inputs=prep['sources']|prep['inputs']|prep['artifacts']
    inputs.update({str(directory/n):h for n,h in r['result']['artifacts'].items()})
    inputs.update({str(directory/n):digest(directory/n) for n in ('preparation.json','report.json','audit.json','native.log')})
    base.verify(inputs)
    return prep,inputs


def program(out,capture,kernels,s,role):
    if role not in ROLES:raise ValueError('invalid role')
    lines=['device',f'attngeom {s["kv_row"]} {s["ptab_row"]}',
           f'xclbin ax {kernels}/final.xclbin',f'kernelx ax ax {kernels}/insts.bin',
           f'buf pool {s["pool"]} {capture}/pool.bin',f'buf consts {s["consts"]} {capture}/consts.bin',
           f'buf xres {s["hidden"]*4}',f'buf act {s["act"]}',f'buf state {s["state"]}',
           f'buf ptab {s["ptab"]} {capture}/ptab.bin',f'load state {out}/zero-state.bin']
    for p in tail.TARGETS:
        act=capture/f'act_{p-1}.bin' if p else out/'zero-act.bin'
        lines += [f'load act {act}',f'load xres {capture}/input_{p}.bin',f'attnpos ax {p}',
                  'run ax pool xres consts state act ptab']
        lines += [f'dump {k} {out}/{role}-{k}_{p}.bin {n}' for k,n in
                  (('act',s['act']),('xres',s['hidden']*4),('state',(p+1)*s['kv_row']))]
    return '\n'.join(lines)+'\n'


def expected_act(data,projections,og,off):
    if len(data)!=off.AA_BYTES or set(projections)!=set(tail.refine.replay.ROLES):raise ValueError('invalid scratch/projections')
    result=bytearray(data)
    regions=[(projections['q'],6144,off.AA_QG),(projections['gate'],6144,off.AA_QG+24576),
             (projections['k'],1024,off.AA_KVN),(projections['v'],1024,off.AA_KVN+4096),(og,6144,off.AA_OG)]
    for v,n,start in regions:
        if v.shape!=(n,) or v.dtype!=np.dtype('<f4') or not np.isfinite(v).all() or start<0 or start+n*4>len(data):
            raise ValueError('invalid finite FP32 scratch region')
        result[start:start+n*4]=v.tobytes()
    return bytes(result)


def prepare(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);r=dict(complete=False,model_acceptance=False,sources=sources())
    try:
        old,inputs=baseline(a.baseline.resolve());builds={}
        for role in ROLES:
            path=getattr(a,role+'_build').resolve();b=build.verify(path)
            if b['role']!=role:raise ValueError('mismatched control/candidate build role')
            inputs.update(b['sources']|b['inputs']);inputs[str(path/'build.json')]=digest(path/'build.json')
            inputs.update({str(path/n):h for n,h in b['sha256'].items()});builds[role]=str(path)
        # Both binaries must derive the same standard geometry as the retained run.
        old_build=json.loads((Path(old['build'])/'build.json').read_text())
        for path in builds.values():
            b=json.loads((Path(path)/'build.json').read_text())
            if b['build_key']!=old_build['build_key']:raise ValueError('incompatible recipe build key')
        s=old['sizes'];capture=Path(old['capture'])
        (out/'zero-state.bin').write_bytes(bytes(s['state']));(out/'zero-act.bin').write_bytes(bytes(s['act']))
        for role in ROLES:(out/f'{role}.cfg').write_text(program(out,capture,Path(builds[role]),s,role))
        artifacts={str(p):digest(p) for p in out.iterdir() if p.is_file()}
        base.verify(inputs);base.verify(r['sources'])
        r.update(complete=True,baseline=str(a.baseline.resolve()),capture=str(capture),attention=old['source'],
                 refinement=old['refinement'],meta=old['meta'],sizes=s,builds=builds,inputs=inputs,artifacts=artifacts)
    except Exception as e:r['error']=str(e);raise
    finally:native.write_report(out/'preparation.json',r)


def verified(out):
    r=json.loads((out/'preparation.json').read_text())
    if r.get('complete') is not True or r.get('model_acceptance') is not False or r['sources']!=sources():
        raise ValueError('incomplete or changed integrated preparation')
    for key in ('inputs','artifacts','sources'):base.verify(r[key])
    _,_,sizes=native.context(r['meta'])
    if sizes!=r['sizes']:raise ValueError('changed geometry')
    for role in ROLES:
        b=build.verify(Path(r['builds'][role]))
        if b['role']!=role or (out/f'{role}.cfg').read_text()!=program(out,Path(r['capture']),Path(r['builds'][role]),sizes,role):
            raise ValueError('changed build role/program')
    if (out/'zero-state.bin').read_bytes()!=bytes(sizes['state']) or (out/'zero-act.bin').read_bytes()!=bytes(sizes['act']):
        raise ValueError('changed zero buffers')
    return r


def evaluate(out,prep):
    tail.inventory(out);capture=Path(prep['capture']);exact=tail.controls(out,capture)
    spec,_,_=native.context(prep['meta']);off=tail.boundary.layout(spec,max_ctx=prep['meta']['max_ctx'])
    old=Path(prep['baseline']);attention=Path(prep['attention']);refinement=Path(prep['refinement'])
    saved=json.loads((old/'report.json').read_text())['result'];rows=[];observed_og=[];control_og=[]
    kv=(attention/'candidate-kv.bin').read_bytes();og=tail.attn.read_output(attention/'candidate-og.bin')
    with np.load(refinement/'numerics.npz',allow_pickle=False) as nums:
        for p in tail.TARGETS:
            c=out/f'candidate-act_{p}.bin'
            projections={k:nums[f'{k}_candidate'][p] for k in tail.refine.replay.ROLES}
            wanted=expected_act((old/f'candidate-act_{p}.bin').read_bytes(),projections,og[p].ravel(),off)
            if c.read_bytes()!=wanted:raise ValueError(f'integrated candidate scratch differs at P{p}')
            if (out/f'candidate-state_{p}.bin').read_bytes()!=kv[:(p+1)*8192]:
                raise ValueError(f'integrated candidate causal KV differs at P{p}')
            if (out/f'candidate-xres_{p}.bin').read_bytes()!=(old/f'candidate-xres_{p}.bin').read_bytes():
                raise ValueError(f'integrated candidate residual differs at P{p}')
            got=base.read(out/f'candidate-xres_{p}.bin',5120);control=base.read(out/f'control-xres_{p}.bin',5120)
            with np.load(old/f'P{p}.npz',allow_pickle=False) as a:
                row=dict(position=p,primary=tail.compare(control,got,a['primary']),
                         model_reference=tail.compare(control,got,a['model_reference']),
                         conditioned=tail.boundary.front.metric(got,a['conditioned']))
                previous=saved['cases'][p]
                if any(row[k]!=previous[k] for k in ('primary','model_reference','conditioned')):
                    raise ValueError('retained residual measurements changed')
                # Complete output/FFN scratch is byte-exact to the independently audited sweep.
                row['passed']=previous['passed'];rows.append(row)
            observed_og.append(tail.boundary.attention.read_attention(c,off,24,4,256,'float32')['og'])
            control_og.append(tail.boundary.attention.read_attention(out/f'control-act_{p}.bin',off,24,4,256,'float32')['og'])
        heads=tail.attn.measure(np.asarray(control_og),np.asarray(observed_og),nums['attention_primary'],nums['attention_candidate'])
    summary=tail.summarize(rows)
    if summary!=saved['summary']:raise ValueError('retained regression summary changed')
    return dict(candidate_passed=summary['passed']==240 and heads['candidate_passed'],exact_controls=exact,
                exact_candidate=dict(projections=960,scratch=240,kv_prefixes=240,residuals=240),summary=summary,cases=rows,
                attention=heads,artifacts={n:digest(out/n) for n in sorted(tail.names())},
                scope='private integrated NPU refinement; own causal KV; original layer-input residuals; no cross-layer/model or performance acceptance')


def run(a):
    out=a.out.resolve()
    if (out/'report.json').exists() or any((out/n).exists() for n in tail.names()):raise ValueError('do not overwrite integrated captures')
    r=dict(complete=False,model_acceptance=False)
    try:
        prep=verified(out);h=a.harness.resolve()
        r.update(preparation_sha256=digest(out/'preparation.json'),harness=str(h),harness_sha256=digest(h),harness_evidence_sha256=native.verify_harness(h))
        for role in ROLES:
            with native.drift.drift.boundary.runner.device_lock(),(out/f'{role}.log').open('w') as log:
                subprocess.run([str(h),str(out/f'{role}.cfg')],stdout=log,stderr=subprocess.STDOUT,
                               env=dict(os.environ,HARNESS_TIMEOUT_MS='30000'),check=True,timeout=1800)
        result=evaluate(out,prep);verified(out)
        if digest(h)!=r['harness_sha256'] or native.verify_harness(h)!=r['harness_evidence_sha256']:raise ValueError('harness changed')
        r.update(complete=True,result=result,logs={role:digest(out/f'{role}.log') for role in ROLES})
    except Exception as e:r['error']=str(e);raise
    finally:native.write_report(out/'report.json',r)
    return 0 if result['candidate_passed'] else 1


def audit(a):
    out=a.out.resolve();proof=dict(complete=False,model_acceptance=False)
    try:
        prep=verified(out);r=json.loads((out/'report.json').read_text())
        if (r.get('complete') is not True or r.get('model_acceptance') is not False or
            r['preparation_sha256']!=digest(out/'preparation.json') or r['harness_sha256']!=digest(r['harness']) or
            r['harness_evidence_sha256']!=native.verify_harness(r['harness']) or
            r['logs']!={role:digest(out/f'{role}.log') for role in ROLES}):raise ValueError('unbound integrated run')
        result=evaluate(out,prep)
        if result!=r['result']:raise ValueError('changed integrated measurements')
        verified(out)
        proof.update(complete=True,report_sha256=digest(out/'report.json'),candidate_passed=result['candidate_passed'],
                     exact_controls=result['exact_controls'],exact_candidate=result['exact_candidate'],artifacts=len(result['artifacts']),summary=result['summary'])
    except Exception as e:proof['error']=str(e);raise
    finally:native.write_report(out/'audit.json',proof)
    return 0 if proof['candidate_passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','audit'])
    for n in ('out','baseline','control-build','candidate-build','harness'):p.add_argument('--'+n,type=Path,required=n=='out')
    a=p.parse_args()
    if a.command=='prepare' and any(getattr(a,n) is None for n in ('baseline','control_build','candidate_build')):p.error('baseline and both builds required')
    if a.command=='run' and a.harness is None:p.error('--harness required')
    raise SystemExit(globals()[a.command](a))
