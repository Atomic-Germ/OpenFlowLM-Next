"""Replay refined L3 projections through current NPU attention and causal paired KV."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


refine=load('_attention_refinement',HERE.parent/'l3-projection-replay/refine.py')
build=load('_attention_refinement_build',HERE/'build.py')
base,digest,native=refine.base,refine.digest,refine.native
COUNT=240
ROLES=('control','candidate')


def sources():
    return refine.sources()|{str(p.resolve()):digest(p) for p in (Path(__file__),HERE/'build.py',HERE/'design.py')}


def schedule(count):
    if type(count) is not int or not 1<=count<=COUNT:raise ValueError('invalid causal extent')
    return [dict(position=p,cache_offset=p*8192,cache_bytes=max(1,p)*8192,
                 input_offset=p*61440,output_offset=p*24576) for p in range(count)]


def pack(projections,meta,ptab,position):
    schedule(position+1)
    if len(meta)!=2048 or len(ptab)!=2048 or tuple(np.frombuffer(ptab,'<i4')[:2])!=(position,max(1,position)):
        raise ValueError('invalid causal attention metadata')
    parts=[meta,ptab]
    if set(projections)!=set(refine.replay.ROLES):raise ValueError('incomplete projections')
    for role,heads in (('q',24),('k',4),('v',4),('gate',24)):
        a=np.asarray(projections[role])
        if a.shape!=(heads,256) or a.dtype!=np.dtype('<f4') or not np.isfinite(a).all():
            raise ValueError('invalid finite FP32 projection')
        parts.append(a.tobytes())
    return b''.join(parts)


def program(out,kernels):
    lines=['device',f'xclbin P {kernels}/final.xclbin',f'kernelx attention P {kernels}/insts.bin',
           'buf input 14745600','buf kv 1966080','buf og 5898240']
    for role in ROLES:
        lines += [f'load input {out}/{role}-input.bin',f'load kv {out}/zero.bin',
                  f'load og {out}/poison.bin','run attention input kv og',
                  f'dump kv {out}/{role}-kv.bin 1966080',f'dump og {out}/{role}-og.bin 5898240']
    return '\n'.join(lines)+'\n'


def validate_source(source):
    prep=refine.verified(source)
    r=json.loads((source/'report.json').read_text());a=json.loads((source/'audit.json').read_text())
    if (r.get('complete') is not True or a.get('complete') is not True or
        r.get('model_acceptance') is not False or a.get('model_acceptance') is not False or
        a['report_sha256']!=digest(source/'report.json') or r['preparation_sha256']!=digest(source/'preparation.json') or
        r['harness_sha256']!=digest(r['harness']) or r['harness_evidence_sha256']!=native.verify_harness(r['harness']) or
        r['log_sha256']!=digest(source/'native.log') or not a['candidate_passed'] or
        (a['control_roles'],a['exact_tables'],a['projection_cases'],a['artifacts'])!=(960,480,960,961)):
        raise ValueError('unbound refinement source')
    hashes={str(source/n):h for n,h in r['result']['artifacts'].items()}
    if set(r['result']['artifacts'])!=refine.names()|{'numerics.npz'}:raise ValueError('incomplete refinement inventory')
    base.verify(hashes)
    hashes.update(prep['sources']|prep['inputs']|prep['artifacts'])
    for n in ('preparation.json','report.json','audit.json','native.log'):hashes[str(source/n)]=digest(source/n)
    h=Path(r['harness']);hashes[str(h)]=digest(h);hashes[str(h.parent/'build-evidence.json')]=native.verify_harness(h)
    original=json.loads((Path(prep['source'])/'preparation.json').read_text())
    capture=Path(original['capture'])
    # Independently re-establish first-pass identity and the actual host FP32 sum.
    high,low,candidate,_=refine.captures(source,Path(prep['source']))
    with np.load(source/'numerics.npz',allow_pickle=False) as arrays:
        start=0
        for role,n in zip(refine.replay.ROLES,refine.replay.ROWS):
            for name,values in (('control',high),('correction',low),('candidate',candidate)):
                v=arrays[f'{role}_{name}'];wanted=values[:,start:start+n]
                if v.shape!=wanted.shape or v.dtype!=wanted.dtype or v.tobytes()!=wanted.tobytes():
                    raise ValueError('changed composed projection')
            start+=n
    return capture,hashes


def packed_inputs(source,capture):
    prep=json.loads((capture/'preparation.json').read_text());spec,_,_=native.context(prep['meta'])
    from recipes.qwen35 import layout
    lay=layout(spec,max_ctx=prep['meta']['max_ctx'])
    const=(capture/'consts.bin').read_bytes();meta=const[lay.CA_META:lay.CA_META+2048]
    ptab=(capture/'ptab.bin').read_bytes()
    with np.load(source/'numerics.npz',allow_pickle=False) as a:
        return {mode:b''.join(pack({role:a[f'{role}_{mode}'][p].reshape(-1,256) for role in refine.replay.ROLES},
                    meta,ptab[p*2048:(p+1)*2048],p) for p in range(COUNT)) for mode in ROLES}


def prepare(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,model_acceptance=False,sources=sources())
    try:
        source=a.source.resolve();kernels=a.build.resolve();b=build.verify(kernels)
        capture,inputs=validate_source(source)
        for mode,data in packed_inputs(source,capture).items():(out/f'{mode}-input.bin').write_bytes(data)
        (out/'zero.bin').write_bytes(bytes(COUNT*8192))
        np.full(COUNT*6144,np.nan,dtype='<f4').tofile(out/'poison.bin')
        (out/'run.cfg').write_text(program(out,kernels))
        artifacts={str(p):digest(p) for p in out.iterdir() if p.is_file()}
        inputs.update({str(kernels/'build.json'):digest(kernels/'build.json')})
        inputs.update({str(kernels/n):h for n,h in b['sha256'].items()})
        inputs.update(b['sources']|b['inputs'])
        base.verify(inputs);base.verify(r['sources'])
        r.update(complete=True,source=str(source),capture=str(capture),build=str(kernels),
                 inputs=inputs,artifacts=artifacts,schedule=schedule(COUNT))
    except Exception as e:r['error']=str(e);raise
    finally:native.write_report(out/'preparation.json',r)


def verified(out):
    p=json.loads((out/'preparation.json').read_text())
    if (p.get('complete') is not True or p.get('model_acceptance') is not False or
        p['sources']!=sources() or p['schedule']!=schedule(COUNT)):
        raise ValueError('incomplete or changed native attention preparation')
    for key in ('inputs','artifacts','sources'):base.verify(p[key])
    build.verify(Path(p['build']))
    if (out/'run.cfg').read_text()!=program(out,Path(p['build'])):raise ValueError('changed attention program')
    for mode,data in packed_inputs(Path(p['source']),Path(p['capture'])).items():
        if (out/f'{mode}-input.bin').read_bytes()!=data:raise ValueError('changed packed projections')
    return p


def read_output(path):
    return refine.read(path,COUNT*6144).reshape(COUNT,24,256)


def controls(got,expected,kv,capture):
    if (got.shape!=(COUNT,24,256) or expected.shape!=got.shape or got.dtype!=np.dtype('<f4') or
        expected.dtype!=got.dtype or not np.isfinite(got).all() or not np.isfinite(expected).all() or
        len(kv)!=COUNT*8192):raise ValueError('invalid native attention control geometry')
    exact=sum(a.tobytes()==b.tobytes() for a,b in zip(got,expected))
    prefix=sum((capture/f'state_{p}.bin').read_bytes()==kv[:(p+1)*8192] for p in range(COUNT))
    if exact!=COUNT or prefix!=COUNT:raise ValueError(f'failed native control: outputs {exact}/240, KV {prefix}/240')
    return dict(outputs=exact,kv_prefixes=prefix)


def measure(control,candidate,primary,conditioned):
    arrays=[np.asarray(v) for v in (control,candidate,primary,conditioned)]
    if any(v.shape!=(240,24,256) or not np.isfinite(v).all() for v in arrays):raise ValueError('invalid attention coverage')
    checks={name:refine.heads.compare_heads(got,ref) for name,got,ref in (
        ('control_primary',control,primary),('candidate_primary',candidate,primary),
        ('candidate_conditioned',candidate,conditioned))}
    summary={k:dict(total=len(v),passed=sum(r['passed'] for r in v),maxabs=max(r['maxabs'] for r in v),
                    min_cosine=min(r['cosine'] for r in v)) for k,v in checks.items()}
    selected=[]
    for p in (125,205,239):
        old=float(np.max(abs(control[p]-primary[p])));new=float(np.max(abs(candidate[p]-primary[p])))
        selected.append(dict(position=p,control_maxabs=old,candidate_maxabs=new,improved=new<old))
    return dict(candidate_passed=all(summary[k]['passed']==5760 for k in ('candidate_primary','candidate_conditioned')),
                summary=summary,selected=selected,heads=checks)


def evaluate(out,prep):
    names={f'{r}-{k}.bin' for r in ROLES for k in ('og','kv')}
    if {p.name for pattern in ('*-og.bin','*-kv.bin') for p in out.glob(pattern)}!=names:
        raise ValueError('incomplete or extra attention capture inventory')
    output={r:read_output(out/f'{r}-og.bin') for r in ROLES}
    cache={r:(out/f'{r}-kv.bin').read_bytes() for r in ROLES}
    from q4nx import bf16_to_f32
    for data in cache.values():
        if len(data)!=COUNT*8192 or not np.isfinite(bf16_to_f32(np.frombuffer(data,'<u2'))).all():
            raise ValueError('invalid paired KV capture')
    capture=Path(prep['capture']);meta=json.loads((capture/'preparation.json').read_text())['meta']
    spec,_,_=native.context(meta)
    from recipes.qwen35 import layout
    lay=layout(spec,max_ctx=meta['max_ctx'])
    expected=np.array([refine.heads.read_attention(capture/f'act_{p}.bin',lay,24,4,256,'float32')['og']
                       for p in range(COUNT)],dtype='<f4')
    exact=controls(output['control'],expected,cache['control'],capture)
    with np.load(Path(prep['source'])/'numerics.npz',allow_pickle=False) as a:
        result=measure(output['control'],output['candidate'],a['attention_primary'],a['attention_candidate'])
    result.update(controls=exact,artifacts={n:digest(out/n) for n in sorted(names)},
                  scope='isolated production NPU attention and causal paired KV on two-pass projections with host FP32 sum; no full-layer/model acceptance')
    return result


def run(a):
    out=a.out.resolve()
    if (out/'report.json').exists() or any(out.glob('*-og.bin')) or any(out.glob('*-kv.bin')):
        raise ValueError('use fresh preparation; do not overwrite native captures')
    r=dict(complete=False,model_acceptance=False)
    try:
        prep=verified(out);h=a.harness.resolve()
        r.update(preparation_sha256=digest(out/'preparation.json'),harness=str(h),harness_sha256=digest(h),
                 harness_evidence_sha256=native.verify_harness(h))
        with native.drift.drift.boundary.runner.device_lock(),(out/'native.log').open('w') as log:
            subprocess.run([str(h),str(out/'run.cfg')],stdout=log,stderr=subprocess.STDOUT,
                           env=dict(os.environ,HARNESS_TIMEOUT_MS='30000'),timeout=1800,check=True)
        result=evaluate(out,prep);verified(out)
        if digest(h)!=r['harness_sha256'] or native.verify_harness(h)!=r['harness_evidence_sha256']:
            raise ValueError('harness changed during attention run')
        r.update(complete=True,result=result,log_sha256=digest(out/'native.log'))
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
            r['log_sha256']!=digest(out/'native.log')):raise ValueError('unbound native attention run')
        result=evaluate(out,prep)
        if result!=r['result']:raise ValueError('changed attention measurements')
        verified(out)
        proof.update(complete=True,report_sha256=digest(out/'report.json'),candidate_passed=result['candidate_passed'],
                     controls=result['controls'],artifacts=len(result['artifacts']))
    except Exception as e:proof['error']=str(e);raise
    finally:native.write_report(out/'audit.json',proof)
    return 0 if proof['candidate_passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','audit'])
    for name in ('out','source','build','harness'):p.add_argument('--'+name,type=Path,required=name=='out')
    a=p.parse_args()
    if a.command=='prepare' and (a.source is None or a.build is None):p.error('--source and --build required')
    if a.command=='run' and a.harness is None:p.error('--harness required')
    raise SystemExit(globals()[a.command](a))
