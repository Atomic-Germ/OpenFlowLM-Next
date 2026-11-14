"""Force audited OG into unchanged full-ax output/FFN stages at three causal positions."""
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
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


attn=load('_tail_attention',HERE.parent/'l3-attention-replay/replay.py')
build=load('_tail_build',HERE/'build.py')
refine=attn.refine
base,digest,native=attn.base,attn.digest,attn.native
boundary=refine.replay.boundary
TARGETS=(125,205,239)
ROLES=('control','candidate')


def sources():
    return attn.sources()|{str(p.resolve()):digest(p) for p in (Path(__file__),HERE/'build.py')}


def forced_input(og):
    if og.shape!=(6144,) or og.dtype!=np.dtype('<f4') or not np.isfinite(og).all():
        raise ValueError('invalid FP32 OG')
    return og.tobytes()


def names():return {f'{r}-{k}_{p}.bin' for r in ROLES for k in ('act','state','xres') for p in TARGETS}


def inventory(out):
    actual={p.name for role in ROLES for kind in ('act','state','xres') for p in out.glob(f'{role}-{kind}_*.bin')}
    if actual!=names():raise ValueError('incomplete or extra tail capture inventory')


def controls(out,capture):
    exact=sum((out/f'control-{k}_{p}.bin').read_bytes()==(capture/f'{k}_{p}.bin').read_bytes()
              for p in TARGETS for k in ('act','state','xres'))
    if exact!=9:raise ValueError(f'failed native full-layer control: {exact}/9')
    return exact


def program(out,capture,kernels,s):
    lines=['device',f'attngeom {s["kv_row"]} {s["ptab_row"]}',
           f'xclbin ax {kernels}/final.xclbin',f'kernelx ax ax {kernels}/insts.bin',
           f'buf pool {s["pool"]} {capture}/pool.bin',f'buf consts {s["consts"]} {capture}/consts.bin',
           f'buf xres {s["hidden"]*4}',f'buf act {s["act"]}',f'buf state {s["state"]}',f'buf ptab {s["ptab"]} {capture}/ptab.bin','buf forced 24576']
    for role in ROLES:
        for p in TARGETS:
            lines += [f'load state {capture}/state_{p-1}.bin',f'load act {capture}/act_{p-1}.bin',
                      f'load xres {capture}/input_{p}.bin',f'load forced {out}/{role}-og_{p}.bin',f'attnpos ax {p}',
                      'run ax pool xres consts state act ptab forced']
            lines += [f'dump {k} {out}/{role}-{k}_{p}.bin {n}' for k,n in
                      (('act',s['act']),('xres',s['hidden']*4),('state',(p+1)*s['kv_row']))]
    return '\n'.join(lines)+'\n'


def validate(source):
    prep=attn.verified(source);r=json.loads((source/'report.json').read_text());a=json.loads((source/'audit.json').read_text())
    if (not r.get('complete') or not a.get('complete') or a['report_sha256']!=digest(source/'report.json') or
        r['model_acceptance'] is not False or a['model_acceptance'] is not False or
        r['preparation_sha256']!=digest(source/'preparation.json') or r['harness_sha256']!=digest(r['harness']) or
        r['harness_evidence_sha256']!=native.verify_harness(r['harness']) or r['log_sha256']!=digest(source/'native.log') or
        a['controls']!=dict(outputs=240,kv_prefixes=240) or a['artifacts']!=4 or not a['candidate_passed']):
        raise ValueError('unbound native attention audit')
    result=attn.evaluate(source,prep)
    if result!=r['result'] or not result['candidate_passed']:raise ValueError('changed native attention measurements')
    inputs=prep['inputs']|prep['artifacts']|prep['sources']
    inputs.update({str(source/n):digest(source/n) for n in ('preparation.json','report.json','audit.json','native.log')})
    inputs.update({str(source/n):h for n,h in result['artifacts'].items()})
    h=Path(r['harness']);inputs[str(h)]=digest(h);inputs[str(h.parent/'build-evidence.json')]=native.verify_harness(h)
    return prep,inputs


def prepared_inputs(source):
    tables={}
    for role in ROLES:
        og=attn.read_output(source/f'{role}-og.bin').reshape(240,6144)
        for p in TARGETS:tables[f'{role}-og_{p}.bin']=forced_input(og[p])
    return tables


def prepare(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);r=dict(complete=False,model_acceptance=False,sources=sources())
    try:
        source=a.source.resolve();kernels=a.build.resolve();ap,inputs=validate(source);b=build.verify(kernels)
        capture=Path(ap['capture']);original=json.loads((capture/'preparation.json').read_text());spec,_,sizes=native.context(original['meta'])
        for n,data in prepared_inputs(source).items():(out/n).write_bytes(data)
        (out/'run.cfg').write_text(program(out,capture,kernels,sizes))
        artifacts={str(p):digest(p) for p in out.iterdir() if p.is_file()}
        inputs.update(b['sources']|b['inputs']);inputs[str(kernels/'build.json')]=digest(kernels/'build.json')
        inputs.update({str(kernels/n):h for n,h in b['sha256'].items()})
        base.verify(inputs);base.verify(r['sources'])
        r.update(complete=True,source=str(source),capture=str(capture),refinement=ap['source'],build=str(kernels),sizes=sizes,
                 meta=original['meta'],inputs=inputs,artifacts=artifacts,positions=list(TARGETS))
    except Exception as e:r['error']=str(e);raise
    finally:native.write_report(out/'preparation.json',r)


def verified(out):
    r=json.loads((out/'preparation.json').read_text())
    if r.get('complete') is not True or r.get('model_acceptance') is not False or r['sources']!=sources() or r['positions']!=list(TARGETS):
        raise ValueError('incomplete or changed tail preparation')
    for k in ('inputs','artifacts','sources'):base.verify(r[k])
    build.verify(Path(r['build']))
    spec,_,sizes=native.context(r['meta'])
    if sizes!=r['sizes'] or (out/'run.cfg').read_text()!=program(out,Path(r['capture']),Path(r['build']),sizes):
        raise ValueError('changed tail geometry/program')
    for n,data in prepared_inputs(Path(r['source'])).items():
        if (out/n).read_bytes()!=data:raise ValueError('changed forced-OG input')
    return r


def compare(control,candidate,reference):
    a,b=[boundary.front.metric(np.asarray(v,dtype=np.float64),np.asarray(reference,dtype=np.float64)) for v in (control,candidate)]
    return dict(control=a,candidate=b,improved=b['maxabs']<a['maxabs'])


def evaluate(out,prep,write):
    inventory(out);capture=Path(prep['capture']);exact=controls(out,capture)
    spec,_,_=native.context(prep['meta']);off=boundary.layout(spec,max_ctx=prep['meta']['max_ctx'])
    model=base.Q4NX(Path(prep['meta']['model'])/'model.q4nx');model.native_q8=lambda name:False
    prefix='model.layers.3.';rows=[];artifacts={n:digest(out/n) for n in sorted(names())}
    try:
        w={k:model.matmul_w(prefix+n,o,i).astype(np.float64) for k,n,o,i in (
            ('out','self_attn.o_proj.weight',5120,6144),('up','mlp.up_proj.weight',17408,5120),
            ('gate','mlp.gate_proj.weight',17408,5120),('down','mlp.down_proj.weight',5120,17408))}
        norm=model.bf16(prefix+'post_attention_layernorm.weight').astype(np.float64)
        def tail(res):
            xn=base.oracle.rms(res,spec.norm_eps)*norm
            return res+w['down']@boundary.front.activation(w['up'],w['gate'],xn)
        refs=[Path(n) for n in prep['inputs'] if n.endswith('/257/ref_res3.bin')]
        if len(refs)!=1:raise ValueError('ambiguous independent model residual reference')
        reference=native.drift.drift.trace.read_trace(refs[0],257,5120)
        candidate_og=attn.read_output(Path(prep['source'])/'candidate-og.bin')
        with np.load(Path(prep['refinement'])/'numerics.npz',allow_pickle=False) as numbers:
            for p in TARGETS:
                print('Scoring forced-OG full layer at',p,flush=True)
                x=base.read(capture/f'input_{p}.bin',5120).astype(np.float64)
                primary=tail(x+w['out']@numbers['attention_primary'][p].ravel())
                og=candidate_og[p].ravel().astype(np.float64);outref=w['out']@og;conditioned=tail(x+outref)
                c=out/f'candidate-act_{p}.bin'
                original=boundary.attention.read_attention(capture/f'act_{p}.bin',off,24,4,256,'float32')
                observed=boundary.attention.read_attention(c,off,24,4,256,'float32')
                if any(original[k].tobytes()!=observed[k].tobytes() for k in original):
                    raise ValueError('forced tail changed upstream native attention')
                if (out/f'candidate-state_{p}.bin').read_bytes()!=(capture/f'state_{p}.bin').read_bytes():
                    raise ValueError('forced tail changed upstream native KV')
                f=boundary.front.read_front(c,off,5120,17408,True,norm_dtype='float32')
                d=boundary.down.read_regions(c,off,5120,17408,True)
                b=boundary.projections.read_boundaries(c,off,5120,'float32')
                old=base.read(out/f'control-xres_{p}.bin',5120);got=base.read(out/f'candidate-xres_{p}.bin',5120)
                fs=boundary.front.compare_front(f['res'],norm,w['up'],w['gate'],f['xm'],f['h'],spec.norm_eps)
                ds=boundary.down.compare_partials(d['h'],w['down'],d['res'],d['partials'],got,tuple(boundary.ffn_geometry(spec).DOWN_SPLIT))
                proj=boundary.front.metric(b['out'],outref);composed=boundary.front.metric(got,conditioned)
                primary_score=compare(old,got,primary)
                row=dict(position=p,primary=primary_score,model_reference=compare(old,got,reference[p]),
                         conditioned=composed,output_projection=proj,front=fs,down=ds,
                         passed=primary_score['candidate']['passed'] and composed['passed'] and proj['passed'] and fs['passed'] and ds['passed'])
                rows.append(row)
                arrays=dict(primary=primary,conditioned=conditioned,control=old,candidate=got,model_reference=reference[p],
                            forced_og=og,output_reference=outref,output=b['out'],residual=f['res'],norm=f['xm'],activation=f['h'],
                            partial0=d['partials'][0],partial1=d['partials'][1])
                path=out/f'P{p}.npz'
                if write:np.savez(path,**arrays)
                else:
                    with np.load(path,allow_pickle=False) as a:
                        if set(a.files)!=set(arrays):raise ValueError('changed numeric inventory')
                        for k,v in arrays.items():
                            if a[k].shape!=v.shape or a[k].dtype!=v.dtype or a[k].tobytes()!=v.tobytes():raise ValueError('numeric replay differs: '+k)
                artifacts[path.name]=digest(path)
    finally:model.mm.close();model.f.close()
    return dict(candidate_passed=all(r['passed'] for r in rows),exact_controls=exact,positions=list(TARGETS),cases=rows,artifacts=artifacts,
                scope='forced precomputed NPU OG through native output/FFN at three positions; upstream native KV restored; no integrated correction or model acceptance')


def run(a):
    out=a.out.resolve()
    if (out/'report.json').exists() or any((out/n).exists() for n in names()):raise ValueError('do not overwrite tail captures')
    r=dict(complete=False,model_acceptance=False)
    try:
        prep=verified(out);h=a.harness.resolve()
        r.update(preparation_sha256=digest(out/'preparation.json'),harness=str(h),harness_sha256=digest(h),harness_evidence_sha256=native.verify_harness(h))
        with native.drift.drift.boundary.runner.device_lock(),(out/'native.log').open('w') as log:
            subprocess.run([str(h),str(out/'run.cfg')],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,HARNESS_TIMEOUT_MS='30000'),check=True,timeout=1800)
        result=evaluate(out,prep,True);verified(out)
        if digest(h)!=r['harness_sha256'] or native.verify_harness(h)!=r['harness_evidence_sha256']:raise ValueError('harness changed during tail run')
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
            r['harness_evidence_sha256']!=native.verify_harness(r['harness']) or r['log_sha256']!=digest(out/'native.log')):
            raise ValueError('unbound tail run')
        result=evaluate(out,prep,False)
        if result!=r['result']:raise ValueError('changed tail measurements')
        verified(out)
        proof.update(complete=True,report_sha256=digest(out/'report.json'),candidate_passed=result['candidate_passed'],exact_controls=result['exact_controls'],artifacts=len(result['artifacts']))
    except Exception as e:proof['error']=str(e);raise
    finally:native.write_report(out/'audit.json',proof)
    return 0 if proof['candidate_passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','audit'])
    for n in ('out','source','build','harness'):p.add_argument('--'+n,type=Path,required=n=='out')
    a=p.parse_args()
    if a.command=='prepare' and (a.source is None or a.build is None):p.error('--source and --build required')
    if a.command=='run' and a.harness is None:p.error('--harness required')
    raise SystemExit(globals()[a.command](a))
