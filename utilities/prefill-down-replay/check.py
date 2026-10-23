"""Prepare, run and audit isolated combined FFN down correction with exact controls."""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


attr=load('_down_attribution',ROOT/'utilities/prefill-combined-attribution/check.py')
prefix=load('_down_prefix_helpers',ROOT/'utilities/prefill-prefix-replay/check.py')
paired,base,proj=prefix.paired,attr.base,prefix.proj
runner=attr.runner;digest,verify=attr.digest,base.verify
N,K,T=5120,17408,256
TARGETS=((63,1),(42,17),(52,17))


def write_reference(path,values):
    values=np.asarray(values,dtype='<f8')
    if not values.size or not np.isfinite(values).all():raise ValueError('invalid primary reference')
    path.write_bytes(np.ascontiguousarray(values).tobytes())


def scores(control,candidate,reference,residual,residual_reference):
    arrays=[np.asarray(v,np.float64) for v in (control,candidate,reference,residual,residual_reference)]
    if (arrays[0].ndim!=1 or not arrays[0].size or any(v.shape!=arrays[0].shape or not np.isfinite(v).all() for v in arrays)):
        raise ValueError('requires complete finite projection/residual vectors')
    c,p=proj.metric(arrays[0],arrays[2]),proj.metric(arrays[1],arrays[2])
    r=prefix.residual_scores(arrays[3],arrays[1],arrays[2],arrays[4])['residual']
    return dict(control=c,candidate=p,projection_passed=p['passed'],
                improved=all(p[k]<c[k] for k in ('maxrel','rel_fro')),residual=r)


def verify_build(path,mode):
    b=paired.verify_build(path,mode)
    expected=dict(GQP_N=str(N),GQP_K=str(K),GQP_T=str(T),GQP_FMT='q4_1',GQP_BREUSE='0')
    if b.get('shape')!=dict(n=N,k=K,t=T) or any(b['environment'].get(k)!=v for k,v in expected.items()):
        raise ValueError('wrong down kernel geometry')
    return b


def harness(path):
    b=json.loads((path/'build-evidence.json').read_text())
    if not b['complete']:raise ValueError('incomplete harness')
    verify(b['sources']);verify(b['compiler_dependencies'])
    verify({str(path/'build/run_kernel'):b['harness_sha256'],str(path/'build/CMakeCache.txt'):b['cmake_cache_sha256']})
    return b


def prepare(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    result=dict(complete=False,model_acceptance=False,artifacts={})
    try:
        attribution=a.attribution.resolve();prior=json.loads((attribution/'report.json').read_text())
        if not prior['complete'] or prior['model_acceptance']:raise ValueError('invalid CPU attribution')
        verify(prior['sources']);verify(prior['inputs']);verify({str(attribution/p):h for p,h in prior['artifacts'].items()})
        capture=a.capture.resolve();audit,meta,refs,inputs=attr.validate(capture,a.audit.resolve())
        spec=base.spec_from_model_dir(Path(meta['model']))
        bundle=proj.load_bundle(Path(meta['kernels']),proj.manifest(spec,max_ctx=meta['max_ctx']))
        cases={c['tag']:c for c in proj.cases(bundle)}
        if any((cases[tag]['n'],cases[tag]['k'])!=(N,K) for tag in ('L0_down','L3_down')):
            raise ValueError('unsupported down geometry')
        model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=N;model.native_q8=lambda name:False
        weights_evidence=[]
        def save(path,x):
            if path.suffix=='.f64':write_reference(path,x)
            else:path.write_bytes(np.ascontiguousarray(x).tobytes())
            result['artifacts'][str(path)]=digest(path)
        for l,p in TARGETS:
            directory=out/f'L{l}_P{p}';directory.mkdir()
            op=cases['L3_down' if l==63 else 'L0_down']['ops'][0]
            weights,e=attr.packed_weights(model,l,op)
            old=next(c for c in prior['layers'] if (c['layer'],c['position'])==(l,p))
            if e!=old['down_weight']:raise ValueError('attributed packed/source weights changed')
            weights_evidence.append(e)
            pool=np.zeros(N*K*5//8,np.uint8);proj.apply_op(dict(op,dst=0),model,l,pool)
            save(directory/'weights.bin',pool)
            with np.load(attribution/f'L{l}_P{p}.npz') as arrays:
                x=arrays['captured_h'].astype(np.float32);ref=arrays['down_exact'].copy()
                native=arrays['captured_down'].astype(np.float32).reshape(N)
                res=arrays['native_res'].astype(np.float32);out_ref=arrays['reference_output'].copy()
                native_output=arrays['native_output'].astype(np.float32)
            if x.shape!=(1,K) or ref.shape!=(1,N):raise ValueError('invalid captured down geometry')
            fresh=attr.rounding.predict(x,weights)['exact']
            prefix.identity(fresh,ref,'independent down reference')
            prefix.identity(np.add(res,native,dtype=np.float32),native_output,'native residual reconstruction')
            for name,values in [('input.bin',x),('reference.f64',ref),('captured.bin',native),
                                ('residual.bin',res),('output-reference.f64',out_ref),('native-output.bin',native_output)]:
                save(directory/name,values)
            for slot in (p,p+128):
                for part,values in zip(('high','low'),paired.control_tool.split_inputs(x)):
                    save(directory/f'{slot}-{part}-x.bin',proj.tile_input(prefix.place(values,slot)))
            del weights,pool
        kernel=Path(meta['kernels'])/cases['L0_down']['kernel']
        files=[Path(__file__),Path(attr.__file__),Path(prefix.__file__),Path(paired.__file__),attribution/'report.json',
               Path(meta['kernels'])/'manifest.json',kernel/'final.xclbin',kernel/'insts.bin']
        inputs.update({str(p.resolve()):digest(p) for p in files})
        result.update(complete=True,targets=TARGETS,inputs=inputs,production_kernel=str(kernel),
                      weights=weights_evidence,scope='captured combined activation rows, isolated down only')
    except Exception as error:
        result.update(complete=False,error=str(error));raise
    finally:(out/'preparation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0


def preflight(prepared,build_root,harness_build):
    p=json.loads((prepared/'preparation.json').read_text())
    if not p['complete'] or p['targets']!=[list(t) for t in TARGETS]:raise ValueError('incomplete preparation')
    verify(p['inputs']);verify(p['artifacts'])
    for mode in ('control','low'):verify_build(build_root/mode,mode)
    harness(harness_build)
    return p


def measure(out,prepared):
    cases=[];outputs={}
    for l,p in TARGETS:
        directory=out/f'L{l}_P{p}';inputs=prepared/directory.name;candidates=[]
        reference=np.fromfile(inputs/'reference.f64','<f8')
        res=base.read(inputs/'residual.bin',N).astype(np.float32)
        refout=np.fromfile(inputs/'output-reference.f64','<f8')
        native=base.read(inputs/'captured.bin',N).astype(np.float32)
        native_out=base.read(inputs/'native-output.bin',N).astype(np.float32)
        if reference.shape!=(N,) or refout.shape!=(N,):raise ValueError('incomplete primary references')
        for slot in (p,p+128):
            products={}
            for mode in ('production','control','low'):
                ys=[]
                for part in ('high','low'):
                    path=directory/f'{slot}-{mode}-{part}.bin'
                    y=base.read(path,N*T).astype(np.float32).reshape(N,T)
                    prefix.extract(y,slot,1);ys.append(y);outputs[str(path)]=digest(path)
                products[mode]=ys
            paired.require_control(*products['control'],*products['production'])
            control=prefix.extract(np.add(*products['control'],dtype=np.float32),slot,1)[0]
            prefix.identity(control,native,'native combined down')
            prefix.identity(np.add(res,control,dtype=np.float32),native_out,'native layer output')
            candidate=prefix.extract(paired.combine(*products['control'],*products['low']),slot,1)[0]
            candidates.append(candidate)
            cases.append(dict(layer=l,position=p,slot=slot,controls_exact=True,zero_inactive_columns=True,
                              **scores(control,candidate,reference,res,refout)))
        prefix.identity(*candidates,'candidate slot placement')
    return dict(cases=cases,outputs=outputs,control_valid=True,slot_identity=True,
                projection_passed=all(c['projection_passed'] for c in cases),
                improved=all(c['improved'] for c in cases),residual_passed=all(c['residual']['passed'] for c in cases))


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    result=dict(complete=False,control_valid=False,model_acceptance=False,dispatches=0)
    try:
        prepared=a.prepared.resolve();builds=a.build_root.resolve();hb=a.harness_build.resolve()
        prep=preflight(prepared,builds,hb)
        evidence=[Path(__file__),prepared/'preparation.json',hb/'build-evidence.json',*[builds/m/'build.json' for m in ('control','low')]]
        with (ROOT/'Models/qwen38-27b/.resident-validation.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runner.device_available()
            for l,p in TARGETS:
                directory=out/f'L{l}_P{p}';directory.mkdir();inputs=prepared/directory.name
                for slot in (p,p+128):
                    products={}
                    for mode in ('production','control','low'):
                        kernel=Path(prep['production_kernel']) if mode=='production' else builds/mode;ys=[]
                        for part in ('high','low'):
                            runner.device_available()
                            tag=f'{slot}-{mode}-{part}';cfg=directory/(tag+'.cfg');dump=directory/(tag+'.bin')
                            cfg.write_text('\n'.join(['device',f'xclbin G {kernel}/final.xclbin',f'kernelx k G {kernel}/insts.bin',
                                f'buf w {N*K*5//8} {inputs}/weights.bin',f'buf x {K*T*2} {inputs}/{slot}-{part}-x.bin',
                                f'buf y {N*T*4}','run k w x y',f'dump y {dump} {N*T*4}','']))
                            with cfg.with_suffix('.log').open('w') as log:
                                subprocess.run([str(hb/'build/run_kernel'),str(cfg)],stdout=log,stderr=subprocess.STDOUT,check=True,
                                               timeout=120,env=os.environ|{'HARNESS_TIMEOUT_MS':'30000'})
                            result['dispatches']+=1;evidence.extend([cfg,cfg.with_suffix('.log')])
                            y=base.read(dump,N*T).astype(np.float32).reshape(N,T);prefix.extract(y,slot,1);ys.append(y)
                        products[mode]=ys
                        if mode=='control':
                            paired.require_control(*ys,*products['production'])
                            native=base.read(inputs/'captured.bin',N).astype(np.float32)
                            prefix.identity(np.add(*ys,dtype=np.float32)[:,slot],native,'native captured down')
                    print('replayed',l,p,'slot',slot,flush=True)
        if result['dispatches']!=36:raise ValueError('incomplete dispatch count')
        result.update(measure(out,prepared))
        result.update(complete=True,prepared=str(prepared),build_root=str(builds),harness_build=str(hb),
                      evidence={str(p):digest(p) for p in evidence},scope='isolated down correction on captured H; no full-model acceptance')
    except Exception as error:
        result.update(complete=False,control_valid=False,error=str(error));raise
    finally:(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['projection_passed'] and result['improved'] and result['residual_passed'] else 1


def audit(a):
    out=a.out.resolve();result=dict(complete=False,model_acceptance=False)
    try:
        report=json.loads((out/'report.json').read_text())
        if not report['complete'] or report['dispatches']!=36:raise ValueError('incomplete replay')
        verify(report['evidence']);verify(report['outputs'])
        prepared=Path(report['prepared']);preflight(prepared,Path(report['build_root']),Path(report['harness_build']))
        measured=measure(out,prepared)
        if any(report.get(k)!=v for k,v in measured.items()):raise ValueError('reported measurements differ')
        result.update(complete=True,report_sha256=digest(out/'report.json'),**{k:v for k,v in measured.items() if k not in ('cases','outputs')})
    except Exception as error:
        result['error']=str(error);raise
    finally:(out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['projection_passed'] and result['improved'] and result['residual_passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    for name in ('out','attribution','capture','audit'):q.add_argument('--'+name,type=Path,required=True)
    q=sub.add_parser('run')
    for name in ('out','prepared','build-root','harness-build'):q.add_argument('--'+name,type=Path,required=True)
    sub.add_parser('audit').add_argument('--out',type=Path,required=True)
    a=p.parse_args();raise SystemExit({'prepare':prepare,'run':run,'audit':audit}[a.command](a))
