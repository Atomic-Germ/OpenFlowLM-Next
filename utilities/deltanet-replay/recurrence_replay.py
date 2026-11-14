"""Prepare, configure and score causal per-head replay with pinned controls."""
import argparse,json
from pathlib import Path
import numpy as np
from recurrence_common import digest,geometry,sources,replay_cfg,score,factored_reference
import deltanet_state as ds
from decode_reference import sfx


def read_f32(path,shape):
    data=path.read_bytes()
    if len(data)!=int(np.prod(shape))*4:raise ValueError('wrong capture size: '+str(path))
    a=np.frombuffer(data,'<f4').reshape(shape);ds.finite(a);return a


def verify_inputs(inputs):
    meta=json.loads((inputs/'fixture.json').read_text())
    if meta['tokens']<2 or meta['heads']!=48 or not meta['layers'] or len(meta['layers'])!=len(set(meta['layers'])):
        raise ValueError('invalid replay coverage')
    names={f'L{l}_H{h}_T{t}.bin' for l in meta['layers'] for h in range(48) for t in range(meta['tokens'])}
    names|={'zero.bin','poison-state.bin','poison-output.bin'}
    if set(meta['input_sha256'])!=names:raise ValueError('incomplete replay inputs')
    for n,h in meta['input_sha256'].items():
        if digest(inputs/n)!=h:raise ValueError('input hash mismatch: '+n)
    return meta


def verify_build(build,meta):
    b=json.loads((build/'build.json').read_text())
    if b['production_build_key']!=meta['build_key']:raise ValueError('build/fixture mismatch')
    expected=sources(b['mode'])
    if set(b['source_sha256'])!=set(expected):raise ValueError('build source coverage mismatch')
    for n,s in expected.items():
        if (build/'source'/n).read_text()!=s or digest(build/'source'/n)!=b['source_sha256'][n]:
            raise ValueError('stale build source: '+n)
    for n,h in b['sha256'].items():
        if digest(build/n)!=h:raise ValueError('binary mismatch')
    if set(b['sha256'])!={'final.xclbin','insts.bin'}:raise ValueError('incomplete binary hashes')
    return b


def prepare(model,kernels,captures,baseline,out):
    current=ds.check(model,kernels,captures);old=json.loads(baseline.read_text())
    for key in ('build_key','spec_hash','capture_sha256','results'):
        if current[key]!=old[key]:raise ValueError('production baseline mismatch: '+key)
    if not current['passed']:raise ValueError('whole-state baseline must pass')
    spec=ds.spec_from_model_dir(model);geometry(spec)
    meta=ds.load_reference(captures);off=ds.layout(spec,max_ctx=meta['max_ctx'])
    selected=ds.load_bundle(kernels,ds.manifest(spec,max_ctx=meta['max_ctx']))
    _,dtype=ds.linear_dtypes(selected)
    out.mkdir(parents=True,exist_ok=True)
    np.zeros((130,128),'<f4').tofile(out/'zero.bin')
    np.full((130,128),np.nan,'<f4').tofile(out/'poison-state.bin')
    np.full(128,np.nan,'<f4').tofile(out/'poison-output.bin')
    hashes={n:digest(out/n) for n in ('zero.bin','poison-state.bin','poison-output.bin')}
    for l in current['layers']:
        for t in range(current['tokens']):
            act=captures/f'y_act{l}{sfx(t)}.bin';state=captures/f'y_state{l}{sfx(t)}.bin'
            for p in (act,state):
                if digest(p)!=current['capture_sha256'][p.name]:raise ValueError('changed capture')
            c=ds.read_boundaries(act.read_bytes(),state.read_bytes(),spec,off,conv_dtype=dtype)
            for h in range(48):
                name=f'L{l}_H{h}_T{t}.bin';r=np.zeros(1280,'<f4');r[:512]=c['records'][h]
                r.tofile(out/name);hashes[name]=digest(out/name)
    record=dict(layers=current['layers'],heads=48,tokens=current['tokens'],build_key=current['build_key'],
                spec_hash=current['spec_hash'],input_sha256=hashes,capture_sha256=current['capture_sha256'],
                captures=str(captures.resolve()),baseline=str(baseline.resolve()),baseline_sha256=digest(baseline),
                model=str(model.resolve()),kernels=str(kernels.resolve()),reference_fixture_sha256=current['fixture_sha256'])
    (out/'fixture.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared',48*len(current['layers'])*current['tokens'],'head steps')


def configure(inputs,build,out):
    out.mkdir(parents=True,exist_ok=True);(out/'run.json').unlink(missing_ok=True)
    m=verify_inputs(inputs);b=verify_build(build,m)
    for layer in m['layers']:
        for head in range(48):
            for token in range(m['tokens']):
                for field in ('state','output'):
                    (out/f'L{layer}_H{head}_T{token}-{field}.bin').unlink(missing_ok=True)
    cfg=out/'run.cfg';cfg.write_text(replay_cfg(build,inputs,out,m['layers'],m['tokens']))
    (out/'run.json').write_text(json.dumps(dict(fixture_sha256=digest(inputs/'fixture.json'),
        build_sha256=digest(build/'build.json'),cfg_sha256=digest(cfg),mode=b['mode']),indent=2)+'\n')


def compare(inputs,build,out):
    m=verify_inputs(inputs);b=verify_build(build,m);run=json.loads((out/'run.json').read_text())
    if (run['fixture_sha256']!=digest(inputs/'fixture.json') or run['build_sha256']!=digest(build/'build.json') or
            run['cfg_sha256']!=digest(out/'run.cfg') or run['mode']!=b['mode']):raise ValueError('run identity mismatch')
    if (out/'run.cfg').read_text()!=replay_cfg(build,inputs,out,m['layers'],m['tokens']):raise ValueError('run program mismatch')
    captures=Path(m['captures']);baseline=Path(m['baseline'])
    if digest(baseline)!=m['baseline_sha256']:raise ValueError('baseline changed')
    if digest(captures/'decode_reference.json')!=m['reference_fixture_sha256']:raise ValueError('reference fixture changed')
    for n,h in m['capture_sha256'].items():
        if digest(captures/n)!=h:raise ValueError('production capture changed')
    spec=ds.spec_from_model_dir(Path(m['model']));meta=ds.load_reference(captures)
    selected=ds.load_bundle(Path(m['kernels']),ds.manifest(spec,max_ctx=meta['max_ctx']))
    if selected['build_key']!=m['build_key'] or selected['spec_hash']!=m['spec_hash']:raise ValueError('export changed')
    off=ds.layout(spec,max_ctx=meta['max_ctx']);_,dtype=ds.linear_dtypes(selected)
    rows=[];hashes={}
    for l in m['layers']:
        ref=np.zeros((48,128,128),np.float64)
        for t in range(m['tokens']):
            cap=ds.read_boundaries((captures/f'y_act{l}{sfx(t)}.bin').read_bytes(),
                (captures/f'y_state{l}{sfx(t)}.bin').read_bytes(),spec,off,conv_dtype=dtype)
            algebra=factored_reference(ref,cap['records']) if b['mode']=='factored' else None
            ref,ref_o=ds.step_reference(ref,cap['records'])
            for h in range(48):
                name=f'L{l}_H{h}_T{t}'
                sp=out/(name+'-state.bin');op=out/(name+'-output.bin')
                got=read_f32(sp,(130,128));o=read_f32(op,(128,));hashes[sp.name]=digest(sp);hashes[op.name]=digest(op)
                row=score(got,o,cap['state'][h],cap['output'][h],ref[h],ref_o[h],control=b['mode']=='control',query=cap['records'][h,128:256])
                if algebra is not None:
                    row['factored_algebra']={'state':ds.metric(algebra[0][h],ref[h]),'output':ds.metric(algebra[1][h],ref_o[h])}
                    if not all(v['passed'] for v in row['factored_algebra'].values()):raise ValueError('factored reference disagrees with primary')
                rows.append(dict(layer=l,token=t,head=h,**row))
    return dict(mode=b['mode'],passed=all(x['passed'] for x in rows),
        head_precision_passed=all(x['state']['passed'] and x['output']['passed'] and x['padding_zero'] for x in rows),
        control_identity_passed=all(x['byte_exact'] for x in rows) if b['mode']=='control' else None,build=b,run=run,
        maxrel_bound=ds.MAXREL,cosine_min=ds.COSINE_MIN,conditioning='production records; independent FP64 state from zero',
        summary=dict(cases=len(rows),byte_exact=sum(x['byte_exact'] for x in rows),
                     state_failures=sum(not x['state']['passed'] for x in rows),
                     output_failures=sum(not x['output']['passed'] for x in rows),
                     padding_failures=sum(not x['padding_zero'] for x in rows)),
        capture_sha256=hashes,results=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','configure','compare'])
    for n in ('model-dir','kernel-dir','captures','baseline','inputs','build','out'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    needed={'prepare':['model_dir','kernel_dir','captures','baseline','out'],'configure':['inputs','build','out'],'compare':['inputs','build','out']}[a.stage]
    for n in needed:
        if getattr(a,n) is None:p.error('missing --'+n.replace('_','-'))
    if a.stage=='prepare':prepare(a.model_dir,a.kernel_dir,a.captures,a.baseline,a.out)
    elif a.stage=='configure':configure(a.inputs,a.build,a.out)
    else:
        result=compare(a.inputs,a.build,a.out);print(json.dumps(result,indent=2,allow_nan=False));return 0 if result['passed'] else 1
    return 0


if __name__=='__main__':raise SystemExit(main())
