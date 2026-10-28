"""Replay 240 current L3 Q/gate/K/V projections before interpreting table arithmetic."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


build=load('_l3_projection_build',HERE/'build.py')
boundary=load('_l3_projection_boundary',ROOT/'utilities/prefill-sequential-drift/l3_boundaries.py')
reference=load('_l3_projection_table_reference',ROOT/'utilities/qkv-precision/qkv_reference.py')
native=boundary.native
base,digest=native.base,native.digest
ROLES=('q','gate','k','v')
ROWS=(6144,6144,1024,1024)
COUNT=240


def sources():
    return boundary.sources()|{str(Path(p).resolve()):digest(p) for p in
        (__file__,build.__file__,HERE/'design.py',reference.__file__)}


def identity(got,expected):
    a,b=np.asarray(got),np.asarray(expected)
    if (a.shape!=(14336,) or b.shape!=a.shape or a.dtype!=np.dtype('<f4') or b.dtype!=a.dtype
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('requires complete finite FP32 Q/gate/K/V captures')
    out={};start=0
    for role,rows in zip(ROLES,ROWS):
        out[role]=a[start:start+rows].tobytes()==b[start:start+rows].tobytes();start+=rows
    return out


def expected_table(x):
    x=np.asarray(x)
    if x.shape!=(5120,) or x.dtype!=np.dtype('<f4'):raise ValueError('requires captured FP32 norm')
    t=reference.activation_table(x,'quantized')
    return (t['xi'].astype('<i2').tobytes()+t['shift'].astype('<i4').tobytes()+
            reference.f32_to_bf16(t['hi']).astype('<u2').tobytes()+
            reference.f32_to_bf16(t['lo']).astype('<u2').tobytes())


def table_identity(x,data):
    if len(data)!=11520:raise ValueError('invalid activation table size')
    return data==expected_table(x)


def summarize(rows):
    if ([r['position'] for r in rows]!=list(range(COUNT)) or
            any(set(r['identity'])!=set(ROLES) or
                any(type(v) is not bool for v in r['identity'].values()) or
                type(r['table_exact']) is not bool for r in rows)):
        raise ValueError('incomplete or reordered projection controls')
    roles=sum(sum(r['identity'].values()) for r in rows)
    tables=sum(r['table_exact'] for r in rows)
    return dict(control_valid=roles==COUNT*4,exact_roles=roles,total_roles=COUNT*4,
                exact_tables=tables,total_tables=COUNT,table_valid=tables==COUNT)


def program(out,kernels):
    lines=['device',f'xclbin P {kernels}/final.xclbin',f'kernelx proj P {kernels}/insts.bin',
           f'buf w 45875200 {out}/weights.bin','buf x 20480','buf y 57344','buf tab 11520']
    for p in range(COUNT):
        lines += [f'load x {out}/x{p}.bin',f'load y {out}/poison-y.bin',f'load tab {out}/poison-tab.bin',
                  'run proj w x y tab',f'dump y {out}/y{p}.bin 57344',f'dump tab {out}/table{p}.bin 11520']
    return '\n'.join(lines)+'\n'


def inventory(out):
    actual={p.name for pattern in ('y*.bin','table*.bin') for p in out.glob(pattern)}
    expected={f'{k}{p}.bin' for k in ('y','table') for p in range(COUNT)}
    if actual!=expected:raise ValueError('incomplete or extra projection capture inventory')


def packed_weights(capture,prep,bundle,model):
    from recipes.pack import q4_chunks_of,std_perm
    ops=bundle['layer_types']['full_attention']['pack']['pool'];parts=[]
    with (capture/'pool.bin').open('rb') as pool:
        for role,rows in zip(ROLES,ROWS):
            tensor=f'model.layers.{{l}}.self_attn.{"q" if role=="gate" else role}_proj.weight'
            chunk0=3840 if role=='gate' else 0
            matches=[o for o in ops if o['tensor']==tensor and o.get('chunk0',0)==chunk0]
            if len(matches)!=1:raise ValueError('ambiguous projection packing')
            op=matches[0];nch=rows//32*20
            if op['op']!='std_perm' or op['in_dim']!=5120 or op['nch']!=nch:
                raise ValueError('unexpected projection packing geometry')
            name=tensor.format(l=3)
            chunks=q4_chunks_of(model,name,model.raw(name)).reshape(-1,5120)[chunk0:chunk0+nch]
            packed=chunks[std_perm(nch,5120)].tobytes()
            pool.seek(op['dst']);actual=pool.read(nch*5120)
            if actual!=packed or len(actual)!=nch*5120:raise ValueError('source tensor/native pool differs')
            parts.append(actual)
    return b''.join(parts)


def prepare(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,model_acceptance=False,sources=sources())
    try:
        capture=a.capture.resolve();kernels=a.build.resolve()
        prep,inputs=boundary.control(capture);spec,bundle,_=native.context(prep['meta'])
        built=build.verify(kernels)
        resources=json.loads(a.resources.read_text())
        if resources['passed'] is not True or resources['build']!=built:raise ValueError('unbound resource audit')
        base.verify(resources['evidence_sha256'])
        inputs.update(built['inputs']);inputs.update(resources['evidence_sha256'])
        inputs.update({str(kernels/n):h for n,h in built['sha256'].items()})
        inputs[str(a.resources.resolve())]=digest(a.resources)
        model=base.Q4NX(Path(prep['meta']['model'])/'model.q4nx')
        try:(out/'weights.bin').write_bytes(packed_weights(capture,prep,bundle,model))
        finally:model.mm.close();model.f.close()
        off=boundary.layout(spec,max_ctx=prep['meta']['max_ctx'])
        for p in range(COUNT):
            path=capture/f'act_{p}.bin'
            c=boundary.attention.read_attention(path,off,spec.num_heads,spec.num_kv_heads,spec.head_dim,'float32')
            x=boundary.projections.read_boundaries(path,off,spec.hidden,'float32')['xn'].astype('<f4')
            expected=np.concatenate([c[k].ravel() for k in ROLES]).astype('<f4')
            identity(expected,expected);expected_table(x)
            x.tofile(out/f'x{p}.bin');expected.tofile(out/f'ref{p}.bin')
        np.full(14336,np.nan,dtype='<f4').tofile(out/'poison-y.bin')
        (out/'poison-tab.bin').write_bytes(bytes([0xa5])*11520)
        (out/'run.cfg').write_text(program(out,kernels))
        artifacts={str(p):digest(p) for p in out.iterdir() if p.is_file()}
        base.verify(inputs);base.verify(r['sources'])
        r.update(complete=True,capture=str(capture),build=str(kernels),inputs=inputs,artifacts=artifacts,
                 production_build_key=bundle['build_key'],scope='current L3 projection control; no precision variant or full-model acceptance')
    except Exception as error:r['error']=str(error);raise
    finally:native.write_report(out/'preparation.json',r)


def verified(out):
    r=json.loads((out/'preparation.json').read_text())
    if r.get('complete') is not True or r.get('model_acceptance') is not False or r['sources']!=sources():
        raise ValueError('incomplete or changed projection preparation')
    for key in ('sources','inputs','artifacts'):base.verify(r[key])
    build.verify(Path(r['build']))
    if (out/'run.cfg').read_text()!=program(out,Path(r['build'])):raise ValueError('changed replay program')
    return r


def score(out):
    inventory(out);rows=[];hashes={}
    for p in range(COUNT):
        if (out/f'y{p}.bin').stat().st_size!=57344:
            raise ValueError('invalid projection capture size')
        got=np.fromfile(out/f'y{p}.bin',dtype='<f4');ref=np.fromfile(out/f'ref{p}.bin',dtype='<f4')
        x=np.fromfile(out/f'x{p}.bin',dtype='<f4');table=(out/f'table{p}.bin').read_bytes()
        rows.append(dict(position=p,identity=identity(got,ref),table_exact=table_identity(x,table)))
        hashes.update({f'{k}{p}.bin':digest(out/f'{k}{p}.bin') for k in ('y','table')})
    return summarize(rows),rows,hashes


def run(a):
    out=a.out.resolve()
    if (out/'report.json').exists() or any(out.glob('y*.bin')) or any(out.glob('table*.bin')):
        raise ValueError('use fresh preparation; never overwrite projection captures')
    r=dict(complete=False,model_acceptance=False)
    try:
        verified(out);h=a.harness.resolve()
        r.update(preparation_sha256=digest(out/'preparation.json'),harness=str(h),harness_sha256=digest(h),
                 harness_evidence_sha256=native.verify_harness(h))
        env=os.environ.copy();env['HARNESS_TIMEOUT_MS']='30000'
        with native.drift.drift.boundary.runner.device_lock(),(out/'native.log').open('w') as log:
            subprocess.run([str(h),str(out/'run.cfg')],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=1800)
        summary,rows,hashes=score(out);verified(out)
        if digest(h)!=r['harness_sha256'] or native.verify_harness(h)!=r['harness_evidence_sha256']:
            raise ValueError('harness changed during run')
        r.update(complete=True,summary=summary,rows=rows,artifacts=hashes,log_sha256=digest(out/'native.log'),
                 interpretation_allowed=summary['control_valid'] and summary['table_valid'])
    except Exception as error:r['error']=str(error);raise
    finally:native.write_report(out/'report.json',r)
    return 0 if r['interpretation_allowed'] else 1


def audit(a):
    out=a.out.resolve();proof=dict(complete=False,model_acceptance=False)
    try:
        verified(out);r=json.loads((out/'report.json').read_text())
        if (r.get('complete') is not True or r.get('model_acceptance') is not False or
                r['preparation_sha256']!=digest(out/'preparation.json') or
                r['harness_sha256']!=digest(r['harness']) or
                r['harness_evidence_sha256']!=native.verify_harness(r['harness']) or
                r['log_sha256']!=digest(out/'native.log')):raise ValueError('unbound native projection run')
        summary,rows,hashes=score(out)
        allowed=summary['control_valid'] and summary['table_valid']
        if (summary!=r['summary'] or rows!=r['rows'] or hashes!=r['artifacts'] or allowed!=r['interpretation_allowed']):
            raise ValueError('changed projection measurements')
        proof.update(complete=True,summary=summary,interpretation_allowed=allowed,artifacts=len(hashes),
                     report_sha256=digest(out/'report.json'))
    except Exception as error:proof['error']=str(error);raise
    finally:native.write_report(out/'audit.json',proof)
    return 0 if proof['interpretation_allowed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','audit']);p.add_argument('--out',type=Path,required=True)
    for name in ('capture','build','resources','harness'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    required=('capture','build','resources') if a.command=='prepare' else ('harness',) if a.command=='run' else ()
    if any(getattr(a,n) is None for n in required):p.error('required: '+', '.join('--'+n for n in required))
    raise SystemExit(globals()[a.command](a))
