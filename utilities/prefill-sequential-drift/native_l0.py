"""Replay production L0 from zero through P242; exact controls precede diagnosis."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np

s = importlib.util.spec_from_file_location('_native_l0_drift', Path(__file__).with_name('check.py'))
drift = importlib.util.module_from_spec(s); s.loader.exec_module(drift)
base, digest, ROOT = drift.base, drift.digest, drift.ROOT
from recipes import pack

COUNT = 243


def extent(count):
    if type(count) is not int or not 1 <= count <= COUNT:
        raise ValueError('replay extent must be 1..243')


def output_names(count):
    extent(count)
    return [f'{kind}_{p}.bin' for p in range(count) for kind in ('xres','act','state')]


def program(out, kernels, sizes, count):
    extent(count)
    if any(type(sizes[k]) is not int or sizes[k] <= 0 for k in ('pool','consts','act','state','hidden')):
        raise ValueError('invalid buffer geometry')
    lines = ['device', f'xclbin lx {kernels}/lx/final.xclbin',
             f'kernelx lx lx {kernels}/lx/insts.bin',
             f'buf pool {sizes["pool"]} {out}/pool.bin',
             f'buf consts {sizes["consts"]} {out}/consts.bin',
             f'buf xres {sizes["hidden"]*4}', f'buf act {sizes["act"]}',
             f'buf state {sizes["state"]}']
    for p in range(count):
        lines += [f'load xres {out}/input_{p}.bin', 'run lx pool xres consts state act']
        lines += [f'dump {kind} {out}/{kind}_{p}.bin {size}' for kind,size in
                  (('xres',sizes['hidden']*4),('act',sizes['act']),('state',sizes['state']))]
    return '\n'.join(lines)+'\n'


def identity(got, ref):
    a,b = np.asarray(got),np.asarray(ref)
    if (a.ndim != 2 or a.shape != b.shape or not a.size or a.dtype != np.dtype('<f4')
            or b.dtype != np.dtype('<f4') or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('complete finite FP32 residual matrices required')
    rows = [x.tobytes() == y.tobytes() for x,y in zip(a,b)]
    return dict(passed=all(rows),positions=len(rows),matching_positions=sum(rows),
                first_mismatch=next((p for p,v in enumerate(rows) if not v),None))


def read_outputs(out, sizes, count):
    rows, hashes = [], {}
    for name in output_names(count):
        kind = name.split('_')[0]; path = out/name
        size = sizes['hidden']*4 if kind == 'xres' else sizes[kind]
        if path.stat().st_size != size: raise ValueError('incomplete capture: '+name)
        hashes[name] = digest(path)
        if kind == 'xres': rows.append(base.read(path,sizes['hidden']).astype('<f4'))
    return np.array(rows,dtype='<f4'), hashes


def sources():
    return drift.sources() | {str(Path(p).resolve()):digest(p) for p in
        (__file__, pack.__file__, ROOT/'open_kernels/harness/run_kernel.cpp',
         ROOT/'open_kernels/harness/CMakeLists.txt')}


def context(meta):
    spec = base.spec_from_model_dir(Path(meta['model']))
    bundle = base.load_bundle(Path(meta['kernels']),base.manifest(spec,max_ctx=meta['max_ctx']))
    if (spec.hidden != 5120 or spec.num_layers != 64 or spec.layer_types[0] != 'linear_attention'
            or any(meta[k] != bundle[k] for k in ('build_key','spec_hash'))):
        raise ValueError('requires matching production 27B linear layer zero')
    b = bundle['layer_types']['linear_attention']['buffers']
    sizes = dict(pool=bundle['layout']['pool_bytes'],consts=b['consts'],act=b['act'],
                 state=b['state']['bytes'],hidden=spec.hidden)
    return spec,bundle,sizes


def write_report(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(a):
    out = a.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    r = dict(complete=False,model_acceptance=False,sources=sources())
    try:
        directory = a.baseline.resolve()
        meta, refs, inputs = drift.validate(directory)
        spec,bundle,sizes = context(meta)
        model = base.Q4NX(Path(meta['model'])/'model.q4nx')
        try:
            plan = dict(pool_bytes=bundle['pack']['pool_bytes'],chunk_bytes=bundle['pack']['chunk_bytes'],
                        layer_types={k:v['pack'] for k,v in bundle['layer_types'].items()})
            pack.build_layer_pool(plan,'linear_attention',model,0).tofile(out/'pool.bin')
            pack.build_consts(plan,'linear_attention',model,0,sizes['consts']).tofile(out/'consts.bin')
            for p,token in enumerate(meta['ids'][:COUNT]):
                x = np.asarray(model.embed(token,spec.hidden),dtype='<f4')
                if x.shape != (spec.hidden,) or not np.isfinite(x).all():
                    raise ValueError('invalid model embedding')
                x.tofile(out/f'input_{p}.bin')
        finally:
            model.mm.close(); model.f.close()
        (out/'run.cfg').write_text(program(out,Path(meta['kernels']),sizes,COUNT))
        artifacts = {str(p):digest(p) for p in out.iterdir() if p.is_file()}
        for name in ('model.q4nx','config.json'):
            p=Path(meta['model'])/name;inputs[str(p)]=digest(p)
        kernel_root=Path(meta['kernels'])
        tc=json.loads((kernel_root/'toolchain.json').read_text())
        for name in ('manifest.json','toolchain.json',*tc['sha256']):
            p=kernel_root/name;inputs[str(p)]=digest(p)
        base.verify(r['sources']);base.verify(inputs)
        r.update(complete=True,baseline=str(directory),meta=meta,sizes=sizes,count=COUNT,
                 inputs=inputs,artifacts=artifacts,
                 scope='one production L0 pool; native causal history; residual identity before boundary interpretation')
    except Exception as error:r['error']=str(error);raise
    finally:write_report(out/'preparation.json',r)
    return 0


def verified(out):
    r=json.loads((out/'preparation.json').read_text())
    if not r['complete'] or r['model_acceptance'] is not False or r['count'] != COUNT or r['sources'] != sources():
        raise ValueError('incomplete or changed replay preparation')
    base.verify(r['sources']);base.verify(r['inputs']);base.verify(r['artifacts'])
    _,_,sizes=context(r['meta'])
    if sizes != r['sizes'] or (out/'run.cfg').read_text() != program(out,Path(r['meta']['kernels']),sizes,COUNT):
        raise ValueError('changed replay program/geometry')
    return r


def score(out, prepared):
    got,hashes=read_outputs(out,prepared['sizes'],COUNT)
    path=Path(prepared['baseline'])/'expanded/257/sequential_res0.bin'
    ref=drift.trace.read_trace(path,258,prepared['sizes']['hidden']).astype('<f4')[:COUNT]
    return identity(got,ref),hashes


def run(a):
    out=a.out.resolve()
    if (out/'report.json').exists() or any((out/n).exists() for n in output_names(COUNT)):
        raise ValueError('use a fresh preparation; do not overwrite native evidence')
    r=dict(complete=False,model_acceptance=False)
    try:
        prep=verified(out); harness=a.harness.resolve()
        r.update(preparation_sha256=digest(out/'preparation.json'),harness=str(harness),
                 harness_sha256=digest(harness),command=[str(harness),str(out/'run.cfg')])
        env=os.environ.copy();env.update(HARNESS_TIMEOUT_MS='30000')
        with drift.boundary.runner.device_lock(), (out/'native.log').open('w') as log:
            subprocess.run(r['command'],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=1800)
        control,hashes=score(out,prep)
        verified(out)
        if digest(harness)!=r['harness_sha256']:raise ValueError('harness changed during execution')
        r.update(complete=True,control_valid=control['passed'],identity=control,artifacts=hashes,
                 log_sha256=digest(out/'native.log'),boundary_interpretation_allowed=control['passed'])
    except Exception as error:r['error']=str(error);raise
    finally:write_report(out/'report.json',r)
    return 0 if r['control_valid'] else 1


def audit(a):
    out=a.out.resolve(); result=dict(complete=False,model_acceptance=False)
    try:
        prep=verified(out);r=json.loads((out/'report.json').read_text())
        if (not r['complete'] or r['model_acceptance'] is not False or
                r['preparation_sha256']!=digest(out/'preparation.json') or
                r['harness_sha256']!=digest(r['harness']) or r['log_sha256']!=digest(out/'native.log')):
            raise ValueError('incomplete or changed native record')
        control,hashes=score(out,prep)
        if (control!=r['identity'] or hashes!=r['artifacts'] or
                control['passed']!=r['control_valid'] or control['passed']!=r['boundary_interpretation_allowed']):
            raise ValueError('native measurement/inventory changed')
        result.update(complete=True,control_valid=control['passed'],report_sha256=digest(out/'report.json'),
                      identity=control,artifacts=len(hashes))
    except Exception as error:result['error']=str(error);raise
    finally:write_report(out/'audit.json',result)
    return 0 if result['control_valid'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','audit'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--harness',type=Path)
    a=p.parse_args()
    if a.command=='prepare' and a.baseline is None:p.error('--baseline required')
    if a.command=='run' and a.harness is None:p.error('--harness required')
    raise SystemExit(globals()[a.command](a))
