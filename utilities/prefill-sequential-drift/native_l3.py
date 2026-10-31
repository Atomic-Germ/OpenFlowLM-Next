"""Replay current production full-ax L3 with causal KV and exact native controls."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np

s = importlib.util.spec_from_file_location('_native_l3_drift', Path(__file__).with_name('post_targets.py'))
drift = importlib.util.module_from_spec(s); s.loader.exec_module(drift)
base, digest, ROOT = drift.base, drift.digest, drift.base.ROOT
from recipes import pack

COUNT = 240
LAYER = 3


def extent(count):
    if type(count) is not int or not 1 <= count <= COUNT:
        raise ValueError('replay extent must be 1..240')


def output_names(count):
    extent(count)
    return [f'{kind}_{p}.bin' for p in range(count) for kind in ('xres','act','state')]


def program(out, kernels, sizes, count):
    extent(count)
    if (any(type(sizes[k]) is not int or sizes[k] <= 0 for k in
            ('pool','consts','act','state','hidden','kv_row','ptab_row','ptab')) or
            sizes['state'] < count*sizes['kv_row'] or sizes['ptab'] < count*sizes['ptab_row']):
        raise ValueError('invalid buffer geometry or causal capacity')
    lines = ['device', f'attngeom {sizes["kv_row"]} {sizes["ptab_row"]}',
             f'xclbin ax {kernels}/ax/final.xclbin', f'kernelx ax ax {kernels}/ax/insts.bin',
             f'buf pool {sizes["pool"]} {out}/pool.bin', f'buf consts {sizes["consts"]} {out}/consts.bin',
             f'buf xres {sizes["hidden"]*4}', f'buf act {sizes["act"]}', f'buf state {sizes["state"]}',
             f'buf ptab {sizes["ptab"]} {out}/ptab.bin']
    for p in range(count):
        lines += [f'load xres {out}/input_{p}.bin', f'attnpos ax {p}',
                  'run ax pool xres consts state act ptab']
        lines += [f'dump {kind} {out}/{kind}_{p}.bin {size}' for kind,size in
                  (('xres',sizes['hidden']*4),('act',sizes['act']),('state',(p+1)*sizes['kv_row']))]
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
    expected=set(output_names(count))
    actual={p.name for kind in ('xres','act','state') for p in out.glob(kind+'_*.bin')}
    if actual != expected:raise ValueError('incomplete or extra native capture inventory')
    for name in output_names(count):
        kind = name.split('_')[0]; path = out/name
        pos=int(name.split('_')[1].split('.')[0])
        size = sizes['hidden']*4 if kind == 'xres' else ((pos+1)*sizes['kv_row'] if kind == 'state' else sizes[kind])
        if path.stat().st_size != size: raise ValueError('incomplete capture: '+name)
        hashes[name] = digest(path)
        if kind == 'xres': rows.append(base.read(path,sizes['hidden']).astype('<f4'))
    return np.array(rows,dtype='<f4'), hashes


def sources():
    return drift.sources() | {str(Path(p).resolve()):digest(p) for p in
        (__file__, pack.__file__, ROOT/'open_kernels/harness/run_kernel.cpp',
         ROOT/'open_kernels/harness/CMakeLists.txt',ROOT/'open_kernels/harness/stream_patch.hpp')}


def context(meta):
    spec = base.spec_from_model_dir(Path(meta['model']))
    bundle = base.load_bundle(Path(meta['kernels']),base.manifest(spec,max_ctx=meta['max_ctx']))
    lay=bundle['layout']
    if (spec.hidden != 5120 or spec.num_layers != 64 or spec.layer_types[LAYER] != 'full_attention'
            or meta['spec_hash'] != bundle['spec_hash'] or meta['max_ctx'] < COUNT or
            lay['attn_kv_dtype'] != 'bfloat16_pair' or lay['attn_output_dtype'] != 'float32' or
            lay['attn_entry_dtype'] != 'float32' or lay['ffn_norm_dtype'] != 'float32'):
        raise ValueError('requires current production 27B full-attention layer three')
    # Reference metadata retains its original build key. load_bundle validates
    # the current export against current source; never relabel the CPU fixture.
    b=bundle['layer_types']['full_attention']['buffers']
    if b['state']['kind'] != 'kv' or b['state']['row'] != lay['kv_row']:
        raise ValueError('inconsistent KV state geometry')
    sizes=dict(pool=lay['pool_bytes'],consts=b['consts'],act=b['act'],hidden=spec.hidden,
               state=b['state']['row']*meta['max_ctx'],kv_row=lay['kv_row'],ptab_row=lay['ptab_row'],
               ptab=meta['max_ctx']*lay['ptab_row'])
    return spec,bundle,sizes


def verify_harness(harness):
    harness=Path(harness).resolve();path=harness.parent/'build-evidence.json'
    record=json.loads(path.read_text());hashes=record['sha256']
    required={str(harness),str(ROOT/'open_kernels/harness/run_kernel.cpp'),
              str(ROOT/'open_kernels/harness/stream_patch.hpp')}
    if not required <= set(hashes):raise ValueError('unbound harness build evidence')
    base.verify(hashes)
    return digest(path)


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
            pack.build_layer_pool(plan,'full_attention',model,LAYER).tofile(out/'pool.bin')
            pack.build_consts(plan,'full_attention',model,LAYER,sizes['consts']).tofile(out/'consts.bin')
            xs=drift.prefix_inputs(LAYER,'sequential',model,meta,refs,directory/'257/trace')
            if xs.shape != (257,spec.hidden) or not np.isfinite(xs).all():
                raise ValueError('invalid retained native input prefix')
            for p in range(COUNT):xs[p].astype('<f4').tofile(out/f'input_{p}.bin')
            g=bundle['globals']['ptab']
            pack.ptab(meta['max_ctx'],spec.rotary_dim,spec.rope_theta,g['per_row'],
                      g.get('inv_freq',spec.rope_inv_freq()),g.get('window',0),g.get('scale',1.0),
                      g.get('long_inv_freq'),g.get('switch_row')).tofile(out/'ptab.bin')
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
        r.update(complete=True,baseline=str(directory),meta=meta,sizes=sizes,count=COUNT,layer=LAYER,
                 export_identity={k:bundle[k] for k in ('build_key','spec_hash')},
                 inputs=inputs,artifacts=artifacts,
                 scope='one current full-ax L3 pool; native causal KV; active-prefix state captures; exact residual control before interpretation')
    except Exception as error:r['error']=str(error);raise
    finally:write_report(out/'preparation.json',r)
    return 0


def verified(out):
    r=json.loads((out/'preparation.json').read_text())
    if not r['complete'] or r['model_acceptance'] is not False or r['count'] != COUNT or r['layer'] != LAYER or r['sources'] != sources():
        raise ValueError('incomplete or changed replay preparation')
    base.verify(r['sources']);base.verify(r['inputs']);base.verify(r['artifacts'])
    _,bundle,sizes=context(r['meta'])
    if r['export_identity']!={k:bundle[k] for k in ('build_key','spec_hash')}:
        raise ValueError('changed current export identity')
    if sizes != r['sizes'] or (out/'run.cfg').read_text() != program(out,Path(r['meta']['kernels']),sizes,COUNT):
        raise ValueError('changed replay program/geometry')
    return r


def score(out, prepared):
    got,hashes=read_outputs(out,prepared['sizes'],COUNT)
    path=Path(prepared['baseline'])/'257/trace/sequential_res3.bin'
    ref=drift.drift.trace.read_trace(path,258,prepared['sizes']['hidden']).astype('<f4')[:COUNT]
    return identity(got,ref),hashes


def run(a):
    out=a.out.resolve()
    if (out/'report.json').exists() or any((out/n).exists() for n in output_names(COUNT)):
        raise ValueError('use a fresh preparation; do not overwrite native evidence')
    r=dict(complete=False,model_acceptance=False)
    try:
        prep=verified(out); harness=a.harness.resolve()
        r.update(preparation_sha256=digest(out/'preparation.json'),harness=str(harness),
                 harness_sha256=digest(harness),harness_evidence_sha256=verify_harness(harness),command=[str(harness),str(out/'run.cfg')])
        env=os.environ.copy();env.update(HARNESS_TIMEOUT_MS='30000')
        with drift.drift.boundary.runner.device_lock(), (out/'native.log').open('w') as log:
            subprocess.run(r['command'],stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=1800)
        control,hashes=score(out,prep)
        verified(out)
        if digest(harness)!=r['harness_sha256'] or verify_harness(harness)!=r['harness_evidence_sha256']:raise ValueError('harness changed during execution')
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
                r['harness_sha256']!=digest(r['harness']) or
                r['harness_evidence_sha256']!=verify_harness(r['harness']) or r['log_sha256']!=digest(out/'native.log')):
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
