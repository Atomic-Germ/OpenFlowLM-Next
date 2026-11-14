"""Signed FP64 projection attribution from audited native tables; no model acceptance."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_projection_attribution_replay',HERE/'replay.py')
replay=importlib.util.module_from_spec(s);s.loader.exec_module(replay)
base,digest=replay.base,replay.digest
from recipes import pack
import q4nx


def sources():
    return replay.sources()|{str(Path(p).resolve()):digest(p) for p in (__file__,pack.__file__,q4nx.__file__)}


def table(data,k):
    if type(k) is not int or k<=0 or k%32 or len(data)!=k*2+k//32*8:
        raise ValueError('invalid table size')
    nb=k//32;xi=np.frombuffer(data,'<i2',count=k).reshape(nb,32).astype(np.float64)
    shift=np.frombuffer(data,'<i4',count=nb,offset=k*2)
    if np.any((shift < -113)|(shift > 126)):raise ValueError('invalid table shift')
    hi,lo=[q4nx.bf16_to_f32(np.frombuffer(data,'<u2',count=nb,offset=k*2+nb*off)).astype(np.float64)
           for off in (4,6)]
    x=np.ldexp(xi,-shift[:,None]).ravel();sums=hi+lo
    if not np.isfinite(x).all() or not np.isfinite(sums).all():raise ValueError('non-finite table')
    return x,sums


def weights(data,rows,k):
    if rows<=0 or rows%64 or k<=0 or k%256 or len(data)!=rows//32*(k//256)*5120:
        raise ValueError('invalid packed weight shape')
    packed=np.frombuffer(data,np.uint8).reshape(-1,5120)
    raster=np.empty_like(packed);raster[pack.std_perm(len(packed),k)]=packed
    q,d,m=replay.reference.unpack_chunks(raster,k)
    w=(q.astype(np.float64)*d.astype(np.float64)[:,:,None]+m.astype(np.float64)[:,:,None]).reshape(rows,k)
    independent=q4nx.dq_chunks_q4_1(raster).reshape(rows//32,k//256,32,256).transpose(0,2,1,3).reshape(rows,k)
    if not np.isfinite(w).all() or not np.array_equal(w,independent):
        raise ValueError('independent weight decoder disagrees')
    return w


def decompose(reference,quantized,native):
    ref,quant,got=[np.asarray(x,np.float64) for x in (reference,quantized,native)]
    if any(x.ndim!=1 or not x.size or x.shape!=ref.shape or not np.isfinite(x).all() for x in (ref,quant,got)):
        raise ValueError('equal finite nonempty projection vectors required')
    v=dict(reference=ref,quantized=quant,native=got,activation_quantization=quant-ref,
           arithmetic_remainder=got-quant,total=got-ref)
    if np.any(abs(v['activation_quantization']+v['arithmetic_remainder']-v['total'])>
              16*np.finfo(np.float64).eps*(abs(ref)+abs(quant)+abs(got))):
        raise ValueError('signed projection attribution does not close')
    # Frozen remainder, not an actual corrected native execution.
    v['without_quantization']=got-v['activation_quantization']
    scale=float(np.max(abs(ref)));metrics={}
    for key in ('activation_quantization','arithmetic_remainder','total','without_quantization'):
        delta=v[key]-ref if key=='without_quantization' else v[key]
        error=float(np.max(abs(delta)))
        metrics[key]=dict(maxabs=error,reference_maxabs=scale,maxrel=error/scale if scale else (0. if error==0 else None),
                          rms=float(np.sqrt(np.mean(delta*delta))))
    return metrics,v


def coverage(rows):
    if [(r['role'],r['position']) for r in rows]!=[(k,p) for k in replay.ROLES for p in range(240)]:
        raise ValueError('incomplete or reordered projection attribution')


def require_control(r,a,sha):
    summary=dict(control_valid=True,exact_roles=960,total_roles=960,exact_tables=240,total_tables=240,table_valid=True)
    if any(x.get('complete') is not True or x.get('model_acceptance') is not False or
           x.get('interpretation_allowed') is not True or x.get('summary')!=summary for x in (r,a)) or (
            a.get('report_sha256')!=sha or a.get('artifacts')!=480):
        raise ValueError('requires complete bound exact projection/table control')


def validate(capture):
    prep=replay.verified(capture)
    r=json.loads((capture/'report.json').read_text());a=json.loads((capture/'audit.json').read_text())
    require_control(r,a,digest(capture/'report.json'))
    if (r['preparation_sha256']!=digest(capture/'preparation.json') or r['log_sha256']!=digest(capture/'native.log') or
            r['harness_sha256']!=digest(r['harness']) or
            r['harness_evidence_sha256']!=replay.native.verify_harness(r['harness'])):
        raise ValueError('changed native projection provenance')
    summary,rows,hashes=replay.score(capture)
    if summary!=r['summary'] or rows!=r['rows'] or hashes!=r['artifacts']:raise ValueError('changed native measurements')
    inputs=prep['inputs']|prep['artifacts']|prep['sources']
    inputs.update({str(capture/n):h for n,h in hashes.items()})
    inputs.update({str(capture/n):digest(capture/n) for n in ('preparation.json','report.json','audit.json','native.log')})
    inputs[r['harness']]=r['harness_sha256']
    inputs[str(Path(r['harness']).parent/'build-evidence.json')]=r['harness_evidence_sha256']
    return inputs


def evaluate(capture,out,write):
    x=np.array([np.fromfile(capture/f'x{p}.bin','<f4') for p in range(240)],dtype=np.float64)
    if x.shape!=(240,5120) or not np.isfinite(x).all():raise ValueError('invalid input norm inventory')
    decoded=[table((capture/f'table{p}.bin').read_bytes(),5120) for p in range(240)]
    xq=np.array([t[0] for t in decoded]);sums=np.array([t[1] for t in decoded])
    ys=np.array([np.fromfile(capture/f'y{p}.bin','<f4') for p in range(240)],dtype=np.float64)
    if ys.shape!=(240,14336):raise ValueError('invalid native projection inventory')
    rows=[];artifacts={};weight_hashes={};start=0
    with (capture/'weights.bin').open('rb') as pool:
        for role,n in zip(replay.ROLES,replay.ROWS):
            print('Attributing',role,'over 240 positions',flush=True)
            w=weights(pool.read(n//32*20*5120),n,5120)
            import hashlib
            weight_hashes[role]=hashlib.sha256(w.tobytes()).hexdigest()
            # Batched FP64 references share a reduction path and the same weights.
            ref=x@w.T;quant=xq@w.T;arrays={}
            for p in range(240):
                metrics,vectors=decompose(ref[p],quant[p],ys[p,start:start+n])
                rows.append(dict(role=role,position=p,metrics=metrics))
                for k,v in vectors.items():arrays.setdefault(k,[]).append(v)
            arrays={k:np.array(v,dtype='<f8') for k,v in arrays.items()}
            path=out/f'{role}.npz'
            if write:np.savez(path,**arrays)
            else:
                with np.load(path,allow_pickle=False) as saved:
                    if set(saved.files)!=set(arrays):raise ValueError('changed attribution array inventory')
                    for k,v in arrays.items():
                        old=saved[k]
                        if old.shape!=v.shape or old.dtype!=v.dtype or old.tobytes()!=v.tobytes():
                            raise ValueError('attribution array replay differs')
            artifacts[path.name]=digest(path);start+=n
            del w,arrays,ref,quant
        if pool.read(1):raise ValueError('extra packed weights')
    coverage(rows)
    worst={role:{k:max((r for r in rows if r['role']==role),key=lambda r:r['metrics'][k]['maxabs'])
                 for k in ('activation_quantization','arithmetic_remainder','total')}
           for role in replay.ROLES}
    return dict(rows=rows,artifacts=artifacts,weight_sha256=weight_hashes,worst_maxabs=worst,
                selected=[r for r in rows if r['position'] in (125,205,239)],
                table_sum_rounding_maxabs=float(np.max(abs(sums-xq.reshape(240,160,32).sum(axis=2)))))


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,model_acceptance=False,sources=sources())
    try:
        capture=a.capture.resolve();inputs=validate(capture)
        r.update(capture=str(capture),inputs=inputs)
        result=evaluate(capture,out,True)
        base.verify(inputs);base.verify(r['sources'])
        r.update(complete=True,result=result,scope='conditioned projections only; signed INT16 quantization and remaining sum/dot/epilogue/accumulation effects; frozen CPU counterfactual, not a corrected native run')
    except Exception as error:r['error']=str(error);raise
    finally:replay.native.write_report(out/'report.json',r)


def audit(a):
    out=a.out.resolve();proof=dict(complete=False,model_acceptance=False)
    try:
        r=json.loads((out/'report.json').read_text())
        if r.get('complete') is not True or r.get('model_acceptance') is not False or r['sources']!=sources():
            raise ValueError('incomplete or changed attribution report')
        base.verify(r['sources']);base.verify(r['inputs'])
        inputs=validate(Path(r['capture']))
        if inputs!=r['inputs']:raise ValueError('changed attribution inputs')
        if {p.name for p in out.glob('*.npz')}!={k+'.npz' for k in replay.ROLES}:raise ValueError('changed artifact inventory')
        result=evaluate(Path(r['capture']),out,False)
        if result!=r['result']:raise ValueError('changed attribution measurements')
        base.verify(inputs);base.verify(r['sources'])
        proof.update(complete=True,report_sha256=digest(out/'report.json'),cases=960,array_sets=4)
    except Exception as error:proof['error']=str(error);raise
    finally:replay.native.write_report(out/'audit.json',proof)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','audit'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--capture',type=Path)
    a=p.parse_args()
    if a.command=='run' and a.capture is None:p.error('--capture required')
    raise SystemExit(globals()[a.command](a))
