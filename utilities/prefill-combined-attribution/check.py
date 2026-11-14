"""CPU attribution of uncorrected combined attention/down projections."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


capture=load('_combined_attribution_capture',ROOT/'utilities/prefill-combined-sublayers/check.py')
rounding=load('_combined_attribution_rounding',ROOT/'utilities/prefill-projection-attribution/check.py')
base,runner,digest=capture.base,capture.runner,capture.digest


def decompose(reference,points):
    reference=np.asarray(reference,np.float64)
    if (reference.ndim!=1 or not reference.size or not points or
        any(np.shape(v)!=reference.shape for v in points.values()) or
        not all(np.isfinite(v).all() for v in [reference,*points.values()])):
        raise ValueError('requires complete finite signed vectors')
    previous=reference;vectors={}
    for key,value in points.items():
        value=np.asarray(value,np.float64);vectors[key]=value-previous;previous=value
    total=previous-reference;gap=np.abs(sum(vectors.values())-total)
    bound=32*np.finfo(np.float64).eps*(np.abs(reference)+sum(np.abs(v) for v in points.values()))
    if np.any(gap>bound):raise ValueError('signed error decomposition failed')
    score=rounding.scan.error_score
    return dict(total=score(total,reference),components={k:score(v,reference) for k,v in vectors.items()},
                closure=dict(passed=True,maxabs=float(gap.max()),max_bound=float(bound.max()))),vectors


def packed_weights(model,layer,op):
    if op['op']!='std_perm' or op.get('split'):raise ValueError('requires unsplit Q4 packing')
    k=op['in_dim'];nch=op['nch'];c0=op.get('chunk0',0)
    name=op['tensor'].format(l=layer);cb=model.chunk_bytes_of(name)
    size=len(model.raw(name))
    if (k<=0 or nch<=0 or c0<0 or size%cb or (c0+nch)*cb>size or
        c0*8192%k or nch*8192%k or (size//cb)*8192%k):
        raise ValueError('invalid packed/source row geometry')
    n=nch*8192//k;row0=c0*8192//k;total=(size//cb)*8192//k
    pool=np.zeros(nch*5120,np.uint8)
    rounding.proj.apply_op(dict(op,dst=0),model,layer,pool)
    w=rounding.proj.dequant_pool(pool,n,k,rs=2)
    source=model.matmul_w(name,total,k)[row0:row0+n]
    if not np.array_equal(w,source):raise ValueError('manifest-packed/source effective weights differ')
    return w,dict(tensor=name,source_row_start=row0,rows=n,columns=k,source_exact=True,
        packed_sha256=hashlib.sha256(pool.tobytes()).hexdigest(),
        effective_sha256=hashlib.sha256(w.tobytes()).hexdigest())


def causal_attention(values,position,qn,kn,spec):
    if (type(position)!=int or position<0 or set(values)!={'q','k','v','gate'} or
        any(np.ndim(v)!=2 or len(v)<=position or not np.isfinite(v).all() for v in values.values())):
        raise ValueError('invalid complete attention prefix')
    arrays={k:np.asarray(v[:position+1],np.float32).astype(np.float64) for k,v in values.items()}
    og,_=capture.attention_reference(arrays,qn,kn,spec.rope_inv_freq(),
        spec.num_heads,spec.num_kv_heads,spec.head_dim,spec.norm_eps)
    return og[position].astype(np.float32).astype(np.float64)


def validate(directory,audit_path):
    audit=json.loads(audit_path.read_text())
    if not audit['capture_valid']:raise ValueError('invalid sublayer audit')
    base.verify(audit['evidence'])
    base.verify({str(directory/name):h for name,h in audit['captures'].items()})
    capture.inventory(directory)
    saved=json.loads((directory/'report.json').read_text())
    if audit['evidence'].get(str((directory/'report.json').resolve()))!=digest(directory/'report.json'):
        raise ValueError('sublayer audit does not bind run report')
    if saved['runner_sha256']!=digest(ROOT/'utilities/prefill-combined-sublayers/run.py'):
        raise ValueError('capture runner changed')
    baseline=Path(saved['baseline']);build=Path(saved['build'])
    original,b,meta,offsets=capture.context(baseline,build)
    if (not saved['complete'] or saved['cli_sha256']!=b['cli_sha256'] or
        saved['baseline_report_sha256']!=digest(baseline/'report.json')):
        raise ValueError('capture build/baseline changed')
    refs=Path(original['prepared'])/'references'
    identity=runner.require_identity(baseline/'combined',directory,len(meta['ids']),meta['layers'])
    if identity!=audit['baseline_identity']:raise ValueError('audited baseline identity changed')
    measured=runner.audit_variant(directory,refs,meta,offsets)
    if measured!=saved['measurements']:raise ValueError('full-depth measurements changed')
    pairs={name:digest(directory/name)==digest(directory/name.replace('hybrid','block_major',1))
           for name in audit['captures'] if name.startswith('hybrid_')}
    if pairs!=audit['hybrid_identity'] or not all(pairs.values()):raise ValueError('hybrid captures differ')
    evidence=audit['evidence']|{str(directory/name):h for name,h in audit['captures'].items()}
    evidence[str(audit_path)]=digest(audit_path)
    return audit,meta,refs,evidence


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    result=dict(complete=False,model_acceptance=False,projection_rows=[],layers=[],artifacts={})
    try:
        directory=a.run.resolve();audit,meta,refs,evidence=validate(directory,a.audit.resolve())
        if meta['layers']!=64 or len(meta['ids'])!=20 or meta['hidden']!=5120:
            raise ValueError('requires complete retained 20-token reference')
        spec=base.spec_from_model_dir(Path(meta['model']));hid=spec.hidden
        bundle=rounding.proj.load_bundle(Path(meta['kernels']),rounding.proj.manifest(spec,max_ctx=meta['max_ctx']))
        if any(bundle[k]!=meta[k] for k in ('build_key','spec_hash')):raise ValueError('export identity changed')
        cases={c['tag']:c for c in rounding.proj.cases(bundle)}
        model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=hid;model.native_q8=lambda name:False
        original_matmul=model.matmul_w;cache={}
        def cached(name,n,k):
            key=(name,n,k)
            if key not in cache:cache[key]=original_matmul(name,n,k)
            return cache[key]
        model.matmul_w=cached
        for l,p in capture.TARGETS:
            cache.clear();arrays={};pre=f'model.layers.{l}.'
            def read(key,width,rows=1):return base.read(directory/f'hybrid_L{l}_P{p}_{key}.bin',rows*width).reshape(rows,width)
            incoming=capture.trace.read_trace(directory/f'hybrid_res{l-1}.bin',21,hid)[:20]
            native_output=capture.trace.read_trace(directory/f'hybrid_res{l}.bin',21,hid)[p]
            ref_inputs=base.read(refs/f'ref_res{l-1}.bin',20*hid).reshape(20,hid)[:p+1].astype(np.float32)
            reference_res=capture.sub.causal_res(model,spec,l,ref_inputs)
            reference_out=reference_res+base.oracle.ffn(model,spec,l,reference_res)
            stored=base.read(refs/f'ref_res{l}.bin',20*hid).reshape(20,hid)[p]
            if reference_out.astype('<f4').tobytes()!=stored.astype('<f4').tobytes():
                raise ValueError('independent causal output differs from primary reference')
            res=read('res',hid)[0];h=read('h',spec.intermediate);down=read('down',hid)
            ideal_ffn=base.oracle.ffn(model,spec,l,res)
            cache.clear()
            op=cases['L3_down' if l==63 else 'L0_down']['ops'][0]
            w,weight_evidence=packed_weights(model,l,op)
            prediction=rounding.predict(h,w);del w;cache.clear()
            rows=rounding.head_scores(down,prediction,1)
            rows[0].update(layer=l,position=p,role='down');result['projection_rows'].extend(rows)
            points=dict(propagated=res+ideal_ffn,front=res+prediction['exact'][0],
                paired_input=res+prediction['paired'][0],weight_rounding=res+prediction['rounded'][0],
                native_remainder=res+down[0],residual_add=native_output)
            scores,vectors=decompose(reference_out,points)
            arrays.update({'down_'+k:v for k,v in prediction.items()})
            arrays.update({'ffn_component_'+k:v for k,v in vectors.items()})
            arrays.update(reference_res=reference_res,reference_output=reference_out,native_output=native_output,
                          captured_h=h,captured_down=down,native_res=res)
            layer=dict(layer=l,position=p,cpu_reference_exact=True,down_weight=weight_evidence,ffn=scores)
            if l==63:
                entry=cases['L3_entry'];observed={};preds={};weights={};packed={}
                xn=read('xn',hid,20)
                for role,op in zip(('q','k','v','gate'),entry['ops']):
                    w,packed[role]=packed_weights(model,l,op)
                    observed[role]=read(role,w.shape[0],20);preds[role]=rounding.predict(xn,w)
                    heads=spec.num_heads if role in ('q','gate') else spec.num_kv_heads
                    rows=rounding.head_scores(observed[role],preds[role],heads)
                    for row in rows:row.update(layer=l,role=role)
                    result['projection_rows'].extend(rows)
                    arrays.update({role+'_'+k:v for k,v in preds[role].items()});arrays[role+'_native']=observed[role]
                    del w;cache.clear()
                ow,packed['out']=packed_weights(model,l,cases['L3_out']['ops'][0])
                og=read('og',spec.num_heads*spec.head_dim,20);native_proj=read('proj',hid)
                outpred=rounding.predict(og[p:p+1],ow)
                rows=rounding.head_scores(native_proj,outpred,1)
                rows[0].update(layer=l,position=p,role='out');result['projection_rows'].extend(rows)
                arrays.update({'out_'+k:v for k,v in outpred.items()})
                qn=model.bf16(pre+'self_attn.q_norm.weight');kn=model.bf16(pre+'self_attn.k_norm.weight')
                exact={k:v['exact'] for k,v in preds.items()}
                def suffix(values):
                    return incoming[p]+causal_attention(values,p,qn,kn,spec)@ow.astype(np.float64).T
                joint=suffix(exact);host=suffix(observed)
                points=dict(propagated_and_norm=joint,entry=host,host_arithmetic=incoming[p]+outpred['exact'][0],
                    paired_input=incoming[p]+outpred['paired'][0],weight_rounding=incoming[p]+outpred['rounded'][0],
                    native_remainder=incoming[p]+native_proj[0],residual_add=res)
                attention,components=decompose(reference_res,points)
                counterfactuals={role:suffix(observed|{role:exact[role]}) for role in observed}
                counterfactuals.update(joint=joint,host=host,out=incoming[p]+outpred['exact'][0])
                layer.update(attention_weights=packed,attention=attention,
                    attention_counterfactuals={k:rounding.scan.error_score(v-reference_res,reference_res) for k,v in counterfactuals.items()})
                arrays.update({'attention_component_'+k:v for k,v in components.items()})
                arrays.update({'attention_counterfactual_'+k:v for k,v in counterfactuals.items()})
                # Recompute an ideal FFN after each attention substitution. These are
                # conditioned suffix diagnostics, not independent model acceptance.
                del ow;cache.clear()
                layer['ideal_suffix_outputs']={}
                for key,value in counterfactuals.items():
                    output=value+base.oracle.ffn(model,spec,l,value)
                    layer['ideal_suffix_outputs'][key]=rounding.scan.error_score(output-reference_out,reference_out)
                    arrays['ideal_suffix_'+key]=output
                cache.clear()
            path=out/f'L{l}_P{p}.npz';np.savez(path,**arrays)
            result['artifacts'][path.name]=digest(path);result['layers'].append(layer)
            print(json.dumps(layer),flush=True)
        rows=result['projection_rows'];expected=20*(2*spec.num_heads+2*spec.num_kv_heads)+4
        if len(rows)!=expected:raise ValueError('incomplete projection coverage')
        summary={}
        for role in ('q','k','v','gate','out','down'):
            group=[r for r in rows if r['role']==role]
            summary[role]=dict(total=len(group),native_passed=sum(r['native']['passed'] for r in group),
                worst_native=max(r['native']['maxrel'] for r in group),
                worst_prediction_gap=max(r['prediction_gap']['maxrel'] for r in group),
                worst_remaining_error_ratio=max(r['remaining_error_ratio'] for r in group))
        base.verify(evidence)
        sources=[Path(__file__),Path(capture.__file__),Path(rounding.__file__),Path(rounding.scan.__file__),
            Path(rounding.proj.__file__),Path(capture.sub.__file__),Path(base.oracle.__file__),Path(capture.rb.__file__),
            ROOT/'open_kernels/model/q4nx.py',ROOT/'open_kernels/q4_1_pack.py',ROOT/'open_kernels/recipes/pack.py']
        result.update(complete=True,summary=summary,inputs=evidence,
            sources={str(p.resolve()):digest(p) for p in sources},hybrid_schedule_representative=True,
            scope='CPU rounding attribution and conditioned suffixes; not byte-exact NPU replay or model acceptance')
    except Exception as error:
        result.update(complete=False,error=str(error));raise
    finally:(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','audit','out'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
