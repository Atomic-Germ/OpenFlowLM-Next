"""Independent causal entry/recurrence and conditioned L0 projection/FFN audit."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

s=importlib.util.spec_from_file_location('_boundary_native_l0',Path(__file__).with_name('native_l0.py'))
native=importlib.util.module_from_spec(s);s.loader.exec_module(native)
base,digest=native.base,native.digest
import deltanet_inputs as di
import deltanet_state as ds
import ffn_front as front
import ffn_partials as down
from recipes.qwen35 import layout, ffn_geometry
from q4nx import bf16_to_f32
TARGETS=(36,37,38,240,241,242)


def require_control(r,a,sha):
    expected=dict(passed=True,positions=243,matching_positions=243,first_mismatch=None)
    if (r.get('complete') is not True or r.get('control_valid') is not True or
            r.get('boundary_interpretation_allowed') is not True or r.get('model_acceptance') is not False or
            r.get('identity')!=expected or a.get('identity')!=expected or
            a.get('complete') is not True or a.get('control_valid') is not True or
            a.get('model_acceptance') is not False or a.get('artifacts')!=729 or a.get('report_sha256')!=sha):
        raise ValueError('requires complete byte-exact independently audited control')


def summarize_delta(rows):
    if len(rows)!=243 or [r['token'] for r in rows]!=list(range(243)):
        raise ValueError('incomplete causal boundary sequence')
    flags=[r['passed'] and r['recurrence']['head_diagnostics_passed'] for r in rows]
    return dict(total=len(rows),passed=sum(flags),first_failure=next((p for p,v in enumerate(flags) if not v),None))


def sources():
    return native.sources() | {str(Path(p).resolve()):digest(p) for p in
        (__file__,di.__file__,ds.__file__,front.__file__,down.__file__)}


class Weights:
    def __init__(self,model):self.model,self.cache=model,{}
    def matmul_w(self,*key):
        if key not in self.cache:self.cache[key]=self.model.matmul_w(*key)
        return self.cache[key]
    def bf16(self,*args):return self.model.bf16(*args)


def projections(data,off,spec,weights,x,c):
    prefix='model.layers.0.';vw=spec.lin_value_width
    def f32(name,n):return ds.finite(np.frombuffer(data,'<f4',count=n,offset=getattr(off,name))).astype(np.float64)
    z,out,res=f32('A_Z',vw),f32('A_OUT',spec.hidden),f32('A_RES',spec.hidden)
    og=ds.finite(bf16_to_f32(np.frombuffer(data,'<u2',count=vw,offset=off.A_OG))).astype(np.float64)
    wz=weights.matmul_w(prefix+'self_attn.gate_proj.weight',vw,spec.hidden).astype(np.float64)
    wo=weights.matmul_w(prefix+'linear_attn.ssm_out_proj.weight',spec.hidden,vw).astype(np.float64)
    norm=weights.bf16(prefix+'input_layernorm.weight').astype(np.float64)
    nw=weights.bf16(prefix+'linear_attn.ssm_norm.weight').astype(np.float64)
    exact=x/np.sqrt(np.mean(x*x)+spec.norm_eps)*norm
    post=base.oracle.rms(c['output'].astype(np.float64),spec.norm_eps)*nw
    metrics=dict(z=front.metric(z,wz@exact),z_conditioned=front.metric(z,wz@c['xn']),
                 post=front.metric(og,post.reshape(vw)*base.oracle.silu(z)),
                 out=front.metric(out,wo@og),residual_add=front.metric(res,x+out))
    return dict(passed=all(v['passed'] for v in metrics.values()),metrics=metrics)


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    result=dict(complete=False,model_acceptance=False,sources=sources())
    try:
        capture=a.capture.resolve();prep=native.verified(capture)
        r=json.loads((capture/'report.json').read_text());audit=json.loads((capture/'audit.json').read_text())
        require_control(r,audit,digest(capture/'report.json'))
        identity,hashes=native.score(capture,prep)
        if identity!=r['identity'] or hashes!=r['artifacts']:raise ValueError('changed native captures')
        result.update(capture=str(capture),inputs={str(capture/n):digest(capture/n)
                      for n in ('preparation.json','report.json','audit.json')},capture_sha256=hashes)
        spec,bundle,_=native.context(prep['meta']);off=layout(spec,max_ctx=prep['meta']['max_ctx'])
        entry_dtype,conv_dtype=ds.linear_dtypes(bundle)
        inputs=[];captures=[]
        for p in range(native.COUNT):
            act=(capture/f'act_{p}.bin').read_bytes()
            c=ds.read_boundaries(act,(capture/f'state_{p}.bin').read_bytes(),spec,off,conv_dtype=conv_dtype)
            c['xn']=di.read_entry(act,off,spec.hidden,norm_dtype=entry_dtype).astype(np.float64)
            captures.append(c);inputs.append(base.read(capture/f'input_{p}.bin',spec.hidden))
        model=base.Q4NX(Path(prep['meta']['model'])/'model.q4nx');model.native_q8=lambda name:False
        try:
            weights,weight_hashes=di.load_weights(model,spec,0)
            print('Checking all 243 entry/conv/recurrent positions',flush=True)
            delta=di.score_sequence(spec,weights,inputs,captures,conv_dtype=conv_dtype)
            del weights
            native.write_report(out/'delta.json',dict(cases=delta,summary=summarize_delta(delta),weights=weight_hashes))
            cached=Weights(model);selected=[];prefix='model.layers.0.'
            for p in TARGETS:
                act=capture/f'act_{p}.bin'
                f=front.read_front(act,off,spec.hidden,spec.intermediate,False,norm_dtype='float32')
                front_score=front.compare_front(f['res'],cached.bf16(prefix+'post_attention_layernorm.weight'),
                    cached.matmul_w(prefix+'mlp.up_proj.weight',spec.intermediate,spec.hidden),
                    cached.matmul_w(prefix+'mlp.gate_proj.weight',spec.intermediate,spec.hidden),f['xm'],f['h'],spec.norm_eps)
                d=down.read_regions(act,off,spec.hidden,spec.intermediate,False)
                down_score=down.compare_partials(d['h'],cached.matmul_w(prefix+'mlp.down_proj.weight',spec.hidden,spec.intermediate),
                    d['res'],d['partials'],base.read(capture/f'xres_{p}.bin',spec.hidden),tuple(ffn_geometry(spec).DOWN_SPLIT))
                projection_score=projections(act.read_bytes(),off,spec,cached,inputs[p],captures[p])
                row=dict(position=p,front=front_score,down=down_score,projections=projection_score)
                row['passed']=all(row[k]['passed'] for k in ('front','down','projections'))
                selected.append(row);print('selected boundary',p,'PASS' if row['passed'] else 'FAIL',flush=True)
            cached.cache.clear()
        finally:model.mm.close();model.f.close()
        base.verify(result['sources']);base.verify(result['inputs'])
        base.verify({str(capture/n):sha for n,sha in hashes.items()})
        summary=summarize_delta(delta)
        result.update(complete=True,delta=summary,selected=selected,
                      passed=summary['passed']==243 and all(row['passed'] for row in selected),
                      artifacts={'delta.json':digest(out/'delta.json')},
                      scope='independent FP64 entry/conv/state from zero; strict recurrence conditioned on captured records; selected Z/out and FFN conditioned on captured inputs; no model acceptance')
    except Exception as error:result['error']=str(error);raise
    finally:native.write_report(out/'report.json',result)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
