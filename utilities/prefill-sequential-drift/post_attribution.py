"""CPU signed attribution of L0 post-output rounding; never model acceptance."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

s=importlib.util.spec_from_file_location('_post_l0_boundaries',Path(__file__).with_name('l0_boundaries.py'))
boundary=importlib.util.module_from_spec(s);s.loader.exec_module(boundary)
native=boundary.native
base,digest,drift=native.base,native.digest,native.drift
di,ds=boundary.di,boundary.ds
from q4nx import f32_to_bf16, bf16_to_f32
TARGETS=boundary.TARGETS
NODES=('reference','oracle','independent','recurrence','z','post_rounded','post_native','projection','residual_add','native')
PARTS=('reference_rounding','reference_math','recurrence','z','post_rounding','post_arithmetic','output_projection','residual_add','ffn')


def exact_reference(got,ref):
    a,b=np.asarray(got),np.asarray(ref)
    if (a.ndim!=1 or a.shape!=b.shape or not a.size or not np.isfinite(a).all()
            or not np.isfinite(b).all() or a.astype('<f4').tobytes()!=b.astype('<f4').tobytes()):
        raise ValueError('independent L0 reference does not reconstruct exactly')


def check_coverage(rows):
    if len(rows)!=len(TARGETS) or [r['position'] for r in rows]!=list(TARGETS):
        raise ValueError('incomplete, duplicate or reordered target inventory')


def decompose(nodes):
    if set(nodes)!=set(NODES):raise ValueError('incomplete attribution nodes')
    values=[np.asarray(nodes[k],np.float64) for k in NODES]
    if any(x.ndim!=1 or not x.size or x.shape!=values[0].shape or not np.isfinite(x).all() for x in values):
        raise ValueError('equal complete finite attribution vectors required')
    vectors={k:b-a for k,a,b in zip(PARTS,values,values[1:])}
    vectors['total']=values[-1]-values[0]
    closure=sum(vectors[k] for k in PARTS)-vectors['total']
    bound=32*np.finfo(np.float64).eps*sum(np.abs(x) for x in values)
    if np.any(np.abs(closure)>bound):raise ValueError('signed attribution does not close')
    metrics={k:drift.scan.error_score(v,values[0]) for k,v in vectors.items()}
    # Frozen downstream errors: a CPU counterfactual, not a corrected NPU run.
    vectors['without_post_rounding']=values[-1]-vectors['post_rounding']
    metrics['without_post_rounding']=drift.scan.error_score(vectors['without_post_rounding']-values[0],values[0])
    return metrics,vectors


def tail(res,w,eps):
    res=np.asarray(res,np.float64)
    xn=base.oracle.rms(res,eps)*w['post_norm']
    up,gate=w['up']@xn,w['gate']@xn
    return res+w['down']@(up*base.oracle.silu(gate))


class Cached:
    def __init__(self,model):self.model,self.cache=model,{}
    def matmul_w(self,*key):
        if key not in self.cache:self.cache[key]=self.model.matmul_w(*key)
        return self.cache[key]
    def bf16(self,*args):return self.model.bf16(*args)
    def f32(self,*args):return self.model.f32(*args)


def sources():
    return boundary.sources() | {str(Path(__file__).resolve()):digest(__file__)}


def validate(capture):
    prep=native.verified(capture)
    r=json.loads((capture/'report.json').read_text());a=json.loads((capture/'audit.json').read_text())
    boundary.require_control(r,a,digest(capture/'report.json'))
    identity,hashes=native.score(capture,prep)
    if identity!=r['identity'] or hashes!=r['artifacts']:raise ValueError('native control changed')
    base.verify({r['harness']:r['harness_sha256'],str(capture/'native.log'):r['log_sha256']})
    inputs=prep['inputs'] | prep['artifacts'] | {str(capture/n):h for n,h in hashes.items()}
    inputs.update({str(capture/n):digest(capture/n) for n in ('preparation.json','report.json','audit.json')})
    refs=[Path(p) for p in prep['inputs'] if p.endswith('/257/ref_res0.bin')]
    if len(refs)!=1:raise ValueError('ambiguous independent L0 reference')
    return prep,inputs,refs[0]


def evaluate(capture,prep,refpath,out=None,expected=None):
    spec,_,_=native.context(prep['meta']);off=boundary.layout(spec,max_ctx=prep['meta']['max_ctx'])
    reference=drift.trace.read_trace(refpath,257,spec.hidden).astype('<f4')
    model=base.Q4NX(Path(prep['meta']['model'])/'model.q4nx');model.native_q8=lambda name:False
    cached=Cached(model);prefix='model.layers.0.'
    rows,artifacts=[],{}
    try:
        dw,_=di.load_weights(model,spec,0)
        w={k:cached.matmul_w(prefix+name,o,i).astype(np.float64) for k,name,o,i in (
            ('z','self_attn.gate_proj.weight',spec.lin_value_width,spec.hidden),
            ('out','linear_attn.ssm_out_proj.weight',spec.hidden,spec.lin_value_width),
            ('up','mlp.up_proj.weight',spec.intermediate,spec.hidden),
            ('gate','mlp.gate_proj.weight',spec.intermediate,spec.hidden),
            ('down','mlp.down_proj.weight',spec.hidden,spec.intermediate))}
        w['post_norm']=model.bf16(prefix+'post_attention_layernorm.weight').astype(np.float64)
        nw=model.bf16(prefix+'linear_attn.ssm_norm.weight').astype(np.float64)
        conv=np.zeros((spec.conv_kernel-1,spec.lin_qkv_dim));state=np.zeros((spec.lin_value_heads,128,128))
        history=conv.copy();independent_state=state.copy()
        for p in range(native.COUNT):
            x=base.read(capture/f'input_{p}.bin',spec.hidden).astype('<f4')
            # The existing mixed-precision oracle is a separate identity gate.
            block,conv,state=base.oracle.linear_decode(cached,spec,0,x,conv,state)
            exact=base.oracle.rms(x,spec.norm_eps)*dw['norm']
            qkv=dw['qkv']@exact
            records,history=di.records_reference(spec,exact,qkv,history,dw)
            independent_state,independent_o=ds.step_reference(independent_state,records)
            if p not in TARGETS:continue
            oracle_res=x+block;oracle=oracle_res+base.oracle.ffn(cached,spec,0,oracle_res)
            exact_reference(oracle,reference[p])
            data=(capture/f'act_{p}.bin').read_bytes()
            c=ds.read_boundaries(data,(capture/f'state_{p}.bin').read_bytes(),spec,off,conv_dtype='fp32')
            def region(name,count):return ds.finite(np.frombuffer(data,'<f4',count=count,offset=getattr(off,name))).astype(np.float64)
            z=region('A_Z',spec.lin_value_width);independent_z=w['z']@exact
            def post(o,z):return (base.oracle.rms(o,spec.norm_eps)*nw).reshape(-1)*base.oracle.silu(z)
            ideal_post=post(independent_o,independent_z)
            recurrent_post=post(c['output'].astype(np.float64),independent_z)
            local_post=post(c['output'].astype(np.float64),z)
            rounded=bf16_to_f32(f32_to_bf16(local_post.astype('<f4'))).astype(np.float64)
            observed=bf16_to_f32(np.frombuffer(data,'<u2',count=spec.lin_value_width,offset=off.A_OG)).astype(np.float64)
            def compose(og):return tail(x.astype(np.float64)+w['out']@og,w,spec.norm_eps)
            nodes=dict(reference=reference[p].astype(np.float64),oracle=oracle.astype(np.float64),
                independent=compose(ideal_post),recurrence=compose(recurrent_post),z=compose(local_post),
                post_rounded=compose(rounded),post_native=compose(observed),
                projection=tail(x.astype(np.float64)+region('A_OUT',spec.hidden),w,spec.norm_eps),
                residual_add=tail(region('A_RES',spec.hidden),w,spec.norm_eps),
                native=base.read(capture/f'xres_{p}.bin',spec.hidden))
            metrics,vectors=decompose(nodes)
            arrays={**{'node_'+k:v for k,v in nodes.items()},**{'effect_'+k:v for k,v in vectors.items()},
                    'post_independent':ideal_post,'post_conditioned':local_post,'post_rounded':rounded,'post_native':observed,
                    'recurrence_independent':independent_o,'recurrence_native':c['output'],
                    'z_independent':independent_z,'z_native':z}
            name=f'P{p}.npz'
            if out is not None:
                np.savez(out/name,**arrays);artifacts[name]=digest(out/name)
            else:
                with np.load(expected/name,allow_pickle=False) as saved:
                    if set(saved.files)!=set(arrays) or any(saved[k].dtype!=v.dtype or saved[k].shape!=v.shape or saved[k].tobytes()!=v.tobytes() for k,v in arrays.items()):
                        raise ValueError('causal attribution re-audit differs: '+name)
            row=dict(position=p,reference_exact=True,metrics=metrics,
                     rounded_post_mismatches=int(np.count_nonzero(observed!=rounded)),
                     post_rounding_prediction=boundary.front.metric(observed,rounded))
            rows.append(row);print('position',p,json.dumps(row),flush=True)
        peak=sum(v.nbytes for v in cached.cache.values())+sum(v.nbytes for v in dw.values())+sum(v.nbytes for v in w.values())
    finally:cached.cache.clear();model.mm.close();model.f.close()
    check_coverage(rows)
    return rows,artifacts,peak


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,model_acceptance=False,sources=sources())
    try:
        capture=a.capture.resolve();prep,inputs,refpath=validate(capture)
        rows,artifacts,peak=evaluate(capture,prep,refpath,out=out)
        base.verify(inputs);base.verify(r['sources'])
        r.update(complete=True,capture=str(capture),inputs=inputs,artifacts=artifacts,cases=rows,
                 materialized_weight_bytes=peak,
                 scope='CPU telescoping decomposition through FP64 nonlinear tail; independent causal histories; frozen downstream counterfactual is not native correction')
    except Exception as error:r['error']=str(error);raise
    finally:native.write_report(out/'report.json',r)
    return 0


def audit(a):
    out=a.out.resolve();result=dict(complete=False,model_acceptance=False)
    try:
        r=json.loads((out/'report.json').read_text())
        if not r['complete'] or r['model_acceptance'] is not False or r['sources']!=sources():raise ValueError('incomplete or changed attribution')
        check_coverage(r['cases']);base.verify(r['sources']);base.verify(r['inputs'])
        names={f'P{p}.npz' for p in TARGETS}
        if set(r['artifacts'])!=names or {p.name for p in out.glob('*.npz')}!=names:raise ValueError('incomplete attribution artifacts')
        base.verify({str(out/n):h for n,h in r['artifacts'].items()})
        capture=Path(r['capture']);prep,inputs,refpath=validate(capture)
        if inputs!=r['inputs']:raise ValueError('changed attribution inputs')
        rows,_,peak=evaluate(capture,prep,refpath,expected=out)
        if rows!=r['cases'] or peak!=r['materialized_weight_bytes']:raise ValueError('attribution measurements changed')
        base.verify(inputs);base.verify(r['sources'])
        result.update(complete=True,cases=len(rows),report_sha256=digest(out/'report.json'))
    except Exception as error:result['error']=str(error);raise
    finally:native.write_report(out/'audit.json',result)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','audit'])
    p.add_argument('--capture',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='run' and a.capture is None:p.error('--capture required')
    raise SystemExit(globals()[a.command](a))
