"""Signed CPU L3 attribution with real causal KV; never model acceptance."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

s=importlib.util.spec_from_file_location('_l3_attribution_boundaries',Path(__file__).with_name('l3_boundaries.py'))
boundary=importlib.util.module_from_spec(s);s.loader.exec_module(boundary)
native=boundary.native
base,digest=native.base,native.digest
targets=native.drift
drift=targets.drift
from q4nx import bf16_to_f32,f32_to_bf16
TARGETS=boundary.TARGETS
NODES=('reference','oracle','independent','entry_norm','entry_projection','native_kv',
       'attention_native','output_projection','residual_add','ffn_norm','ffn_activation','ffn_down','native')
PARTS=('propagated','reference_math','entry_norm','entry_projection','kv_path','attention_remainder',
       'output_projection','residual_add','ffn_norm','ffn_activation','ffn_down','closing_add')


def exact(got,expected):
    a,b=np.asarray(got),np.asarray(expected)
    if (a.shape!=b.shape or a.dtype!=b.dtype or not a.size or not np.isfinite(a).all()
            or not np.isfinite(b).all() or a.tobytes()!=b.tobytes()):
        raise ValueError('retained control differs byte-for-byte')


def coverage(rows):
    if [r['position'] for r in rows]!=list(TARGETS):raise ValueError('incomplete or reordered attribution targets')


def read_native_kv(capture,count,heads,dim):
    if any(type(x) is not int or x<=0 for x in (count,heads,dim)):raise ValueError('invalid KV geometry')
    row=8*heads*dim;raw=(capture/f'state_{count-1}.bin').read_bytes()
    if len(raw)!=count*row:raise ValueError('incorrect KV size')
    for p in range(count):
        if (capture/f'state_{p}.bin').read_bytes()!=raw[:(p+1)*row]:
            raise ValueError('native KV prefix changed across snapshots')
    v=bf16_to_f32(np.frombuffer(raw,'<u2')).astype(np.float64).reshape(count,2,2,heads,dim)
    if not np.isfinite(v).all():raise ValueError('non-finite native KV')
    return v[:,:,0]+v[:,:,1]


def native_kv_attention(q,gate,qnorm,kv,spec,pos):
    q,gate,qnorm,kv=[np.asarray(x,np.float64) for x in (q,gate,qnorm,kv)]
    if (q.shape!=(spec.num_heads,spec.head_dim) or gate.shape!=q.shape or
            qnorm.shape!=(spec.head_dim,) or kv.shape!=(pos+1,2,spec.num_kv_heads,spec.head_dim) or
            spec.num_heads%spec.num_kv_heads or not all(np.isfinite(x).all() for x in (q,gate,qnorm,kv))):
        raise ValueError('invalid native-KV attention geometry or history')
    qr=base.oracle.rope(base.oracle.rms(q,spec.norm_eps)*qnorm,pos,spec.rotary_dim,spec.rope_inv_freq())
    out=np.empty_like(q);e=np.exp(-np.abs(gate));sigmoid=np.where(gate>=0,1/(1+e),e/(1+e))
    for h in range(spec.num_heads):
        kh=h//(spec.num_heads//spec.num_kv_heads)
        scores=kv[:,0,kh]@qr[h]/np.sqrt(spec.head_dim)
        weights=np.exp(scores-scores.max());weights/=weights.sum()
        out[h]=(weights@kv[:,1,kh])*sigmoid[h]
    return out


def decompose(nodes):
    if set(nodes)!=set(NODES):raise ValueError('incomplete attribution nodes')
    values=[np.asarray(nodes[k],np.float64) for k in NODES]
    if any(x.ndim!=1 or not x.size or x.shape!=values[0].shape or not np.isfinite(x).all() for x in values):
        raise ValueError('equal complete finite attribution vectors required')
    vectors={k:b-a for k,a,b in zip(PARTS,values,values[1:])}
    vectors['total']=values[-1]-values[0];vectors['local_total']=values[-1]-values[1]
    closure=sum(vectors[k] for k in PARTS)-vectors['total']
    if np.any(np.abs(closure)>32*np.finfo(np.float64).eps*sum(np.abs(x) for x in values)):
        raise ValueError('signed attribution does not close')
    metrics={k:drift.scan.error_score(v,values[0]) for k,v in vectors.items()}
    for part in PARTS[1:]:
        # Freeze every other signed effect; this is not a corrected native run.
        vectors['without_'+part]=values[-1]-vectors[part]
        metrics['without_'+part]=drift.scan.error_score(vectors['without_'+part]-values[0],values[0])
    return metrics,vectors


class Cached:
    def __init__(self,model):self.model,self.cache=model,{}
    def matmul_w(self,*key):
        if key not in self.cache:self.cache[key]=self.model.matmul_w(*key)
        return self.cache[key]
    def bf16(self,*args):return self.model.bf16(*args)


def tail(res,w,eps):
    res=np.asarray(res,np.float64);xn=base.oracle.rms(res,eps)*w['post_norm']
    return res+w['down']@boundary.front.activation(w['up'],w['gate'],xn)


def sources():
    return boundary.sources()|{str(Path(__file__).resolve()):digest(__file__)}


def require_audit(r,a,sha,current):
    if (r.get('complete') is not True or a.get('complete') is not True or
            r.get('model_acceptance') is not False or a.get('model_acceptance') is not False or
            a.get('report_sha256')!=sha or r.get('sources')!=current):
        raise ValueError('unbound or stale diagnostic audit')


def validate(boundaries,baseline):
    b=json.loads((boundaries/'report.json').read_text());ba=json.loads((boundaries/'audit.json').read_text())
    require_audit(b,ba,digest(boundaries/'report.json'),boundary.sources())
    if b['passed'] is not True or ba['passed'] is not True or ba['summary']!=b['result']['summary']:
        raise ValueError('requires passing audited L3 boundaries')
    capture=Path(b['capture']);prep,inputs=boundary.control(capture)
    if inputs!=b['inputs']:raise ValueError('changed boundary input inventory')
    r=json.loads((baseline/'report.json').read_text());a=json.loads((baseline/'audit.json').read_text())
    require_audit(r,a,digest(baseline/'report.json'),targets.sources())
    wanted={str(l):list(TARGETS) for l in range(64)}
    if (r['targets']!=wanted or r['baseline']!=prep['baseline'] or a['cases']!=384 or
            a['reference_reconstructions']!=192 or r['reference_reconstructions']!=192 or
            any(inputs.get(p)!=h for p,h in r['inputs'].items()) or
            targets.summarize(r['cases'],{int(k):tuple(v) for k,v in wanted.items()})!=r['summary']):
        raise ValueError('incompatible all-layer attribution control')
    names={f'{route}_L{l}_P{p}.npz' for l in range(64) for p in TARGETS for route in drift.ROUTES}
    if set(r['artifacts'])!=names:raise ValueError('incomplete all-layer artifacts')
    inputs.update({str(baseline/n):h for n,h in r['artifacts'].items()})
    inputs.update({str(d/n):digest(d/n) for d in (boundaries,baseline) for n in ('report.json','audit.json')})
    refs=[Path(p) for p in inputs if p.endswith('/257/ref_res2.bin')]
    if len(refs)!=1:raise ValueError('ambiguous reference input prefix')
    base.verify(inputs);base.verify(sources())
    return capture,prep,inputs,refs[0]


def evaluate(capture,prep,refpath,baseline,out=None,expected=None):
    spec,_,_=native.context(prep['meta']);off=boundary.layout(spec,max_ctx=prep['meta']['max_ctx'])
    xs=np.array([base.read(capture/f'input_{p}.bin',spec.hidden) for p in range(native.COUNT)],dtype='<f4')
    rx=drift.trace.read_trace(refpath,257,spec.hidden).astype('<f4')
    kv=read_native_kv(capture,native.COUNT,spec.num_kv_heads,spec.head_dim)
    captures=[]
    for p in range(native.COUNT):
        path=capture/f'act_{p}.bin'
        c=boundary.attention.read_attention(path,off,spec.num_heads,spec.num_kv_heads,spec.head_dim,'float32')
        c.update(boundary.projections.read_boundaries(path,off,spec.hidden,'float32'));captures.append(c)
    model=base.Q4NX(Path(prep['meta']['model'])/'model.q4nx');model.native_q8=lambda name:False
    cached=Cached(model);prefix=f'model.layers.{native.LAYER}.';rows=[];artifacts={}
    try:
        print('Reconstructing independent and native-input mixed-oracle controls',flush=True)
        oracle,_=drift.replay(cached,spec,3,xs,TARGETS)
        ref_oracle,_=drift.replay(cached,spec,3,rx,TARGETS)
        controls={}
        for p in TARGETS:
            with np.load(baseline/f'sequential_L3_P{p}.npz',allow_pickle=False) as data:
                controls[p]={k:data[k].copy() for k in data.files}
            saved=controls[p]
            for got,old in ((oracle[p],saved['ideal_native']),(ref_oracle[p],saved['ideal_reference']),
                            (ref_oracle[p],saved['reference'].astype('<f4')),
                            (xs[p],saved['native_input']),(rx[p],saved['reference_input']),
                            (base.read(capture/f'xres_{p}.bin',spec.hidden),saved['native'])):exact(got,old)
        w={k:cached.matmul_w(prefix+name,o,i).astype(np.float64) for k,name,o,i in (
            ('qg','self_attn.q_proj.weight',2*spec.attn_q_width,spec.hidden),
            ('k','self_attn.k_proj.weight',spec.attn_kv_width,spec.hidden),
            ('v','self_attn.v_proj.weight',spec.attn_kv_width,spec.hidden),
            ('out','self_attn.o_proj.weight',spec.hidden,spec.attn_q_width),
            ('up','mlp.up_proj.weight',spec.intermediate,spec.hidden),
            ('gate','mlp.gate_proj.weight',spec.intermediate,spec.hidden),
            ('down','mlp.down_proj.weight',spec.hidden,spec.intermediate))}
        for key,name in (('entry_norm','input_layernorm.weight'),('post_norm','post_attention_layernorm.weight'),
                         ('qnorm','self_attn.q_norm.weight'),('knorm','self_attn.k_norm.weight')):
            w[key]=model.bf16(prefix+name).astype(np.float64)
        trajectories={key:[] for key in ('independent','entry_norm','entry_projection')}
        print('Advancing FP64 entry/projection histories and real native KV',flush=True)
        for p,c in enumerate(captures):
            ideal=base.oracle.rms(xs[p],spec.norm_eps)*w['entry_norm']
            for key,norm in (('independent',ideal),('entry_norm',c['xn'])):
                qg=w['qg']@norm;qw=spec.attn_q_width
                trajectories[key].append(dict(q=qg[:qw].reshape(spec.num_heads,spec.head_dim),
                    gate=qg[qw:].reshape(spec.num_heads,spec.head_dim),
                    k=(w['k']@norm).reshape(spec.num_kv_heads,spec.head_dim),
                    v=(w['v']@norm).reshape(spec.num_kv_heads,spec.head_dim)))
            trajectories['entry_projection'].append(c)
        og={key:boundary.attention.attention_reference(*(np.stack([c[k] for c in seq]) for k in ('q','k','v','gate')),
                 w['qnorm'],w['knorm'],spec.norm_eps,spec.rotary_dim,spec.rope_inv_freq())
            for key,seq in trajectories.items()}
        for p in TARGETS:
            x=xs[p].astype(np.float64);c=captures[p]
            def compose(o):return tail(x+w['out']@o.ravel(),w,spec.norm_eps)
            actual_kv_og=native_kv_attention(c['q'],c['gate'],w['qnorm'],kv[:p+1],spec,p)
            act=capture/f'act_{p}.bin'
            f=boundary.front.read_front(act,off,spec.hidden,spec.intermediate,True,norm_dtype='float32')
            d=boundary.down.read_regions(act,off,spec.hidden,spec.intermediate,True)
            nodes=dict(reference=controls[p]['reference'],oracle=oracle[p].astype(np.float64),
                **{key:compose(value[p]) for key,value in og.items()},native_kv=compose(actual_kv_og),
                attention_native=compose(c['og']),output_projection=tail(x+c['out'],w,spec.norm_eps),
                residual_add=tail(c['res'],w,spec.norm_eps),
                ffn_norm=c['res']+w['down']@boundary.front.activation(w['up'],w['gate'],f['xm']),
                ffn_activation=c['res']+w['down']@f['h'],
                ffn_down=c['res']+d['partials'][0]+d['partials'][1],native=controls[p]['native'])
            metrics,vectors=decompose(nodes)
            exact(vectors['local_total'],controls[p]['implementation_gap'])
            exact(vectors['total'],controls[p]['total'])
            arrays={**{'node_'+k:v for k,v in nodes.items()},**{'effect_'+k:v for k,v in vectors.items()},
                    'native_kv':kv[:p+1],'native_kv_og':actual_kv_og,'native_og':c['og'],
                    **{'og_'+key:v[p] for key,v in og.items()}}
            name=f'P{p}.npz'
            if out is not None:np.savez(out/name,**arrays);artifacts[name]=digest(out/name)
            else:
                with np.load(expected/name,allow_pickle=False) as data:
                    if set(data.files)!=set(arrays):raise ValueError('incomplete attribution arrays')
                    for k,v in arrays.items():exact(v,data[k])
            rows.append(dict(position=p,reference_exact=True,local_control_exact=True,metrics=metrics))
            print('position',p,json.dumps(rows[-1]),flush=True)
        peak=sum(v.nbytes for v in cached.cache.values())+sum(v.nbytes for v in w.values())
    finally:cached.cache.clear();model.mm.close();model.f.close()
    coverage(rows)
    return rows,artifacts,peak


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);r=dict(complete=False,model_acceptance=False,sources=sources())
    try:
        boundaries=a.boundaries.resolve();baseline=a.baseline.resolve()
        capture,prep,inputs,refpath=validate(boundaries,baseline)
        rows,artifacts,peak=evaluate(capture,prep,refpath,baseline,out=out)
        base.verify(inputs);base.verify(r['sources'])
        r.update(complete=True,boundaries=str(boundaries),baseline=str(baseline),inputs=inputs,cases=rows,
                 artifacts=artifacts,materialized_weight_bytes=peak,
                 scope='ordered CPU telescoping decomposition on common stored-reference scale; real native KV; attention remainder includes unobserved Q/gating/softmax arithmetic; FFN activation combines unobserved up/gate/SiLU; frozen downstream predictions are not NPU corrections')
    except Exception as error:r['error']=str(error);raise
    finally:native.write_report(out/'report.json',r)
    return 0


def audit(a):
    out=a.out.resolve();proof=dict(complete=False,model_acceptance=False)
    try:
        r=json.loads((out/'report.json').read_text())
        if not r['complete'] or r['model_acceptance'] is not False or r['sources']!=sources():raise ValueError('incomplete or changed attribution')
        coverage(r['cases']);base.verify(r['sources']);base.verify(r['inputs'])
        names={f'P{p}.npz' for p in TARGETS}
        if set(r['artifacts'])!=names or {p.name for p in out.glob('*.npz')}!=names:raise ValueError('incomplete attribution artifacts')
        base.verify({str(out/n):h for n,h in r['artifacts'].items()})
        capture,prep,inputs,refpath=validate(Path(r['boundaries']),Path(r['baseline']))
        if inputs!=r['inputs']:raise ValueError('changed attribution inventory')
        rows,_,peak=evaluate(capture,prep,refpath,Path(r['baseline']),expected=out)
        if rows!=r['cases'] or peak!=r['materialized_weight_bytes']:raise ValueError('attribution re-audit differs')
        base.verify(inputs);base.verify(r['sources'])
        proof.update(complete=True,report_sha256=digest(out/'report.json'),cases=len(rows),reference_reconstructions=6)
    except Exception as error:proof['error']=str(error);raise
    finally:native.write_report(out/'audit.json',proof)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','audit'])
    p.add_argument('--boundaries',type=Path);p.add_argument('--baseline',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='run' and (a.boundaries is None or a.baseline is None):p.error('--boundaries and --baseline required')
    raise SystemExit(globals()[a.command](a))
