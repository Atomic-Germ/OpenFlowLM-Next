"""Conditioned FP64 full-ax boundaries after exact current L3 native replay."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

s=importlib.util.spec_from_file_location('_boundary_native_l3',Path(__file__).with_name('native_l3.py'))
native=importlib.util.module_from_spec(s);s.loader.exec_module(native)
base,digest=native.base,native.digest
import attention_projections as projections
import attention_heads as attention
import ffn_front as front
import ffn_partials as down
from recipes.qwen35 import layout,ffn_geometry
TARGETS=(125,205,239)


def require_control(r,a,sha):
    identity=dict(passed=True,positions=240,matching_positions=240,first_mismatch=None)
    if (r.get('complete') is not True or r.get('control_valid') is not True or
            r.get('boundary_interpretation_allowed') is not True or r.get('model_acceptance') is not False or
            r.get('identity')!=identity or a.get('identity')!=identity or
            a.get('complete') is not True or a.get('control_valid') is not True or
            a.get('model_acceptance') is not False or a.get('artifacts')!=720 or a.get('report_sha256')!=sha):
        raise ValueError('requires exact independently audited current L3 control')


def sources():
    return native.sources() | {str(Path(p).resolve()):digest(p) for p in
        (__file__,projections.__file__,attention.__file__,front.__file__,down.__file__)}


def summarize(rows,heads,selected):
    if ([(r['layer'],r['token']) for r in rows]!=[(3,p) for p in range(240)] or
            [(r['token'],r['head']) for r in heads]!=[(p,h) for p in range(240) for h in range(24)] or
            [r['position'] for r in selected]!=list(TARGETS)):
        raise ValueError('incomplete or duplicate causal boundary coverage')
    counts={k:dict(total=len(v),passed=sum(r['passed'] for r in v)) for k,v in
            (('projections',rows),('attention_heads',heads),('selected_ffn',selected))}
    return dict(passed=all(v['passed']==v['total'] for v in counts.values()),**counts)


def control(capture):
    prep=native.verified(capture)
    r=json.loads((capture/'report.json').read_text());a=json.loads((capture/'audit.json').read_text())
    require_control(r,a,digest(capture/'report.json'))
    if (r['preparation_sha256']!=digest(capture/'preparation.json') or
            r['harness_sha256']!=digest(r['harness']) or
            r['harness_evidence_sha256']!=native.verify_harness(r['harness']) or
            r['log_sha256']!=digest(capture/'native.log')):
        raise ValueError('changed native control provenance')
    identity,hashes=native.score(capture,prep)
    if identity!=r['identity'] or hashes!=r['artifacts']:raise ValueError('changed native captures')
    inputs=dict(prep['inputs']) | prep['artifacts'] | prep['sources']
    inputs.update({str(capture/n):digest(capture/n) for n in ('preparation.json','report.json','audit.json','native.log')})
    inputs.update({str(capture/n):h for n,h in hashes.items()})
    inputs[r['harness']]=r['harness_sha256']
    inputs[str(Path(r['harness']).parent/'build-evidence.json')]=r['harness_evidence_sha256']
    return prep,inputs


def evaluate(capture,prep):
    spec,bundle,_=native.context(prep['meta']);off=layout(spec,max_ctx=prep['meta']['max_ctx'])
    inputs=[];captures=[]
    for p in range(native.COUNT):
        act=capture/f'act_{p}.bin'
        inputs.append(base.read(capture/f'input_{p}.bin',spec.hidden))
        c=attention.read_attention(act,off,spec.num_heads,spec.num_kv_heads,spec.head_dim,'float32')
        c.update(projections.read_boundaries(act,off,spec.hidden,'float32'));captures.append(c)
    model=base.Q4NX(Path(prep['meta']['model'])/'model.q4nx');model.native_q8=lambda name:False
    try:
        print('Scoring 240 independent entry/attention/output positions',flush=True)
        rows,weights=projections.score_layer(spec,model,native.LAYER,inputs,captures)
        prefix=f'model.layers.{native.LAYER}.'
        norms=[model.bf16(prefix+f'self_attn.{k}_norm.weight') for k in ('q','k')]
        ref=attention.attention_reference(*(np.stack([c[k] for c in captures]) for k in ('q','k','v','gate')),
            *norms,spec.norm_eps,spec.rotary_dim,spec.rope_inv_freq())
        heads=attention.compare_heads(np.stack([c['og'] for c in captures]),ref)
        # Decode the three FFN matrices once, only for this one layer.
        up=model.matmul_w(prefix+'mlp.up_proj.weight',spec.intermediate,spec.hidden)
        gate=model.matmul_w(prefix+'mlp.gate_proj.weight',spec.intermediate,spec.hidden)
        wd=model.matmul_w(prefix+'mlp.down_proj.weight',spec.hidden,spec.intermediate)
        norm=model.bf16(prefix+'post_attention_layernorm.weight');selected=[]
        for p in TARGETS:
            act=capture/f'act_{p}.bin'
            f=front.read_front(act,off,spec.hidden,spec.intermediate,True,norm_dtype='float32')
            fs=front.compare_front(f['res'],norm,up,gate,f['xm'],f['h'],spec.norm_eps)
            d=down.read_regions(act,off,spec.hidden,spec.intermediate,True)
            ds=down.compare_partials(d['h'],wd,d['res'],d['partials'],
                base.read(capture/f'xres_{p}.bin',spec.hidden),tuple(ffn_geometry(spec).DOWN_SPLIT))
            selected.append(dict(position=p,front=fs,down=ds,passed=fs['passed'] and ds['passed']))
        return dict(summary=summarize(rows,heads,selected),projections=rows,attention_heads=heads,
                    selected_ffn=selected,projection_weight_sha256=weights)
    finally:model.mm.close();model.f.close()


def run(a):
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(complete=False,model_acceptance=False)
    try:
        capture=a.capture.resolve();prep,inputs=control(capture)
        r.update(capture=str(capture),sources=sources(),inputs=inputs)
        result=evaluate(capture,prep)
        base.verify(r['sources']);base.verify(inputs)
        r.update(complete=True,passed=result['summary']['passed'],result=result,
                 scope='FP64 entry and causal attention from native layer inputs; attention heads conditioned on captured Q/K/V/gate; selected FFN conditioned on captured residual/activation; no model acceptance or signed root-cause attribution')
    except Exception as error:r['error']=str(error);raise
    finally:native.write_report(out/'report.json',r)
    return 0 if r['passed'] else 1


def audit(a):
    out=a.out.resolve();proof=dict(complete=False,model_acceptance=False)
    try:
        r=json.loads((out/'report.json').read_text())
        if not r['complete'] or r['model_acceptance'] is not False or r['sources']!=sources():
            raise ValueError('incomplete or changed boundary report')
        base.verify(r['sources']);base.verify(r['inputs'])
        capture=Path(r['capture']);prep,inputs=control(capture)
        if inputs!=r['inputs']:raise ValueError('changed boundary inputs')
        result=evaluate(capture,prep)
        if result!=r['result'] or result['summary']['passed']!=r['passed']:
            raise ValueError('boundary re-audit differs')
        base.verify(r['sources']);base.verify(inputs)
        proof.update(complete=True,passed=r['passed'],report_sha256=digest(out/'report.json'),summary=result['summary'])
    except Exception as error:proof['error']=str(error);raise
    finally:native.write_report(out/'audit.json',proof)
    return 0 if proof['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','audit']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--capture',type=Path);a=p.parse_args()
    if a.command=='run' and a.capture is None:p.error('--capture required')
    raise SystemExit(globals()[a.command](a))
