"""Separate propagated input drift from conditioned FFN error in the weighted candidate."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
weighted=load('_weighted_sublayer_audit',ROOT/'utilities/prefill-weighted-cli/audit.py')
sub=load('_weighted_sublayer_math',ROOT/'utilities/prefill-sublayer-trace/check.py')
base,trace=weighted.base,weighted.trace
TARGETS=((55,7),(51,17))
KEYS=('input','res','norm','upgate','h','down')

def expected_captures():
 expected={f'sequential_L{l}_P{p}_act.bin' for l,p in TARGETS}
 expected.update(f'{r}_L{l}_P{p}_{k}.bin' for r in ('hybrid','block_major') for l,p in TARGETS for k in KEYS)
 return expected

def inventory(run):
 if {p.name for p in run.glob('*_L*_P*.bin')}!=expected_captures():raise ValueError('incomplete or unexpected sublayer captures')

def residual_inventory(run,layers):
 expected={f'{route}_res{l}.bin' for route in trace.ROUTES for l in range(layers)}
 # Selected pre-FFN residual captures share the suffix with layer traces.
 # Exclude only the exact declared capture names, never arbitrary prefixes.
 actual={p.name for p in run.glob('*_res*.bin') if not p.name.startswith('ref_')}-expected_captures()
 if actual!=expected:raise ValueError('incomplete or unexpected layer residual inventory')

def check(a):
 result=dict(capture_valid=False,conditioned_ffn_passed=False,model_acceptance=False)
 try:
  build=weighted.build(a.build.resolve(),'weighted-sublayer')
  baseline_build=weighted.build(a.baseline.parent.resolve(),'candidate')
  baseline_audit=a.baseline.parent.parent/'comparison.json'
  prior=json.loads(baseline_audit.read_text())
  if not prior['capture_valid']:raise ValueError('invalid weighted baseline')
  base.verify(prior['evidence'])
  bm,br=trace.paired.validate_run(a.baseline.resolve())
  meta,report=trace.paired.validate_run(a.run.resolve())
  if meta!=bm or report['cli_sha256']!=build['cli_sha256'] or br['cli_sha256']!=baseline_build['cli_sha256']:
   raise ValueError('reference or CLI mismatch')
  if prior['evidence'].get(str((a.baseline/'report.json').resolve()))!=base.digest(a.baseline/'report.json'):
   raise ValueError('baseline report is not bound to prior audit')
  identities=trace.require_identity(a.baseline,a.run,len(meta['ids']))
  residual_inventory(a.run,meta['layers'])
  baseline_hashes=prior['trace_hashes']['candidate']
  base.verify({str(a.baseline/name):h for name,h in baseline_hashes.items()})
  residual_identity={name:base.digest(a.run/name)==h for name,h in baseline_hashes.items()}
  if not all(residual_identity.values()):raise ValueError('sublayer instrumentation changed residuals')
  inventory(a.run);hashes={name:base.digest(a.run/name) for name in expected_captures()}
  for l,p in TARGETS:
   for key in KEYS:
    if hashes[f'hybrid_L{l}_P{p}_{key}.bin']!=hashes[f'block_major_L{l}_P{p}_{key}.bin']:
     raise ValueError('hybrid sublayer schedules differ')
  spec=base.spec_from_model_dir(Path(meta['model']));offsets=base.layout(spec,max_ctx=meta['max_ctx'])
  hidden,n=meta['hidden'],len(meta['ids'])
  if hidden!=5120 or meta['layers']!=64 or n!=20:raise ValueError('requires retained full-depth short prompt')
  model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=hidden;model.native_q8=lambda name:False
  plain=model.matmul_w;cache={}
  def cached(name,o,i):
   if name not in cache:cache[name]=plain(name,o,i)
   return cache[name]
  model.matmul_w=cached;cases=[];cpu_hashes={}
  for l,p in TARGETS:
   cache.clear();pre=f'model.layers.{l}.'
   ref_inputs=base.read(a.run/f'ref_res{l-1}.bin',n*hidden).reshape(n,hidden)[:p+1].astype(np.float32)
   independent_res=sub.causal_res(model,spec,l,ref_inputs)
   independent_out=independent_res+base.oracle.ffn(model,spec,l,independent_res)
   expected_out=base.read(a.run/f'ref_res{l}.bin',n*hidden).reshape(n,hidden)[p]
   if independent_out.astype('<f4').tobytes()!=expected_out.astype('<f4').tobytes():
    raise ValueError('independent causal reconstruction differs from retained CPU output')
   for key,v in [('res',independent_res),('output',independent_out)]:
    path=a.run/f'cpu_L{l}_P{p}_{key}.f64';path.write_bytes(np.asarray(v,'<f8').tobytes());cpu_hashes[str(path.resolve())]=base.digest(path)
   nw=model.bf16(pre+'post_attention_layernorm.weight').astype(np.float64)
   up=cached(pre+'mlp.up_proj.weight',spec.intermediate,hidden).astype(np.float64)
   gate=cached(pre+'mlp.gate_proj.weight',spec.intermediate,hidden).astype(np.float64)
   down=cached(pre+'mlp.down_proj.weight',hidden,spec.intermediate).astype(np.float64)
   for route in trace.ROUTES:
    inputs=trace.read_trace(a.run/f'{route}_res{l-1}.bin',n+1,hidden)[:p+1].astype(np.float32)
    output=trace.read_trace(a.run/f'{route}_res{l}.bin',n+1,hidden)[p]
    c=sub.selected_captures(a.run,route,l,p,meta,offsets)
    if 'input' in c and not np.array_equal(c['input'],inputs[-1]):raise ValueError('input differs from preceding layer')
    if np.any(np.abs(output-(c['res']+c['down']))>4*np.finfo(np.float32).eps*c['sum_bound']):
     raise ValueError('residual/down do not reconstruct layer output')
    ideal=sub.causal_res(model,spec,l,inputs)
    metrics=sub.ffn_metrics(c,nw,up,gate,down,spec.norm_eps)
    propagation=sub.output_diagnostics(output,c['res'],base.oracle.ffn(model,spec,l,c['res']),independent_out)
    case=dict(route=route,layer=l,position=p,cpu_reference_exact=True,
              input=sub.metric(inputs[-1],ref_inputs[-1]),pre_ffn=sub.metric(c['res'],independent_res),
              output=sub.metric(output,independent_out),ideal_state_from_native_input=sub.metric(c['res'],ideal),
              ffn=metrics,ffn_propagation=propagation)
    cases.append(case);print(json.dumps(case),flush=True)
   del up,gate,down;cache.clear()
  files=[Path(__file__),Path(sub.__file__),Path(weighted.__file__),Path(base.oracle.__file__),
         ROOT/'open_kernels/model/ffn_front.py',a.build/'build-evidence.json',a.baseline.parent/'build-evidence.json',
         baseline_audit,a.run/'reference.json',a.run/'report.json']
  result.update(capture_valid=True,conditioned_ffn_passed=all(m['passed'] for c in cases for m in c['ffn'].values()),
                scope='selected causal sublayers; ideal state driven by native inputs is diagnostic only',
                logit_identity=identities,residual_identity=residual_identity,capture_hashes=hashes,cases=cases,
                cpu_outputs=cpu_hashes,evidence={str(p.resolve()):base.digest(p) for p in files})
 except Exception as e:result.update(capture_valid=False,conditioned_ffn_passed=False,error=str(e));raise
 finally:a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return 0 if result['conditioned_ffn_passed'] else 1

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('build','baseline','run','out'):p.add_argument('--'+name,type=Path,required=True)
 raise SystemExit(check(p.parse_args()))
