"""Controlled sublayer capture audit and FP64 FFN checks on propagated inputs."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('_sublayer_trace_audit',ROOT/'utilities/prefill-residual-trace/audit.py')
trace=importlib.util.module_from_spec(s);s.loader.exec_module(trace)
base=trace.base
from ffn_front import metric,read_front
TARGETS=((63,7),(60,38),(35,80))


def output_diagnostics(output,res,ideal_ffn,reference):
 ideal_output=np.asarray(res,np.float64)+np.asarray(ideal_ffn,np.float64)
 return dict(propagated_input_only=metric(ideal_output,reference),
             local_ffn_output_error=metric(output,ideal_output))


def ffn_metrics(c,nw,up,gate,down,eps):
 res,norm,h,y=[np.asarray(c[k],np.float64) for k in ('res','norm','h','down')]
 hidden=res.size;ff=h.size
 if (res.shape!=(hidden,) or norm.shape!=res.shape or y.shape!=res.shape or
     np.shape(nw)!=res.shape or np.shape(up)!=(ff,hidden) or np.shape(gate)!=(ff,hidden) or
     np.shape(down)!=(hidden,ff) or not np.isfinite(eps) or eps<=0):raise ValueError('invalid FFN shape/epsilon')
 if not all(np.isfinite(x).all() for x in [res,norm,h,y,nw,up,gate,down]):raise ValueError('non-finite FFN input')
 xr=base.oracle.rms(res,eps)*nw
 u=up@xr;g=gate@xr;hr=base.oracle.silu(g)*u
 result=dict(norm=metric(norm,xr),front_h=metric(h,hr),down_from_captured_h=metric(y,down@h),
             ffn_from_captured_res=metric(y,down@hr))
 if 'upgate' in c:
  ug=np.asarray(c['upgate'],np.float64)
  if ug.shape!=(2*ff,) or not np.isfinite(ug).all():raise ValueError('invalid up/gate capture')
  result.update(up=metric(ug[:ff],u),gate=metric(ug[ff:],g),
                activation_from_captured_upgate=metric(h,base.oracle.silu(ug[ff:])*ug[:ff]))
 return result


def causal_res(model,spec,layer,xs):
 """Independent ideal state driven by the supplied complete input prefix."""
 conv=np.zeros((spec.conv_kernel-1,spec.lin_qkv_dim))
 state=np.zeros((spec.lin_value_heads,spec.lin_value_dim,spec.lin_value_dim))
 k=np.zeros((0,spec.num_kv_heads,spec.head_dim));v=k.copy()
 for pos,x in enumerate(xs):
  if spec.layer_types[layer]=='full_attention':out,k,v=base.oracle.attn_decode(model,spec,layer,x,k,v,pos)
  else:out,conv,state=base.oracle.linear_decode(model,spec,layer,x,conv,state)
 return xs[-1]+out


def selected_captures(run,route,layer,pos,meta,offsets):
 prefix=f'{route}_L{layer}_P{pos}_';hidden=meta['hidden']
 spec=base.spec_from_model_dir(Path(meta['model']));ff=spec.intermediate
 if route=='sequential':
  path=run/(prefix+'act.bin');full=spec.layer_types[layer]=='full_attention'
  c=read_front(path,offsets,hidden,ff,full,norm_dtype='float32')
  c['norm']=c.pop('xm');data=path.read_bytes();tag='AA_' if full else 'A_'
  parts=[np.frombuffer(data,'<f4',count=hidden,offset=getattr(offsets,tag+name)).astype(np.float64) for name in ('OUT2','OUT2B')]
  if not all(np.isfinite(x).all() for x in parts):raise ValueError('non-finite down partial')
  c['down']=parts[0]+parts[1];c['sum_bound']=np.abs(c['res'])+np.abs(parts[0])+np.abs(parts[1])
 else:
  counts=dict(input=hidden,res=hidden,norm=hidden,upgate=2*ff,h=ff,down=hidden)
  c={key:base.read(run/(prefix+key+'.bin'),count) for key,count in counts.items()}
  c['sum_bound']=np.abs(c['res'])+np.abs(c['down'])
 return c


def check(a):
 result=dict(capture_valid=False,conditioned_ffn_passed=False,scope='selected propagated sublayers; conditioned ideal-state replay is diagnostic, not a replacement reference')
 try:
  build=trace.verify_build(a.build.resolve(),'paired-sublayer-trace');base.verify(build['compiler_dependencies'])
  baseline=json.loads((a.baseline/'trace-audit.json').read_text())
  if not baseline['capture_valid']:raise ValueError('invalid residual baseline')
  base.verify(baseline['evidence']);base.verify({str(a.baseline/p):h for p,h in baseline['trace_hashes'].items()})
  meta,report=trace.paired.validate_run(a.run.resolve())
  if meta!=json.loads((a.baseline/'reference.json').read_text()) or report['cli_sha256']!=build['cli_sha256']:
   raise ValueError('sublayer reference/CLI differs')
  identities=trace.require_identity(a.baseline,a.run,len(meta['ids']))
  residual_identity={name:base.digest(a.run/name)==h for name,h in baseline['trace_hashes'].items()}
  if not all(residual_identity.values()):raise ValueError('residual trace differs from baseline')
  expected={f'sequential_L{l}_P{p}_act.bin' for l,p in TARGETS}
  expected.update(f'{r}_L{l}_P{p}_{key}.bin' for r in ('hybrid','block_major') for l,p in TARGETS for key in ('input','res','norm','upgate','h','down'))
  if {p.name for p in a.run.glob('*_L*_P*.bin')}!=expected:raise ValueError('incomplete sublayer capture inventory')
  hashes={name:base.digest(a.run/name) for name in expected}
  for l,p in TARGETS:
   for key in ('input','res','norm','upgate','h','down'):
    if hashes[f'hybrid_L{l}_P{p}_{key}.bin']!=hashes[f'block_major_L{l}_P{p}_{key}.bin']:
     raise ValueError('hybrid sublayer schedules differ')
  spec=base.spec_from_model_dir(Path(meta['model']));offsets=base.layout(spec,max_ctx=meta['max_ctx']);hidden=meta['hidden'];n=len(meta['ids'])
  model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=hidden;model.native_q8=lambda name:False
  plain=model.matmul_w;cache={}
  def cached(name,o,i):
   if name not in cache:cache[name]=plain(name,o,i)
   return cache[name]
  model.matmul_w=cached
  cases=[]
  for l,p in TARGETS:
   cache.clear();pre=f'model.layers.{l}.'
   ref_inputs=base.read(a.run/f'ref_res{l-1}.bin',n*hidden).reshape(n,hidden)[:p+1].astype(np.float32)
   independent_res=causal_res(model,spec,l,ref_inputs)
   independent_out=independent_res+base.oracle.ffn(model,spec,l,independent_res)
   expected_out=base.read(a.run/f'ref_res{l}.bin',n*hidden).reshape(n,hidden)[p]
   if independent_out.astype('<f4').tobytes()!=expected_out.astype('<f4').tobytes():raise ValueError('causal CPU reconstruction differs from retained reference')
   nw=model.bf16(pre+'post_attention_layernorm.weight').astype(np.float64)
   up=cached(pre+'mlp.up_proj.weight',spec.intermediate,hidden).astype(np.float64)
   gate=cached(pre+'mlp.gate_proj.weight',spec.intermediate,hidden).astype(np.float64)
   down=cached(pre+'mlp.down_proj.weight',hidden,spec.intermediate).astype(np.float64)
   for route in trace.ROUTES:
    inputs=trace.read_trace(a.run/f'{route}_res{l-1}.bin',n+1,hidden)[:p+1].astype(np.float32)
    output=trace.read_trace(a.run/f'{route}_res{l}.bin',n+1,hidden)[p]
    c=selected_captures(a.run,route,l,p,meta,offsets)
    if 'input' in c and not np.array_equal(c['input'],inputs[-1]):raise ValueError('hybrid input differs from previous residual')
    error=np.abs(output-(c['res']+c['down']))
    if np.any(error>4*np.finfo(np.float32).eps*c['sum_bound']):raise ValueError('captured residual/down do not reconstruct layer output')
    ideal=causal_res(model,spec,l,inputs)
    metrics=ffn_metrics(c,nw,up,gate,down,spec.norm_eps)
    propagation=output_diagnostics(output,c['res'],base.oracle.ffn(model,spec,l,c['res']),independent_out)
    cases.append(dict(route=route,layer=l,position=p,cpu_reference_exact=True,
        input=metric(inputs[-1],ref_inputs[-1]),pre_ffn=metric(c['res'],independent_res),
        output=metric(output,independent_out),
        ideal_state_from_native_input=metric(c['res'],ideal),ffn=metrics,ffn_propagation=propagation))
    print(route,'L',l,'P',p,'pre_ffn',cases[-1]['pre_ffn']['maxrel'],'output',cases[-1]['output']['maxrel'],'ffn', {k:v['maxrel'] for k,v in metrics.items()},flush=True)
   del up,gate,down;cache.clear()
  evidence=[Path(__file__),ROOT/'open_kernels/model/ffn_front.py',a.build/'build-evidence.json',a.baseline/'trace-audit.json',a.run/'reference.json',a.run/'report.json']
  result.update(capture_valid=True,conditioned_ffn_passed=all(m['passed'] for c in cases for m in c['ffn'].values()),
                model_acceptance=False,logit_identity=identities,residual_identity=residual_identity,capture_hashes=hashes,cases=cases,
                evidence={str(p.resolve()):base.digest(p) for p in evidence})
 except Exception as e:
  result.update(capture_valid=False,conditioned_ffn_passed=False,error=str(e));raise
 finally:a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return 0 if result['conditioned_ffn_passed'] else 1

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('build','baseline','run','out'):p.add_argument('--'+name,type=Path,required=True)
 raise SystemExit(check(p.parse_args()))
