"""Diagnose early propagated drift with actual projection prefixes and block state."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('_prefix_selected_audit',ROOT/'utilities/prefill-weighted-sublayers/check.py')
rb=load('_prefix_block_oracle',ROOT/'open_kernels/model/replica_block.py')
base,trace=old.base,old.trace
TARGETS=((34,17),(37,7),(53,17))
old.TARGETS=TARGETS
old.KEYS=old.KEYS+('xn','qkv','z','og','state_before','state_after','proj')
verify_build=old.weighted.build
# Private imported module only: preserve the historical checker and pinned files.
def build(directory,variant):
 return verify_build(directory,'prefix-sublayer' if variant=='weighted-sublayer' else variant)
old.weighted.build=build

def metrics(got,ref):
 got,ref=np.asarray(got,np.float64),np.asarray(ref,np.float64)
 if got.shape!=ref.shape or not got.size:raise ValueError('incomplete projection arrays')
 return old.sub.metric(got,ref)

def states(before,after,offsets,spec):
 if len(before)!=offsets.STATE_BYTES or len(after)!=len(before) or any(before):
  raise ValueError('wrong state extent or nonzero initial state')
 a=np.frombuffer(after,'<f4').astype(np.float64)
 if not np.isfinite(a).all():raise ValueError('non-finite block state')
 n=(spec.conv_kernel-1)*spec.lin_qkv_dim
 if offsets.STATE_S_OFF!=n*4:raise ValueError('requires FP32 conv history')
 count=spec.lin_value_heads*offsets.S_ROWS*spec.lin_value_dim
 state=a[n:n+count].reshape(spec.lin_value_heads,offsets.S_ROWS,spec.lin_value_dim)
 if np.any(state[:,spec.lin_value_dim:]) or np.any(a[n+count:]):raise ValueError('nonzero state padding')
 return a[:n].reshape(spec.conv_kernel-1,spec.lin_qkv_dim),state

def check(a):
 result=dict(capture_valid=False,model_acceptance=False,conditioned_boundaries_passed=False)
 try:
  old.check(a)
  ffn=json.loads(a.out.read_text())
  meta=json.loads((a.run/'reference.json').read_text());n=len(meta['ids']);hidden=meta['hidden']
  if n!=20 or hidden!=5120 or meta['layers']!=64:raise ValueError('requires full-depth short prompt')
  spec=base.spec_from_model_dir(Path(meta['model']));offsets=base.layout(spec,max_ctx=meta['max_ctx'])
  model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=hidden;model.native_q8=lambda name:False
  cases=[]
  for l,p in TARGETS:
   pre=f'model.layers.{l}.'
   def weight(name,o,i):return model.matmul_w(pre+name,o,i).astype(np.float64)
   qkvw=weight('linear_attn.qkv_proj.weight',spec.lin_qkv_dim,hidden)
   zw=weight('self_attn.gate_proj.weight',spec.lin_value_heads*spec.lin_value_dim,hidden)
   ow=weight('linear_attn.ssm_out_proj.weight',hidden,spec.lin_value_heads*spec.lin_value_dim)
   norm=model.bf16(pre+'input_layernorm.weight').astype(np.float64)
   constants=[model.bf16(pre+'linear_attn.'+name).astype(np.float64) for name in
              ('ssm_conv1d.weight','ssm_alpha_proj.bf16.weight','ssm_beta_proj.bf16.weight')]
   constants[1]=constants[1].T;constants[2]=constants[2].T
   constants += [model.f32(pre+'linear_attn.ssm_a').astype(np.float64),model.f32(pre+'linear_attn.ssm_dt.bias').astype(np.float64),model.bf16(pre+'linear_attn.ssm_norm.weight').astype(np.float64)]
   for route in ('hybrid','block_major'):
    prefix=a.run/f'{route}_L{l}_P{p}_'
    def path(key):return Path(str(prefix)+key+'.bin')
    def read(key,width):return base.read(path(key),n*width).reshape(n,width)
    xn,qkv,z,og=(read(k,w) for k,w in [('xn',hidden),('qkv',spec.lin_qkv_dim),('z',zw.shape[0]),('og',zw.shape[0])])
    inputs=trace.read_trace(a.run/f'{route}_res{l-1}.bin',n+1,hidden)[:n]
    exact_norm=base.oracle.rms(inputs,spec.norm_eps)*norm
    conv,state=states(path('state_before').read_bytes(),path('state_after').read_bytes(),offsets,spec)
    ideal_conv=np.zeros_like(conv);ideal_state=np.zeros_like(state)
    ideal_og=rb.deltanet_block(qkv,z,xn,*constants,ideal_conv,ideal_state,key_heads=spec.lin_key_heads,
                              value_heads=spec.lin_value_heads,head_dim=spec.lin_value_dim,eps=spec.norm_eps,conv_f32=True)
    if not np.array_equal(conv,qkv[-(spec.conv_kernel-1):]):raise ValueError('conv carry is not exact captured QKV tail')
    proj=base.read(path('proj'),hidden);res=base.read(path('res'),hidden)
    if not np.array_equal(res.astype(np.float32),(inputs[p].astype(np.float32)+proj.astype(np.float32))):
     raise ValueError('projection/input do not reconstruct pre-FFN residual bytewise')
    expected=dict(norm=exact_norm,qkv_from_input=exact_norm@qkvw.T,z_from_input=exact_norm@zw.T,
                  qkv_from_norm=xn@qkvw.T,z_from_norm=xn@zw.T,og_from_captured_prefix=ideal_og)
    observed=dict(norm=xn,qkv_from_input=qkv,z_from_input=z,qkv_from_norm=qkv,z_from_norm=z,og_from_captured_prefix=og)
    rows=[dict(position=t,metrics={k:metrics(observed[k][t],v[t]) for k,v in expected.items()}) for t in range(n)]
    selected=dict(output_projection=metrics(proj,og[p]@ow.T),
                  block_end_state=metrics(state[:,:spec.lin_value_dim],ideal_state[:,:spec.lin_value_dim]))
    case=dict(route=route,layer=l,position=p,rows=rows,selected=selected,conv_tail_exact=True,
              residual_reconstruction_exact=True,state_position=n-1)
    cases.append(case);print(json.dumps(case),flush=True)
   del qkvw,zw,ow
  files=(Path(__file__),Path(rb.__file__))
  result.update(ffn)
  result.update(projection_cases=cases,conditioned_boundaries_passed=all(m['passed'] for c in cases for m in [*c['selected'].values(),*(m for row in c['rows'] for m in row['metrics'].values())]),
                scope='selected FFN and projections; causal captured-prefix host diagnostics, block-end state at P19; no model acceptance')
  result['evidence'].update({str(p.resolve()):base.digest(p) for p in files})
 except Exception as e:
  result.update(capture_valid=False,conditioned_boundaries_passed=False,error=str(e));raise
 finally:a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return 0 if result['conditioned_ffn_passed'] and result['conditioned_boundaries_passed'] else 1

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('build','baseline','run','out'):p.add_argument('--'+name,type=Path,required=True)
 raise SystemExit(check(p.parse_args()))
