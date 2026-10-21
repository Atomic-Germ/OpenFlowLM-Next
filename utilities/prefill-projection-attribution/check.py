"""CPU QKV/Z/out rounding attribution on captured causal prefixes; no model acceptance."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prefix=load('_attribution_prefix',ROOT/'utilities/prefill-prefix-sublayers/check.py')
scan=load('_attribution_scan',ROOT/'utilities/prefill-prefix-scan/check.py')
proj=load('_attribution_pack',ROOT/'utilities/prefill-projections/check.py')
scan.sub=prefix.old  # Validate the new 81-capture inventory and private build variant.
base=prefix.base

def round_bf16(x):
 y=proj.bf16_to_f32(proj.f32_to_bf16(x))
 if not np.isfinite(y).all():raise ValueError('outside finite BF16 range')
 return y

def predict(x,w):
 x,w=np.asarray(x,np.float32),np.asarray(w,np.float32)
 if (x.ndim!=2 or w.ndim!=2 or not x.size or not w.size or x.shape[1]!=w.shape[1]
     or not np.isfinite(x).all() or not np.isfinite(w).all()):raise ValueError('invalid projection inputs')
 hi=round_bf16(x);lo=round_bf16(np.subtract(x,hi,dtype=np.float32))
 paired=hi.astype(np.float64)+lo
 result={k:np.empty((len(x),len(w)),np.float64) for k in ('exact','paired','rounded')}
 for start in range(0,len(w),256):
  sl=slice(start,start+256);ww=w[sl].astype(np.float64)
  result['exact'][:,sl]=x.astype(np.float64)@ww.T
  result['paired'][:,sl]=paired@ww.T
  result['rounded'][:,sl]=paired@round_bf16(w[sl]).astype(np.float64).T
 return result

def head_scores(native,predictions,heads):
 native=np.asarray(native,np.float64)
 if (type(heads)!=int or heads<1 or native.ndim!=2 or not native.size or native.shape[1]%heads
     or set(predictions)!= {'exact','paired','rounded'}
     or any(np.shape(x)!=native.shape for x in predictions.values())
     or not all(np.isfinite(x).all() for x in [native,*predictions.values()])):
  raise ValueError('requires complete finite head coverage')
 width=native.shape[1]//heads;rows=[]
 for t in range(len(native)):
  for h in range(heads):
   sl=slice(h*width,(h+1)*width)
   exact,paired,rounded=(predictions[k][t,sl] for k in ('exact','paired','rounded'))
   error=native[t,sl]-exact;gap=native[t,sl]-rounded
   denominator=float(np.linalg.norm(error));remainder=float(np.linalg.norm(gap))
   rows.append(dict(position=t,head=h,native=scan.error_score(error,exact),
     paired_input=scan.error_score(paired-exact,exact),weight_rounding=scan.error_score(rounded-paired,exact),
     prediction_gap=scan.error_score(gap,exact),
     remaining_error_ratio=remainder/denominator if denominator else (0. if remainder==0 else None)))
 return rows

def causal_middle(spec,constants,qkv,z,xn,position):
 if (type(position)!=int or position<0 or np.shape(qkv)!=(len(xn),spec.lin_qkv_dim)
     or np.shape(z)!=(len(xn),spec.lin_value_heads*spec.lin_value_dim)
     or np.shape(xn)!=(len(xn),spec.hidden) or position>=len(xn)
     or not all(np.isfinite(a).all() for a in (qkv,z,xn))):raise ValueError('invalid causal prefix')
 conv=np.zeros((spec.conv_kernel-1,spec.lin_qkv_dim))
 state=np.zeros((spec.lin_value_heads,spec.lin_value_dim,spec.lin_value_dim))
 # Substituted projections cross the same FP32 storage boundary as native GEMM.
 arrays=[np.asarray(a[:position+1],np.float32).astype(np.float64) for a in (qkv,z,xn)]
 return prefix.rb.deltanet_block(*arrays,*constants,conv,state,key_heads=spec.lin_key_heads,
   value_heads=spec.lin_value_heads,head_dim=spec.lin_value_dim,eps=spec.norm_eps,conv_f32=True)[position]

def decompose(native,reference,variants):
 keys={'joint','qkv','z','host','out','projection'}
 if (set(variants)!=keys or np.ndim(native)!=1 or not np.size(native)
     or any(np.shape(x)!=np.shape(native) for x in [reference,*variants.values()])
     or not all(np.isfinite(x).all() for x in [native,reference,*variants.values()])):
  raise ValueError('invalid residual decomposition')
 native,reference=np.asarray(native,np.float64),np.asarray(reference,np.float64)
 v={k:np.asarray(x,np.float64) for k,x in variants.items()}
 components=dict(propagated=v['joint']-reference,
  qkv=v['z']-v['joint'],z=v['qkv']-v['joint'],
  interaction=v['host']-v['z']-v['qkv']+v['joint'],
  host_gap=v['out']-v['host'],output_projection=v['projection']-v['out'],
  residual_add=native-v['projection'])
 total=native-reference;discrepancy=np.abs(sum(components.values())-total)
 bound=32*np.finfo(np.float64).eps*(np.abs(native)+np.abs(reference)+sum(np.abs(x) for x in v.values()))
 if np.any(discrepancy>bound):raise ValueError('signed residual decomposition failed')
 return dict(total=scan.error_score(total,reference),components={k:scan.error_score(x,reference) for k,x in components.items()},
             closure=dict(passed=True,maxabs=float(discrepancy.max()),max_bound=float(bound.max()))),components

def packed_weights(model,layer,op):
 n,k=op['nch']*8192//op['in_dim'],op['in_dim']
 pool=np.zeros(op['nch']*5120,np.uint8)
 proj.apply_op(dict(op,dst=0),model,layer,pool)
 weights=proj.dequant_pool(pool,n,k,rs=2)
 source=model.matmul_w(op['tensor'].format(l=layer),n,k)
 if not np.array_equal(weights,source):raise ValueError('manifest-packed/source effective weights differ')
 return weights,dict(packed_sha256=hashlib.sha256(pool.tobytes()).hexdigest(),
   effective_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),source_exact=True)

def run(a):
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 result=dict(complete=False,model_acceptance=False)
 try:
  directory=a.run.resolve();audit_path=directory/'sublayer-audit.json'
  audit=json.loads(audit_path.read_text())
  meta,evidence=scan.validate(directory,a.baseline.resolve(),a.build.resolve())
  if meta['layers']!=64 or len(meta['ids'])!=20 or meta['hidden']!=5120:raise ValueError('requires retained full-depth short prompt')
  spec=base.spec_from_model_dir(Path(meta['model']))
  bundle=proj.load_bundle(Path(meta['kernels']),proj.manifest(spec,max_ctx=meta['max_ctx']))
  if any(meta[k]!=bundle[k] for k in ('build_key','spec_hash')):raise ValueError('export identity mismatch')
  cases={c['tag']:c for c in proj.cases(bundle)}
  ops=cases['L0_entry']['ops']+cases['L0_out']['ops']
  model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=spec.hidden;model.native_q8=lambda name:False
  result.update(evidence=evidence,projection_rows=[],layers=[],artifacts={})
  for l,p in prefix.TARGETS:
   pre=f'model.layers.{l}.';weights={};weight_evidence={}
   for key,op in zip(('qkv','z','out'),ops):weights[key],weight_evidence[key]=packed_weights(model,l,op)
   def read(key,width):return base.read(directory/f'hybrid_L{l}_P{p}_{key}.bin',20*width).reshape(20,width)
   xn=read('xn',spec.hidden);qkv=read('qkv',spec.lin_qkv_dim)
   z=read('z',spec.lin_value_heads*spec.lin_value_dim);og=read('og',z.shape[1])
   observed_out=base.read(directory/f'hybrid_L{l}_P{p}_proj.bin',spec.hidden)[None,:]
   inputs=prefix.trace.read_trace(directory/f'hybrid_res{l-1}.bin',21,spec.hidden)[:20]
   # The prior verified audit binds raw state; verify zero initial state and ABI again.
   prefix.states((directory/f'hybrid_L{l}_P{p}_state_before.bin').read_bytes(),
                 (directory/f'hybrid_L{l}_P{p}_state_after.bin').read_bytes(),
                 base.layout(spec,max_ctx=meta['max_ctx']),spec)
   predictions={k:predict(x,weights[k]) for k,x in [('qkv',xn),('z',xn),('out',og[p:p+1])]}
   kw=spec.lin_key_heads*spec.lin_key_dim;vw=spec.lin_value_heads*spec.lin_value_dim
   roles=[('q',qkv,slice(0,kw),'qkv',spec.lin_key_heads),('k',qkv,slice(kw,2*kw),'qkv',spec.lin_key_heads),
          ('v',qkv,slice(2*kw,2*kw+vw),'qkv',spec.lin_value_heads),('z',z,slice(None),'z',spec.lin_value_heads),
          ('out',observed_out,slice(None),'out',1)]
   arrays={}
   for role,got,sl,key,heads in roles:
    pred={name:x[:,sl] for name,x in predictions[key].items()}
    rows=head_scores(got[:,sl],pred,heads)
    for row in rows:
     if role=='out':row['position']=p
     row.update(layer=l,role=role)
    result['projection_rows'].extend(rows)
    arrays.update({f'{role}_{name}':x for name,x in pred.items()});arrays[role+'_native']=got[:,sl]
   constants=[model.bf16(pre+'linear_attn.'+name).astype(np.float64) for name in
     ('ssm_conv1d.weight','ssm_alpha_proj.bf16.weight','ssm_beta_proj.bf16.weight')]
   constants[1]=constants[1].T;constants[2]=constants[2].T
   constants += [model.f32(pre+'linear_attn.ssm_a').astype(np.float64),model.f32(pre+'linear_attn.ssm_dt.bias').astype(np.float64),model.bf16(pre+'linear_attn.ssm_norm.weight').astype(np.float64)]
   q0,z0=predictions['qkv']['exact'],predictions['z']['exact']
   variants={}
   for name,qq,zz in [('host',qkv,z),('qkv',q0,z),('z',qkv,z0),('joint',q0,z0)]:
    mid=causal_middle(spec,constants,qq,zz,xn,p)
    variants[name]=inputs[p]+mid@weights['out'].astype(np.float64).T
   # Q/K/V-only changes are counterfactuals, not additive components.
   for name,sl in [('q',slice(0,kw)),('k',slice(kw,2*kw)),('v',slice(2*kw,2*kw+vw))]:
    qq=qkv.copy();qq[:,sl]=q0[:,sl]
    mid=causal_middle(spec,constants,qq,z,xn,p)
    arrays['counterfactual_'+name]=inputs[p]+mid@weights['out'].astype(np.float64).T
   variants['out']=inputs[p]+predictions['out']['exact'][0]
   variants['projection']=inputs[p]+observed_out[0]
   reference=np.fromfile(directory/f'cpu_L{l}_P{p}_res.f64','<f8')
   native=base.read(directory/f'hybrid_L{l}_P{p}_res.bin',spec.hidden)
   if reference.shape!=native.shape:raise ValueError('incomplete independent residual')
   scores,vectors=decompose(native,reference,variants)
   arrays.update({'residual_'+k:v for k,v in variants.items()})
   arrays.update({'component_'+k:v for k,v in vectors.items()})
   arrays.update(reference=reference,native_residual=native)
   cf={k:scan.error_score(v-reference,reference) for k,v in variants.items()}
   cf.update({k:scan.error_score(arrays['counterfactual_'+k]-reference,reference) for k in ('q','k','v')})
   path=out/f'L{l}_P{p}.npz';np.savez(path,**arrays)
   result['artifacts'][path.name]=base.digest(path)
   layer=dict(layer=l,position=p,weights=weight_evidence,residual=scores,counterfactuals=cf)
   result['layers'].append(layer);print(json.dumps(layer),flush=True)
  expected=3*(20*(2*spec.lin_key_heads+2*spec.lin_value_heads)+1)
  if len(result['projection_rows'])!=expected:raise ValueError('incomplete head/position coverage')
  rows=result['projection_rows'];summary={}
  for role in ('q','k','v','z','out'):
   selected=[r for r in rows if r['role']==role]
   summary[role]=dict(total=len(selected),native_passed=sum(r['native']['passed'] for r in selected),
     worst_native=max(selected,key=lambda r:r['native']['maxrel'] if r['native']['maxrel'] is not None else float('inf')),
     worst_prediction_gap=max(r['prediction_gap']['maxrel'] for r in selected),
     worst_remaining_error_ratio=max(r['remaining_error_ratio'] for r in selected if r['remaining_error_ratio'] is not None))
  files=[Path(__file__),Path(prefix.__file__),Path(prefix.rb.__file__),Path(scan.__file__),Path(proj.__file__),
         ROOT/'open_kernels/model/q4nx.py',ROOT/'open_kernels/q4_1_pack.py',ROOT/'open_kernels/recipes/pack.py',audit_path]
  result.update(complete=True,summary=summary,hybrid_schedule_representative=True,
    scope='CPU rounding predictions and ideal-suffix causal substitutions; no byte-exact NPU replay or model acceptance',
    sources={str(p.resolve()):base.digest(p) for p in files})
 except Exception as e:result.update(complete=False,error=str(e));raise
 finally:(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return 0

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','build','baseline','out'):p.add_argument('--'+name,type=Path,required=True)
 raise SystemExit(run(p.parse_args()))
