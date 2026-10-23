"""Compose full captured QKV/Z prefixes, native DeltaNet and original NPU out."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
entry=load('_prefix_entry_replay',ROOT/'utilities/prefill-entry-replay/check.py')
host=load('_prefix_host_bridge',ROOT/'utilities/prefill-boundaries/check.py')
attr,paired,base,proj=entry.attr,entry.paired,entry.base,entry.proj
prefix=attr.prefix
digest,verify=entry.digest,entry.verify
TARGETS=((34,17),(37,7),(53,17))
OFFSETS=(0,128)
T,ROWS=256,20

def identity(got,ref,label):
 got,ref=np.asarray(got),np.asarray(ref)
 if got.shape!=ref.shape or got.dtype!=ref.dtype or got.tobytes()!=ref.tobytes():raise ValueError(label+' differs bytewise')

def place(x,offset):
 x=np.asarray(x,np.float32)
 if (x.ndim!=2 or not x.size or type(offset)!=int or offset<0 or offset+len(x)>T
     or not np.isfinite(x).all()):raise ValueError('invalid prefix placement')
 y=np.zeros((T,x.shape[1]),np.float32);y[offset:offset+len(x)]=x;return y

def extract(y,offset,count):
 y=np.asarray(y,np.float32)
 if (y.ndim!=2 or not y.size or y.shape[1]!=T or type(offset)!=int or type(count)!=int
     or count<1 or offset<0 or offset+count>T or not np.isfinite(y).all()):raise ValueError('invalid full product')
 if np.any(y[:,:offset]) or np.any(y[:,offset+count:]):raise ValueError('nonzero inactive columns')
 return y[:,offset:offset+count].T.copy()

def entry_scores(control,candidate,ref):
 if any(np.shape(x)!=(ROWS,16384) for x in (control,candidate,ref)):raise ValueError('incomplete prefix coverage')
 rows=[entry.scores(c,p,r) for c,p,r in zip(control,candidate,ref)]
 return dict(rows=rows,checks=ROWS*128,passed=all(r['candidate_passed'] for r in rows),improved=all(r['improved'] for r in rows))

def residual_scores(x,y,ideal,ref):
 arrays=[np.asarray(v,np.float64) for v in (x,y,ideal,ref)]
 if (arrays[0].ndim!=1 or not arrays[0].size or any(v.shape!=arrays[0].shape or not np.isfinite(v).all() for v in arrays)):
  raise ValueError('complete finite residual vectors required')
 result=np.add(arrays[0].astype(np.float32),arrays[1].astype(np.float32),dtype=np.float32)
 return dict(projection=proj.metric(arrays[1],arrays[2]),residual=proj.metric(result,arrays[3]))

def host_run(lib,c,g,qkv,z,chunks):
 qkv,z=np.asarray(qkv,np.float32),np.asarray(z,np.float32)
 n=len(qkv);nch=(2*g['kh']+g['vh'])*g['dim'];vw=g['vh']*g['dim']
 if (qkv.shape!=(n,nch) or z.shape!=(n,vw) or c['xn'].shape!=(n,g['hid']) or not chunks
     or any(type(k)!=int or not 1<=k<=T for k in chunks) or sum(chunks)!=n
     or not all(np.isfinite(v).all() for v in (qkv,z,c['xn']))):raise ValueError('invalid causal host prefix')
 shapes=dict(convw=(g['taps'],nch),Wa=(g['hid'],g['lanes']),Wb=(g['hid'],g['lanes']),A=(g['vh'],),dtb=(g['vh'],),nw=(g['dim'],))
 if any(np.shape(c[k])!=shape or not np.isfinite(c[k]).all() for k,shape in shapes.items()):raise ValueError('invalid host constant shape/stride')
 conv=np.zeros((g['taps']-1,nch),np.float32);state=np.zeros((g['vh'],g['rows'],g['dim']),np.float32)
 start=0;outputs=[]
 for count in chunks:
  args=[]
  for x in (qkv,z,c['xn']):
   padded=np.full((T,x.shape[1]),np.nan,np.float32);padded[:count]=x[start:start+count];args.append(padded)
  y=np.full((T,vw),np.nan,np.float32)
  geom=np.array([T,count,*(g[k] for k in ('hid','kh','vh','dim','taps','lanes','rows'))],np.uintp)
  constants=[np.ascontiguousarray(c[k],np.float32) for k in ('convw','Wa','Wb','A','dtb','nw')]
  if lib.prefill_delta(geom,*args,*constants,conv,state,y):raise RuntimeError('native DeltaNet failed')
  if np.any(y[count:]) or np.any(state[:,g['dim']:]):raise ValueError('host padding differs')
  if not all(np.isfinite(v).all() for v in (y,conv,state)):raise ValueError('nonfinite host output')
  outputs.append(y[:count].copy());start+=count
 identity(conv,qkv[-(g['taps']-1):],'conv carry')
 return dict(og=np.concatenate(outputs),conv=conv,state=state)

def host_reference(c,g,qkv,z):
 conv=np.zeros((g['taps']-1,qkv.shape[1]),np.float64);state=np.zeros((g['vh'],g['rows'],g['dim']),np.float64)
 arrays=[np.asarray(x,np.float32).astype(np.float64) for x in (qkv,z,c['xn'])]
 constants=[np.asarray(c[k],np.float64) for k in ('convw','Wa','Wb','A','dtb','nw')]
 y=prefix.rb.deltanet_block(*arrays,*constants,conv,state,key_heads=g['kh'],value_heads=g['vh'],head_dim=g['dim'],conv_f32=True)
 return dict(og=y,conv=conv,state=state)

def host_scores(got,ref,g):
 identity(got['conv'],ref['conv'].astype(np.float32),'reference conv')
 outputs=[dict(position=t,head=h,**host.metric(got['og'][t,h*g['dim']:(h+1)*g['dim']],ref['og'][t,h*g['dim']:(h+1)*g['dim']])) for t in range(ROWS) for h in range(g['vh'])]
 states=[dict(head=h,**host.metric(got['state'][h,:g['dim']],ref['state'][h,:g['dim']])) for h in range(g['vh'])]
 return dict(outputs=outputs,states=states,passed=all(r['passed'] for r in outputs+states),state_position=ROWS-1)

def run(a):
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 r=dict(complete=False,control_valid=False,model_acceptance=False,dispatches=0,artifacts={},evidence={},cases=[])
 def pin(path):r['evidence'][str(path.resolve())]=digest(path)
 def save(path,values):
  path.write_bytes(np.ascontiguousarray(values).tobytes());r['artifacts'][str(path)]=digest(path);return path
 def npz(path,**arrays):
  np.savez(path,**arrays);r['artifacts'][str(path)]=digest(path)
 try:
  prior_path=a.attribution.resolve()/'report.json';prior=json.loads(prior_path.read_text())
  if not prior['complete'] or prior['model_acceptance']:raise ValueError('invalid attribution')
  verify(prior['sources']);verify({str(a.attribution.resolve()/p):h for p,h in prior['artifacts'].items()});pin(prior_path)
  capture=a.capture.resolve();meta,capture_evidence=attr.scan.validate(capture,a.baseline.resolve(),a.capture_build.resolve())
  pin(capture/'sublayer-audit.json');r['capture_evidence']=capture_evidence
  if prior['sources'].get(str(capture/'sublayer-audit.json'))!=digest(capture/'sublayer-audit.json'):raise ValueError('capture differs from attribution')
  spec=base.spec_from_model_dir(Path(meta['model']));g=host.geometry()
  if spec.hidden!=5120 or spec.norm_eps!=1e-6 or len(meta['ids'])!=ROWS or meta['layers']!=64:raise ValueError('unsupported capture geometry')
  bundle=proj.load_bundle(Path(meta['kernels']),proj.manifest(spec,max_ctx=meta['max_ctx']))
  if any(meta[k]!=bundle[k] for k in ('build_key','spec_hash')):raise ValueError('bundle differs')
  cases={c['tag']:c for c in proj.cases(bundle)};ec,oc=cases['L0_entry'],cases['L0_out']
  if (ec['n'],ec['k'],oc['n'],oc['k'])!=(16384,5120,5120,6144):raise ValueError('unsupported GEMM geometry')
  for mode in ('control','low'):entry.verify_build(a.build_root.resolve()/mode,mode);pin(a.build_root/mode/'build.json')
  for c in (ec,oc):
   for name in ('final.xclbin','insts.bin'):pin(Path(meta['kernels'])/c['kernel']/name)
  hb=a.harness_build.resolve();h=json.loads((hb/'build-evidence.json').read_text())
  if not h['complete']:raise ValueError('invalid harness')
  verify(h['sources']);verify(h['compiler_dependencies']);verify({str(hb/'build/run_kernel'):h['harness_sha256'],str(hb/'build/CMakeCache.txt'):h['cmake_cache_sha256']});pin(hb/'build-evidence.json')
  bridge=a.bridge.resolve();b=json.loads((bridge/'build.json').read_text())
  verify({str(ROOT/p):v for p,v in b['sources'].items()});verify({str(bridge/'libprefill_boundaries.so'):b['binary_sha256']});pin(bridge/'build.json');pin(bridge/'libprefill_boundaries.so')
  lib=host.native_library(bridge/'libprefill_boundaries.so')
  model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=5120;model.native_q8=lambda name:False
  def dispatch(directory,tag,kernel,weights,x,n,offset):
   products=[]
   for part,values in zip(('high','low'),paired.control_tool.split_inputs(x)):
    xp=save(directory/f'{tag}-{part}-x.bin',proj.tile_input(place(values,offset)))
    yp=directory/f'{tag}-{part}-y.bin';cfg=directory/f'{tag}-{part}.cfg'
    cfg.write_text('\n'.join(['device',f'xclbin G {kernel}/final.xclbin',f'kernelx k G {kernel}/insts.bin',f'buf w {weights.stat().st_size} {weights}',f'buf x {xp.stat().st_size} {xp}',f'buf y {n*T*4}','run k w x y',f'dump y {yp} {n*T*4}','']))
    pin(cfg)
    with cfg.with_suffix('.log').open('w') as f:subprocess.run([str(hb/'build/run_kernel'),str(cfg)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120,env=os.environ|{'HARNESS_TIMEOUT_MS':'30000'})
    r['dispatches']+=1;pin(cfg.with_suffix('.log'));r['artifacts'][str(yp)]=digest(yp)
    products.append(extract(base.read(yp,n*T).astype(np.float32).reshape(n,T),offset,ROWS))
   return products
  for l,p in TARGETS:
   directory=out/f'L{l}_P{p}';directory.mkdir();pre=f'model.layers.{l}.'
   def read(key,width):return base.read(capture/f'hybrid_L{l}_P{p}_{key}.bin',ROWS*width).astype(np.float32).reshape(ROWS,width)
   xn,qkv,z,og=read('xn',5120),read('qkv',10240),read('z',6144),read('og',6144)
   constants=dict(xn=xn,convw=model.bf16(pre+'linear_attn.ssm_conv1d.weight'),Wa=model.bf16(pre+'linear_attn.ssm_alpha_proj.bf16.weight').T,Wb=model.bf16(pre+'linear_attn.ssm_beta_proj.bf16.weight').T,A=model.f32(pre+'linear_attn.ssm_a'),dtb=model.f32(pre+'linear_attn.ssm_dt.bias'),nw=model.bf16(pre+'linear_attn.ssm_norm.weight'))
   # Native core derives the alpha/beta stride from tensor shape, not padded NPU lanes.
   g=dict(g,lanes=constants['Wa'].shape[1])
   control_host=host_run(lib,constants,g,qkv,z,[ROWS])
   conv,state=prefix.states((capture/f'hybrid_L{l}_P{p}_state_before.bin').read_bytes(),(capture/f'hybrid_L{l}_P{p}_state_after.bin').read_bytes(),base.layout(spec,max_ctx=meta['max_ctx']),spec)
   for key,expected in [('og',og),('conv',conv.astype(np.float32)),('state',state.astype(np.float32))]:identity(control_host[key],expected,'native '+key)
   retained=next(x['weights'] for x in prior['layers'] if x['layer']==l)
   pools={};ow=None
   for kind,c in [('entry',ec),('out',oc)]:
    pool=np.zeros(c['n']*c['k']*5//8,np.uint8);offset=0
    for key,op in zip(('qkv','z') if kind=='entry' else ('out',),c['ops']):
     weights,evidence=attr.packed_weights(model,l,op)
     if evidence!=retained[key]:raise ValueError('packed/source identity differs')
     if key=='out':ow=weights.astype(np.float64)
     del weights
     proj.apply_op(dict(op,dst=offset),model,l,pool);offset+=op['nch']*5120
    if offset!=len(pool):raise ValueError('incomplete weight packing')
    pools[kind]=save(directory/(kind+'-weights.bin'),pool)
   with np.load(a.attribution/f'L{l}_P{p}.npz') as refs:
    exact=np.concatenate([refs[role+'_exact'] for role,_,_ in entry.ROLES],axis=1)
    native=np.concatenate([refs[role+'_native'] for role,_,_ in entry.ROLES],axis=1).astype(np.float32)
    ref_res=refs['reference'].copy()
   identity(native,np.concatenate((qkv,z),axis=1),'attribution native')
   npz(directory/'inputs.npz',**constants,native=native,reference=exact,residual_reference=ref_res)
   candidates=[];entry_results=[]
   for offset in OFFSETS:
    products={}
    for mode in ('production','control','low'):
     kernel=Path(meta['kernels'])/ec['kernel'] if mode=='production' else a.build_root.resolve()/mode
     products[mode]=dispatch(directory,f'entry-{mode}-{offset}',kernel,pools['entry'],xn,16384,offset)
     if mode=='production':identity(np.add(*products[mode],dtype=np.float32),native,'production prefix')
     if mode=='control':paired.require_control(*products[mode],*products['production'])
    candidate=paired.combine(*products['control'],*products['low']);candidates.append(candidate)
    entry_results.append(dict(offset=offset,**entry_scores(native,candidate,exact)))
    save(directory/f'entry-candidate-{offset}.bin',candidate)
   identity(*candidates,'entry placement')
   candidate=candidates[0];candidate_host=host_run(lib,constants,g,candidate[:,:10240],candidate[:,10240:],[ROWS])
   hs={};out_results=[]
   incoming=prefix.trace.read_trace(capture/f'hybrid_res{l-1}.bin',21,5120)[:ROWS].astype(np.float32)
   captured_out=base.read(capture/f'hybrid_L{l}_P{p}_proj.bin',5120).astype(np.float32)
   captured_res=base.read(capture/f'hybrid_L{l}_P{p}_res.bin',5120).astype(np.float32)
   for name,qq,zz,got in [('control',qkv,z,control_host),('candidate',candidate[:,:10240],candidate[:,10240:],candidate_host)]:
    sequential=host_run(lib,constants,g,qq,zz,[1]*ROWS)
    for key in got:identity(got[key],sequential[key],name+' host partition '+key)
    ref=host_reference(constants,g,qq,zz);hs[name]=host_scores(got,ref,g)
    npz(directory/(name+'-host.npz'),**got,**{'reference_'+k:v for k,v in ref.items()})
    expected=got['og'].astype(np.float64)@ow.T
    results=[]
    for offset in OFFSETS:
     ys=dispatch(directory,f'out-{name}-{offset}',Path(meta['kernels'])/oc['kernel'],pools['out'],got['og'],5120,offset)
     y=np.add(*ys,dtype=np.float32);results.append(y)
     res=np.add(incoming[p],y[p],dtype=np.float32)
     if name=='control':identity(y[p],captured_out,'native output projection');identity(res,captured_res,'native residual')
     save(directory/f'out-{name}-{offset}.bin',y);save(directory/f'residual-{name}-{offset}.bin',res)
     out_results.append(dict(variant=name,offset=offset,rows=[proj.metric(a,b) for a,b in zip(y,expected)],selected=residual_scores(incoming[p],y[p],expected[p],ref_res)))
    identity(*results,name+' output placement')
   # Ideal-entry state is diagnostic only, separate from conditioned native-host gates.
   ideal=host_reference(constants,g,exact[:,:10240],exact[:,10240:])
   npz(directory/'ideal-entry-host.npz',**ideal)
   ideal_effect={name:dict(og=proj.metric(got['og'].astype(np.float64),ideal['og']),state=proj.metric(got['state'].astype(np.float64),ideal['state'])) for name,got in [('control',control_host),('candidate',candidate_host)]}
   case=dict(layer=l,position=p,entry=entry_results,host=hs,output=out_results,ideal_entry_diagnostic=ideal_effect,controls_exact=True,partition_exact=True,slot_identity=True)
   r['cases'].append(case)
   print(json.dumps(dict(layer=l,position=p,entry_passed=all(c['passed'] for c in entry_results),host_passed=all(c['passed'] for c in hs.values()),residuals=[dict(variant=c['variant'],offset=c['offset'],**c['selected']['residual']) for c in out_results])),flush=True)
  if r['dispatches']!=60 or len(r['cases'])!=3:raise ValueError('incomplete replay coverage')
  for path in (Path(__file__),Path(entry.__file__),Path(host.__file__),Path(prefix.rb.__file__)):pin(path)
  r.update(complete=True,control_valid=True,entry_passed=all(c['passed'] for layer in r['cases'] for c in layer['entry']),host_passed=all(c['passed'] for layer in r['cases'] for c in layer['host'].values()),
   projection_passed=all(row['passed'] for layer in r['cases'] for c in layer['output'] for row in c['rows']),
   residual_passed=all(c['selected']['residual']['passed'] for layer in r['cases'] for c in layer['output'] if c['variant']=='candidate'),
   scope='three conditioned 20-row prefixes; original incoming residual/norm, no corrected FFN or model propagation')
  verify(r['evidence']);verify(r['artifacts'])
 except Exception as e:r.update(complete=False,control_valid=False,error=str(e));raise
 finally:(out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 return 0 if all(r[k] for k in ('entry_passed','host_passed','projection_passed','residual_passed')) else 1

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('out','attribution','capture','baseline','capture-build','build-root','harness-build','bridge'):p.add_argument('--'+name,type=Path,required=True)
 raise SystemExit(run(p.parse_args()))
