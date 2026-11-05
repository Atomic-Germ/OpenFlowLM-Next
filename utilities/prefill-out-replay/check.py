"""Paired-weight out replay with historical inputs and fresh post-merge controls."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
FIXTURE_REV='aef4a93'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prefix=load('_out_prefix_replay',ROOT/'utilities/prefill-prefix-replay/check.py')
paired,base,proj=prefix.paired,prefix.base,prefix.proj
N,K,T,ROWS=5120,6144,256,20
TARGETS=((34,17),(37,7),(53,17))
OFFSETS=(0,128)
digest,verify=prefix.digest,prefix.verify

def git_bytes(revision,name):
 return subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT,stderr=subprocess.DEVNULL)

def verify_historical(path,expected):
 path=Path(path).resolve();current=digest(path)
 if current==expected:return dict(historical=False,sha256=expected)
 # Historical exceptions are only for tracked source, never model/build artifacts.
 if not path.is_relative_to(ROOT) or path.is_relative_to(ROOT/'Models') or path.suffix not in ('.py','.cpp','.hpp','.h','.cc','.txt','.json'):
  raise ValueError('changed historical artifact/dependency: '+str(path))
 name=str(path.relative_to(ROOT))
 try:
  old=hashlib.sha256(git_bytes(FIXTURE_REV,name)).hexdigest()
  head=hashlib.sha256(git_bytes('HEAD',name)).hexdigest()
 except subprocess.CalledProcessError as e:raise ValueError('untracked historical source') from e
 if old!=expected or head!=current:raise ValueError('historical source or current HEAD differs: '+name)
 return dict(historical=True,revision=FIXTURE_REV,sha256=old,current_sha256=current)

def history_graph(path):
 """Verify pinned records recursively without modifying them or executing old code."""
 files={};records=set()
 maps=('sources','evidence','inputs','artifacts','outputs','compiler_dependencies','dependencies','generated','binaries','kernel_inputs','staged_sources','sha256','capture_hashes','cpu_outputs','residual_hashes')
 def check(p,h):
  p=Path(p).resolve();name=str(p)
  if name in files:
   if files[name]['sha256']!=h:raise ValueError('conflicting historical hashes: '+name)
  else:files[name]=verify_historical(p,h)
  if p.suffix=='.json' and name not in records and not files[name]['historical']:walk(p)
 def walk(p):
  records.add(str(p));r=json.loads(p.read_text())
  if not isinstance(r,dict):return
  for key in maps:
   values=r.get(key,{})
   if not isinstance(values,dict):continue
   for name,h in values.items():
    if name=='sublayer_audit_sha256':continue  # Advisory digest; audit file is separately pinned.
    if not isinstance(h,str) or len(h)!=64:continue
    target=Path(name)
    if not target.is_absolute():target=(ROOT if key=='sources' else p.parent)/target
    check(target,h)
  if 'harness_sha256' in r:check(p.parent/'build/run_kernel',r['harness_sha256'])
  if 'cli_sha256' in r and p.name=='build-evidence.json':check(p.parent/'build/open_qwen36_cli',r['cli_sha256'])
  if 'cmake_cache_sha256' in r:check(p.parent/'build/CMakeCache.txt',r['cmake_cache_sha256'])
  if 'binary_sha256' in r:
   candidates=[p.parent/n for n in ('libprefill_boundaries.so','bridge.so') if (p.parent/n).exists()]
   if len(candidates)!=1:raise ValueError('ambiguous historical library')
   check(candidates[0],r['binary_sha256'])
 path=Path(path).resolve();check(path,digest(path))
 return dict(revision=FIXTURE_REV,files=files,records=sorted(records))

def scores(control,candidate,reference,incoming,residual_reference,position):
 arrays=[np.asarray(v,np.float64) for v in (control,candidate,reference)]
 if (any(v.shape!=(ROWS,N) or not np.isfinite(v).all() for v in arrays)
     or type(position)!=int or not 0<=position<ROWS):raise ValueError('complete finite output prefix required')
 control,candidate,reference=arrays
 rows=[dict(position=t,control=proj.metric(c,r),candidate=proj.metric(p,r)) for t,(c,p,r) in enumerate(zip(control,candidate,reference))]
 residual=prefix.residual_scores(incoming,candidate[position],reference[position],residual_reference)['residual']
 return dict(rows=rows,projection_passed=all(r['candidate']['passed'] for r in rows),
  improved=all(r['candidate'][k]<r['control'][k] for r in rows for k in ('maxrel','rel_fro')),residual=residual)

def verify_build(path,mode):
 b=paired.verify_build(path,mode)
 if b.get('shape')!=dict(n=N,k=K,t=T) or any(b['environment'].get(k)!=v for k,v in dict(GQP_N=str(N),GQP_K=str(K),GQP_T=str(T),GQP_FMT='q4_1',GQP_BREUSE='0').items()):raise ValueError('wrong output kernel geometry')
 return b

def prepare(a):
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);r=dict(complete=False,artifacts={})
 try:
  baseline=a.baseline.resolve();old=json.loads((baseline/'report.json').read_text())
  if not old['complete'] or not old['control_valid'] or old['dispatches']!=60 or old['model_acceptance']:raise ValueError('invalid prefix baseline')
  graph=history_graph(baseline/'report.json')
  capture=a.capture.resolve();meta=json.loads((capture/'reference.json').read_text())
  audit=capture/'sublayer-audit.json'
  if old['evidence'].get(str(audit))!=digest(audit):raise ValueError('unbound native capture')
  # Bind incoming residual files to the prefix measurement's validated inventory.
  prefix.verify({str(capture/k):v for k,v in old['capture_evidence']['residual_hashes'].items()})
  spec=base.spec_from_model_dir(Path(meta['model']));manifest_path=Path(meta['kernels'])/'manifest.json'
  manifest=json.loads(manifest_path.read_text())
  c=next(c for c in proj.cases(manifest) if c['tag']=='L0_out')
  if (c['n'],c['k'])!=(N,K) or len(c['ops'])!=1:raise ValueError('unsupported output projection')
  model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=N;model.native_q8=lambda name:False
  def save(path,x):path.write_bytes(np.ascontiguousarray(x).tobytes());r['artifacts'][str(path)]=digest(path)
  for l,p in TARGETS:
   directory=out/f'L{l}_P{p}';directory.mkdir();previous=baseline/directory.name
   op=c['ops'][0];weights,_=prefix.attr.packed_weights(model,l,op)
   pool=np.zeros(N*K*5//8,np.uint8);proj.apply_op(dict(op,dst=0),model,l,pool)
   if pool.tobytes()!=(previous/'out-weights.bin').read_bytes():raise ValueError('packed output weights changed')
   save(directory/'weights.bin',pool)
   incoming=prefix.prefix.trace.read_trace(capture/f'hybrid_res{l-1}.bin',21,N)[p].astype(np.float32)
   save(directory/'incoming.bin',incoming)
   with np.load(previous/'inputs.npz') as data:ref=data['residual_reference'].copy()
   save(directory/'residual-reference.f64',ref.astype('<f8'))
   for variant in ('control','candidate'):
    with np.load(previous/(variant+'-host.npz')) as data:x=data['og'].copy()
    if x.shape!=(ROWS,K) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('invalid retained gated prefix')
    save(directory/(variant+'-input.bin'),x)
    save(directory/(variant+'-reference.f64'),x.astype(np.float64)@weights.astype(np.float64).T)
    for offset in OFFSETS:
     for part,values in zip(('high','low'),paired.control_tool.split_inputs(x)):
      save(directory/f'{variant}-{offset}-{part}-x.bin',proj.tile_input(prefix.place(values,offset)))
   del weights,pool
  kernel=Path(meta['kernels'])/c['kernel']
  files=[Path(__file__),Path(prefix.__file__),Path(paired.__file__),baseline/'report.json',capture/'reference.json',manifest_path,kernel/'final.xclbin',kernel/'insts.bin']
  r.update(complete=True,baseline=str(baseline),production_kernel=str(kernel),targets=TARGETS,history=graph,
   inputs={str(p.resolve()):digest(p) for p in files},scope='retained pre-merge gated prefixes; fresh output projection only')
 except Exception as e:r.update(complete=False,error=str(e));raise
 finally:(out/'preparation.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 return 0

def run(a):
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);r=dict(complete=False,control_valid=False,model_acceptance=False,dispatches=0,cases=[],artifacts={})
 try:
  prepared=a.prepared.resolve();pp=prepared/'preparation.json';prep=json.loads(pp.read_text())
  if not prep['complete'] or prep['targets']!=[list(x) for x in TARGETS]:raise ValueError('invalid preparation')
  verify(prep['inputs']);verify(prep['artifacts'])
  for p,h in prep['history']['files'].items():
   if verify_historical(p,h['sha256'])!=h:raise ValueError('historical/current identity changed')
  for mode in ('control','low'):verify_build(a.build_root.resolve()/mode,mode)
  hb=a.harness_build.resolve();h=json.loads((hb/'build-evidence.json').read_text())
  if not h['complete']:raise ValueError('incomplete harness')
  verify(h['sources']);verify(h['compiler_dependencies']);verify({str(hb/'build/run_kernel'):h['harness_sha256'],str(hb/'build/CMakeCache.txt'):h['cmake_cache_sha256']})
  evidence=[Path(__file__),pp,hb/'build-evidence.json',*[a.build_root.resolve()/m/'build.json' for m in ('control','low')]]
  baseline=Path(prep['baseline'])
  for l,p in TARGETS:
   directory=out/f'L{l}_P{p}';directory.mkdir();inputs=prepared/directory.name;previous=baseline/directory.name
   incoming=base.read(inputs/'incoming.bin',N).astype(np.float32);refres=np.fromfile(inputs/'residual-reference.f64','<f8')
   for variant in ('control','candidate'):
    reference=np.fromfile(inputs/(variant+'-reference.f64'),'<f8').reshape(ROWS,N);candidates=[]
    for offset in OFFSETS:
     products={}
     for mode in ('production','control','low'):
      kernel=Path(prep['production_kernel']) if mode=='production' else a.build_root.resolve()/mode
      ys=[]
      for part in ('high','low'):
       tag=f'{variant}-{offset}-{mode}-{part}';cfg=directory/(tag+'.cfg');dump=directory/(tag+'.bin')
       cfg.write_text('\n'.join(['device',f'xclbin G {kernel}/final.xclbin',f'kernelx k G {kernel}/insts.bin',f'buf w {N*K*5//8} {inputs}/weights.bin',f'buf x {K*T*2} {inputs}/{variant}-{offset}-{part}-x.bin',f'buf y {N*T*4}','run k w x y',f'dump y {dump} {N*T*4}','']))
       with cfg.with_suffix('.log').open('w') as f:subprocess.run([str(hb/'build/run_kernel'),str(cfg)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120,env=os.environ|{'HARNESS_TIMEOUT_MS':'30000'})
       r['dispatches']+=1;evidence.extend([cfg,cfg.with_suffix('.log')]);r['artifacts'][str(dump)]=digest(dump)
       y=base.read(dump,N*T).astype(np.float32).reshape(N,T);ys.append(y);prefix.extract(y,offset,ROWS)
       if mode=='production':
        old=base.read(previous/f'out-{variant}-{offset}-{part}-y.bin',N*T).astype(np.float32).reshape(N,T)
        prefix.identity(y,old,'retained individual output product')
      products[mode]=ys
      if mode=='control':paired.require_control(*ys,*products['production'])
     control=prefix.extract(np.add(*products['control'],dtype=np.float32),offset,ROWS)
     old=base.read(previous/f'out-{variant}-{offset}.bin',ROWS*N).astype(np.float32).reshape(ROWS,N);prefix.identity(control,old,'retained summed output')
     candidate=prefix.extract(paired.combine(*products['control'],*products['low']),offset,ROWS);candidates.append(candidate)
     for name,x in [('output',candidate),('residual',np.add(incoming,candidate[p],dtype=np.float32))]:
      path=directory/f'{variant}-{offset}-{name}.bin';path.write_bytes(x.tobytes());r['artifacts'][str(path)]=digest(path)
     case=dict(layer=l,position=p,input_variant=variant,offset=offset,controls_exact=True,**scores(control,candidate,reference,incoming,refres,p));r['cases'].append(case)
     print(json.dumps({k:v for k,v in case.items() if k!='rows'}),flush=True)
    prefix.identity(*candidates,variant+' candidate placement')
  if r['dispatches']!=72 or len(r['cases'])!=12:raise ValueError('incomplete output replay')
  r.update(complete=True,control_valid=True,slot_identity=True,projection_passed=all(c['projection_passed'] for c in r['cases']),improved=all(c['improved'] for c in r['cases']),residual_passed=all(c['residual']['passed'] for c in r['cases']),
   evidence={str(p.resolve()):digest(p) for p in evidence},scope='output correction on two retained gated prefixes per layer; no fresh full-model or cross-layer execution')
 except Exception as e:r.update(complete=False,control_valid=False,error=str(e));raise
 finally:(out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 return 0 if all(r[k] for k in ('projection_passed','improved','residual_passed')) else 1

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('prepare')
 for name in ('out','baseline','capture'):a.add_argument('--'+name,type=Path,required=True)
 a=sub.add_parser('run')
 for name in ('out','prepared','build-root','harness-build'):a.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();raise SystemExit(prepare(a) if a.command=='prepare' else run(a))
