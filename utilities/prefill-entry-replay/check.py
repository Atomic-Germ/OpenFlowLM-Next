"""Isolated entry QKV/Z replay with exact native controls and unchanged head gates."""
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
attr=load('_entry_attribution',ROOT/'utilities/prefill-projection-attribution/check.py')
paired=load('_entry_paired_weights',ROOT/'utilities/prefill-paired-weights/run.py')
base,proj=attr.base,attr.proj
digest,verify=base.digest,base.verify
N,K,T=16384,5120,256
TARGETS=((34,6,17),(37,0,7))
ROLES=(('q',0,2048),('k',2048,4096),('v',4096,10240),('z',10240,N))

def selected(hi,lo,captured,slot):
 hi,lo,captured=[np.asarray(x,np.float32) for x in (hi,lo,captured)]
 if (hi.ndim!=2 or hi.shape[1]!=T or hi.shape!=lo.shape or captured.shape!=(hi.shape[0],)
     or not hi.size or type(slot)!=int or not 0<=slot<T or not all(np.isfinite(x).all() for x in (hi,lo,captured))):raise ValueError('invalid control extent')
 if np.any(np.delete(hi,slot,axis=1)) or np.any(np.delete(lo,slot,axis=1)):raise ValueError('nonzero inactive control columns')
 y=np.add(hi[:,slot],lo[:,slot],dtype=np.float32)
 if y.tobytes()!=captured.tobytes():raise ValueError('native paired-input capture differs bytewise')
 return y

def scores(control,candidate,reference):
 control,candidate,reference=[np.asarray(x,np.float64) for x in (control,candidate,reference)]
 if any(x.shape!=(N,) or not np.isfinite(x).all() for x in (control,candidate,reference)):raise ValueError('complete finite entry vectors required')
 heads=[]
 for role,start,end in ROLES:
  for h,lo in enumerate(range(start,end,128)):
   sl=slice(lo,lo+128)
   heads.append(dict(role=role,head=h,control=proj.metric(control[sl],reference[sl]),candidate=proj.metric(candidate[sl],reference[sl])))
 return dict(heads=heads,candidate_passed=all(r['candidate']['passed'] for r in heads),
   improved=all(r['candidate']['maxrel']<r['control']['maxrel'] and r['candidate']['rel_fro']<r['control']['rel_fro'] for r in heads))

def prepare(a):
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);result=dict(complete=False)
 try:
  attribution=a.attribution.resolve();prior=json.loads((attribution/'report.json').read_text())
  if not prior['complete'] or prior['model_acceptance']:raise ValueError('invalid retained attribution')
  verify(prior['sources']);verify({str(attribution/n):h for n,h in prior['artifacts'].items()})
  capture=a.capture.resolve()
  meta,evidence=attr.scan.validate(capture,a.baseline.resolve(),a.capture_build.resolve())
  if prior['sources'].get(str(capture/'sublayer-audit.json'))!=digest(capture/'sublayer-audit.json'):raise ValueError('capture differs from attribution')
  spec=base.spec_from_model_dir(Path(meta['model']))
  bundle=proj.load_bundle(Path(meta['kernels']),proj.manifest(spec,max_ctx=meta['max_ctx']))
  if any(meta[k]!=bundle[k] for k in ('build_key','spec_hash')):raise ValueError('export mismatch')
  case=next(c for c in proj.cases(bundle) if c['tag']=='L0_entry')
  if (case['n'],case['k'])!=(N,K) or len(case['ops'])!=2:raise ValueError('unsupported entry geometry')
  model=base.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=K;model.native_q8=lambda name:False
  artifacts={};targets=[]
  def save(path,values):
   path.write_bytes(np.ascontiguousarray(values).tobytes());artifacts[str(path)]=digest(path)
  for l,p,source_pos in TARGETS:
   directory=out/f'L{l}_P{p}';directory.mkdir()
   expected=next(x['weights'] for x in prior['layers'] if x['layer']==l)
   pool=np.zeros(N*K*5//8,np.uint8);offset=0
   for key,op in zip(('qkv','z'),case['ops']):
    weights,identity=attr.packed_weights(model,l,op)
    if identity!=expected[key]:raise ValueError('effective weights differ from attribution')
    del weights
    proj.apply_op(dict(op,dst=offset),model,l,pool);offset+=op['nch']*5120
   if offset!=len(pool):raise ValueError('incomplete fused weight packing')
   save(directory/'weights.bin',pool)
   x=base.read(capture/f'hybrid_L{l}_P{source_pos}_xn.bin',20*K).reshape(20,K)[p].astype(np.float32)
   qkv=base.read(capture/f'hybrid_L{l}_P{source_pos}_qkv.bin',20*10240).reshape(20,10240)[p]
   z=base.read(capture/f'hybrid_L{l}_P{source_pos}_z.bin',20*6144).reshape(20,6144)[p]
   save(directory/'captured.bin',np.concatenate((qkv,z)).astype('<f4'));save(directory/'input.bin',x)
   with np.load(attribution/f'L{l}_P{source_pos}.npz') as reference:
    for kind in ('exact','paired'):
     save(directory/f'{kind}.f64',np.concatenate([reference[f'{role}_{kind}'][p] for role,_,_ in ROLES]).astype('<f8'))
    retained=np.concatenate([reference[f'{role}_native'][p] for role,_,_ in ROLES]).astype('<f4')
    if retained.tobytes()!=(directory/'captured.bin').read_bytes():raise ValueError('native row differs from attribution')
   parts=paired.control_tool.split_inputs(x)
   for slot in (p,p+128):
    for name,part in zip(('high','low'),parts):
     full=np.zeros((T,K),np.float32);full[slot]=part
     save(directory/f'slot{slot}-{name}-x.bin',proj.tile_input(full).astype('<u2'))
   targets.append(dict(layer=l,position=p,source_position=source_pos,slots=[p,p+128]))
  kernel=(Path(meta['kernels'])/case['kernel']).resolve()
  files=[Path(__file__),Path(attr.__file__),Path(paired.__file__),attribution/'report.json',capture/'sublayer-audit.json',
         kernel/'final.xclbin',kernel/'insts.bin',Path(meta['kernels'])/'manifest.json']
  result.update(complete=True,n=N,k=K,t=T,targets=targets,production_kernel=str(kernel),capture_evidence=evidence,
    artifacts=artifacts,inputs={str(p.resolve()):digest(p) for p in files})
 except Exception as e:result.update(complete=False,error=str(e));raise
 finally:(out/'preparation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return 0

def verify_build(directory,mode):
 b=paired.verify_build(directory,mode)
 if b.get('shape')!=dict(n=N,k=K,t=T) or any(b['environment'].get(k)!=v for k,v in dict(GQP_N=str(N),GQP_K=str(K),GQP_T=str(T),GQP_BREUSE='0',GQP_FMT='q4_1').items()):raise ValueError('wrong entry kernel geometry')
 return b

def run(a):
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 result=dict(control_valid=False,candidate_passed=False,model_acceptance=False,dispatches=0)
 try:
  prepared=a.prepared.resolve();prep_path=prepared/'preparation.json';prep=json.loads(prep_path.read_text())
  if not prep['complete'] or (prep['n'],prep['k'],prep['t'])!=(N,K,T):raise ValueError('invalid prepared replay')
  expected=[dict(layer=l,position=p,source_position=s,slots=[p,p+128]) for l,p,s in TARGETS]
  if prep['targets']!=expected:raise ValueError('incomplete target/slot coverage')
  verify(prep['inputs']);verify(prep['artifacts'])
  builds={m:verify_build((a.build_root/m).resolve(),m) for m in ('control','low')}
  hb=a.harness_build.resolve();h=json.loads((hb/'build-evidence.json').read_text())
  if not h['complete']:raise ValueError('incomplete harness')
  verify(h['sources']);verify(h['compiler_dependencies'])
  verify({str(hb/'build/run_kernel'):h['harness_sha256'],str(hb/'build/CMakeCache.txt'):h['cmake_cache_sha256']})
  artifacts={};cases=[];cfgs=[]
  for target in expected:
   l,p=target['layer'],target['position'];inputs=prepared/f'L{l}_P{p}';directory=out/f'L{l}_P{p}';directory.mkdir()
   captured=base.read(inputs/'captured.bin',N).astype(np.float32)
   refs={k:np.fromfile(inputs/f'{k}.f64','<f8') for k in ('exact','paired')}
   candidates=[]
   for slot in target['slots']:
    outputs={}
    for mode in ('production','control','low'):
     kernel=Path(prep['production_kernel']) if mode=='production' else (a.build_root/mode).resolve()
     ys=[]
     for part in ('high','low'):
      tag=f'{mode}-slot{slot}-{part}';dump=directory/(tag+'.bin');cfg=directory/(tag+'.cfg')
      cfg.write_text('\n'.join(['device',f'xclbin G {kernel}/final.xclbin',f'kernelx k G {kernel}/insts.bin',
       f'buf w {N*K*5//8} {inputs}/weights.bin',f'buf x {K*T*2} {inputs}/slot{slot}-{part}-x.bin',
       f'buf y {N*T*4}','run k w x y',f'dump y {dump} {N*T*4}','']))
      cfgs.append(cfg)
      with cfg.with_suffix('.log').open('w') as f:
       subprocess.run([str(hb/'build/run_kernel'),str(cfg)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120,
                      env=os.environ|{'HARNESS_TIMEOUT_MS':'30000'})
      result['dispatches']+=1
      y=base.read(dump,N*T).astype(np.float32).reshape(N,T)
      if np.any(np.delete(y,slot,axis=1)):raise ValueError('nonzero inactive column')
      ys.append(y);artifacts[str(dump)]=digest(dump)
     outputs[mode]=ys
     if mode=='production':selected(*ys,captured,slot)
     if mode=='control':
      paired.require_control(*ys,*outputs['production']);selected(*ys,captured,slot)
    hh,hl=outputs['control'];lh,ll=outputs['low']
    candidate=paired.combine(hh[:,slot],hl[:,slot],lh[:,slot],ll[:,slot]);candidates.append(candidate)
    path=directory/f'candidate-slot{slot}.bin';path.write_bytes(candidate.tobytes());artifacts[str(path)]=digest(path)
    case=dict(layer=l,position=p,slot=slot,individual_control_exact=True,native_capture_exact=True,zero_inactive_columns=True,
              primary=scores(captured,candidate,refs['exact']),paired_input=scores(captured,candidate,refs['paired']))
    cases.append(case);print(json.dumps(case),flush=True)
   if candidates[0].tobytes()!=candidates[1].tobytes():raise ValueError('candidate slot identity differs')
  files=[Path(__file__),Path(paired.__file__),prep_path,hb/'build-evidence.json',*[(a.build_root/m/'build.json').resolve() for m in builds],*cfgs,*[p.with_suffix('.log') for p in cfgs]]
  result.update(control_valid=True,candidate_passed=all(c['primary']['candidate_passed'] for c in cases),
   improved=all(c['primary']['improved'] for c in cases),slot_identity=True,cases=cases,outputs=artifacts,
   scope='isolated fused QKV/Z on two captured rows; no corrected causal state/model execution',
   evidence={str(p.resolve()):digest(p) for p in files})
 except Exception as e:result.update(control_valid=False,candidate_passed=False,error=str(e));raise
 finally:(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return 0 if result['candidate_passed'] and result['improved'] else 1

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('prepare')
 for k in ('out','attribution','capture','capture-build','baseline'):a.add_argument('--'+k,type=Path,required=True)
 a=sub.add_parser('run')
 for k in ('out','prepared','build-root','harness-build'):a.add_argument('--'+k,type=Path,required=True)
 args=p.parse_args();raise SystemExit(prepare(args) if args.command=='prepare' else run(args))
