"""Compose retained up/gate with native host activation and production down GEMM."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bridge import Host

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
paired=load('_composed_paired',ROOT/'utilities/prefill-paired-weights/run.py')
parent=paired.control_tool
proj=parent.proj;base=parent.attribution.base
H,F,T=5120,17408,256
SLOTS=(7,135)
digest,verify=parent.digest,parent.verify
DEFAULT=ROOT/'Models/qwen38-27b'

def identity(got,reference,label):
 got,reference=np.asarray(got),np.asarray(reference)
 if got.shape!=reference.shape or got.dtype!=reference.dtype or got.tobytes()!=reference.tobytes():
  raise ValueError(label+' is not byte-exact')

def score(res,down,ideal_ffn,reference):
 arrays=[np.asarray(v,np.float64) for v in (res,down,ideal_ffn,reference)]
 if not arrays[0].size or arrays[0].ndim!=1 or any(v.shape!=arrays[0].shape or not np.isfinite(v).all() for v in arrays):
  raise ValueError('complete finite equal-sized FFN vectors required')
 output=np.add(arrays[0].astype(np.float32),arrays[1].astype(np.float32),dtype=np.float32)
 return dict(ffn=proj.metric(arrays[1],arrays[2]),output=proj.metric(output,arrays[3]))

def prepare(a):
 out=a.out.resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);record=dict(complete=False)
 try:
  host=Host(a.bridge)
  meta,_=parent.attribution.verify_inputs(SimpleNamespace(run=a.capture.resolve(),build=a.capture_build.resolve(),baseline=a.baseline.resolve()))
  r=json.loads((a.paired/'report.json').read_text())
  if not r['control_valid'] or not r['candidate_passed'] or not r['slot_identity']:raise ValueError('invalid paired-weight input')
  verify(r['evidence']);verify(r['outputs'])
  for mode in ('control','low'):paired.verify_build(a.kernel_builds/mode,mode)
  spec=proj.load_spec(a.kernels/'spec.json');m=proj.load_bundle(a.kernels,proj.manifest(spec,max_ctx=meta['max_ctx']))
  if meta['build_key']!=m['build_key'] or meta['spec_hash']!=m['spec_hash'] or spec.hidden!=H or spec.intermediate!=F:raise ValueError('model/export differs')
  case=next(c for c in proj.cases(m) if c['tag']=='L3_down')
  if (case['n'],case['k'],case['storage'])!=(H,F,'pool'):raise ValueError('unsupported down layout')
  model=proj.Q4NX(Path(meta['model'])/'model.q4nx');model.hidden=H;model.native_q8=lambda name:False
  res=base.read(a.capture/'hybrid_L63_P7_res.bin',H).astype(np.float32)
  norm=host.norm(res,model.bf16('model.layers.63.post_attention_layernorm.weight'),spec.norm_eps)
  identity(norm,base.read(a.capture/'hybrid_L63_P7_norm.bin',H).astype(np.float32),'norm')
  captured_ug=base.read(a.capture/'hybrid_L63_P7_upgate.bin',2*F).astype(np.float32)
  candidate=base.read(a.paired/'candidate-slot7.bin',2*F).astype(np.float32)
  identity(candidate,base.read(a.paired/'candidate-slot135.bin',2*F).astype(np.float32),'candidate up/gate slots')
  acts={name:host.activation(ug) for name,ug in [('control',captured_ug),('candidate',candidate)]}
  identity(acts['control'],base.read(a.capture/'hybrid_L63_P7_h.bin',F).astype(np.float32),'control activation')
  artifacts={}
  def save(name,array):
   path=out/name;path.write_bytes(np.ascontiguousarray(array).tobytes());artifacts[str(path)]=digest(path)
  for name,values in acts.items():save(name+'-h.bin',values)
  save('res.bin',res);save('norm.bin',norm)
  save('control-down.bin',base.read(a.capture/'hybrid_L63_P7_down.bin',H).astype(np.float32))
  n=len(meta['ids'])
  save('control-output.bin',base.read(a.capture/'hybrid_res63.bin',(n+1)*H).reshape(n+1,H)[7].astype(np.float32))
  save('reference.bin',base.read(a.capture/'ref_res63.bin',n*H).reshape(n,H)[7].astype(np.float32))
  save('ideal-ffn.bin',base.oracle.ffn(model,spec,63,res).astype('<f8'))
  prior={str(Path(p).resolve()):h for p,h in json.loads(a.original.read_text()).items()}
  pool_path=a.pools.resolve()/'pool_L63.bin'
  if prior.get(str(pool_path))!=digest(pool_path):raise ValueError('changed production pool')
  pool=np.zeros(H*F*5//8,np.uint8);dst=0
  with pool_path.open('rb') as f:
   for op in case['ops']:
    size=op['nch']*5120;proj.apply_op(dict(op,dst=dst),model,63,pool);f.seek(op['dst'])
    if pool[dst:dst+size].tobytes()!=f.read(size):raise ValueError('down packed bytes differ')
    dst+=size
  if dst!=pool.size:raise ValueError('incomplete down packing')
  w=proj.dequant_pool(pool,H,F,rs=2);original=model.matmul_w('model.layers.63.mlp.down_proj.weight',H,F)
  if not np.array_equal(w,original):raise ValueError('down effective weights differ')
  del w,original
  save('weights.bin',pool);del pool
  kernel=a.kernels.resolve()/case['kernel'];commands=[]
  for name,h in acts.items():
   for slot in SLOTS:
    for part,values in zip(('high','low'),parent.split_inputs(h)):
     tag=f'{name}-slot{slot}-{part}';full=np.zeros((T,F),np.float32);full[slot]=values
     save(tag+'-x.bin',proj.tile_input(full))
     cfg=out/(tag+'.cfg');cfg.write_text('\n'.join(['device',f'xclbin G {kernel}/final.xclbin',f'kernelx k G {kernel}/insts.bin',
      f'buf w {H*F*5//8} {out}/weights.bin',f'buf x {F*T*2} {out}/{tag}-x.bin',f'buf y {H*T*4}',
      'run k w x y',f'dump y {out}/{tag}-y.bin {H*T*4}','']))
     commands.append(str(cfg));artifacts[str(cfg)]=digest(cfg)
  files=[Path(__file__),a.bridge/'build.json',a.capture/'sublayer-audit.json',a.capture/'reference.json',a.paired/'report.json',
         a.original,pool_path,Path(meta['model'])/'model.q4nx',a.kernels/'manifest.json',a.kernels/'toolchain.json',
         kernel/'final.xclbin',kernel/'insts.bin']
  files += [a.capture/('hybrid_L63_P7_'+key+'.bin') for key in ('res','norm','upgate','h','down')]
  files += [a.capture/'hybrid_res63.bin',a.capture/'ref_res63.bin']
  record.update(complete=True,norm_exact=True,activation_exact=True,down_pool_exact=True,effective_weight_exact=True,
    scope='conditioned L63/P7 FFN, retained causal input; attention/state not replayed',slots=SLOTS,
    artifacts=artifacts,commands=commands,inputs={str(p.resolve()):digest(p) for p in files})
 finally:(out/'audit.json').write_text(json.dumps(record,indent=2)+'\n')

def run(a):
 out=a.out.resolve();result=dict(passed=False,control_valid=False,model_acceptance=False)
 try:
  audit=json.loads((out/'audit.json').read_text());Host(a.bridge)
  if not audit['complete'] or audit['slots']!=list(SLOTS):raise ValueError('invalid preparation')
  verify(audit['inputs']);verify(audit['artifacts'])
  hb=a.harness_build.resolve();record=json.loads((hb/'build-evidence.json').read_text())
  if not record['complete']:raise ValueError('invalid harness')
  verify(record['sources']);verify(record['compiler_dependencies'])
  harness=hb/'build/run_kernel';verify({str(harness):record['harness_sha256'],str(hb/'build/CMakeCache.txt'):record['cmake_cache_sha256']})
  for name in ('control','candidate'):
   for slot in SLOTS:
    for part in ('high','low'):(out/f'{name}-slot{slot}-{part}-y.bin').unlink(missing_ok=True)
  # Execute and validate control first, before any candidate dispatch.
  cases=[];hashes={};res=base.read(out/'res.bin',H).astype(np.float32)
  ref=base.read(out/'reference.bin',H);ideal=np.fromfile(out/'ideal-ffn.bin','<f8')
  for name in ('control','candidate'):
   selected=[]
   for slot in SLOTS:
    ys=[]
    for part in ('high','low'):
     cfg=out/f'{name}-slot{slot}-{part}.cfg'
     if str(cfg) not in audit['commands']:raise ValueError('missing dispatch config')
     with cfg.with_suffix('.log').open('w') as f:
      subprocess.run([str(harness),str(cfg)],check=True,timeout=120,stdout=f,stderr=subprocess.STDOUT,
                     env=os.environ|{'HARNESS_TIMEOUT_MS':'30000'})
     path=out/f'{name}-slot{slot}-{part}-y.bin';y=base.read(path,H*T).astype(np.float32).reshape(H,T)
     if np.any(np.delete(y,slot,axis=1)!=0):raise ValueError('nonzero inactive output columns')
     ys.append(y[:,slot]);hashes[str(path)]=digest(path)
    down=np.add(*ys,dtype=np.float32);output=np.add(res,down,dtype=np.float32)
    if name=='control':
     identity(down,base.read(out/'control-down.bin',H).astype(np.float32),'control down')
     identity(output,base.read(out/'control-output.bin',H).astype(np.float32),'control residual')
    selected.append(down)
    for kind,values in (('down',down),('output',output)):
     path=out/f'{name}-slot{slot}-{kind}.bin';path.write_bytes(values.tobytes());hashes[str(path)]=digest(path)
    case=dict(variant=name,slot=slot,zero_inactive_columns=True,**score(res,down,ideal,ref));cases.append(case)
    print(json.dumps(case),flush=True)
   identity(*selected,name+' slot identity')
  verify(audit['artifacts'])
  candidates=[c for c in cases if c['variant']=='candidate']
  controls=[c for c in cases if c['variant']=='control']
  passed=all(c['ffn']['passed'] and c['output']['passed'] for c in candidates)
  evidence=[Path(__file__),out/'audit.json',hb/'build-evidence.json',a.bridge/'build.json']
  evidence += [Path(c).with_suffix('.log') for c in audit['commands']]
  result.update(passed=passed,control_valid=True,cases=cases,slot_identity=True,dispatches=len(audit['commands']),
    output_improved=all(c['output']['maxrel']<b['output']['maxrel'] for c,b in zip(candidates,controls)),
    scope=audit['scope'],outputs=hashes,evidence={str(p.resolve()):digest(p) for p in evidence})
 except Exception as e:result.update(passed=False,control_valid=False,error=str(e));raise
 finally:(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return 0 if result['passed'] else 1

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
 prep=subs.add_parser('prepare');r=subs.add_parser('run')
 for sub in (prep,r):
  sub.add_argument('--out',type=Path,required=True);sub.add_argument('--bridge',type=Path,required=True)
 for name,default in {'capture':'prefill-sublayer-trace/run-257','capture-build':'prefill-sublayer-trace/verified',
  'baseline':'prefill-residual-trace/run-257','paired':'prefill-paired-weights/replay','kernel-builds':'prefill-paired-weights/v2',
  'kernels':'production-consistent/kernels','pools':'production-full/pools','original':'production-consistent/input-sha256.json'}.items():
  prep.add_argument('--'+name,type=Path,default=DEFAULT/default)
 r.add_argument('--harness-build',type=Path,default=DEFAULT/'prefill-upgate-replay/harness')
 args=p.parse_args();raise SystemExit(globals()[args.command](args))
