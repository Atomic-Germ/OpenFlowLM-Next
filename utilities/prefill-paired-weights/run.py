"""Replay private paired weights with exact production controls and fixed references."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('_paired_weight_control',ROOT/'utilities/prefill-upgate-replay/check.py')
control_tool=importlib.util.module_from_spec(s);s.loader.exec_module(control_tool)
digest,verify=control_tool.digest,control_tool.verify
N,K,T=control_tool.N,control_tool.K,control_tool.T

def combine(hh,hl,lh,ll):
 values=[np.asarray(v,np.float32) for v in (hh,hl,lh,ll)]
 if not values[0].size or any(v.shape!=values[0].shape or not np.isfinite(v).all() for v in values):
  raise ValueError('complete finite equal-shaped products required')
 with np.errstate(over='raise',invalid='raise'):
  return np.add(np.add(values[0],values[1],dtype=np.float32),np.add(values[2],values[3],dtype=np.float32),dtype=np.float32)

def require_control(hi,lo,old_hi,old_lo):
 for got,old in ((hi,old_hi),(lo,old_lo)):
  if got.shape!=old.shape or got.dtype!=old.dtype or got.tobytes()!=old.tobytes():
   raise ValueError('individual production control differs bytewise')

def scores(control,candidate,reference):
 c=control_tool.projection_metrics(control,reference);p=control_tool.projection_metrics(candidate,reference)
 return dict(control=c,candidate=p,candidate_passed=all(v['passed'] for v in p.values()),
             improved=all(p[k]['maxrel']<c[k]['maxrel'] and p[k]['rel_fro']<c[k]['rel_fro'] for k in c))

def verify_build(path,mode):
 record=json.loads((path/'build.json').read_text())
 if not record['complete'] or record['mode']!=mode:raise ValueError('incomplete or wrong kernel build')
 for key in ('sources','generated','compiler_dependencies','binaries'):verify(record[key])
 verify(record['resources']['evidence'])
 return record

def run(a):
 out=a.out.resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 result=dict(control_valid=False,candidate_passed=False,model_acceptance=False)
 try:
  baseline=a.baseline.resolve();old=json.loads((baseline/'report.json').read_text());audit=json.loads((baseline/'audit.json').read_text())
  if not old['control_valid'] or not audit['complete']:raise ValueError('invalid retained projection control')
  verify(old['evidence']);verify(old['outputs']);verify(audit['inputs']);verify(audit['artifacts'])
  builds={mode:verify_build((a.build_root/mode).resolve(),mode) for mode in ('control','low')}
  hb=a.harness_build.resolve();h=json.loads((hb/'build-evidence.json').read_text())
  if not h['complete']:raise ValueError('incomplete harness')
  verify(h['sources']);verify(h['compiler_dependencies']);verify({str(hb/'build/run_kernel'):h['harness_sha256'],str(hb/'build/CMakeCache.txt'):h['cmake_cache_sha256']})
  hashes={};cases=[];cfgs=[]
  captured=control_tool.attribution.base.read(baseline/'captured.bin',N).astype(np.float32)
  ref=np.fromfile(baseline/'exact-weight-ref.bin','<f8').reshape(N,4)
  for slot in control_tool.SLOTS:
   outputs={}
   for mode in ('control','low'):
    kernel=(a.build_root/mode).resolve();ys=[]
    for part in ('high','low'):
     tag=f'{mode}-slot{slot}-{part}';dump=out/(tag+'.bin');cfg=out/(tag+'.cfg')
     cfg.write_text('\n'.join(['device',f'xclbin G {kernel}/final.xclbin',f'kernelx k G {kernel}/insts.bin',
      f'buf w {N*K*5//8} {baseline}/weights.bin',f'buf x {K*T*2} {baseline}/slot{slot}-{part}-x.bin',
      f'buf y {N*T*4}','run k w x y',f'dump y {dump} {N*T*4}','']))
     cfgs.append(cfg)
     with cfg.with_suffix('.log').open('w') as f:
      subprocess.run([str(hb/'build/run_kernel'),str(cfg)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120,
                     env=os.environ|{'HARNESS_TIMEOUT_MS':'30000'})
     y=control_tool.attribution.base.read(dump,N*T).astype(np.float32).reshape(N,T)
     if np.any(np.delete(y,slot,axis=1)!=0):raise ValueError('nonzero inactive output column')
     ys.append(y);hashes[str(dump)]=digest(dump)
    outputs[mode]=ys
    if mode=='control':
     previous=[control_tool.attribution.base.read(baseline/f'slot{slot}-{part}-y.bin',N*T).astype(np.float32).reshape(N,T) for part in ('high','low')]
     require_control(*ys,*previous)
     control_tool.control(*ys,captured,slot)
   hh,hl=outputs['control'];lh,ll=outputs['low']
   paired=combine(hh[:,slot],hl[:,slot],lh[:,slot],ll[:,slot])
   path=out/f'candidate-slot{slot}.bin';path.write_bytes(paired.tobytes());hashes[str(path)]=digest(path)
   case=dict(slot=slot,individual_control_exact=True,zero_inactive_columns=True,
             primary=scores(captured,paired,ref[:,0]),paired_input=scores(captured,paired,ref[:,1]))
   cases.append(case);print(json.dumps(case),flush=True)
  if (out/'candidate-slot7.bin').read_bytes()!=(out/'candidate-slot135.bin').read_bytes():raise ValueError('candidate slot identity differs')
  files=[Path(__file__),baseline/'report.json',baseline/'audit.json',hb/'build-evidence.json']
  files += [(a.build_root/m/'build.json').resolve() for m in builds]+cfgs+[p.with_suffix('.log') for p in cfgs]
  result.update(control_valid=True,candidate_passed=all(c['primary']['candidate_passed'] for c in cases),
                improved=all(c['primary']['improved'] for c in cases),slot_identity=True,cases=cases,dispatches=len(cfgs),outputs=hashes,
                scope='isolated L63/P7 up/gate; no propagated FFN/model correction',evidence={str(p):digest(p) for p in files})
 except Exception as e:result.update(control_valid=False,candidate_passed=False,error=str(e));raise
 finally:(out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return 0 if result['candidate_passed'] and result['improved'] else 1

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('out','baseline','build-root','harness-build'):p.add_argument('--'+name,type=Path,required=True)
 raise SystemExit(run(p.parse_args()))
