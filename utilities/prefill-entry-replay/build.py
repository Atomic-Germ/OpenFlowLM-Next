"""Private N16384/K5120 entry control and low-weight kernels, unchanged production ABI."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('_entry_weight_builder',ROOT/'utilities/prefill-paired-weights/build.py')
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
digest=parent.digest

def environment():
 env={k:v for k,v in os.environ.items() if not k.startswith('GQP_')}
 peano=ROOT/f'ironvenv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/llvm-aie/bin'
 env.update(GQP_N='16384',GQP_K='5120',GQP_T='256',GQP_FMT='q4_1',GQP_BREUSE='0',
  OFLM_KEEP_FAILED='1',XILINX_XRT='/opt/xilinx/xrt',OPENBLAS_NUM_THREADS='1',
  PATH=str(peano)+':/opt/xilinx/xrt/bin:'+env['PATH'])
 return env

def build(a):
 out=a.out.resolve()
 if out.exists():raise FileExistsError(out)
 if a.mode not in ('control','low') or not out.is_relative_to(ROOT/'Models'):raise ValueError('private mode/output required')
 out.mkdir(parents=True);record=dict(complete=False,mode=a.mode,shape=dict(n=16384,k=5120,t=256))
 try:
  generated=out/'generated';generated.mkdir()
  files=[parent.DESIGN/'gemm_q4_prefill.py',parent.DESIGN/'gemm_q4_dequant.h',Path(__file__),Path(parent.__file__),ROOT/'open_kernels/build_design.py']
  record['sources']={str(p):digest(p) for p in files}
  (generated/'design.py').write_text(files[0].read_text())
  header=files[1].read_text()
  (generated/'gemm_q4_dequant.h').write_text(parent.low_header(header) if a.mode=='low' else header)
  env=environment();record['environment']={k:env[k] for k in ('GQP_N','GQP_K','GQP_T','GQP_FMT','GQP_BREUSE','PATH')}
  cmd=[sys.executable,str(ROOT/'open_kernels/build_design.py'),str(generated/'design.py'),str(out)]
  record['command']=cmd;subprocess.run(cmd,env=env,check=True)
  deps=set();depfiles=list((out/'final.prj').glob('*.o.d'))
  if len(depfiles)<4:raise ValueError('missing compiler dependencies')
  for path in depfiles:
   deps.update(Path(n).resolve() for n in shlex.split(path.read_text().replace('\\\n',' ').split(':',1)[1]))
  if generated/'gemm_q4_dequant.h' not in deps:raise ValueError('private header not compiled')
  record['compiler_dependencies']={str(p):digest(p) for p in sorted(deps)}
  record['generated']={str(p):digest(p) for p in generated.iterdir() if p.is_file()}
  for p,h in record['sources'].items():
   if digest(p)!=h:raise ValueError('source changed during build')
  record['resources']=parent.resources(out)
  record.update(complete=True,binaries={str(out/n):digest(out/n) for n in ('final.xclbin','insts.bin','insts.elf')})
 finally:(out/'build.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['control','low'],required=True)
 build(p.parse_args())
