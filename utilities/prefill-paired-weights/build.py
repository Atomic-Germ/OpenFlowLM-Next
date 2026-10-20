"""Build private unchanged/low-weight GEMM variants; never alter production sources."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
DESIGN=ROOT/'open_kernels/designs/gemm_q4_prefill'
sys.path.insert(0,str(ROOT/'utilities'))
from core_sizes import sections

def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def low_header(src):
 old='''        const aie::vector<bfloat16, 64> we = se.to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> wo = so.to_vector<bfloat16>();'''
 new='''        // Private low-weight experiment: preserve the FP32 dequant residual.
        const auto he = se.to_vector<bfloat16>();
        const auto ho = so.to_vector<bfloat16>();
        aie::accum<accfloat, 64> he32, ho32, re, ro;
        he32.from_vector(he);
        ho32.from_vector(ho);
        re.from_vector(aie::sub(se.to_vector<float>(), he32.to_vector<float>()));
        ro.from_vector(aie::sub(so.to_vector<float>(), ho32.to_vector<float>()));
        const aie::vector<bfloat16, 64> we = re.to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> wo = ro.to_vector<bfloat16>();'''
 if src.count(old)!=1:raise ValueError('dequant conversion anchor changed')
 return src.replace(old,new)

def resources(out):
 prj=out/'final.prj';elfs=sorted(prj.glob('elfs_main_core_*/*.elf'))
 text={re.search(r'(\d+_\d+)$',p.stem)[1]:sections(p.read_bytes())['.text'] for p in elfs}
 l1={};mlir=prj/'input_with_addresses.mlir'
 for line in mlir.read_text().splitlines():
  if not re.search(r'aie.buffer\(%tile_\d+_[2-5]\)',line):continue
  m=re.search(r'aie.buffer\(%tile_(\d+_[2-5])\).*address = (\d+).*memref<([\dx]+)x(bf16|f32|i32|ui8)>',line)
  if not m:raise ValueError('unknown compute buffer layout')
  tile,addr,dims,ty=m.groups();end=int(addr)+math.prod(map(int,dims.split('x')))*{'bf16':2,'f32':4,'i32':4,'ui8':1}[ty]
  l1[tile]=max(l1.get(tile,0),end)
 if len(text)!=32 or set(text)!=set(l1):raise ValueError('incomplete 32-core resource coverage')
 if max(text.values())>=16384 or max(l1.values())>65536:raise ValueError('core memory budget exceeded')
 return dict(text=text,l1_buffer_end=l1,stack_bytes=4096,
             evidence={str(p):digest(p) for p in [mlir,*elfs]})

def build(a):
 out=a.out.resolve()
 if out.exists():raise FileExistsError(out)
 if a.mode not in ('control','low') or not out.is_relative_to(ROOT/'Models'):raise ValueError('private mode/output required')
 out.mkdir(parents=True);record=dict(complete=False,mode=a.mode)
 try:
  generated=out/'generated';generated.mkdir()
  paths=[DESIGN/'gemm_q4_prefill.py',DESIGN/'gemm_q4_dequant.h',Path(__file__),ROOT/'open_kernels/build_design.py']
  record['sources']={str(p):digest(p) for p in paths}
  (generated/'design.py').write_text((DESIGN/'gemm_q4_prefill.py').read_text())
  header=(DESIGN/'gemm_q4_dequant.h').read_text()
  (generated/'gemm_q4_dequant.h').write_text(low_header(header) if a.mode=='low' else header)
  env={k:v for k,v in os.environ.items() if not k.startswith('GQP_')}
  peano=ROOT/f'ironvenv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/llvm-aie/bin'
  env.update(GQP_N='34816',GQP_K='5120',GQP_T='256',GQP_FMT='q4_1',GQP_BREUSE='0',
             OFLM_KEEP_FAILED='1',XILINX_XRT='/opt/xilinx/xrt',OPENBLAS_NUM_THREADS='1',
             PATH=str(peano)+':/opt/xilinx/xrt/bin:'+env['PATH'])
  record['environment']={k:env[k] for k in ('GQP_N','GQP_K','GQP_T','GQP_FMT','GQP_BREUSE','PATH')}
  cmd=[sys.executable,str(ROOT/'open_kernels/build_design.py'),str(generated/'design.py'),str(out)]
  record['command']=cmd
  subprocess.run(cmd,env=env,check=True)
  deps=set();depfiles=list((out/'final.prj').glob('*.o.d'))
  if len(depfiles)<4:raise ValueError('missing kernel compiler dependencies')
  for p in depfiles:
   names=shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1])
   deps.update(Path(n).resolve() for n in names)
  if generated/'gemm_q4_dequant.h' not in deps:raise ValueError('private header was not compiled')
  record['compiler_dependencies']={str(p):digest(p) for p in sorted(deps)}
  record['generated']={str(p):digest(p) for p in generated.iterdir() if p.is_file()}
  for p,h in record['sources'].items():
   if digest(p)!=h:raise ValueError('source changed during build')
  record['resources']=resources(out)
  record.update(complete=True,binaries={str(out/n):digest(out/n) for n in ('final.xclbin','insts.bin','insts.elf')})
 finally:(out/'build.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['control','low'],required=True)
 build(p.parse_args())
