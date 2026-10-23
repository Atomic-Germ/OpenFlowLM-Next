"""Private bridge extracted from production host FFN arithmetic."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def source(core,host):
 start=core.index('void Core::ffn_block(');end=core.index('void Core::block_layer_linear(',start)
 body=core[start:end]
 expression='for (size_t j = 0; j < ff; ++j) ho[j] = g[j] / (1.f + std::exp(-g[j])) * u[j];'
 if body.count(expression)!=1:raise ValueError('FFN activation source anchor changed')
 anchor='void rms_vec(const float* x, size_t d, const float* w, double eps, float* out) {'
 if host.count(anchor)!=1:raise ValueError('RMSNorm source anchor changed')
 start=host.index(anchor);end=host.index('\n}',start)+2
 norm=host[start:end]
 return '#include <cmath>\n#include <cstddef>\n'+norm+'''\nextern "C" void norm(const float* x,const float* w,float* out,size_t d,double eps) { rms_vec(x,d,w,eps,out); }
extern "C" void activation(const float* ug,float* ho,size_t ff) {
 const float* u=ug; const float* g=u+ff;
 '''+expression+'\n}\n'

def build(out):
 out=Path(out).resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);r=dict(complete=False,commands=[])
 try:
  paths=[ROOT/'src/open_qwen36/core.cpp',ROOT/'src/open_qwen36/block_host.cpp',Path(__file__).resolve()]
  r['sources']={str(p):digest(p) for p in paths}
  cpp=out/'bridge.cpp';cpp.write_text(source(paths[0].read_text(),paths[1].read_text()))
  flags=['c++','-std=c++17','-O3','-mavx2','-mfma','-ffp-contract=off','-fPIC']
  cmds=[flags+['-MD','-MF',str(out/'bridge.d'),'-c',str(cpp),'-o',str(out/'bridge.o')],
        ['c++','-shared',str(out/'bridge.o'),'-o',str(out/'bridge.so')]]
  for cmd in cmds:r['commands'].append(cmd);subprocess.run(cmd,check=True)
  names=shlex.split((out/'bridge.d').read_text().replace('\\\n',' ').split(':',1)[1])
  r.update(complete=True,dependencies={str(Path(p).resolve()):digest(p) for p in names},
           binary_sha256=digest(out/'bridge.so'),compiler=subprocess.check_output(['c++','--version'],text=True))
 finally:(out/'build.json').write_text(json.dumps(r,indent=2)+'\n')

class Host:
 def __init__(self,out):
  out=Path(out).resolve();r=json.loads((out/'build.json').read_text())
  if not r['complete']:raise ValueError('incomplete host bridge')
  for p,h in (r['sources']|r['dependencies']|{str(out/'bridge.so'):r['binary_sha256']}).items():
   if digest(p)!=h:raise ValueError('host bridge provenance differs: '+p)
  self.lib=ctypes.CDLL(str(out/'bridge.so'))
  pointer=np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS')
  self.lib.norm.argtypes=[pointer,pointer,pointer,ctypes.c_size_t,ctypes.c_double];self.lib.norm.restype=None
  self.lib.activation.argtypes=[pointer,pointer,ctypes.c_size_t];self.lib.activation.restype=None
 @staticmethod
 def array(x):
  x=np.ascontiguousarray(x,dtype=np.float32)
  if x.ndim!=1 or not x.size or not np.isfinite(x).all():raise ValueError('finite nonempty vector required')
  return x
 def norm(self,x,w,eps):
  x,w=self.array(x),self.array(w)
  if x.shape!=w.shape or not np.isfinite(eps) or eps<=0:raise ValueError('invalid norm shape/epsilon')
  y=np.empty_like(x);self.lib.norm(x,w,y,x.size,eps)
  return self.array(y)
 def activation(self,ug):
  ug=self.array(ug)
  if ug.size%2:raise ValueError('up/gate must contain two complete projections')
  y=np.empty(ug.size//2,np.float32);self.lib.activation(ug,y,y.size)
  return self.array(y)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);build(p.parse_args().out)
