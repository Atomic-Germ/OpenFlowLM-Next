"""Capture selected sublayers in the private paired-weight candidate, without arithmetic changes."""
import argparse
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
parent=load('_weighted_capture_parent',ROOT/'utilities/prefill-weighted-cli/build.py')
capture=load('_weighted_capture_patch',ROOT/'utilities/prefill-sublayer-trace/build.py')

def patch(cpp,cli):
 cpp,cli=capture.patch(cpp,cli)
 old='#include "'+str(capture.HERE/'capture.hpp')+'"'
 new='#include "'+str(HERE/'capture.hpp')+'"'
 return parent.once(cpp,old,new),parent.once(cli,old,new)

def build(a):
 if a.out.exists():raise FileExistsError(a.out)
 original=parent.trace.patch
 def combined(cpp,cli):return patch(*original(cpp,cli))
 parent.trace.patch=combined
 a.variant='candidate'
 try:parent.build(a)
 finally:
  parent.trace.patch=original
  p=a.out.resolve()/'build-evidence.json'
  if p.exists():
   e=json.loads(p.read_text());e['variant']='weighted-sublayer'
   e['sources'].update({str(p.resolve()):parent.digest(p) for p in (Path(__file__),HERE/'capture.hpp',Path(capture.__file__))})
   p.write_text(json.dumps(e,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('out','xrt-include','xrt-lib'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--low',type=Path,default=ROOT/'Models/qwen38-27b/prefill-paired-weights/v2/low')
 p.add_argument('--jobs',type=int,default=4);build(p.parse_args())
