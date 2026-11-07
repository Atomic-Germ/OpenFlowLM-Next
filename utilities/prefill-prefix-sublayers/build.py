"""Build a private weighted CLI with early-drift prefix/projection/state captures."""
import argparse
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
parent=load('_prefix_capture_parent',ROOT/'utilities/prefill-weighted-cli/build.py')
capture=load('_prefix_capture_patch',ROOT/'utilities/prefill-weighted-sublayers/build.py')

def patch(cpp,cli):
 cpp,cli=capture.patch(cpp,cli)
 old='#include "'+str(capture.HERE/'capture.hpp')+'"'
 new='#include "'+str(HERE/'capture.hpp')+'"'
 cpp=parent.once(cpp,old,new);cli=parent.once(cli,old,new)
 start=cpp.index('void Core::block_layer_linear(');end=cpp.index('void Core::attention_npu(',start)
 body=cpp[start:end];pos='static_cast<size_t>(pos_) + row0'
 anchor='    uint8_t* sp = st.map<uint8_t*>();'
 body=parent.once(body,anchor,anchor+f'''
    sublayer_trace::rows("xn", l, {pos}, t_real, hid, xn);
    sublayer_trace::rows("qkv", l, {pos}, t_real, nch, qkv);
    sublayer_trace::rows("z", l, {pos}, t_real, vw, z);
    sublayer_trace::snapshot("state_before", l, {pos}, t_real, sp, lt.state_bytes);''')
 anchor='    timing_.dn_conv_ms += phase[0];'
 body=parent.once(body,anchor,f'''
    sublayer_trace::rows("og", l, {pos}, t_real, vw, og);
    sublayer_trace::snapshot("state_after", l, {pos}, t_real, sp, lt.state_bytes);
'''+anchor)
 anchor='        const std::vector<float>& out = gout_;'
 body=parent.once(body,anchor,anchor+f'\n        sublayer_trace::row("proj", l, {pos}, t_real, hid, out.data());')
 return cpp[:start]+body+cpp[end:],cli

def build(a):
 if a.out.exists():raise FileExistsError(a.out)
 original=parent.trace.patch
 def combined(cpp,cli):return patch(*original(cpp,cli))
 parent.trace.patch=combined;a.variant='candidate'
 try:parent.build(a)
 finally:
  parent.trace.patch=original
  p=a.out.resolve()/'build-evidence.json'
  if p.exists():
   e=json.loads(p.read_text());e['variant']='prefix-sublayer'
   helpers=(Path(__file__),HERE/'capture.hpp',Path(capture.__file__),Path(capture.capture.__file__))
   e['sources'].update({str(h.resolve()):parent.digest(h) for h in helpers})
   p.write_text(json.dumps(e,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('out','xrt-include','xrt-lib'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--low',type=Path,default=ROOT/'Models/qwen38-27b/prefill-paired-weights/v2/low')
 p.add_argument('--jobs',type=int,default=4);build(p.parse_args())
