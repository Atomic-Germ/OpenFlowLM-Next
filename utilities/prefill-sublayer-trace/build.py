"""Extend a private residual tracer with selected sublayer captures; no arithmetic change."""
import argparse
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_sublayer_parent',ROOT/'utilities/prefill-residual-trace/build.py')
parent=importlib.util.module_from_spec(s);s.loader.exec_module(parent)
once=parent.once


def patch(cpp,cli):
 if "sublayer_trace::" in cpp or "sublayer_trace::" in cli:raise ValueError("sublayer capture already patched")
 anchor='#include "open_qwen36/trace.hpp"'
 include=anchor+'\n#include "'+str(HERE/'capture.hpp')+'"'
 cpp=once(cpp,anchor,include);cli=once(cli,anchor,include)
 cli=once(cli,'        open_qwen36::residual_trace::writer =',
          '        open_qwen36::sublayer_trace::prefix = a.dump_prefix;\n        open_qwen36::residual_trace::writer =')
 anchor='            residual_trace::writer.append(l, static_cast<size_t>(pos_), 1, w_->hidden, xres.map<const float*>());'
 cpp=once(cpp,anchor,anchor+'''
            if (sublayer_trace::selected(l, static_cast<size_t>(pos_))) {
                read_back(act_[l], types_[l]->act_bytes, 0);
                sublayer_trace::save("act", l, pos_, act_[l].map<const uint8_t*>(), types_[l]->act_bytes);
            }''')
 # Restrict edits to these two functions; their causal state code stays intact.
 for name,end,pos in [('block_layer_linear','attention_npu','static_cast<size_t>(pos_) + row0'),
                      ('block_layer_full','gemm_phase','pos0')]:
  start=cpp.index('void Core::'+name+'(');stop=cpp.index('void Core::'+end+'(',start)
  body=cpp[start:stop]
  anchor='    host::rmsnorm_rows(res, T, hid, hc.postln.data(), gb.eps, xm);'
  body=once(body,anchor,f'''    sublayer_trace::row("input", l, {pos}, t_real, hid, xres);
    sublayer_trace::row("res", l, {pos}, t_real, hid, res);
'''+anchor+f'''
    sublayer_trace::row("norm", l, {pos}, t_real, hid, xm);
    sublayer_trace::start = {pos}; sublayer_trace::real_rows = t_real;''')
  cpp=cpp[:start]+body+cpp[stop:]
 start=cpp.index('void Core::ffn_block(');stop=cpp.index('void Core::block_layer_linear(',start)
 body=cpp[start:stop]
 body=once(body,'    const std::vector<float>& ug = sg_ug_;',
           '    const std::vector<float>& ug = sg_ug_;\n    sublayer_trace::stage("upgate", l, 2 * ff, ug.data());')
 body=once(body,'    gemm(prog[1], h, T, ff, hid, l, sg_y_);',
           '    sublayer_trace::stage("h", l, ff, h);\n    gemm(prog[1], h, T, ff, hid, l, sg_y_);')
 body=once(body,'    const std::vector<float>& y = sg_y_;',
           '    const std::vector<float>& y = sg_y_;\n    sublayer_trace::stage("down", l, hid, y.data());')
 return cpp[:start]+body+cpp[stop:],cli


def build(a):
 if a.out.exists():raise FileExistsError(a.out)
 original=parent.patch
 def combined(cpp,cli):return patch(*original(cpp,cli))
 parent.patch=combined
 try:
  parent.build(a)
 finally:
  parent.patch=original
  path=a.out.resolve()/'build-evidence.json'
  if path.exists():
   e=json.loads(path.read_text());e['variant']='paired-sublayer-trace'
   e['sources'].update({str(p):parent.digest(p) for p in [Path(__file__).resolve(),HERE/'capture.hpp']})
   path.write_text(json.dumps(e,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('out','xrt-include','xrt-lib'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--jobs',type=int,default=4);build(p.parse_args())
