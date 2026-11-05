"""Targeted causal captures must retain real prefixes, state and exact controls."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
TOOL=ROOT/'utilities/prefill-prefix-sublayers'
def load(name):
 s=importlib.util.spec_from_file_location('prefix_sublayers_'+name,TOOL/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_capture_real_prefix_and_selected_rows(tmp_path):
 cpp=tmp_path/'probe.cpp';cpp.write_text(r'''
#include "capture.hpp"
using namespace open_qwen36::sublayer_trace;
int main(int argc,char**argv){
 prefix=argv[1];float x[40]={};for(int i=0;i<40;++i)x[i]=i;
 rows("xn",34,0,17,2,x); // target is padding
 rows("xn",34,0,20,2,x);row("res",37,0,20,2,x);
 row("res",53,0,20,2,x);row("res",55,0,20,2,x);
 try{rows("xn",34,0,20,2,x);return 1;}catch(const std::runtime_error&){}
 try{rows("og",34,16,4,2,x);return 2;}catch(const std::runtime_error&){}
}
''');exe=tmp_path/'probe';subprocess.run(['c++','-std=c++17','-I',str(TOOL),str(cpp),'-o',str(exe)],check=True)
 subprocess.run([str(exe),str(tmp_path/'run')],check=True)
 assert {p.name for p in tmp_path.glob('*.bin')}=={'run_L34_P17_xn.bin','run_L37_P7_res.bin','run_L53_P17_res.bin'}
 np.testing.assert_array_equal(np.fromfile(tmp_path/'run_L34_P17_xn.bin','<f4'),np.arange(40))

def test_patch_keeps_weighted_arithmetic_and_rejects_repeat():
 m=load('build');cpp=(ROOT/'src/open_qwen36/core.cpp').read_text();hpp=(ROOT/'src/open_qwen36/core.hpp').read_text();cli=(ROOT/'src/open_qwen36/cli.cpp').read_text()
 cpp,hpp=m.parent.patch(cpp,hpp,Path('/tmp/low'));cpp,cli=m.parent.trace.patch(cpp,cli)
 result,entry=m.patch(cpp,cli)
 a=cpp.index('const float* Core::gemm_run(');b=cpp.index('\nvoid Core::gemm(',a)
 assert cpp[a:b] in result
 assert result.index('snapshot("state_before"')<result.index('host::deltanet_block(g, qkv')<result.index('snapshot("state_after"')
 for key in ('xn','qkv','z','og'):assert f'rows("{key}"' in result
 assert str(TOOL/'capture.hpp') in result and str(TOOL/'capture.hpp') in entry
 with pytest.raises(ValueError):m.patch(result,entry)

def test_existing_build_preserved(tmp_path):
 p=tmp_path/'build-evidence.json';p.write_text('keep')
 with pytest.raises(FileExistsError):load('build').build(SimpleNamespace(out=tmp_path))
 assert p.read_text()=='keep'

@pytest.mark.parametrize('problem',['missing','extra','old'])
def test_exact_capture_inventory(tmp_path,problem):
 m=load('check');names=m.old.expected_captures();assert len(names)==81
 for n in names:(tmp_path/n).touch()
 m.old.inventory(tmp_path)
 if problem=='missing':(tmp_path/next(iter(names))).unlink()
 else:(tmp_path/('hybrid_L34_P17_extra.bin' if problem=='extra' else 'hybrid_L55_P7_xn.bin')).touch()
 with pytest.raises(ValueError):m.old.inventory(tmp_path)

def test_state_requires_exact_extent_zero_start_and_zero_padding():
 m=load('check');offsets=SimpleNamespace(STATE_BYTES=40,STATE_S_OFF=8,S_ROWS=2)
 spec=SimpleNamespace(conv_kernel=2,lin_qkv_dim=2,lin_value_heads=2,lin_value_dim=1)
 after=np.zeros(10,'<f4');after[:2]=[1,2];after[2:6]=[3,0,4,0]
 conv,state=m.states(bytes(40),after.tobytes(),offsets,spec)
 assert conv.tolist()==[[1,2]] and state.shape==(2,2,1)
 for start,end in [(bytes(39),after.tobytes()),(after.tobytes(),after.tobytes())]:
  with pytest.raises(ValueError):m.states(start,end,offsets,spec)
 after[3]=1
 with pytest.raises(ValueError):m.states(bytes(40),after.tobytes(),offsets,spec)

def test_projection_metric_refuses_shape_broadcast():
 m=load('check')
 with pytest.raises(ValueError):m.metrics(np.zeros((2,3)),np.zeros((1,3)))

def test_failure_persists_no_success(tmp_path):
 p=tmp_path/'audit.json'
 with pytest.raises(FileNotFoundError):load('check').check(SimpleNamespace(build=tmp_path/'missing',out=p))
 r=json.loads(p.read_text());assert not r['capture_valid'] and not r['model_acceptance']

def test_runner_validates_build_before_creating_output(tmp_path):
 out=tmp_path/'run'
 with pytest.raises(FileNotFoundError):load('run').run(SimpleNamespace(out=out,build=tmp_path/'missing'))
 assert not out.exists()
