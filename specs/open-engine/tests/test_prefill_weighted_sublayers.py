"""Weighted candidate captures must preserve native history and complete evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
TOOL=ROOT/'utilities/prefill-weighted-sublayers'
def load(name):
 s=importlib.util.spec_from_file_location('weighted_sublayer_'+name,TOOL/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_capture_targets_real_rows_and_rejects_duplicates(tmp_path):
 cpp=tmp_path/'test.cpp';cpp.write_text(r'''
#include "capture.hpp"
using namespace open_qwen36::sublayer_trace;
int main(int argc,char**argv) {
 prefix=argv[1];float x[]={1,2,3,4};
 row("res",55,6,1,2,x); // target in padding
 row("res",55,6,2,2,x); // target now real
 row("res",51,16,2,2,x);
 row("res",63,6,2,2,x); // old target must not be captured
 try{row("res",51,17,1,2,x);return 1;}catch(const std::runtime_error&){}
}
''');exe=tmp_path/'test';subprocess.run(['c++','-std=c++17','-I',str(TOOL),str(cpp),'-o',str(exe)],check=True)
 subprocess.run([str(exe),str(tmp_path/'run')],check=True)
 assert {p.name for p in tmp_path.glob('*.bin')}=={'run_L55_P7_res.bin','run_L51_P17_res.bin'}
 assert (tmp_path/'run_L51_P17_res.bin').read_bytes()==np.array([3,4],'<f4').tobytes()

def test_patch_preserves_weighted_dispatch_and_refuses_reapplication():
 m=load('build');cpp=(ROOT/'src/open_qwen36/core.cpp').read_text();hpp=(ROOT/'src/open_qwen36/core.hpp').read_text();cli=(ROOT/'src/open_qwen36/cli.cpp').read_text()
 cpp,hpp=m.parent.patch(cpp,hpp,Path('/tmp/low'));cpp,cli=m.parent.trace.patch(cpp,cli)
 patched,entry=m.patch(cpp,cli)
 start=cpp.index('const float* Core::gemm_run(');end=cpp.index('\nvoid Core::gemm(',start)
 assert cpp[start:end] in patched
 assert str(TOOL/'capture.hpp') in patched and 'sublayer_trace::prefix' in entry
 assert str(m.capture.HERE/'capture.hpp') not in patched
 with pytest.raises(ValueError):m.patch(patched,entry)

def test_existing_build_preserves_evidence(tmp_path):
 p=tmp_path/'build-evidence.json';p.write_text('keep')
 with pytest.raises(FileExistsError):load('build').build(SimpleNamespace(out=tmp_path))
 assert p.read_text()=='keep'

@pytest.mark.parametrize('problem',['missing','extra','old_target'])
def test_capture_inventory_refuses_incomplete_or_wrong_targets(tmp_path,problem):
 m=load('check')
 for name in m.expected_captures():(tmp_path/name).touch()
 m.inventory(tmp_path)
 if problem=='missing':(tmp_path/next(iter(m.expected_captures()))).unlink()
 elif problem=='extra':(tmp_path/'hybrid_L55_P7_extra.bin').touch()
 else:(tmp_path/'sequential_L63_P7_act.bin').touch()
 with pytest.raises(ValueError):m.inventory(tmp_path)

def test_audit_failure_invalidates_success(tmp_path):
 p=tmp_path/'audit.json';p.write_text('{"capture_valid":true,"conditioned_ffn_passed":true}')
 with pytest.raises(FileNotFoundError):load('check').check(SimpleNamespace(build=tmp_path/'missing',out=p))
 r=json.loads(p.read_text());assert not r['capture_valid'] and not r['conditioned_ffn_passed'] and not r['model_acceptance']


def test_layer_residual_inventory_allows_selected_sublayer_residuals(tmp_path):
 m=load('check')
 for route in m.trace.ROUTES:
  for layer in range(2):(tmp_path/f'{route}_res{layer}.bin').touch()
 for name in m.expected_captures():(tmp_path/name).touch()
 m.residual_inventory(tmp_path,2)
 (tmp_path/'hybrid_L55_P7_res_extra.bin').touch()
 with pytest.raises(ValueError):m.residual_inventory(tmp_path,2)
 (tmp_path/'hybrid_L55_P7_res_extra.bin').unlink();(tmp_path/'hybrid_res1.bin').unlink()
 with pytest.raises(ValueError):m.residual_inventory(tmp_path,2)
