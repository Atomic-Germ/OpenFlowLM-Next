"""Selected causal sublayer boundaries: native capture and conditioned gates."""
import importlib.util
from pathlib import Path
import subprocess
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
TOOL=ROOT/'utilities/prefill-sublayer-trace'
def load(name):
 s=importlib.util.spec_from_file_location('sublayer_'+name,TOOL/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_native_capture_selects_absolute_real_position_and_refuses_overwrite(tmp_path):
 cpp=tmp_path/'test.cpp';cpp.write_text(r'''
#include "capture.hpp"
#include <limits>
using namespace open_qwen36::sublayer_trace;
int main(int argc,char**argv){
 prefix=argv[1]; float x[]={1,2,3,4,std::numeric_limits<float>::quiet_NaN(),0};
 row("res",63,6,1,2,x); // P7 is padding: no file
 row("res",63,6,2,2,x); // P7 is now real
 row("res",62,6,2,2,x); // other layer ignored
 try{row("res",63,7,1,2,x);return 1;}catch(const std::runtime_error&){}
 try{row("bad",63,7,1,2,x+4);return 2;}catch(const std::runtime_error&){}
}
''');exe=tmp_path/'test';subprocess.run(['c++','-std=c++17','-I',str(TOOL),str(cpp),'-o',str(exe)],check=True);subprocess.run([str(exe),str(tmp_path/'run')],check=True)
 assert (tmp_path/'run_L63_P7_res.bin').read_bytes()==np.array([3,4],dtype='<f4').tobytes()
 assert len(list(tmp_path.glob('*.bin')))==1

def test_patch_is_private_and_refuses_drift():
 m=load('build');cpp=(ROOT/'src/open_qwen36/core.cpp').read_text();cli=(ROOT/'src/open_qwen36/cli.cpp').read_text()
 cpp,cli=m.parent.patch(cpp,cli)
 changed,entry=m.patch(cpp,cli)
 assert 'sublayer_trace::row' in changed and 'sublayer_trace::prefix' in entry
 with pytest.raises(ValueError):m.patch(changed,entry)

@pytest.mark.parametrize('bad',['shape','nan'])
def test_conditioned_ffn_refuses_bad_capture(bad):
 m=load('check');x=np.ones(2);w=np.eye(2);c=dict(res=x,norm=x,upgate=np.ones(4),h=x,down=x)
 if bad=='shape':c['h']=np.ones(3)
 else:c['res']=np.array([1.,np.nan])
 with pytest.raises(ValueError):m.ffn_metrics(c,x,w,w,w,1e-6)

def test_conditioned_ffn_exposes_projection_error_without_changing_reference():
 m=load('check');x=np.array([1.,2.]);w=np.eye(2);norm=x/np.sqrt(np.mean(x*x)+1e-6);h=norm*norm/(1+np.exp(-norm))
 c=dict(res=x,norm=norm,upgate=np.concatenate((norm,norm)),h=h,down=h)
 r=m.ffn_metrics(c,np.ones(2),w,w,w,1e-6);assert all(v['passed'] for v in r.values())
 c['upgate']=c['upgate'].copy();c['upgate'][0]+=.1
 r=m.ffn_metrics(c,np.ones(2),w,w,w,1e-6);assert not r['up']['passed'] and r['norm']['passed']


def test_builder_refuses_existing_directory_without_mutating_evidence(tmp_path):
 import json
 from types import SimpleNamespace
 m=load('build');e=tmp_path/'build-evidence.json';original='{"complete":true,"variant":"sentinel","sources":{}}\n';e.write_text(original)
 with pytest.raises(FileExistsError):m.build(SimpleNamespace(out=tmp_path))
 assert e.read_text()==original


def test_check_error_invalidates_previous_success(tmp_path):
 import json
 from types import SimpleNamespace
 m=load('check');out=tmp_path/'audit.json';out.write_text('{"capture_valid":true,"conditioned_ffn_passed":true}')
 with pytest.raises(FileNotFoundError):m.check(SimpleNamespace(build=tmp_path/'missing',out=out))
 r=json.loads(out.read_text());assert not r['capture_valid'] and not r['conditioned_ffn_passed']


def test_output_diagnostic_keeps_propagation_separate_from_local_ffn_error():
 m=load('check');res=np.array([.5,1.]);ideal_ffn=np.array([.6,1.2]);out=res+ideal_ffn
 r=m.output_diagnostics(out,res,ideal_ffn,np.array([1.,2.]))
 assert r['local_ffn_output_error']['maxrel']==0
 assert not r['propagated_input_only']['passed']
 assert r['propagated_input_only']['maxrel']==pytest.approx(.1)
