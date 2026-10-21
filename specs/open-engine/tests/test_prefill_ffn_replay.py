"""Composed FFN gates distinguish local math from propagated output accuracy."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
def load(name):
 s=importlib.util.spec_from_file_location('composed_ffn_'+name,ROOT/'utilities/prefill-ffn-replay'/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_bridge_uses_current_host_arithmetic_and_refuses_source_drift():
 m=load('bridge');core=(ROOT/'src/open_qwen36/core.cpp').read_text();host=(ROOT/'src/open_qwen36/block_host.cpp').read_text()
 cpp=m.source(core,host)
 assert 'std::exp(-g[j])' in cpp and 'static_cast<double>(x[j]) * x[j]' in cpp
 with pytest.raises(ValueError):m.source(core.replace('std::exp(-g[j])','std::exp(g[j])'),host)

def test_bridge_refuses_existing_output_without_mutation(tmp_path):
 (tmp_path/'build.json').write_text('keep')
 with pytest.raises(FileExistsError):load('bridge').build(tmp_path)
 assert (tmp_path/'build.json').read_text()=='keep'

@pytest.fixture(scope='module')
def host(tmp_path_factory):
 m=load('bridge');out=tmp_path_factory.mktemp('ffn')/'bridge';m.build(out);return m.Host(out)

def test_native_bridge_norm_and_activation(host):
 x=np.array([1.,2.,-1.],np.float32);w=np.ones(3,np.float32)
 np.testing.assert_allclose(host.norm(x,w,1e-6),x/np.sqrt(np.mean(x*x)+1e-6),rtol=1e-6)
 ug=np.array([2.,3.,-2.,1.],np.float32)
 np.testing.assert_allclose(host.activation(ug),ug[:2]*ug[2:]/(1+np.exp(-ug[2:])),rtol=1e-6)

@pytest.mark.parametrize('bad',['shape','nan','epsilon'])
def test_bridge_rejects_invalid_norm(host,bad):
 x=np.ones(2,np.float32);w=x.copy();eps=1e-6
 if bad=='shape':w=np.ones(3,np.float32)
 elif bad=='nan':x[0]=np.nan
 else:eps=0
 with pytest.raises(ValueError):host.norm(x,w,eps)

def test_one_ulp_control_error_is_rejected():
 m=load('check');x=np.ones(4,np.float32);y=x.copy();y[0]=np.nextafter(y[0],np.float32(2))
 with pytest.raises(ValueError):m.identity(y,x,'control')

def test_passing_local_ffn_cannot_hide_bad_global_output():
 m=load('check');r=m.score(np.ones(2),np.ones(2),np.ones(2),np.ones(2))
 assert r['ffn']['passed'] and not r['output']['passed']

def test_failed_run_invalidates_report(tmp_path):
 m=load('check');p=tmp_path/'report.json';p.write_text('{"passed":true}')
 with pytest.raises(FileNotFoundError):m.run(SimpleNamespace(out=tmp_path,bridge=tmp_path/'missing'))
 r=json.loads(p.read_text());assert not r['passed'] and not r['control_valid'] and not r['model_acceptance']
