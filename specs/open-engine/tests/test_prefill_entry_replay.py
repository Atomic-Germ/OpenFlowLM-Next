"""Fused QKV/Z replay keeps exact controls and independent per-head references."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
def load(name):
 s=importlib.util.spec_from_file_location('entry_replay_'+name,ROOT/'utilities/prefill-entry-replay'/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_build_geometry_is_entry_and_experiments_are_sanitized(monkeypatch):
 m=load('build');monkeypatch.setenv('GQP_N','34816');monkeypatch.setenv('GQP_BREUSE','1');monkeypatch.setenv('GQP_BAD','1')
 e=m.environment();assert [e['GQP_N'],e['GQP_K'],e['GQP_T'],e['GQP_BREUSE']]==['16384','5120','256','0']
 assert 'GQP_BAD' not in e

def test_existing_build_preserves_evidence(tmp_path):
 p=tmp_path/'build.json';p.write_text('keep')
 with pytest.raises(FileExistsError):load('build').build(SimpleNamespace(out=tmp_path,mode='low'))
 assert p.read_text()=='keep'

def test_low_header_preserves_stream_body():
 m=load('build');s=(m.parent.DESIGN/'gemm_q4_dequant.h').read_text();low=m.parent.low_header(s)
 assert low[:low.index('gqd_fused_ky')]==s[:s.index('gqd_fused_ky')]
 assert low[low.index('// One entry point:'):]==s[s.index('// One entry point:'):]

def test_control_requires_exact_native_sum_and_inactive_zero():
 m=load('check');hi=np.zeros((4,256),np.float32);lo=hi.copy();hi[:,6]=1;lo[:,6]=.125
 np.testing.assert_array_equal(m.selected(hi,lo,np.full(4,1.125,np.float32),6),np.full(4,1.125,np.float32))
 with pytest.raises(ValueError):m.selected(hi,lo,np.ones(4,np.float32),6)
 hi[0,7]=1
 with pytest.raises(ValueError):m.selected(hi,lo,np.full(4,1.125,np.float32),6)

def test_individual_control_cannot_cancel_changed_products():
 m=load('check');x=np.ones(4,np.float32)
 with pytest.raises(ValueError):m.paired.require_control(x+.25,-x-.25,x,-x)

def test_head_gate_preserves_small_head_failure():
 m=load('check');ref=np.ones(16384);ref[:128]=1000;control=ref*1.02;candidate=ref.copy();candidate[128]=1.01
 r=m.scores(control,candidate,ref)
 assert len(r['heads'])==128 and r['improved'] and not r['candidate_passed']
 assert not r['heads'][1]['candidate']['passed']

@pytest.mark.parametrize('bad',['shape','nan'])
def test_heads_require_complete_finite_values(bad):
 m=load('check');ref=np.ones(16384);got=ref.copy()
 if bad=='shape':got=got[:-1]
 else:got[0]=np.nan
 with pytest.raises(ValueError):m.scores(ref,got,ref)

def test_equal_result_is_not_strict_improvement():
 m=load('check');ref=np.ones(16384);got=ref*1.001
 assert not m.scores(got,got,ref)['improved']

def test_four_product_sum_retains_native_order():
 m=load('check');hi=np.array([2**24],np.float32);one=np.ones(1,np.float32)
 np.testing.assert_array_equal(m.paired.combine(hi,one,-hi,one),one)

def test_prepare_failure_is_explicit_and_never_overwrites(tmp_path):
 m=load('check');out=tmp_path/'out'
 with pytest.raises(FileNotFoundError):m.prepare(SimpleNamespace(out=out,attribution=tmp_path/'missing'))
 assert not json.loads((out/'preparation.json').read_text())['complete']
 with pytest.raises(FileExistsError):m.prepare(SimpleNamespace(out=out))

def test_run_failure_cannot_leave_success(tmp_path):
 m=load('check');out=tmp_path/'out'
 with pytest.raises(FileNotFoundError):m.run(SimpleNamespace(out=out,prepared=tmp_path/'missing'))
 r=json.loads((out/'report.json').read_text());assert not r['control_valid'] and not r['candidate_passed'] and not r['model_acceptance']
