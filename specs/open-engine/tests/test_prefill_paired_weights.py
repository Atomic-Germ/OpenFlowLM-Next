"""Private weight precision experiment retains strict controls and FP32 summation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
def load(name):
 s=importlib.util.spec_from_file_location('paired_weights_'+name,ROOT/'utilities/prefill-paired-weights'/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_low_weight_header_changes_only_fused_conversion_and_rejects_repatch():
 m=load('build');src=(ROOT/'open_kernels/designs/gemm_q4_prefill/gemm_q4_dequant.h').read_text()
 low=m.low_header(src)
 assert low!=src and 'aie::sub(se.to_vector<float>()' in low
 assert low[:low.index('gqd_fused_ky')]==src[:src.index('gqd_fused_ky')]
 assert low[low.index('// One entry point:'):]==src[src.index('// One entry point:'):]
 with pytest.raises(ValueError):m.low_header(low)

def test_builder_refuses_existing_directory_before_changing_record(tmp_path):
 p=tmp_path/'build.json';p.write_text('keep')
 with pytest.raises(FileExistsError):load('build').build(SimpleNamespace(out=tmp_path,mode='low'))
 assert p.read_text()=='keep'

def test_pair_sums_each_weight_pass_before_combining():
 m=load('run');hh=np.array([2**24],np.float32);hl=np.array([1.],np.float32);lh=-hh;ll=hl.copy()
 # (hh+hl)+(lh+ll) differs from flat left-to-right summation.
 np.testing.assert_array_equal(m.combine(hh,hl,lh,ll),np.array([1.],np.float32))

@pytest.mark.parametrize('bad',['shape','nan'])
def test_pair_refuses_bad_outputs(bad):
 x=np.ones(4,np.float32);y=np.ones(3,np.float32) if bad=='shape' else np.full(4,np.nan,np.float32)
 with pytest.raises(ValueError):load('run').combine(x,x,x,y)

def test_new_control_must_match_both_individual_old_outputs():
 m=load('run');a=np.ones(4,np.float32);b=-a
 with pytest.raises(ValueError):m.require_control(a+.25,b-.25,a,b)
 m.require_control(a,b,a,b)

def test_existing_numerical_failure_is_not_candidate_success():
 m=load('run');ref=np.ones(4);control=ref*1.001;candidate=ref*1.01
 assert not m.scores(control,candidate,ref)['candidate_passed']

def test_improvement_does_not_replace_original_reference():
 m=load('run');ref=np.ones(4);control=ref*1.02;candidate=ref*1.01
 r=m.scores(control,candidate,ref)
 assert r['improved'] and not r['candidate_passed']

def test_identical_candidate_cannot_be_reported_as_improvement():
 m=load('run');ref=np.ones(4);control=ref*1.001
 assert not m.scores(control,control.copy(),ref)['improved']
