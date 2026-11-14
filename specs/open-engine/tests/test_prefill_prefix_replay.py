"""Causal prefix replay must retain native controls and independent residual gates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
def module():
 s=importlib.util.spec_from_file_location('prefix_replay',ROOT/'utilities/prefill-prefix-replay/check.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_prefix_placement_preserves_order_and_zero_padding():
 m=module();x=np.arange(60,dtype=np.float32).reshape(20,3)
 full=m.place(x,128)
 np.testing.assert_array_equal(full[128:148],x)
 assert not np.any(full[:128]) and not np.any(full[148:])

@pytest.mark.parametrize('offset',[True,-1,237])
def test_invalid_prefix_placement(offset):
 with pytest.raises(ValueError):module().place(np.ones((20,3),np.float32),offset)

def test_products_require_all_inactive_columns_zero():
 m=module();y=np.zeros((3,256),np.float32);y[:,128:148]=1
 np.testing.assert_array_equal(m.extract(y,128,20),np.ones((20,3)))
 y[0,149]=1
 with pytest.raises(ValueError):m.extract(y,128,20)

def test_nonfinite_product_rejected():
 y=np.zeros((3,256),np.float32);y[0,0]=np.nan
 with pytest.raises(ValueError):module().extract(y,0,20)

def test_exact_control_checks_dtype_and_small_changes():
 m=module();x=np.ones(3,np.float32)
 with pytest.raises(ValueError):m.identity(x,x.astype(np.float64),'dtype')
 with pytest.raises(ValueError):m.identity(x,np.nextafter(x,np.float32(2)),'changed')

def test_prefix_scores_cover_each_position_and_head():
 m=module();ref=np.ones((20,16384));control=ref*1.01;candidate=ref.copy();candidate[19,128]=1.006
 r=m.entry_scores(control,candidate,ref)
 assert len(r['rows'])==20 and r['checks']==2560 and not r['passed'] and r['improved']

@pytest.mark.parametrize('shape',[(19,16384),(20,16383)])
def test_incomplete_prefix_cannot_pass(shape):
 with pytest.raises(ValueError):module().entry_scores(np.ones(shape),np.ones(shape),np.ones(shape))

def test_residual_gate_does_not_accept_conditioned_projection_alone():
 m=module();x=np.ones(5,np.float32);y=x.copy()
 r=m.residual_scores(x,y,y,np.ones(5)*2.02)
 assert r['projection']['passed'] and not r['residual']['passed']

def test_host_padding_corruption_rejected():
 m=module();g=dict(hid=4,kh=1,vh=1,dim=2,taps=2,lanes=1,rows=4)
 c=dict(xn=np.ones((20,4),np.float32),convw=np.ones((2,6),np.float32),Wa=np.ones((4,1),np.float32),Wb=np.ones((4,1),np.float32),A=np.ones(1,np.float32),dtb=np.ones(1,np.float32),nw=np.ones(2,np.float32))
 class Fake:
  def prefill_delta(self,*args):
   args[-1].fill(0);args[-1][-1]=1;return 0
 with pytest.raises(ValueError,match='padding'):m.host_run(Fake(),c,g,np.ones((20,6)),np.ones((20,2)),[20])

def test_run_failure_persists_and_preserves_directory(tmp_path):
 m=module();out=tmp_path/'out'
 with pytest.raises(FileNotFoundError):m.run(SimpleNamespace(out=out,attribution=tmp_path/'missing'))
 r=json.loads((out/'report.json').read_text());assert not r['complete'] and not r['control_valid'] and not r['model_acceptance']
 with pytest.raises(FileExistsError):m.run(SimpleNamespace(out=out))

def test_host_rejects_wrong_constant_stride_before_ffi():
 m=module();g=dict(hid=4,kh=1,vh=1,dim=2,taps=2,lanes=2,rows=4)
 c=dict(xn=np.ones((20,4),np.float32),convw=np.ones((2,6),np.float32),Wa=np.ones((4,1),np.float32),Wb=np.ones((4,1),np.float32),A=np.ones(1,np.float32),dtb=np.ones(1,np.float32),nw=np.ones(2,np.float32))
 class Forbidden:
  def prefill_delta(self,*args):pytest.fail('wrong stride reached native code')
 with pytest.raises(ValueError,match='constant'):m.host_run(Forbidden(),c,g,np.ones((20,6)),np.ones((20,2)),[20])
