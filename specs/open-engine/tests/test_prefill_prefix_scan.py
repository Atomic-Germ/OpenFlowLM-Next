"""Causal prefix scans distinguish inherited drift from an ideal-history diagnostic."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
def load():
 s=importlib.util.spec_from_file_location('prefix_scan',ROOT/'utilities/prefill-prefix-scan/check.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_replay_carries_every_prefix_row_and_resets_between_routes(monkeypatch):
 m=load();calls=[]
 def attn(model,spec,layer,x,k,v,pos):
  calls.append(pos);k=np.concatenate((k,x.reshape(1,1,1)))
  return np.array([k.sum()]),k,k.copy()
 monkeypatch.setattr(m.base.oracle,'attn_decode',attn)
 monkeypatch.setattr(m.base.oracle,'ffn',lambda *args:np.zeros(1))
 spec=SimpleNamespace(layer_types=['full_attention'],conv_kernel=2,lin_qkv_dim=1,lin_value_heads=1,lin_value_dim=1,num_kv_heads=1,head_dim=1)
 xs=np.array([[1],[2],[3]],np.float32);original=xs.copy()
 r=m.replay(None,spec,0,xs,(1,2));assert calls==[0,1,2]
 assert r[1][0]==5 and r[2][0]==9
 assert m.replay(None,spec,0,xs,(2,))[2][0]==9
 np.testing.assert_array_equal(xs,original)

@pytest.mark.parametrize('positions',[(),(2,2),(-1,),(3,),(1.5,)])
def test_replay_refuses_invalid_prefix_coverage(positions):
 with pytest.raises(ValueError):load().replay(None,None,0,np.ones((3,1)),positions)

def test_common_scale_vectors_close_and_do_not_hide_cancellation():
 m=load();ref=np.array([10.,2.]);ideal=np.array([12.,3.]);native=np.array([11.,2.5])
 score,v=m.decompose(native,ideal,ref)
 np.testing.assert_allclose(v['propagated']+v['ideal_history_gap'],v['total'])
 assert score['propagated']['maxrel']==pytest.approx(.2)
 assert score['ideal_history_gap']['maxrel']==pytest.approx(.1)
 assert score['total']['maxrel']==pytest.approx(.1)
 assert not score['total']['passed']

@pytest.mark.parametrize('bad',['shape','nan'])
def test_vectors_require_complete_finite_input(bad):
 x=np.ones(3);y=x.copy()
 if bad=='shape':y=np.ones(2)
 else:y[0]=np.nan
 with pytest.raises(ValueError):load().decompose(x,y,x)

def test_summary_rejects_missing_or_duplicate_cases():
 m=load();rows=[dict(layer=l,position=p,route=r,metrics={k:dict(maxrel=.001,passed=True) for k in ('total','propagated','ideal_history_gap')}) for l in range(2) for p in (1,2) for r in m.ROUTES]
 assert m.summarize(rows,2,(1,2))['hybrid']['passed']==4
 with pytest.raises(ValueError):m.summarize(rows[:-1],2,(1,2))
 with pytest.raises(ValueError):m.summarize(rows+[rows[0]],2,(1,2))

def test_failed_scan_preserves_explicit_failure(tmp_path):
 m=load();out=tmp_path/'scan'
 with pytest.raises(FileNotFoundError):m.run(SimpleNamespace(out=out,run=tmp_path/'missing'))
 r=json.loads((out/'report.json').read_text());assert not r['complete'] and not r['model_acceptance']
 with pytest.raises(FileExistsError):m.run(SimpleNamespace(out=out))

def test_linear_replay_carries_conv_and_recurrence_for_omitted_targets(monkeypatch):
 m=load();calls=[];ffn_calls=[]
 def linear(model,spec,layer,x,conv,state):
  calls.append((float(conv.sum()),float(state.sum())))
  conv=conv+x;state=state+conv.sum()
  return np.array([state.sum()]),conv,state
 monkeypatch.setattr(m.base.oracle,'linear_decode',linear)
 monkeypatch.setattr(m.base.oracle,'ffn',lambda *args:ffn_calls.append(1) or np.zeros(1))
 spec=SimpleNamespace(layer_types=['linear_attention'],conv_kernel=2,lin_qkv_dim=1,lin_value_heads=1,lin_value_dim=1,num_kv_heads=1,head_dim=1)
 r=m.replay(None,spec,0,np.array([[1.],[2.],[3.]]),(2,))
 assert calls==[(0.,0.),(1.,1.),(3.,4.)] and len(ffn_calls)==1
 assert r[2][0]==13

def test_zero_reference_failure_remains_json_serializable():
 m=load();scores,_=m.decompose(np.ones(2),np.zeros(2),np.zeros(2))
 assert not scores['total']['passed'] and scores['total']['maxrel'] is None
 assert scores['propagated']['passed']
 json.dumps(scores,allow_nan=False)
