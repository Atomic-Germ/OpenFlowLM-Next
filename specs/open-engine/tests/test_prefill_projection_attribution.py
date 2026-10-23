"""CPU rounding hypotheses must preserve head scales, causality and signed errors."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
def load():
 s=importlib.util.spec_from_file_location('projection_attribution',ROOT/'utilities/prefill-projection-attribution/check.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_prediction_separates_paired_input_and_weight_rounding():
 m=load();x=np.array([[1.001234,-.123456]],np.float32);w=np.array([[.12513,.78137],[-.20231,.3111]],np.float32)
 r=m.predict(x,w)
 hi=m.round_bf16(x);lo=m.round_bf16(x-hi)
 np.testing.assert_allclose(r['exact'],x.astype('f8')@w.astype('f8').T,rtol=0,atol=0)
 np.testing.assert_allclose(r['paired'],(hi.astype('f8')+lo)@w.astype('f8').T,rtol=0,atol=0)
 np.testing.assert_allclose(r['rounded'],(hi.astype('f8')+lo)@m.round_bf16(w).astype('f8').T,rtol=0,atol=0)
 assert np.max(abs(r['paired']-r['rounded']))>1e-5

@pytest.mark.parametrize('bad',['shape','nan','empty'])
def test_prediction_rejects_invalid_inputs(bad):
 x=np.ones((2,3));w=np.ones((4,3))
 if bad=='shape':w=np.ones((4,2))
 elif bad=='nan':x[0,0]=np.nan
 else:x=x[:0]
 with pytest.raises(ValueError):load().predict(x,w)

def test_small_head_error_is_not_hidden_by_large_other_head():
 m=load();exact=np.array([[1000.,1000.,.1,.1]]);got=exact.copy();got[0,2]+=.01
 rows=m.head_scores(got,dict(exact=exact,paired=exact,rounded=got),2)
 assert len(rows)==2 and rows[0]['native']['passed'] and not rows[1]['native']['passed']
 assert rows[1]['prediction_gap']['maxrel']==0 and rows[1]['remaining_error_ratio']==0

@pytest.mark.parametrize('bad',[0,3])
def test_head_partition_requires_exact_coverage(bad):
 a=np.ones((2,4))
 with pytest.raises(ValueError):load().head_scores(a,dict(exact=a,paired=a,rounded=a),bad)

def test_zero_error_diagnostic_is_json_finite():
 a=np.zeros((1,2));r=load().head_scores(a,dict(exact=a,paired=a,rounded=a),1)
 assert r[0]['remaining_error_ratio']==0
 json.dumps(r,allow_nan=False)

def test_causal_substitutions_reset_and_exclude_future(monkeypatch):
 m=load();calls=[]
 def middle(q,z,x,*args,**kw):
  conv,state=args[-2:];assert not np.any(conv) and not np.any(state)
  calls.append(len(q));conv[:]=99;state[:]=99
  return np.cumsum(q[:,:1],axis=0)
 monkeypatch.setattr(m.prefix.rb,'deltanet_block',middle)
 spec=SimpleNamespace(conv_kernel=2,lin_qkv_dim=3,lin_value_heads=1,lin_value_dim=1,lin_key_heads=1,hidden=2,norm_eps=1e-6)
 q=np.arange(12,dtype=float).reshape(4,3);z=np.ones((4,1));x=np.ones((4,2))
 a=m.causal_middle(spec,[],q,z,x,1);q[2:]=999
 b=m.causal_middle(spec,[],q,z,x,1)
 assert calls==[2,2];np.testing.assert_array_equal(a,b)

def test_signed_decomposition_keeps_interaction_and_cancellation():
 m=load();ref=np.array([10.,2.]);native=np.array([11.,2.])
 variants={k:ref+d for k,d in dict(joint=1.,qkv=2.,z=-1.,host=3.,out=2.5,projection=2.).items()}
 metrics,v=m.decompose(native,ref,variants)
 np.testing.assert_allclose(sum(v.values()),native-ref)
 assert np.any(v['interaction']) and metrics['closure']['passed']
 assert not metrics['total']['passed']

def test_failed_run_preserves_failure_and_rejects_overwrite(tmp_path):
 m=load();out=tmp_path/'out'
 with pytest.raises(FileNotFoundError):m.run(SimpleNamespace(out=out,run=tmp_path/'missing'))
 assert not json.loads((out/'report.json').read_text())['complete']
 with pytest.raises(FileExistsError):m.run(SimpleNamespace(out=out))
