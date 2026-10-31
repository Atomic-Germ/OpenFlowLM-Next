"""Rounding attribution must preserve packing slices, signed errors and provenance."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('combined_attribution',ROOT/'utilities/prefill-combined-attribution/check.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_signed_chain_retains_cancellation_and_closes():
    m=load();ref=np.array([10.,2.]);points={'propagated':ref+1,'rounding':ref-2,'remainder':ref+.25}
    report,vectors=m.decompose(ref,points)
    np.testing.assert_array_equal(sum(vectors.values()),points['remainder']-ref)
    assert report['closure']['passed'] and np.any(vectors['rounding']<0)
    assert not report['total']['passed']


@pytest.mark.parametrize('bad',['empty','shape','nan'])
def test_signed_chain_refuses_invalid_vectors(bad):
    ref=np.ones(2);points={'x':ref.copy()}
    if bad=='empty':points={}
    elif bad=='shape':points['x']=np.ones((1,2))
    else:points['x'][0]=np.nan
    with pytest.raises(ValueError):load().decompose(ref,points)


def test_manifest_slice_selects_gate_half_not_q(monkeypatch):
    m=load();source=np.arange(64*256,dtype=np.float32).reshape(64,256)
    model=SimpleNamespace(chunk_bytes_of=lambda name:5120,raw=lambda name:bytes(2*5120),
                          matmul_w=lambda name,n,k:source)
    monkeypatch.setattr(m.rounding.proj,'apply_op',lambda op,model,l,pool:None)
    monkeypatch.setattr(m.rounding.proj,'dequant_pool',lambda pool,n,k,rs:source[32:])
    op=dict(op='std_perm',tensor='layer.{l}.qgate',nch=1,chunk0=1,in_dim=256,dst=123)
    w,e=m.packed_weights(model,63,op)
    np.testing.assert_array_equal(w,source[32:]);assert e['source_row_start']==32 and e['source_exact']
    monkeypatch.setattr(m.rounding.proj,'dequant_pool',lambda pool,n,k,rs:source[:32])
    with pytest.raises(ValueError,match='source'):m.packed_weights(model,63,op)


def test_causal_attention_counterfactual_excludes_future_and_stores_fp32(monkeypatch):
    m=load();seen=[]
    def oracle(values,*args):
        seen.append(values)
        return np.cumsum(values['v'],axis=0),None
    monkeypatch.setattr(m.capture,'attention_reference',oracle)
    values={k:np.array([[.1000000001],[2.],[300.]]) for k in ('q','k','v','gate')}
    spec=SimpleNamespace(num_heads=1,num_kv_heads=1,head_dim=1,norm_eps=1e-6,rope_inv_freq=lambda:[1.])
    a=m.causal_attention(values,1,np.ones(1),np.ones(1),spec)
    values['v'][2]=1e9;b=m.causal_attention(values,1,np.ones(1),np.ones(1),spec)
    np.testing.assert_array_equal(a,b);assert len(seen[0]['v'])==2
    assert seen[0]['v'][0,0]==np.float64(np.float32(.1000000001))


def test_native_error_not_confused_with_prediction_fit():
    m=load();ref=np.array([[1000.,1000.,.1,.1]]);native=ref.copy();native[0,2]+=.01
    rows=m.rounding.head_scores(native,dict(exact=ref,paired=ref,rounded=native),2)
    assert not rows[1]['native']['passed'] and rows[1]['prediction_gap']['passed']


def test_validation_rejects_changed_captures_before_math(tmp_path):
    m=load();run=tmp_path/'run';run.mkdir()
    (run/'hybrid_L63_P1_h.bin').write_bytes(b'wrong')
    audit=tmp_path/'audit.json'
    audit.write_text(json.dumps(dict(capture_valid=True,captures={'hybrid_L63_P1_h.bin':'0'*64},evidence={})))
    with pytest.raises(ValueError,match='provenance'):m.validate(run,audit)


def test_failure_is_recorded_and_previous_output_refused(tmp_path):
    m=load();out=tmp_path/'out'
    with pytest.raises(FileNotFoundError):m.run(SimpleNamespace(out=out,run=tmp_path/'missing',audit=tmp_path/'audit'))
    r=json.loads((out/'report.json').read_text());assert not r['complete'] and not r['model_acceptance']
    with pytest.raises(FileExistsError):m.run(SimpleNamespace(out=out))
