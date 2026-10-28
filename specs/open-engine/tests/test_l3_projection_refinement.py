"""Residual refinement must retain exact first-pass controls and causal history."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('l3_refinement',ROOT/'utilities/l3-projection-replay/refine.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_residual_uses_captured_quantization_and_is_exact_fp32():
    m=load();x=np.linspace(-1.3,2.1,5120,dtype='<f4')
    data=m.replay.expected_table(x)
    lo=m.residual(x,data);hi,_=m.attr.table(data,5120)
    assert lo.dtype==np.dtype('<f4') and np.any(lo)
    np.testing.assert_array_equal(hi+lo.astype(np.float64),x.astype(np.float64))
    assert np.max(abs(lo))<1e-4


def test_residual_refuses_table_for_another_input():
    m=load();x=np.ones(5120,dtype='<f4')
    with pytest.raises(ValueError):m.residual(x,m.replay.expected_table(x*2))


def test_host_composition_is_one_fp32_addition():
    m=load();hi=np.array([1.,-1.],dtype='<f4');lo=np.array([2**-24,2**-24],dtype='<f4')
    actual=m.combine(hi,lo)
    assert actual.dtype==np.dtype('<f4')
    np.testing.assert_array_equal(actual,np.array([1.,-1+2**-24],dtype='<f4'))


@pytest.mark.parametrize('fault',['shape','dtype','nan'])
def test_composition_refuses_invalid_output(fault):
    m=load();a=np.ones(4,dtype='<f4');b=a.copy()
    if fault=='shape':b=b[:-1]
    elif fault=='dtype':b=b.astype(float)
    else:b[0]=np.nan
    with pytest.raises(ValueError):m.combine(a,b)


def test_program_repeats_both_passes_and_poisons_each_buffer(tmp_path):
    m=load();lines=m.program(tmp_path,tmp_path/'source',tmp_path/'build').splitlines()
    assert lines.count('run proj w x y tab')==480
    assert lines.count(f'load y {tmp_path}/source/poison-y.bin')==480
    assert lines.count(f'load tab {tmp_path}/source/poison-tab.bin')==480
    for p in range(240):
        assert f'load x {tmp_path}/source/x{p}.bin' in lines
        assert f'load x {tmp_path}/low{p}.bin' in lines
        assert lines.index(f'dump tab {tmp_path}/hi-table{p}.bin 11520')<lines.index(f'load x {tmp_path}/low{p}.bin')


def test_f32_reader_refuses_extra_partial_float(tmp_path):
    m=load();p=tmp_path/'y.bin';p.write_bytes(np.ones(4,dtype='<f4').tobytes()+b'\x00')
    with pytest.raises(ValueError,match='size'):m.read(p,4)


def test_control_failure_prevents_interpretation():
    m=load();m.require_controls(960,480)
    with pytest.raises(ValueError):m.require_controls(959,480)
    with pytest.raises(ValueError):m.require_controls(960,479)


def test_causal_attention_uses_each_trajectory_full_prefix():
    m=load()
    from types import SimpleNamespace
    spec=SimpleNamespace(num_heads=2,num_kv_heads=1,head_dim=2,norm_eps=1e-6,rotary_dim=0,rope_inv_freq=lambda:np.array([]))
    q=np.ones((3,4));k=np.ones((3,2));v=np.array([[2.,4.],[6.,8.],[10.,12.]])
    got=m.attention(dict(q=q,gate=np.zeros_like(q),k=k,v=v),np.ones(2),np.ones(2),spec)
    expected=np.stack([np.tile(v[:p+1].mean(axis=0)/2,(2,1)) for p in range(3)])
    np.testing.assert_allclose(got,expected,atol=1e-14)
    changed=v.copy();changed[-1]=100
    other=m.attention(dict(q=q,gate=np.zeros_like(q),k=k,v=changed),np.ones(2),np.ones(2),spec)
    np.testing.assert_array_equal(other[:2],got[:2])


def test_improvement_is_separate_from_threshold_pass():
    m=load();ref=np.array([1.,2.]);old=ref+1e-4;new=ref+2e-4
    r=m.compare(old,new,ref)
    assert r['candidate']['passed'] and not r['improved']


def test_capture_inventory_rejects_extra_position(tmp_path):
    m=load()
    for name in m.names():(tmp_path/name).write_bytes(b'')
    (tmp_path/'lo-y240.bin').write_bytes(b'')
    with pytest.raises(ValueError,match='inventory'):m.captures(tmp_path,tmp_path/'source')


def test_run_refuses_partial_existing_capture(tmp_path):
    m=load();(tmp_path/'hi-table239.bin').write_bytes(b'partial')
    with pytest.raises(ValueError,match='overwrite'):m.run(SimpleNamespace(out=tmp_path))


def test_audit_preserves_numerical_failure(tmp_path,monkeypatch):
    m=load();result=dict(candidate_passed=False,artifacts={'numerics.npz':'hash'})
    r=dict(complete=True,model_acceptance=False,preparation_sha256='hash',harness='runner',
        harness_sha256='hash',harness_evidence_sha256='hash',log_sha256='hash',result=result)
    (tmp_path/'report.json').write_text(json.dumps(r))
    monkeypatch.setattr(m,'verified',lambda p:{})
    monkeypatch.setattr(m,'digest',lambda p:'hash')
    monkeypatch.setattr(m.native,'verify_harness',lambda p:'hash')
    monkeypatch.setattr(m,'evaluate',lambda *args:result)
    assert m.audit(SimpleNamespace(out=tmp_path))==1
    proof=json.loads((tmp_path/'audit.json').read_text())
    assert proof['complete'] and not proof['candidate_passed'] and not proof['model_acceptance']
    r['log_sha256']='changed';(tmp_path/'report.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='unbound'):m.audit(SimpleNamespace(out=tmp_path))
