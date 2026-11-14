"""Isolated down replay must preserve exact controls and independent residual gates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load(name):
    s=importlib.util.spec_from_file_location('down_replay_'+name,ROOT/'utilities/prefill-down-replay'/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_geometry_and_environment_are_pinned(monkeypatch):
    monkeypatch.setenv('GQP_K','7');monkeypatch.setenv('GQP_BREUSE','1');monkeypatch.setenv('GQP_INJECT','bad')
    e=load('build').environment()
    assert (e['GQP_N'],e['GQP_K'],e['GQP_T'],e['GQP_BREUSE'])==('5120','17408','256','0')
    assert 'GQP_INJECT' not in e


def test_existing_build_is_not_overwritten(tmp_path):
    (tmp_path/'build.json').write_text('keep')
    with pytest.raises(FileExistsError):load('build').build(SimpleNamespace(out=tmp_path))
    assert (tmp_path/'build.json').read_text()=='keep'


def test_projection_pass_does_not_hide_propagated_residual_failure():
    m=load('check');ref=np.ones(4);control=ref+0.01;candidate=ref.copy()
    result=m.scores(control,candidate,ref,np.ones(4),np.ones(4)*1.8)
    assert result['projection_passed'] and result['improved'] and not result['residual']['passed']


def test_equal_accuracy_is_not_improvement():
    m=load('check');x=np.ones(4)
    assert not m.scores(x,x,x,x,2*x)['improved']


@pytest.mark.parametrize('kind',['shape','nan'])
def test_invalid_projection_or_residual_rejected(kind):
    m=load('check');x=np.ones(4);candidate=x.copy()
    if kind=='shape':candidate=candidate[:3]
    else:candidate[0]=np.nan
    with pytest.raises(ValueError):m.scores(x,candidate,x,x,x)


def test_single_row_placement_and_inactive_column_gate():
    m=load('check');x=np.array([[1.,2.]],np.float32)
    y=m.prefix.place(x,129).T
    np.testing.assert_array_equal(m.prefix.extract(y,129,1),x)
    y[0,0]=1
    with pytest.raises(ValueError):m.prefix.extract(y,129,1)


def test_mismatched_individual_control_cannot_hide_in_sum():
    m=load('check');a=np.ones(4,np.float32);b=2*a
    with pytest.raises(ValueError):m.paired.require_control(a+1,b-1,a,b)


def test_failed_prepare_preserves_failure_and_refuses_overwrite(tmp_path):
    m=load('check');out=tmp_path/'out'
    with pytest.raises(FileNotFoundError):m.prepare(SimpleNamespace(out=out,attribution=tmp_path/'missing'))
    assert not json.loads((out/'preparation.json').read_text())['complete']
    with pytest.raises(FileExistsError):m.prepare(SimpleNamespace(out=out))


def test_reference_writer_uses_explicit_fp64_for_fp32_oracle(tmp_path):
    m=load('check');x=np.array([1.25,-3.5],np.float32);p=tmp_path/'reference.f64'
    m.write_reference(p,x)
    assert p.stat().st_size==16
    np.testing.assert_array_equal(np.fromfile(p,'<f8'),x.astype(np.float64))
    with pytest.raises(ValueError):m.write_reference(tmp_path/'bad.f64',np.array([np.nan]))
