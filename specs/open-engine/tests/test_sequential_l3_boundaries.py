"""Interpret L3 boundaries only after an exact independently audited control."""
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('l3_boundaries',ROOT/'utilities/prefill-sequential-drift/l3_boundaries.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


@pytest.mark.parametrize('fault',['none','hash','identity','permission','count'])
def test_requires_exact_bound_native_control(fault):
    m=load();identity=dict(passed=True,positions=240,matching_positions=240,first_mismatch=None)
    r=dict(complete=True,control_valid=True,boundary_interpretation_allowed=True,model_acceptance=False,identity=identity)
    a=dict(complete=True,control_valid=True,model_acceptance=False,identity=identity,artifacts=720,report_sha256='hash')
    if fault=='hash':a['report_sha256']='changed'
    if fault=='identity':a['identity']=dict(identity,matching_positions=239)
    if fault=='permission':r['boundary_interpretation_allowed']=False
    if fault=='count':a['artifacts']=719
    if fault=='none':m.require_control(r,a,'hash')
    else:
        with pytest.raises(ValueError):m.require_control(r,a,'hash')


def test_summary_requires_all_positions_heads_and_selected_ffn():
    m=load();rows=[dict(layer=3,token=p,passed=True) for p in range(240)]
    heads=[dict(token=p,head=h,passed=True) for p in range(240) for h in range(24)]
    ffn=[dict(position=p,passed=True) for p in m.TARGETS]
    assert m.summarize(rows,heads,ffn)['passed']
    rows[205]['passed']=False
    assert not m.summarize(rows,heads,ffn)['passed']
    for args in ((rows[:-1],heads,ffn),(rows,heads+[heads[0]],ffn),(rows,heads,ffn[:-1])):
        with pytest.raises(ValueError):m.summarize(*args)
