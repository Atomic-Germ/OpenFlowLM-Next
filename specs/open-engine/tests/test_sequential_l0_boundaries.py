"""Boundary interpretation requires an exact, audited causal control."""
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('l0_boundaries',ROOT/'utilities/prefill-sequential-drift/l0_boundaries.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def valid():
    identity=dict(passed=True,positions=243,matching_positions=243,first_mismatch=None)
    r=dict(complete=True,model_acceptance=False,control_valid=True,boundary_interpretation_allowed=True,identity=identity)
    a=dict(complete=True,model_acceptance=False,control_valid=True,identity=identity,report_sha256='sha',artifacts=729)
    return r,a


def test_accepts_complete_audited_control():
    load().require_control(*valid(),'sha')


@pytest.mark.parametrize('field',['control_valid','complete','boundary_interpretation_allowed','model_acceptance'])
def test_rejects_incomplete_or_promoted_control(field):
    r,a=valid();r[field]=not r[field]
    with pytest.raises(ValueError):load().require_control(r,a,'sha')


def test_rejects_unbound_or_incomplete_audit():
    for field,value in [('report_sha256','changed'),('artifacts',728),('complete',False)]:
        r,a=valid();a[field]=value
        with pytest.raises(ValueError):load().require_control(r,a,'sha')


def test_summary_preserves_failed_head_gate_and_full_coverage():
    m=load()
    rows=[dict(token=p,passed=True,recurrence=dict(head_diagnostics_passed=True)) for p in range(243)]
    rows[37]['recurrence']['head_diagnostics_passed']=False
    r=m.summarize_delta(rows)
    assert r['passed']==242 and r['first_failure']==37
    with pytest.raises(ValueError):m.summarize_delta(rows[:-1])
    rows[38]['token']=37
    with pytest.raises(ValueError):m.summarize_delta(rows)
