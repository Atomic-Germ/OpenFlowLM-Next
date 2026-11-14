"""Selected causal rows must preserve native controls and separate acceptance gates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('ffn_sweep', ROOT / 'utilities/prefill-ffn-replay/sweep.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_slots_preserve_original_position_and_other_half():
    m = load()
    assert m.slots(38) == (38, 166)
    assert m.slots(80) == (80, 208)
    assert m.slots(200) == (200, 72)


@pytest.mark.parametrize('bad', [-1, 256, True, 1.5])
def test_slots_reject_invalid_position(bad):
    with pytest.raises(ValueError):
        load().slots(bad)


def test_manifest_selects_linear_and_attention_pool_offsets(monkeypatch):
    m = load()
    cases = [dict(tag=f'L{i}_up_gate', n=2*m.F, k=m.H, storage='pool', marker=i) for i in (0, 3)]
    monkeypatch.setattr(m.proj, 'cases', lambda _: cases)
    assert m.operation({}, 60, 'up_gate')['marker'] == 0
    assert m.operation({}, 35, 'up_gate')['marker'] == 3
    cases[0]['storage'] = 'other'
    with pytest.raises(ValueError):
        m.operation({}, 60, 'up_gate')


def rows(m):
    return [dict(layer=l, position=p, slot=s, control_exact=True, slot_exact=True,
                 projections_passed=True, ffn_passed=True, output_passed=True)
            for l, p in m.TARGETS for s in m.slots(p)]


def test_global_failure_is_not_hidden_by_local_pass():
    m = load(); cases = rows(m); cases[0]['output_passed'] = False
    r = m.summarize(cases)
    assert r['control_valid'] and r['local_passed'] and not r['passed']
    assert not r['model_acceptance']


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'extra', 'control', 'slot'])
def test_incomplete_or_invalid_controls_refuse_summary(problem):
    m = load(); cases = rows(m)
    if problem == 'missing': cases.pop()
    elif problem == 'duplicate': cases[-1] = cases[0].copy()
    elif problem == 'extra': cases.append(cases[0].copy())
    elif problem == 'control': cases[0]['control_exact'] = False
    else: cases[0]['slot_exact'] = False
    with pytest.raises(ValueError): m.summarize(cases)


def test_failed_run_invalidates_stale_success(tmp_path):
    m = load(); report = tmp_path / 'report.json'; report.write_text('{"passed":true}')
    with pytest.raises(FileNotFoundError): m.run(SimpleNamespace(out=tmp_path))
    r = json.loads(report.read_text())
    assert not r['passed'] and not r['control_valid'] and not r['model_acceptance']


def test_prepare_refuses_existing_directory_without_mutation(tmp_path):
    marker = tmp_path / 'audit.json'; marker.write_text('keep')
    with pytest.raises(FileExistsError): load().prepare(SimpleNamespace(out=tmp_path))
    assert marker.read_text() == 'keep'
