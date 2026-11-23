"""Behavioural checks for the fresh final numerical gate without NPU access."""
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def driver(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / 'utilities/prefill-block'))
    return importlib.import_module('fold_dedup_final')


def test_existing_output_is_never_reused(driver, tmp_path):
    with pytest.raises(FileExistsError):
        driver.numerical(tmp_path)


def test_control_failure_stops_before_candidate(driver, tmp_path, monkeypatch):
    calls = []

    def fail(arm, prompts, name):
        calls.append(arm)
        raise driver.E.EvidenceError('control unavailable')

    monkeypatch.setattr(driver.L, 'list_capture', fail)
    monkeypatch.setattr(driver.L, 'OUT', tmp_path / 'unused')
    with pytest.raises(driver.E.EvidenceError, match='control unavailable'):
        driver.numerical(tmp_path / 'fresh')
    assert calls == ['original']
    assert not (tmp_path / 'fresh/numerical-passed.json').exists()


def test_isolation_mismatch_is_rejected(driver):
    fresh = {'outputs': {'p0_logits.f32': {'sha256': 'a'}, 'p0_gen.txt': {'sha256': 'g'}},
             'runtime': {'before': {}}}
    aba = {'outputs': {'p0_logits.f32': {'sha256': 'a'}, 'p2_logits.f32': {'sha256': 'b'}}}
    with pytest.raises(driver.E.EvidenceError, match='reset/isolation mismatch'):
        driver.L.compare_isolation(fresh, aba)


def test_timing_requires_application_gate(driver, tmp_path):
    timing = importlib.import_module('fold_dedup_final_timing')
    numerical = tmp_path / 'numerical-v1'
    numerical.mkdir()
    (numerical / 'numerical-passed.json').write_text('{"passed":true}')
    with pytest.raises((FileNotFoundError, driver.E.EvidenceError)):
        timing.campaign(tmp_path)
    assert not (tmp_path / 'timing-v1').exists()


@pytest.fixture
def campaign(driver, tmp_path, monkeypatch):
    timing = importlib.import_module('fold_dedup_final_timing')
    root = tmp_path / 'campaign'
    (root / 'numerical-v1').mkdir(parents=True)
    timing.C.write_json(root / 'numerical-v1/numerical-passed.json', {'passed': True})
    timing.C.write_json(root / 'application-passed.json', {'passed': True})
    monkeypatch.setattr(timing, 'SCENARIOS', [('prefill', 256, 1, 2), ('decode', 32, 33, 2)])
    monkeypatch.setattr(timing.P, 'conditions', lambda: {})
    monkeypatch.setattr(timing.E, 'audit_capture', lambda _: {'valid': True, 'errors': []})
    calls = []

    def run(cli, phase, tokens, outputs, out, ctx, power, budget, **kwargs):
        assert ctx >= tokens + outputs - 1
        assert budget == ('0' if phase == 'prefill' else f'decode:{outputs - 1}')
        calls.append((phase, kwargs['dedup_input_split']))
        out.mkdir()
        record = dict(runtime={'before': {'runtime-library': 'digest'}},
                      generated_ids=[7] * outputs, prompt_wall_ms=10,
                      generated_ms_per_token=1 if phase == 'decode' else None)
        timing.C.write_json(out / 'capture.json', record)
        return record

    monkeypatch.setattr(timing.C, 'run', run)
    return timing, root, calls, run


def test_campaign_executes_balanced_pairs_and_marks_warmups(campaign):
    timing, root, calls, _ = campaign
    timing.campaign(root)
    # Each scenario starts with an excluded candidate-first pair, then alternates.
    expected_order = [True, False, False, True, True, False]
    assert calls == [(phase, enabled) for phase in ('prefill', 'decode') for enabled in expected_order]
    selected = timing.E.read_json(root / 'timing-v1/selection.json')['samples']
    assert len(selected) == 12
    assert len([s for s in selected if s['warmup']]) == 4
    assert all(s['warmup'] == (s['rep'] < 0) for s in selected)
    completed = timing.E.read_json(root / 'timing-v1/completed.json')
    assert completed['completed'] is True
    assert completed['measured_pairs'] == 4


@pytest.mark.parametrize('failure', ['trajectory', 'runtime', 'audit'])
def test_campaign_stops_without_completion_on_invalid_pair(campaign, monkeypatch, failure):
    timing, root, calls, run = campaign

    def invalid_run(*args, **kwargs):
        record = run(*args, **kwargs)
        if len(calls) == 2:
            if failure == 'trajectory':
                record['generated_ids'] = [8]
            elif failure == 'runtime':
                record['runtime']['before'] = {'runtime-library': 'replaced'}
            else:
                monkeypatch.setattr(timing.E, 'audit_capture',
                                    lambda _: {'valid': False, 'errors': ['incomplete capture']})
        return record

    monkeypatch.setattr(timing.C, 'run', invalid_run)
    with pytest.raises(timing.E.EvidenceError):
        timing.campaign(root)
    assert len(calls) == 2
    assert not (root / 'timing-v1/completed.json').exists()
    assert timing.E.read_json(root / 'timing-v1/stopped.json')['completed'] is False


def test_fixed_bootstrap_and_pair_direction(driver):
    report = importlib.import_module('fold_dedup_final_report')
    result = report.paired([100.0]*14, [98.0]*14, bootstrap=True)
    assert result['median_reduction_percent'] == 2.0
    assert result['median_absolute_difference_ms'] == 2.0
    assert result['improving_pairs'] == 14
    assert result['bootstrap']['ci95_percent'] == [2.0, 2.0]
    assert result['bootstrap']['seed'] == 26026
    assert result['bootstrap']['resamples'] == 100000
    assert report.paired([100.0], [101.0])['median_reduction_percent'] == -1.0


def test_primary_rejects_incomplete_pairs(driver):
    report = importlib.import_module('fold_dedup_final_report')
    with pytest.raises(driver.E.EvidenceError, match='exactly 14'):
        report.paired([100.0]*13, [98.0]*13, bootstrap=True)
