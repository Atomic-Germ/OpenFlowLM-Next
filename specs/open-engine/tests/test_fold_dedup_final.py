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


def test_fixed_campaign_counts(driver):
    timing = importlib.import_module('fold_dedup_final_timing')
    assert timing.SCENARIOS == [('prefill', 256, 1, 14), ('prefill', 512, 1, 6),
                                ('prefill', 1024, 1, 6), ('decode', 32, 33, 6)]


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
