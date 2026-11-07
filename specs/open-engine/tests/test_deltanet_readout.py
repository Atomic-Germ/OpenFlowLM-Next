"""Readout diagnostics must separate state error without replacing the oracle."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'utilities'))
import deltanet_readout as mod
from test_deltanet_state import sequence, encode, geometry, bundle


def test_exact_readout_orientation_and_scale():
    s = np.zeros((1, 128, 128)); q = np.zeros((1, 128))
    s[0, 2, 7] = 3; q[0, 2] = 2
    got = mod.readout(s, q)
    assert got[0, 7] == 6 / np.sqrt(128)
    assert np.count_nonzero(got) == 1


def test_pair_four_terms_restore_low_low_product():
    a = np.array([1.00390625], dtype=np.float32)
    h, l = mod.split_pair(a)
    assert h[0] == 1 and l[0] == 2**-8
    s = np.zeros((1, 128, 128)); q = np.zeros((1, 128))
    s[0, 0, 0] = q[0, 0] = a[0]
    exact = mod.pair_dot(s, q, include_low_low=True)
    three = mod.pair_dot(s, q, include_low_low=False)
    assert exact[0, 0] == float(a[0])**2
    assert exact[0, 0] - three[0, 0] == 2**-16


def test_output_error_and_state_error_are_separated():
    caps = sequence()
    caps[1]['state'][0, 0, 1] += .1
    rows = mod.diagnose_sequence(caps)
    row = rows[2]  # token 1, head 0
    assert row['primary_output']['passed']
    assert not row['captured_state_readout']['passed']
    assert not row['state_effect']['passed']
    # Captured state must never feed the following primary reference.
    assert rows[4]['primary_output']['maxabs'] == 0
    caps = sequence(); caps[1]['output'][0, 1] += .1
    row = mod.diagnose_sequence(caps)[2]
    assert row['state_effect']['maxabs'] == 0
    assert not row['primary_output']['passed']
    assert not row['captured_state_readout']['passed']


def test_cancellation_is_measured_without_relaxing_acceptance():
    s = np.zeros((1, 128, 128)); q = np.zeros((1, 128))
    s[0, :2, 0] = [1, -1 + 2**-20]; q[0, :2] = 1
    stats = mod.cancellation(s, q)
    assert stats[0] > 1e6
    assert mod.cancellation(np.zeros_like(s), q) == [None]


@pytest.mark.parametrize('bad', ['nan', 'shape', 'empty'])
def test_readout_rejects_invalid_inputs(bad):
    s = np.zeros((1, 128, 128)); q = np.zeros((1, 128))
    if bad == 'nan': s[0, 0, 0] = np.nan
    if bad == 'shape': q = q[:, :-1]
    if bad == 'empty': s = s[:0]; q = q[:0]
    with pytest.raises(ValueError): mod.readout(s, q)


def test_bundle_keeps_primary_results_and_complete_coverage(bundle):
    path, _ = bundle
    result = mod.check(path, path, path)
    assert result['passed'] and result['primary_whole_passed']
    assert len(result['results']) == 18
    assert len(result['capture_sha256']) == 19
    assert result['diagnostic_only']


def test_missing_capture_is_not_silently_skipped(bundle):
    path, _ = bundle
    (path / 'y_state2_t2.bin').unlink()
    with pytest.raises(OSError): mod.check(path, path, path)


def test_head_failure_keeps_nonzero_status_even_if_whole_passes(bundle, geometry):
    path, _ = bundle
    spec, offsets = geometry
    caps = sequence()
    for t, cap in enumerate(caps):
        cap['records'][1, 257] *= 1e-6
        cap['state'][1] *= 1e-6; cap['output'][1] *= 1e-6
        if t == 2: cap['output'][1, 1] *= 1.001
        act, state = encode(cap, spec, offsets)
        suffix = '' if t == 0 else f'_t{t}'
        (path / f'y_act2{suffix}.bin').write_bytes(act)
        (path / f'y_state2{suffix}.bin').write_bytes(state)
    result = mod.check(path, path, path)
    assert result['primary_whole_passed'] and not result['passed']
    assert any(not x['primary_output']['passed'] for x in result['results'])


def test_representation_estimates_never_replace_primary_acceptance(bundle, monkeypatch):
    path, _ = bundle
    monkeypatch.setattr(mod, 'pair_dot', lambda s, q, **kw: np.zeros_like(q))
    result = mod.check(path, path, path)
    assert result['passed']
    assert not result['results'][0]['representation_estimates']['three_term']['to_exact_readout']['passed']


def test_changed_capture_after_baseline_validation_is_refused(bundle, monkeypatch):
    path, _ = bundle
    original = mod.ds.check
    def changed(*args):
        result = original(*args)
        p = path / 'y_act0.bin'
        a = bytearray(p.read_bytes()); a[-1] ^= 1; p.write_bytes(a)
        return result
    monkeypatch.setattr(mod.ds, 'check', changed)
    with pytest.raises(ValueError, match='changed'): mod.check(path, path, path)


def test_cli_keeps_complete_json_and_failure_status(monkeypatch, capsys):
    import json
    monkeypatch.setattr(sys, 'argv', ['deltanet_readout', '--model-dir', '.', '--kernel-dir', '.', '--out', '.'])
    expected = dict(passed=False, primary_whole_passed=True, results=[{'head': 1}])
    monkeypatch.setattr(mod, 'check', lambda *args: expected)
    assert mod.main() == 1
    assert json.loads(capsys.readouterr().out) == expected


def test_summary_counts_both_sources_and_keeps_all_heads():
    caps = sequence(); caps[1]['state'][0, 0, 1] += .1
    rows = mod.diagnose_sequence(caps)
    summary = mod.summarize(rows)
    assert summary['heads_checked'] == 6
    assert summary['primary_output_failures'] == 0
    assert summary['primary_state_failures'] == 1
    assert summary['captured_state_readout_failures'] == 1
    assert summary['state_effect_failures'] == 1


def test_summary_includes_head_failure_even_for_small_absolute_error():
    caps = sequence()
    for c in caps:
        c['records'][1, 257] *= 1e-6
        c['state'][1] *= 1e-6; c['output'][1] *= 1e-6
    caps[1]['output'][1, 1] *= 1.001
    summary = mod.summarize(mod.diagnose_sequence(caps))
    assert summary['primary_output_failures'] == 1
    assert summary['primary_state_failures'] == 0
