"""Portable behavioural tests for the isolated kernel comparison driver."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('q4_gemm_compare', ROOT / 'utilities/q4-gemm/compare.py')
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_timings_require_complete_successful_records():
    log = 'device ready\nrun k [3 bufs] -> state 4 (2.500 ms)\nrun k [3 bufs] -> state 4 (1.250 ms)\n'
    assert TOOL.timings(log, 2) == [2.5, 1.25]


@pytest.mark.parametrize('log,count', [
    ('', 1),
    ('run k [3 bufs] -> state 8 (2.000 ms)\n', 1),
    ('run k [3 bufs] -> state 4 (nan ms)\n', 1),
    ('run k [3 bufs] -> state 4 (inf ms)\n', 1),
    ('run k [3 bufs] -> state 4 (0.000 ms)\n', 1),
    ('run k [3 bufs] -> state 4 (-1.000 ms)\n', 1),
    ('run k [3 bufs] -> state 4 (1.000 ms)\n', 2),
])
def test_timings_reject_failures_and_missing_dispatches(log, count):
    with pytest.raises(ValueError):
        TOOL.timings(log, count)


def test_output_gate_checks_full_shape_finiteness_and_existing_tolerance(tmp_path):
    got, ref = tmp_path / 'got', tmp_path / 'ref'
    np.array([1, 2, 3, 4], '<f4').tofile(ref)
    np.array([1, 2, 3, 4.001], '<f4').tofile(got)
    metrics = TOOL.output_metrics(got, ref, 4)
    assert 0 < metrics['rel_fro'] < 5e-3
    assert metrics['elements'] == 4
    with pytest.raises(ValueError, match='incomplete'):
        TOOL.output_metrics(got, ref, 5)
    np.array([1, 2, 3, 5], '<f4').tofile(got)
    with pytest.raises(ValueError, match='numerical gate'):
        TOOL.output_metrics(got, ref, 4)
    np.array([1, 2, 3, np.nan], '<f4').tofile(got)
    with pytest.raises(ValueError, match='nonfinite output'):
        TOOL.output_metrics(got, None, 4)


def test_zero_reference_does_not_hide_an_error(tmp_path):
    got, ref = tmp_path / 'got', tmp_path / 'ref'
    np.zeros(4, '<f4').tofile(ref)
    np.zeros(4, '<f4').tofile(got)
    assert TOOL.output_metrics(got, ref, 4)['rel_fro'] == 0
    np.ones(4, '<f4').tofile(got)
    with pytest.raises(ValueError, match='numerical gate'):
        TOOL.output_metrics(got, ref, 4)


def test_summary_uses_pair_medians_and_counts_wins():
    pairs = [{'before': {'median_ms': b}, 'after': {'median_ms': a}}
             for b, a in [(10, 8), (11, 9), (9, 7), (12, 13), (10, 8)]]
    result = TOOL.summary(pairs)
    assert result['before'] == dict(median_ms=10, min_ms=9, max_ms=12)
    assert result['after'] == dict(median_ms=8, min_ms=7, max_ms=13)
    assert result['change_percent'] == pytest.approx(-20)
    assert result['paired_wins'] == 4


def test_config_preserves_buffer_sizes_and_dispatch_count(tmp_path):
    cfg = TOOL.config(tmp_path / 'kernel', tmp_path / 'w', tmp_path / 'x',
                      tmp_path / 'y', 14336, 5120, 256, 11)
    assert 'buf w 45875200 ' in cfg
    assert 'buf x 2621440 ' in cfg
    assert 'buf y 14680064\n' in cfg
    assert cfg.splitlines().count('run k w x y') == 11
    with pytest.raises(ValueError, match='whitespace'):
        TOOL.config(tmp_path / 'bad path', tmp_path / 'w', tmp_path / 'x',
                    tmp_path / 'y', 256, 256, 256, 1)
