"""The long-context verdict must tell system drift from a tail outlier.

A real tail regression hides inside a healthy-looking mean: at 513 tokens the
median defect is 1.1e-6 while one position sits at 8.9e-5. Averaging those
together reports "fine" for a defect that is 60x the body. These tests pin the
separation, and pin that a body which IS drifting is not excused as a tail.
"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('long_context', ROOT/'utilities/prefill-long-context/check.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_correlation_is_pearson_over_the_real_vocabulary():
    m = load()
    a = np.array([1.0, 2.0, 3.0, 1000.0])
    assert m.correlation(a, a, 3) == pytest.approx(1.0)
    flipped = np.array([3.0, 2.0, 1.0, 1000.0])
    assert m.correlation(flipped, a, 3) == pytest.approx(-1.0)
    # the padding past real_vocab must not affect the score
    assert m.correlation(a, a * 5.0, 3) == pytest.approx(1.0)


def test_degenerate_or_non_finite_vectors_are_refused():
    m = load()
    assert m.correlation(np.ones(4), np.ones(4), 4) is None
    bad = np.array([1.0, np.nan, 2.0])
    with pytest.raises(ValueError):
        m.correlation(bad, np.ones(3), 3)


def test_a_flat_body_with_one_outlier_is_a_tail_not_drift():
    m = load()
    corr = {p: 1.0 - 1.1e-6 for p in range(512)}
    corr[494] = 1.0 - 8.9e-5          # 80x the body, one position
    out = m.summarise(corr)
    assert out['body_stable'] is True
    assert out['verdict'] == 'tail-outliers'
    assert out['worst'][0]['position'] == 494


def test_a_rising_body_is_drift_and_is_not_excused_as_a_tail():
    m = load()
    corr = {}
    for p in range(512):
        # every position gets worse as the prompt grows
        corr[p] = 1.0 - (1e-6 + 1e-6 * p)
    out = m.summarise(corr)
    assert out['body_stable'] is False
    assert out['verdict'] == 'system-drift'


def test_a_clean_run_reports_clean():
    m = load()
    out = m.summarise({p: 1.0 - 1e-6 for p in range(512)})
    assert out['verdict'] == 'clean'
    assert out['failing'] == []
    assert out['tail_ratio'] == pytest.approx(1.0)


def test_failing_positions_use_the_project_correlation_gate():
    m = load()
    corr = {p: 1.0 - 1e-6 for p in range(512)}
    corr[300] = 0.9998                # below the 0.9999 bound
    out = m.summarise(corr)
    assert out['failing'] == [300]


def test_blocks_are_counted_at_the_compiled_width():
    m = load()
    assert m.block_of(0) == 0 and m.block_of(255) == 0
    assert m.block_of(256) == 1 and m.block_of(511) == 1
    out = m.summarise({p: 1.0 - 1e-6 for p in range(513)})
    assert out['blocks'] == 3


def test_every_worst_position_in_a_later_block_is_reported():
    """The localisation that matters: outliers appearing only once the carried
    state has crossed a block boundary."""
    m = load()
    corr = {p: 1.0 - 1e-6 for p in range(256)}
    corr.update({p: 1.0 - 1e-6 for p in range(256, 512)})
    corr[494] = 1.0 - 8.9e-5
    assert m.summarise(corr)['tail_is_in_later_blocks'] is True
