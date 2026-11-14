"""The stage probe must name the first operation to fix, not the biggest number.

An engineer reads this to decide where to spend a kernel change, so the two ways
it can mislead are pinned here: reporting the largest error instead of the first
one in pipeline order, and treating a stage it could not observe as perfect.
"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('step_probe', ROOT/'utilities/prefill-long-context/step_probe.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_first_stage_in_pipeline_order_wins_not_the_largest():
    """qkv is worse than xn here, but xn comes first: a fix must start at xn,
    because everything downstream is conditioned on it."""
    m = load()
    stages = dict(xn=8.5e-6, qkv=2e-5, z=1e-6, o=1e-6)
    assert m.first_stage_over(stages, 4.1e-6) == 'xn'


def test_a_stage_at_or_below_the_ceiling_is_not_named():
    m = load()
    assert m.first_stage_over(dict(xn=4.0e-6, qkv=4.0e-6, z=1e-7, o=1e-7), 4.129e-6) is None


def test_unobserved_stages_are_skipped_not_treated_as_perfect():
    """A missing capture must not silently pass; the next observed offender is
    named instead, and a wholly unobserved layer reports no verdict."""
    m = load()
    assert m.first_stage_over(dict(xn=None, qkv=9e-6), 4.1e-6) == 'qkv'
    assert m.first_stage_over(dict.fromkeys(m.STAGES), 4.1e-6) is None


def test_verdict_flags_a_clean_layer_as_within_block_accuracy():
    m = load()
    out = m.verdict(dict(xn=1e-7, qkv=1e-7, z=1e-7, o=1e-7), 4.129e-6)
    assert out['within_block_route_accuracy'] is True
    assert out['first_stage_over_ceiling'] is None


def test_rel_error_uses_the_reference_scale_and_refuses_mismatch():
    m = load()
    # max |diff| = 2 over max |ref| = 1
    assert m.rel(np.array([1.0, 3.0]), np.array([1.0, 1.0])) == pytest.approx(2.0)
    # the scale is the REFERENCE's, not the observed vector's
    assert m.rel(np.array([1.0, 3.0]), np.array([1.0, 10.0])) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        m.rel(np.ones(2), np.ones(3))


def test_a_zero_reference_reports_absent_rather_than_perfect():
    m = load()
    assert m.rel(np.ones(3), np.zeros(3)) is None


def test_stage_order_matches_the_pipeline():
    """The order IS the semantics; a reorder would silently change which stage
    the tool blames."""
    m = load()
    assert m.STAGES == ('xn', 'qkv', 'z', 'o')
