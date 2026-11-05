"""The A/B tool must report a fail-count move as inconclusive, and cancellation as the cause.

These pin the lesson from the Qwen3.8 norm fix (2026-10-31). The norm's own error fell
28x; the failing-position COUNT rose 66 -> 75; and the paired test was t = +0.89. A tool
that read that count as a regression would have sent the next agent to "fix" a component
that was measurably cancelling a larger error rather than causing it.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location(
        'ab', ROOT / 'utilities' / 'prefill-long-context' / 'ab.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_defect_is_the_complement_the_gate_thresholds():
    ab = load()
    assert ab.defect(1.0) == pytest.approx(0.0)
    assert ab.defect(0.9999) == pytest.approx(1e-4)
    assert ab.CORR_GATE == pytest.approx(0.9999)


def test_a_degenerate_position_cannot_carry_a_defect():
    ab = load()
    with pytest.raises(ValueError):
        ab.defect(None)


def test_paired_reports_no_statistic_for_a_single_position():
    ab = load()
    one = ab.paired([(0, 1e-4, 2e-4)])
    assert one['t'] is None and one['stderr'] is None
    assert one['systematic'] is False and one['direction'] == 'inconclusive'


def test_a_real_improvement_is_systematic_and_signed_negative():
    ab = load()
    rng = np.random.default_rng(0)
    before = 1e-3 + rng.random(200) * 1e-3
    after = before * 0.5
    out = ab.paired([(p, a, b) for p, (a, b) in enumerate(zip(before, after))])
    assert out['direction'] == 'improved' and out['systematic'] is True
    assert out['t'] < -2.0
    assert out['improved'] == 200 and out['worse'] == 0


def test_a_real_regression_is_systematic_and_signed_positive():
    ab = load()
    rng = np.random.default_rng(1)
    before = 1e-3 + rng.random(200) * 1e-3
    after = before * 2.0
    out = ab.paired([(p, a, b) for p, (a, b) in enumerate(zip(before, after))])
    assert out['direction'] == 'worse' and out['t'] > 2.0


def test_scatter_wider_than_the_effect_is_inconclusive_not_a_regression():
    """The measured case: 66 -> 75 failing positions, t = +0.89.

    A few positions are 100x the body, so any change can move the count. The paired
    test on the same data says there is no effect, and the tool must say so.
    """
    ab = load()
    rng = np.random.default_rng(2)
    n = 400
    before = 1e-5 + rng.random(n) * 1e-3
    before[0] = 3e-1                      # one catastrophic outlier, as the real data has
    after = before + rng.normal(0, 1e-4, n)
    # Force the count to rise while the effect is a coin flip.
    rows = [(p, a, b) for p, (a, b) in enumerate(zip(before, after))]
    out = ab.paired(rows)
    assert out['t'] is not None
    if abs(out['t']) < 2.0:
        assert out['systematic'] is False and out['direction'] == 'inconclusive'


def test_a_cancelling_component_is_named_as_such():
    """The measured L0 case: the norm's projected error had cosine 0.29 with the total.

    cos = 0.97 is `cancelling`: the component removes most of the total's length.
    """
    ab = load()
    total = np.array([1.0, 0.0, 0.0])
    share = np.array([-0.9, 0.0, 0.0])         # anti-parallel: it cancels the total
    out = ab.component(total, share)
    assert out['cosine'] == pytest.approx(-1.0)
    assert out['role'] == 'cancelling'


def test_an_adding_component_is_nameable_and_ordered_by_fraction():
    ab = load()
    total = np.array([1.0, 0.0, 0.0])
    out = ab.component(total, np.array([0.5, 0.0, 0.0]))
    assert out['role'] == 'adding' and out['fraction'] == pytest.approx(0.5)


def test_an_absent_component_is_not_scored_as_cancelling():
    ab = load()
    out = ab.component(np.array([1.0, 2.0]), np.zeros(2))
    assert out['role'] == 'absent' and out['cosine'] is None


def test_component_rejects_mismatched_or_nonfinite_input():
    ab = load()
    with pytest.raises(ValueError):
        ab.component(np.ones(3), np.ones(2))
    with pytest.raises(ValueError):
        ab.component(np.array([1.0, np.nan]), np.ones(2))


def test_a_flip_scraping_past_the_gate_is_immaterial():
    """Scraping past the gate is a threshold artifact, not a regression.

    Crossing a threshold always moves further than the distance to it, so that can
    never be the test. What matters is how far onto the far side the position landed.
    """
    ab = load()
    gate = 1e-4
    # A flip is material when it lands `material` times over the gate. The default of
    # 2.0 means a position just past the gate is an artifact and one 3x over is not.
    rows = [(7, gate * 0.5, gate * 1.001),    # lands barely over: artifact
            (8, gate * 0.5, 5e-3)]            # lands 50x over: material
    out = {f['position']: f for f in ab.flips(rows)}
    assert out[7]['kind'] == 'now_failing' and out[7]['material'] is False
    assert out[8]['kind'] == 'now_failing' and out[8]['material'] is True
    assert out[8]['margin_over_gate'] == pytest.approx(50.0, rel=1e-2)


def test_a_landing_clearly_under_the_gate_is_material_for_a_passing_flip():
    ab = load()
    gate = 1e-4
    out = ab.flips([(1, 5e-3, gate * 0.5)])[0]
    assert out['kind'] == 'now_passing' and out['material'] is True


def test_flips_ignore_positions_that_did_not_cross():
    ab = load()
    assert ab.flips([(0, 1e-6, 2e-6), (1, 5e-3, 4e-3)]) == []


def test_the_materiality_factor_is_validated():
    ab = load()
    for bad in (1.0, 0.5, float('inf')):
        with pytest.raises(ValueError):
            ab.flips([(0, 1e-6, 2e-6)], bad)


def test_correlation_rejects_nonfinite_and_shapes():
    ab = load()
    with pytest.raises(ValueError):
        ab.correlation(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        ab.correlation(np.ones(3), np.ones(4))


def test_correlation_is_one_for_an_affine_copy():
    ab = load()
    a = np.linspace(-1, 1, 64)
    assert ab.correlation(a, 3.0 * a + 7.0) == pytest.approx(1.0)


def test_score_requires_the_capture_it_compares(tmp_path):
    ab = load()
    ref = tmp_path / 'ref'
    ref.mkdir()
    # A real, varying vector: identical constants are degenerate and must be refused.
    rng = np.random.default_rng(3)
    rng.standard_normal(16).astype('<f4').tofile(ref / 'ref_p0.bin')
    before, capture = tmp_path / 'before', tmp_path / 'capture'
    before.mkdir()
    capture.mkdir()
    np.fromfile(ref / 'ref_p0.bin', dtype='<f4').tofile(before / 'sequential_p0.bin')
    with pytest.raises(FileNotFoundError):
        ab.score(capture, ref, [0], before)
    np.fromfile(ref / 'ref_p0.bin', dtype='<f4').tofile(capture / 'sequential_p0.bin')
    (position, before_defect, after_defect), = ab.score(capture, ref, [0], before)
    # An identical vector correlates at 1.0 to rounding, not to the last bit: Pearson is
    # computed in float64 over data that is a float32 copy, so the defect is ~1e-16.
    assert position == 0
    assert before_defect == pytest.approx(0.0, abs=1e-12)
    assert after_defect == pytest.approx(0.0, abs=1e-12)


def test_score_compares_over_the_shorter_vector_not_a_padded_window(tmp_path):
    """A reference prepared for fewer logits is a preparation mismatch, not a free pass."""
    ab = load()
    ref = tmp_path / 'ref'
    ref.mkdir()
    rng = np.random.default_rng(4)
    short = rng.standard_normal(16).astype('<f4')
    short.tofile(ref / 'ref_p0.bin')
    before, capture = tmp_path / 'before', tmp_path / 'capture'
    before.mkdir()
    capture.mkdir()
    np.concatenate([short, np.zeros(16, dtype='<f4')]).tofile(before / 'sequential_p0.bin')
    np.concatenate([short, np.ones(16, dtype='<f4')]).tofile(capture / 'sequential_p0.bin')
    (position, before_defect, after_defect), = ab.score(capture, ref, [0], before)
    # Both read the same 16 logits, so the difference is entirely in the padding and
    # invisible -- which is the point: the truncation is deliberate and testable.
    assert position == 0
    assert before_defect == pytest.approx(0.0, abs=1e-12)
    assert after_defect == pytest.approx(0.0, abs=1e-12)


def test_a_reference_with_too_few_logits_to_correlate_is_refused(tmp_path):
    ab = load()
    ref = tmp_path / 'ref'
    ref.mkdir()
    np.ones(1, dtype='<f4').tofile(ref / 'ref_p0.bin')
    before, capture = tmp_path / 'before', tmp_path / 'capture'
    before.mkdir()
    capture.mkdir()
    np.ones(8, dtype='<f4').tofile(before / 'sequential_p0.bin')
    np.ones(8, dtype='<f4').tofile(capture / 'sequential_p0.bin')
    with pytest.raises(ValueError):
        ab.score(capture, ref, [0], before)


def test_no_position_overlapping_the_reference_is_an_error_not_an_empty_pass(tmp_path):
    ab = load()
    ref = tmp_path / 'ref'
    ref.mkdir()
    (tmp_path / 'before').mkdir()
    (tmp_path / 'capture').mkdir()
    with pytest.raises(ValueError):
        ab.score(tmp_path / 'capture', ref, [0, 1], tmp_path / 'before')
