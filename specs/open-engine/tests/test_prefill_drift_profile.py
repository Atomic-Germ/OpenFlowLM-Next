"""Phase 20 root-cause profile: where the sequential route loses precision.

Both prefill routes execute the SAME audited capture with the same weights and
kernel set; only the schedule differs. `sequential` fails every late gate while
`hybrid` passes all of them, from the same embeddings, so the pair is an
existence proof that the model CAN be computed acceptably on this hardware.

The profile has to distinguish two very different causes, which is what these
tests pin:
  * a per-token arithmetic deficit -- the error is already there at position 0
    and does not grow with the prompt;
  * history accumulation -- the error grows with position.
Treating one as the other would send the fix to the wrong place.
"""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('drift_profile', ROOT/'utilities/prefill-drift-profile/profile.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def matrix(layers, rows, value):
    return np.full((layers, rows), value, dtype=np.float64)


def test_rel_error_uses_the_reference_scale():
    m = load()
    ref = np.array([1.0, -2.0, 0.0])
    got = np.array([1.5, -2.0, 0.0])
    # max abs diff 0.5 over max abs ref 2.0
    assert m.rel_error(got, ref) == pytest.approx(0.25)


def test_rel_error_ignores_an_all_zero_reference_row():
    """A zero row has no relative scale; it must be reported absent, not 0 or inf."""
    m = load()
    assert m.rel_error(np.zeros(3), np.zeros(3)) is None


def test_growth_classifies_a_flat_deficit_as_local():
    """Error present at P0 and no larger at the last position is per-token
    arithmetic, not accumulation -- the hybrid/sequential distinction."""
    m = load()
    rel = {"sequential": matrix(4, 8, 1e-5), "hybrid": matrix(4, 8, 1e-6)}
    verdict = m.classify(rel, positions=list(range(8)))
    assert verdict['sequential']['kind'] == 'per-token'
    assert verdict['sequential']['p0_over_median'] > 0.5


def test_growth_classifies_a_ramp_as_accumulation():
    m = load()
    base = np.geomspace(1e-7, 1e-2, 8)
    rel = {"sequential": np.tile(base, (4, 1)), "hybrid": matrix(4, 8, 1e-6)}
    verdict = m.classify(rel, positions=list(range(8)))
    assert verdict['sequential']['kind'] == 'accumulation'


def test_ratio_reports_how_much_better_the_passing_route_is():
    m = load()
    rel = {"sequential": matrix(2, 4, 3e-5), "hybrid": matrix(2, 4, 1e-6)}
    verdict = m.classify(rel, positions=list(range(4)))
    assert verdict['ratio'] == pytest.approx(30.0)


def test_first_failing_layer_is_the_shallowest_crossing():
    """4e-3 is BELOW the 5e-3 bound and must not count as a failure; 6e-3 is the
    first genuine crossing."""
    m = load()
    seq = np.array([1e-6, 1e-5, 4e-3, 6e-3, 9e-3])
    rel = {"sequential": seq.reshape(-1, 1).repeat(2, axis=1)}
    info = m.summarise_route(rel['sequential'], bound=5e-3)
    assert info['first_failing_layer'] == 3
    assert info['worst_layer'] == 4
    assert info['passing_layers'] == 3
