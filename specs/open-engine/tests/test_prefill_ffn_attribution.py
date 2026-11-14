"""FFN attribution must close as vectors, including nonlinear interaction."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location(
        'ffn_attribution', ROOT / 'utilities/prefill-sublayer-trace/attribute.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    res = np.array([1., 1.])
    nw = np.full(2, np.sqrt(2.))
    w = np.eye(2)
    norm = res.copy()
    h = norm / (1 + np.exp(-norm))
    c = dict(res=res, norm=norm, upgate=np.r_[norm, norm], h=h, down=h.copy())
    return c, nw, w


@pytest.mark.parametrize('stage', ['up', 'gate', 'activation', 'down', 'residual_add'])
def test_one_perturbation_is_attributed_to_its_stage(stage):
    m = load()
    c, nw, w = fixture()
    if stage in ('up', 'gate'):
        c['upgate'][0 if stage == 'up' else 2] += .25
        c['h'] = c['upgate'][:2] * m.silu(c['upgate'][2:])
        c['down'] = c['h'].copy()
    if stage == 'activation':
        c['h'][0] += .25
        c['down'] = c['h'].copy()
    if stage == 'down':
        c['down'][0] += .25
    output = c['res'] + c['down']
    if stage == 'residual_add':
        output[0] += .25
    result, vectors = m.decompose(c, nw, w, w, w, 1., output, np.ones(2))
    assert result['closure']['passed']
    assert np.max(np.abs(vectors[stage])) > .01
    for other in ('up', 'gate', 'activation', 'down', 'residual_add', 'interaction'):
        if other != stage:
            assert np.max(np.abs(vectors[other])) < 1e-14
    np.testing.assert_allclose(sum(vectors.values()), output - 1., atol=1e-14)


def test_nonlinear_up_gate_interaction_is_explicit():
    m = load()
    c, nw, w = fixture()
    c['upgate'][0] += .3
    c['upgate'][2] += .4
    c['h'] = c['upgate'][:2] * m.silu(c['upgate'][2:])
    c['down'] = c['h'].copy()
    result, vectors = m.decompose(c, nw, w, w, w, 1., c['res'] + c['down'], np.ones(2))
    assert vectors['interaction'][0] == pytest.approx(.3 * (m.silu(1.4) - m.silu(1.)))
    assert result['closure']['passed']


def test_cancelling_errors_use_signed_components_and_common_scale():
    m = load()
    c, nw, w = fixture()
    c['h'][0] += .2
    output = c['res'] + c['down']
    result, vectors = m.decompose(c, nw, w, w, w, 1., output, np.array([2., 2.]))
    np.testing.assert_allclose(vectors['activation'], -vectors['down'], atol=1e-15)
    assert result['components']['activation']['maxrel'] == pytest.approx(.1)
    assert result['components']['down']['maxrel'] == pytest.approx(.1)
    assert result['components']['activation']['at_native_worst'] > 0
    assert result['components']['down']['at_native_worst'] < 0
    assert result['reference_scale'] == 2.


def test_norm_and_oracle_rounding_are_separate_from_projection():
    m = load()
    c, nw, w = fixture()
    c['norm'][0] += .1
    c['upgate'] = np.r_[c['norm'], c['norm']]
    c['h'] = c['norm'] * m.silu(c['norm'])
    c['down'] = c['h'].copy()
    result, v = m.decompose(c, nw, w, w, w, 1., c['res'] + c['down'], np.ones(2))
    assert np.max(np.abs(v['norm'])) > .01
    assert np.max(np.abs(v['oracle_activation_rounding_bridge'])) > 0
    assert np.max(np.abs(v['up'])) == 0
    assert np.max(np.abs(v['gate'])) == 0
    assert result['closure']['passed']


@pytest.mark.parametrize('bad', ['shape', 'nan', 'epsilon', 'weights', 'empty'])
def test_invalid_numeric_inputs_fail(bad):
    m = load()
    c, nw, w = fixture()
    eps = 1.
    if bad == 'shape':
        c['upgate'] = np.ones(3)
    elif bad == 'nan':
        c['down'][0] = np.nan
    elif bad == 'epsilon':
        eps = 0.
    elif bad == 'weights':
        nw = np.ones(3)
    else:
        c['res'] = np.array([])
    with pytest.raises(ValueError):
        m.decompose(c, nw, w, w, w, eps, np.ones(2), np.ones(2))


def test_zero_reference_never_divides_by_zero_or_claims_pass():
    m = load()
    c, nw, w = fixture()
    result, _ = m.decompose(c, nw, w, w, w, 1., c['res'] + c['down'], np.zeros(2))
    assert result['native']['maxrel'] is None
    assert not result['native']['passed']
    json.dumps(result, allow_nan=False)


def test_rectangular_weights_close_and_up_counterfactual_includes_interaction():
    m = load()
    rng = np.random.default_rng(32)
    res, nw = rng.normal(size=3), rng.normal(size=3)
    up, gate, down = rng.normal(size=(5, 3)), rng.normal(size=(5, 3)), rng.normal(size=(3, 5))
    norm = (m.base.oracle.rms(res, .1) * nw).astype(np.float32).astype(np.float64)
    un, gn = up @ norm, gate @ norm
    uc, gc = un + .1, gn + .2
    h = uc * m.silu(gc) + .03
    y = down @ h + .04
    c = dict(res=res, norm=norm, upgate=np.r_[uc, gc], h=h, down=y)
    output, ref = res + y, np.array([1., 2., 3.])
    r, v = m.decompose(c, nw, up, gate, down, .1, output, ref)
    np.testing.assert_allclose(sum(v.values()), output - ref, atol=1e-13)
    repaired = res + down @ (un * m.silu(gc)) + down @ np.full(5, .03) + .04
    assert r['frozen_downstream_removal']['up']['maxrel'] == pytest.approx(
        np.max(np.abs(repaired - ref)) / 3.)
    ideal_h = (m.base.oracle.silu(gate @ norm) * (up @ norm)).astype(np.float32)
    ideal = res + down @ ideal_h
    np.testing.assert_array_equal(v['propagated'], ideal - ref)
    assert r['ideal_suffix']['ideal_ffn']['maxrel'] == pytest.approx(
        np.max(np.abs(ideal - ref)) / 3.)


def test_stale_capture_hash_invalidates_old_success(tmp_path):
    m = load()
    captured = tmp_path / 'capture.bin'
    captured.write_bytes(b'original')
    audit = dict(capture_valid=True, conditioned_ffn_passed=True,
                 evidence={}, capture_hashes={'capture.bin': m.base.digest(captured)})
    (tmp_path / 'sublayer-audit.json').write_text(json.dumps(audit))
    captured.write_bytes(b'modified')
    out = tmp_path / 'attribution.json'
    out.write_text('{"attribution_valid":true}')
    with pytest.raises(ValueError):
        m.analyze(SimpleNamespace(run=tmp_path, out=out, baseline=tmp_path, build=tmp_path))
    result = json.loads(out.read_text())
    assert not result['attribution_valid'] and not result['model_acceptance']
