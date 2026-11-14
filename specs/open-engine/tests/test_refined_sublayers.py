"""Fail-closed controls for candidate causal linear-layer diagnostics."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location(
        'refined_sublayers', ROOT/'utilities/refined-sublayers/replay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analysis():
    return load().load('refined_analysis', ROOT/'utilities/refined-sublayers/analyze.py')


def test_fixed_targets_and_causal_program(tmp_path):
    m = load()
    assert m.TARGETS == {50: 205, 54: 125, 60: 220}
    sizes = dict(pool=64, consts=32, act=128, state=256, hidden=8)
    for layer, position in m.TARGETS.items():
        text = m.program(tmp_path, tmp_path/'kernels', sizes, layer)
        assert text.count('run lx pool xres consts state act\n') == position+1
        assert text.count('buf state ') == 1
        assert 'load state ' not in text
        assert f'load xres {tmp_path}/input_0.bin' in text
        assert f'dump state {tmp_path}/state_{position}.bin 256' in text
        assert len(m.names(layer)) == 3*(position+1)
    with pytest.raises(ValueError, match='target'):
        m.program(tmp_path, tmp_path, sizes, 51)


def test_identity_covers_every_position_and_signed_zero():
    m = load()
    a = np.zeros((206, 4), dtype='<f4')
    b = a.copy()
    assert m.identity(a, b, 50)['passed']
    b[100, 2] = -0.0
    assert m.identity(a, b, 50)['first_mismatch'] == 100
    with pytest.raises(ValueError, match='coverage'):
        m.identity(a[:-1], b[:-1], 50)
    b[0, 0] = np.nan
    with pytest.raises(ValueError):
        m.identity(a, b, 50)


def test_historical_fail_is_not_a_waiver():
    m = load()
    report = dict(complete=True, candidate_passed=False, model_acceptance=False)
    audit = dict(complete=True, candidate_passed=False, model_acceptance=False,
                 report_sha256='bound')
    m.require_baseline(report, audit, 'bound')
    for field in ('complete', 'model_acceptance', 'candidate_passed'):
        changed = audit | {field: not audit[field]}
        with pytest.raises(ValueError, match='audit'):
            m.require_baseline(report, changed, 'bound')
    with pytest.raises(ValueError, match='audit'):
        m.require_baseline(report, audit, 'changed')


def test_full_capture_inventory_required(tmp_path):
    m = load()
    # Refuse incomplete or extra captures before hashing/reading their contents.
    (tmp_path/'xres_0.bin').write_bytes(b'')
    with pytest.raises(ValueError, match='inventory'):
        m.read_outputs(tmp_path, dict(hidden=4, act=8, state=8), 50)


def test_native_run_never_overwrites_evidence(tmp_path):
    m = load()
    (tmp_path/'report.json').write_text('{}')
    with pytest.raises(FileExistsError):
        m.run(SimpleNamespace(out=tmp_path))


def test_all_targets_required_for_boundary_interpretation():
    m = load()
    rows = {str(l): dict(passed=True, positions=p+1, matching_positions=p+1,
                        first_mismatch=None) for l,p in m.TARGETS.items()}
    assert m.complete_identity(rows)
    assert not m.complete_identity(rows | {'50': rows['50'] | {'passed': False}})
    with pytest.raises(ValueError, match='coverage'):
        m.complete_identity({'50': rows['50']})


def test_fp32_post_reader_preserves_low_bits_and_checks_extent():
    m = analysis()
    values = np.array([0, 1.000123, -2.000321, 0], dtype='<f4')
    offsets = SimpleNamespace(A_BYTES=16, A_OG=4)
    assert np.array_equal(m.region(values.tobytes(), offsets, 'A_OG', 2), values[1:3])
    for data, off, size in ((values.tobytes()[:-1], offsets, 2),
                            (values.tobytes(), SimpleNamespace(A_BYTES=16, A_OG=3), 2),
                            (values.tobytes(), offsets, 4)):
        with pytest.raises(ValueError, match='region'):
            m.region(data, off, 'A_OG', size)
    values[1] = np.inf
    with pytest.raises(ValueError):
        m.region(values.tobytes(), offsets, 'A_OG', 2)


def test_signed_attribution_closes_and_retains_cancellation():
    m = analysis()
    nodes = {name: np.array([2., 4.]) for name in m.NODES}
    nodes['current_input'] += [1., -2.]
    nodes['input_history'] += [2., -1.]
    nodes['native'] += [.01, -.02]
    scores, vectors = m.decompose(nodes)
    np.testing.assert_allclose(sum(vectors[p] for p in m.PARTS), vectors['total'], atol=1e-14)
    np.testing.assert_allclose(vectors['propagated']+vectors['local'], vectors['total'])
    assert scores['total']['passed'] is False  # Strict maxrel < .005, not <=.
    assert scores['propagated']['maxrel'] > scores['total']['maxrel']
    assert scores['local']['maxrel'] > scores['total']['maxrel']
    with pytest.raises(ValueError, match='nodes'):
        m.decompose({'reference': nodes['reference']})
    nodes['z'] = np.array([np.nan, 0.])
    with pytest.raises(ValueError, match='finite'):
        m.decompose(nodes)


def test_analysis_coverage_and_bytewise_artifact_audit(tmp_path):
    m = analysis()
    rows = [dict(layer=l, position=p) for l,p in m.replay.TARGETS.items()]
    m.coverage(rows)
    for bad in (rows[:-1], rows[::-1], rows+rows[:1]):
        with pytest.raises(ValueError, match='coverage'):
            m.coverage(bad)
    arrays = dict(native=np.array([0., 1.], dtype='<f8'))
    m.store(tmp_path, None, 'target.npz', arrays)
    m.store(None, tmp_path, 'target.npz', arrays)
    arrays['native'][0] = -0.
    with pytest.raises(ValueError, match='re-audit'):
        m.store(None, tmp_path, 'target.npz', arrays)
