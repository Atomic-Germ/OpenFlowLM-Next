"""Conditional DeltaNet diagnostics must preserve independent model acceptance."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[3]


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'utilities'))
    spec = importlib.util.spec_from_file_location('dn_boundary', ROOT / 'utilities/diagnose-wide-deltanet-boundary.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_error_decomposition_keeps_local_and_propagated_overlap(monkeypatch):
    m = module(monkeypatch)
    ref = np.array([1, 2, 3, 4], bfloat16)
    local = np.array([1, 2.5, 3, 4.5], bfloat16)
    got = np.array([1.5, 2.5, 3, 4], bfloat16)
    r = m.differences(got, local, ref)
    assert (r['device_vs_reference'], r['local'], r['propagated']) == (2, 2, 2)
    assert r['local_indices'] == [0, 3]
    assert r['propagated_indices'] == [1, 3]


def test_post_reference_uses_per_head_norm_and_fp32_before_bf16(monkeypatch):
    m = module(monkeypatch)
    o = np.ones((48, 128), np.float32)
    o[1] *= -2
    z = np.zeros((48, 128), np.float32)
    z[:, 7] = .25
    got = m.post_reference(o, z, np.ones(128, bfloat16)).reshape(48, 128)
    assert got.dtype == np.dtype(bfloat16)
    assert np.count_nonzero(got) == 48
    expected = np.float32(.25 / (1 + np.exp(-.25)) / np.sqrt(1 + 1e-6))
    assert got[0, 7] == bfloat16(expected)
    assert got[1, 7] < 0


def test_capture_rejects_corruption_and_nonfinite(monkeypatch, tmp_path):
    m = module(monkeypatch)
    p = tmp_path / 'capture.bin'
    for raw in (b'', np.array([1], np.float32).tobytes() + bytes(64),
                np.array([np.nan], np.float32).tobytes() + m.M.GUARD):
        p.write_bytes(raw)
        with pytest.raises(ValueError):
            m.capture(p, 4, np.float32)
    p.write_bytes(np.array([1], np.float32).tobytes() + m.M.GUARD)
    np.testing.assert_array_equal(m.capture(p, 4, np.float32), [1])


def test_capture_checks_only_logical_activation_not_poison_padding(monkeypatch, tmp_path):
    m = module(monkeypatch)
    p = tmp_path / 'padded.bin'
    p.write_bytes(np.array([1, np.nan], np.float32).tobytes() + m.M.GUARD)
    np.testing.assert_array_equal(m.capture(p, 8, np.float32, count=1), [1])


def test_diagnose_validates_fixtures_and_never_claims_model_acceptance(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    m = module(monkeypatch)
    layout = SimpleNamespace(STATE_S_OFF=3*10240*2, S_HEAD_BYTES=128*128*4)
    layout.STATE_BYTES = layout.STATE_S_OFF + 48*layout.S_HEAD_BYTES
    monkeypatch.setattr(m.M.S.A, 'geometry', lambda rows: (None, layout))
    projections = dict(qkv=np.full(10240, .125, np.float32), z=np.full(6144, .25, np.float32),
                       out=np.zeros(5120, np.float32))
    monkeypatch.setattr(m, 'projection', lambda pool, const, l, kind, name, x: projections[name])
    tag = 'cold-0-layer10'
    directory = tmp_path/'layer10'; directory.mkdir()
    (directory/'pool.bin').write_bytes(bytes(4)); (directory/'const.bin').write_bytes(bytes(4))
    params = dict(lnw=np.ones(5120, bfloat16), postw=np.ones(5120, bfloat16),
                  nw=np.ones(128, bfloat16), weights=np.zeros((2,48,5120), bfloat16),
                  a=np.full(48, -.1, np.float32), dt=np.zeros(48, np.float32),
                  convw=np.ones((4,10240), bfloat16))
    np.savez(directory/'params.npz', **{n:v.astype(np.float32) for n,v in params.items()})
    x = np.ones(5120, np.float32); xn = m.norm(x, params['lnw'])
    ab = m.ab_reference(xn, params['weights'], params['a'], params['dt']).astype(np.float32)
    conv, vec = m.glue_reference(projections['qkv'], np.zeros((3,10240), bfloat16), params['convw'], ab)
    vec = vec.astype(np.float32)
    so, o = m.step_reference(np.zeros((48,128,128), np.float32), vec)
    so, o = so.astype(np.float32), o.astype(np.float32)
    ref = dict(xn=xn, qkv=projections['qkv'], z=projections['z'], ab=ab, conv=conv, vec=vec,
               so=so, o=o, og=m.post_reference(o, projections['z'], params['nw']),
               projout=projections['out'], res1=x, xm=xn)
    np.savez(tmp_path/f'{tag}-ref.npz', **{n:v.astype(np.float32) for n,v in ref.items()})
    for name, value in ref.items():
        (tmp_path/f'{tag}-got-{name}.bin').write_bytes(value.tobytes()+m.M.GUARD)
    (tmp_path/f'{tag}-input-x.bin').write_bytes(x.tobytes()+m.M.GUARD)
    (tmp_path/f'{tag}-input-state.bin').write_bytes(bytes(layout.STATE_BYTES)+m.M.GUARD)
    names = ['layer10/pool.bin', 'layer10/const.bin', 'layer10/params.npz', f'{tag}-ref.npz']
    meta = dict(rows=257, fixtures={n:m.M.sha(tmp_path/n) for n in names}, kernels={},
                outputs={'d':{n:v.nbytes for n,v in ref.items()}},
                cases=[dict(tag=tag, kind=m.LINEAR, layer=10)])
    (tmp_path/'slice-fixture.json').write_text(json.dumps(meta))
    result = m.diagnose(tmp_path, tag)
    assert result['diagnostic_only'] and not result['passed']
    assert all(r['local'] == 0 and r['propagated'] == 0 for r in result['boundaries'].values())
    assert len(result['post_counterfactuals']) == 8
    assert all(r['local'] == 0 and r['propagated'] == 0 for r in result['post_counterfactuals'].values())
    (directory/'params.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='fixture changed'):
        m.diagnose(tmp_path, tag)
