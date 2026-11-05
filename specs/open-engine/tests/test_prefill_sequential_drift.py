"""Causal drift attribution must isolate current input without corrupting history."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('sequential_drift', ROOT/'utilities/prefill-sequential-drift/check.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def fake_spec(kind='linear_attention'):
    return SimpleNamespace(hidden=1, layer_types=[kind], conv_kernel=2, lin_qkv_dim=1,
                           lin_value_heads=1, lin_value_dim=1, num_kv_heads=1, head_dim=1)


def test_reference_history_forks_do_not_mutate_later_snapshots(monkeypatch):
    m = load(); calls = []
    def linear(model, spec, layer, x, conv, state):
        calls.append(float(x[0])); conv += x; state += conv.sum()
        return np.array([state.sum()]), conv, state
    monkeypatch.setattr(m.base.oracle, 'linear_decode', linear)
    monkeypatch.setattr(m.base.oracle, 'ffn', lambda *a: np.zeros(1))
    xs = np.array([[1.], [2.], [3.]], np.float32)
    output, states = m.replay(None, fake_spec(), 0, xs, (1,2))
    assert calls == [1,2,3] and output[1][0] == 6 and output[2][0] == 13
    saved = [[a.copy() for a in states[p]] for p in (1,2)]
    assert m.on_history(None, fake_spec(), 0, np.array([20.]), states[1], 1)[0] == 42
    for p, expected in zip((1,2), saved):
        for a, b in zip(states[p], expected): np.testing.assert_array_equal(a, b)
    assert m.on_history(None, fake_spec(), 0, xs[2], states[2], 2)[0] == 13
    np.testing.assert_array_equal(xs, [[1],[2],[3]])


def test_attention_carries_every_position_and_resets(monkeypatch):
    m = load(); calls = []
    def attention(model, spec, layer, x, k, v, pos):
        calls.append(pos); k = np.concatenate((k, x.reshape(1,1,1)))
        return np.array([k.sum()]), k, k.copy()
    monkeypatch.setattr(m.base.oracle, 'attn_decode', attention)
    monkeypatch.setattr(m.base.oracle, 'ffn', lambda *a: np.zeros(1))
    xs = np.array([[1.],[2.],[3.]], np.float32)
    output, states = m.replay(None, fake_spec('full_attention'), 0, xs, (2,))
    assert calls == [0,1,2] and output[2][0] == 9
    assert m.on_history(None,fake_spec('full_attention'),0,np.array([4.]),states[2],2)[0] == 11
    again, _ = m.replay(None,fake_spec('full_attention'),0,xs,(2,))
    assert again[2][0] == 9


@pytest.mark.parametrize('positions', [(), (1,1), (-1,), (3,), (1.5,), (True,)])
def test_invalid_target_positions_are_rejected(positions):
    with pytest.raises(ValueError): load().replay(None, fake_spec(), 0, np.ones((3,1)), positions)


def test_signed_components_close_with_cancellation_and_rounding():
    m = load(); ref = np.array([10., 2.]); ideal_ref = np.array([10.1, 2.])
    mixed = np.array([12., 3.]); ideal_native = np.array([11., 2.5]); native = np.array([10.5,2.2])
    scores, vectors = m.decompose(native, ideal_native, mixed, ideal_ref, ref)
    np.testing.assert_allclose(sum(vectors[k] for k in m.PARTS), vectors['total'])
    assert scores['current_input']['maxrel'] > scores['total']['maxrel']
    assert scores['reference_rounding']['maxrel'] == pytest.approx(.01)
    assert scores['implementation_gap']['maxrel'] == pytest.approx(.05)


def test_decomposition_refuses_nonfinite_or_incompatible_arrays():
    m = load(); x = np.ones(2)
    for bad in (np.ones(3), np.array([np.nan, 1.])):
        with pytest.raises(ValueError): m.decompose(bad,x,x,x,x)
    scores, _ = m.decompose(x,np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2))
    assert scores['total']['maxrel'] is None and not scores['total']['passed']
    json.dumps(scores, allow_nan=False)


def test_summary_requires_all_targets_and_routes():
    m = load(); rows = [dict(layer=l, position=p, route=r, metrics={k:dict(maxrel=.001,passed=True) for k in (*m.PARTS,'propagated','total')}) for l,ps in m.TARGETS.items() for p in ps for r in m.ROUTES]
    assert m.summarize(rows)['sequential']['passed'] == 6
    with pytest.raises(ValueError): m.summarize(rows[:-1])
    with pytest.raises(ValueError): m.summarize(rows + [rows[0]])


def test_failed_input_validation_is_persisted_without_overwrite(tmp_path):
    m = load(); a = SimpleNamespace(out=tmp_path/'out', baseline=tmp_path/'missing')
    with pytest.raises(FileNotFoundError): m.run(a)
    r = json.loads((a.out/'report.json').read_text())
    assert not r['complete'] and not r['model_acceptance']
    with pytest.raises(FileExistsError): m.run(a)
