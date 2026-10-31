"""Reject cancellation-hidden FFN errors and incomplete diagnostic captures."""
from types import SimpleNamespace

import numpy as np
import pytest

from ffn_partials import compare_partials, read_regions


def fixture():
    h = np.array([1., -2., 3., 4.])
    w = np.array([[2., 1., -1., 3.], [-1., 2., 4., 1.]])
    res = np.array([2., -3.])
    parts = [w[:, :2] @ h[:2], w[:, 2:] @ h[2:]]
    return h, w, res, parts, res + sum(parts), (2, 2)


def test_exact_partials_and_sum_pass():
    result = compare_partials(*fixture())
    assert result['passed']
    assert set(result['metrics']) == {'partial0', 'partial1', 'sum', 'closing_residual'}
    assert all(m['maxrel'] == 0 for m in result['metrics'].values())


def test_cancelling_partial_errors_cannot_hide_behind_correct_sum():
    h, w, res, parts, closed, split = fixture()
    parts[0] += 1
    parts[1] -= 1
    result = compare_partials(h, w, res, parts, closed, split)
    assert not result['passed']
    assert result['metrics']['sum']['passed']
    assert not result['metrics']['partial0']['passed']


def test_closing_residual_is_enforced():
    h, w, res, parts, closed, split = fixture()
    assert not compare_partials(h, w, res, parts, closed + 1, split)['passed']


@pytest.mark.parametrize('split', [(1, 2), (0, 4), (2, -2), (1, 1, 2)])
def test_invalid_partition_refused(split):
    args = list(fixture()); args[-1] = split
    with pytest.raises(ValueError):
        compare_partials(*args)


@pytest.mark.parametrize('index', [0, 1, 2, 3, 4])
def test_nonfinite_input_refused(index):
    args = list(fixture())
    arr = args[index][0] if index == 3 else args[index]
    arr.flat[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        compare_partials(*args)


@pytest.mark.parametrize('index', [0, 1, 2, 3, 4])
def test_wrong_shape_refused(index):
    args = list(fixture()); args[index] = args[index][:-1]
    with pytest.raises(ValueError):
        compare_partials(*args)


def test_zero_reference_requires_exact_zero():
    h = np.zeros(4); w = np.ones((2, 4)); res = np.zeros(2)
    assert compare_partials(h, w, res, [res, res], res, (2, 2))['passed']
    assert not compare_partials(h, w, res, [res + 1e-20, res], res, (2, 2))['passed']


@pytest.mark.parametrize('full', [False, True])
def test_reads_separate_linear_and_attention_offsets(tmp_path, full):
    layout = SimpleNamespace(A_BYTES=40, A_RES=0, A_H=8, A_OUT2=24, A_OUT2B=32,
                             AA_BYTES=48, AA_RES=8, AA_H=16, AA_OUT2=32, AA_OUT2B=40)
    a = np.arange(12 if full else 10, dtype=np.float32)
    p = tmp_path / 'act.bin'; p.write_bytes(a.tobytes())
    regions = read_regions(p, layout, 2, 4, full)
    base = 2 if full else 0
    np.testing.assert_array_equal(regions['res'], a[base:base+2])
    np.testing.assert_array_equal(regions['h'], a[base+2:base+6])
    np.testing.assert_array_equal(regions['partials'][1], a[base+8:base+10])
    p.write_bytes(a.tobytes()[:-1])
    with pytest.raises(ValueError, match='bytes'):
        read_regions(p, layout, 2, 4, full)


def test_missing_capture_refused(tmp_path):
    with pytest.raises(OSError):
        read_regions(tmp_path/'missing', SimpleNamespace(A_BYTES=40), 2, 4, False)


def test_strict_boundary_not_inclusive():
    h = np.ones(2); w = np.array([[200., 200.]])
    result = compare_partials(h, w, np.zeros(1), [np.array([201.]), np.array([200.])],
                              np.array([400.]), (1, 1))
    assert result['metrics']['partial0']['maxrel'] == 0.005
    assert not result['passed']


def test_region_nonfinite_refused(tmp_path):
    a = np.ones(10, dtype=np.float32); a[8] = np.inf
    p = tmp_path/'act'; p.write_bytes(a.tobytes())
    layout = SimpleNamespace(A_BYTES=40, A_RES=0, A_H=8, A_OUT2=24, A_OUT2B=32)
    with pytest.raises(ValueError, match='finite'):
        read_regions(p, layout, 2, 4, False)


@pytest.fixture
def tiny_fixture(tmp_path, monkeypatch):
    import ffn_partials as mod
    h, w, res, parts, closed, split = fixture()
    offsets = SimpleNamespace(A_BYTES=40, A_RES=0, A_H=8, A_OUT2=24, A_OUT2B=32)
    meta = dict(spec_hash='spec', build_key='export', max_ctx=4, hidden=2, layers=1, tokens=2)
    spec = SimpleNamespace(family='qwen35', hidden=2, intermediate=4, num_layers=1, layer_types=['linear'])
    monkeypatch.setattr(mod, 'spec_from_model_dir', lambda _: spec)
    monkeypatch.setattr(mod, 'manifest', lambda *a, **k: meta.copy())
    monkeypatch.setattr(mod, 'load_bundle', lambda *a: meta.copy())
    monkeypatch.setattr(mod, 'load_reference', lambda _: meta.copy())
    monkeypatch.setattr(mod, 'layout', lambda *a, **k: offsets)
    monkeypatch.setattr(mod, 'ffn_geometry', lambda _: SimpleNamespace(DOWN_SPLIT=split))
    monkeypatch.setattr(mod, 'Q4NX', lambda _: SimpleNamespace(matmul_w=lambda *a: w))
    (tmp_path/'decode_reference.json').write_text('{}')
    for suffix in ('', '_t1'):
        np.concatenate([res, h, *parts]).astype(np.float32).tofile(tmp_path/f'y_act0{suffix}.bin')
        closed.astype(np.float32).tofile(tmp_path/f'y_res0{suffix}.bin')
    return mod, tmp_path, meta


def test_every_declared_token_checked(tiny_fixture):
    mod, out, meta = tiny_fixture
    result = mod.check(out, out, out, [0])
    assert result['passed'] and len(result['results']) == 2
    assert len(result['capture_sha256']) == 4
    (out/'y_act0_t1.bin').unlink()
    with pytest.raises(OSError):
        mod.check(out, out, out, [0])


@pytest.mark.parametrize('layers', [[], [-1], [1], [0, 0]])
def test_uncovered_layer_refused(tiny_fixture, layers):
    mod, out, meta = tiny_fixture
    with pytest.raises(ValueError, match='indices'):
        mod.check(out, out, out, layers)


def test_stale_export_identity_refused(tiny_fixture, monkeypatch):
    mod, out, meta = tiny_fixture
    monkeypatch.setattr(mod, 'load_bundle', lambda *a: dict(meta, build_key='other'))
    with pytest.raises(ValueError, match='build_key'):
        mod.check(out, out, out, [0])


def test_rounded_input_diagnostic_cannot_turn_failure_into_pass():
    h = np.array([1.001, 1., 1., 1.], dtype=np.float32)
    w = np.array([[1., -1., 1., 1.]])
    result = compare_partials(h, w, np.zeros(1), [np.zeros(1), np.array([2.])],
                              np.array([2.]), (2, 2), diagnose_bf16=True)
    assert not result['passed']
    assert result['bf16_input_diagnostic']['passed']
