"""Shared WideDeltaNet geometry and standalone AB dispatch contracts."""
import pytest
import ast
from pathlib import Path
import numpy as np

from recipes.wide_deltanet import WideDeltaNet


@pytest.mark.parametrize("hidden,chunks,tiles", [(5120, 3, (32, 32, 16)), (2560, 2, (32, 8))])
def test_shared_geometry(hidden, chunks, tiles):
    g = WideDeltaNet(hidden=hidden, key_heads=16, value_heads=48, key_dim=128, value_dim=128)
    assert g.ab_bank_width == 32 and g.ab_banks == 2
    assert g.banks == ((0, 32), (32, 16))
    assert g.xn_chunks == chunks and g.ab_tiles == tiles
    assert g.group_size == 3
    assert [g.key_head(h) for h in range(48)] == [h for h in range(16) for _ in range(3)]
    assert g.ab_input_dma_channels == 2
    assert g.side_bytes == 2 * (2 * hidden * 32 * 2 + 4096)
    assert g.result_bytes == 4 * 48 * 4
    for bad in (-1, 48, 63):
        with pytest.raises(ValueError, match="value head"):
            g.key_head(bad)


@pytest.mark.parametrize("tail", range(1, 33))
def test_all_bank_tails(tail):
    g = WideDeltaNet(hidden=64, key_heads=1, value_heads=32 + tail, key_dim=128, value_dim=128)
    assert g.banks == ((0, 32), (32, tail))


@pytest.mark.parametrize("change", [dict(hidden=0), dict(hidden=65), dict(key_heads=0),
                                    dict(value_heads=47), dict(key_dim=0), dict(value_dim=64)])
def test_invalid_geometry_fails_before_build(change):
    args = dict(hidden=5120, key_heads=16, value_heads=48, key_dim=128, value_dim=128)
    with pytest.raises(ValueError):
        WideDeltaNet(**(args | change))


@pytest.mark.parametrize("hidden", [2560, 5120])
def test_standalone_ab_worker_streams_and_reuses_accumulators(hidden):
    from test_qwen35_wide_glue import Input
    g = WideDeltaNet(hidden, 16, 48, 128, 128)
    rng = np.random.default_rng(17)
    x = rng.integers(-2, 3, hidden).astype(np.float32)
    weights = rng.integers(-2, 3, (2, 48, hidden)).astype(np.float32)
    side, chunks = [], []
    for base, active in g.banks:
        for w in weights:
            bank = np.zeros((hidden, 32), np.float32)
            bank[:, :active] = w[base:base + active].T
            side.extend(bank.reshape(-1, 64, 32))
            for h in range(g.xn_chunks):
                chunk = np.zeros(2048, np.float32)
                n = min(2048, hidden - h * 2048)
                chunk[:n] = x[h * 2048:h * 2048 + n]
                chunks.append(chunk)
        side.append("small")
    sin, xin = Input(side), Input(chunks)
    result = np.full((4, 48), np.nan, np.float32)
    class Output:
        def acquire(self, n):
            assert n == 1
            return result
        def release(self, n):
            assert n == 1 and np.isfinite(result).all()
    acc_a, acc_b, xn = np.empty(32), np.empty(32), np.empty(2048)
    ids, calls = set(), []
    def copy(src, dst):
        dst[:] = src
    def ab(w, x, acc, tile, first):
        ids.add(id(acc))
        if first and tile == 0:
            acc[:] = 0
        acc[:] += x[tile * 64:(tile + 1) * 64] @ w
    def store(sm, a, b, out, base, active):
        assert sm == "small"
        calls.append((base, active))
        out[0, base:base + active] = a[:active]
        out[1, base:base + active] = b[:active]
        out[2:, base:base + active] = 0  # C++ helper and hardware test check nonlinearity.
    path = Path(__file__).resolve().parents[3] / "open_kernels/designs/wide_deltanet/ab.py"
    tree = ast.parse(path.read_text())
    design = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "wide_ab")
    fn = next(n for n in design.body if isinstance(n, ast.FunctionDef) and n.name == "core_body")
    ns = dict(G=g, range_=range)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    ns["core_body"](sin, xin, Output(), acc_a, acc_b, xn, ab, store, copy)
    np.testing.assert_array_equal(result[:2], weights @ x)
    assert ids == {id(acc_a), id(acc_b)} and calls == list(g.banks)
    assert not sin.values and not xin.values and sin.held == xin.held == 0
