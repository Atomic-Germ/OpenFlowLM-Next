"""Wide xn/xm must prepare from a depth-two FIFO without retaining three elements."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "open_kernels/designs/layer_x/xcommon.py"


class Fifo:
    def __init__(self, count):
        self.values = list(range(count))
        self.held = 0
        self.events = []

    def acquire(self, n):
        assert 0 < n <= 2, "acquisition exceeds depth-two FIFO"
        assert self.held == 0 and len(self.values) >= n
        self.held = n
        self.events.append(("acquire", n))
        return self.values[0] if n == 1 else self.values[:n]

    def release(self, n):
        assert n == self.held
        del self.values[:n]
        self.held = 0
        self.events.append(("release", n))


def functions(ns):
    tree = ast.parse(SOURCE.read_text())
    names = {"prep_bands", "ffn_body", "prep_stream"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), ns)


@pytest.mark.parametrize("width", [1024, 2048, 2560, 4096, 5120, 6144, 8192])
def test_projection_stream_prepares_all_blocks_and_preserves_legacy_lifetimes(width):
    chunks = (width + 2047) // 2048
    x = Fifo(chunks)
    tab = np.full(width, -1)
    calls = []
    def prep(element, table, k, i):
        assert element == i and k == width and x.held > 0
        table[i * 2048:min(k, (i + 1) * 2048)] = i
        calls.append(i)
    def gemv(*args):
        np.testing.assert_array_equal(tab, np.arange(width) // 2048)
        assert x.held == (chunks if chunks <= 2 else 0)
        x.events.append(("gemv", 1))
    ns = dict(range_=range, role_gemv_bands=gemv)
    functions(ns)
    ns["prep_bands"](None, x, None, {"tab": tab}, {"prep": prep}, width, chunks, 1)
    assert calls == list(range(chunks)) and not x.values and not x.held
    if chunks <= 2:
        assert x.events == [("acquire", chunks), ("gemv", 1), ("release", chunks)]
    else:
        assert x.events == [(event, 1) for _ in range(chunks) for event in ("acquire", "release")] + [("gemv", 1)]


@pytest.mark.parametrize("width", [2048, 4096, 5120])
def test_ffn_xm_releases_wide_input_before_up_gate_and_then_consumes_h(width):
    chunks = (width + 2047) // 2048
    x = Fifo(chunks + 1)
    weights = Fifo(2)
    class Output:
        def acquire(self, n):
            assert n == 1
            return np.zeros(64)
        def release(self, n):
            assert n == 1
    calls = []
    def prep(e, tab, k, i):
        assert e == i and k == width
        calls.append(("xm", i))
    def gms(*args):
        assert x.held == (chunks if chunks <= 2 else 0)
        assert calls == [("xm", i) for i in range(chunks)]
    def prepf(e, tab, k, i):
        assert e == chunks and i == 0 and k == 1024
        calls.append(("h", i))
    def down(*args):
        assert not x.values and not x.held
        calls.append(("down", 0))
    ns = dict(range_=range, FFN=SimpleNamespace(XM_ELEMS=chunks, UP_PC=1, H_ELEMS=1, DOWN_PC=1),
              HID=width, FF=1024, Q8=(), MIXED=False, C=SimpleNamespace(MS_U=0, MS_G=64),
              per_band=lambda k: 1, n_groups=lambda k: 1, role_gemv_bands=down)
    functions(ns)
    ns["ffn_body"](weights, x, Output(), {"tab": None, "ms": None},
                   {"prep": prep, "gms": gms, "act": lambda *a: None, "prepf": prepf})
    assert calls[-2:] == [("h", 0), ("down", 0)]
    assert not weights.values
