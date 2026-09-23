"""Execute lx's actual glue worker with checked FIFOs, without pretending to run IRON.

This tests stream order, accumulator reuse and the conv/record continuation.
Hardware placement and AIE arithmetic remain separate validation gates.
"""
import ast
from collections import deque
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from recipes import qwen35 as Q35, qwen36moe as Q36
from recipes.catalogue import OpRangeError
from recipes.spec import ModelSpec

ROOT = Path(__file__).resolve().parents[3]
LX = ROOT / "open_kernels/designs/layer_x/lx.py"


def spec27():
    return ModelSpec.from_hf_config(json.loads(
        (Path(__file__).parent / "fixtures/config_qwen38_27b.json").read_text()))


def worker(namespace):
    tree = ast.parse(LX.read_text())
    lx = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "lx")
    body = next(n for n in lx.body if isinstance(n, ast.FunctionDef) and n.name == "glue_body")
    exec(compile(ast.Module(body=[body], type_ignores=[]), str(LX), "exec"), namespace)
    return namespace["glue_body"]


@pytest.mark.parametrize("heads,hidden", [(48, 5120), (32, 4096), (32, 2560), (16, 2048), (16, 1024)])
def test_actual_glue_declarations_preserve_legacy_and_budget_wide_buffers(heads, hidden):
    """Evaluate the real design's type/FIFO declarations with recording constructors.

    This is declared storage only: IRON alignment/placement is not simulated.
    """
    s = ModelSpec.from_dict(dict(spec27().to_dict(), hidden=hidden,
                                 lin_value_heads=heads, intermediate=8192))
    d, layout = Q36.linear(s), Q35.layout(s)
    fifos = {}

    def fifo(ty, name, depth):
        result = SimpleNamespace(ty=ty, name=name, depth=depth)
        fifos[name] = result
        return result

    def external(name, **kwargs):
        return SimpleNamespace(name=name, **kwargs)

    def size(ty):
        shape, dtype = ty.__args__
        return np.prod(shape) * np.dtype(dtype.__args__[0]).itemsize

    x = SimpleNamespace(
        types=lambda: dict(elem=object(), y=object(), x=object()),
        ln_types=lambda: {"u8_ln": np.ndarray[(layout.ELN,), np.dtype[np.uint8]]},
        kernels=lambda *args: {}, ln_kernels=lambda *args: {}, LN=Path("ln"), RT=Path("router"))
    ns = dict(layout.constants(), np=np, bfloat16=np.uint16, D=d, SPEC=s, X=x,
              ELEM=4096, HID=hidden, NHEAD=heads, TILE=1024, N_CORES=8,
              DENSE=True, WIDE_GLUE=heads > 32, ObjectFifo=fifo, ExternalFunction=external,
              include_dirs=lambda: [], GEMV=Path("gemv"), GLUE=Path("glue"), POST=Path("post"),
              HERE=LX.parent, GLUE_FLAGS={})
    tree = ast.parse(LX.read_text())
    lx = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "lx")
    setup = []
    for node in lx.body:
        if isinstance(node, ast.FunctionDef):
            break
        setup.append(node)
    exec(compile(ast.Module(body=setup, type_ignores=[]), str(LX), "exec"), ns)
    assert set(fifos) == ({f"w{i}" for i in range(8)} | {f"y{i}" for i in range(8)} |
                          {"x", "lni", "lno", "side", "gact", "gout", "pin", "pout"} |
                          ({"xn_side"} if heads > 32 else set()))
    assert (fifos["side"].depth, fifos["gact"].depth, fifos["gout"].depth) == (2, 5, 3)
    if heads > 32:
        assert size(ns["f_acc"]) == 32 * 4
        assert size(ns["f32"]) == 48 * 4
        assert ns["f_small"].name == "glue_small_bank_fn"
        assert fifos["xn_side"].depth == 1
        fifo_bytes = sum(size(fifos[n].ty) * fifos[n].depth for n in ("side", "xn_side", "gact", "gout"))
        private_bytes = 2 * size(ns["f_acc"]) + 2 * size(ns["f32"]) + sum(size(ns[n]) for n in ("fqk", "fvt", "fxn"))
        assert fifo_bytes + private_bytes + 0x1800 == 60032 <= Q36.L1_BUDGET
    else:
        assert ns["f_acc"] == ns["f32"]
        assert ns["f_small"].name == "glue_small_fn"


class Input:
    def __init__(self, values):
        self.values = deque(values)
        self.held = 0
        self.count = 0

    def acquire(self, count):
        assert self.held == 0, "an input was acquired before releasing its previous elements"
        assert len(self.values) >= count, "worker consumed beyond the scheduled stream"
        self.held = count
        self.count += count
        values = [self.values.popleft() for _ in range(count)]
        return values[0] if count == 1 else values

    def release(self, count):
        assert count == self.held
        self.held = 0


class Output:
    def __init__(self):
        self.count = 0
        self.held = 0

    def acquire(self, count):
        assert self.held == 0
        self.held = count
        values = [np.zeros(512) for _ in range(count)]
        return values[0] if count == 1 else values

    def release(self, count):
        assert count == self.held
        self.held = 0
        self.count += count


@pytest.mark.parametrize("hidden,heads,dense", [(5120, 48, True), (4096, 32, True), (2560, 32, True),
                                              (2048, 16, True), (1024, 16, True), (2048, 32, False)])
def test_glue_stream_order_reuses_accumulators_and_continues_to_records(hidden, heads, dense):
    wide = heads > 32
    banks = (heads + 31) // 32
    tiles = [min(2048, hidden - off) // 64 for off in range(0, hidden, 2048)]
    rng = np.random.default_rng(81)
    x = rng.integers(-2, 3, hidden).astype(np.float64)
    weights = [rng.integers(-2, 3, (heads, hidden)).astype(np.float64) for _ in range(2)]
    side, xs = ([] if dense else [x.copy()]), []
    for bank in range(banks):
        for w in weights:
            padded = np.zeros((32, hidden))
            active = min(32, heads - bank * 32)
            padded[:active] = w[bank * 32:bank * 32 + active]
            off = 0
            for nt in tiles:
                chunk = np.zeros(2048)
                chunk[:nt * 64] = x[off:off + nt * 64]
                if dense:
                    (xs if wide else side).append(chunk)
                for tile in range(nt):
                    side.append(padded[:, off + tile * 64:off + (tile + 1) * 64].T.copy())
                off += nt * 64
        side.append("small")
    # Exactly 4 key tiles and heads/8 value tiles; a marker catches premature conv.
    conv_tiles = 4 + heads // 8
    side.extend(["conv"] * (conv_tiles * 2))
    sin, xin, ain, out = Input(side), Input(xs), Input([None] * (conv_tiles * 5)), Output()
    acc_a, acc_b = np.full(32, np.nan), np.full(32, np.nan)
    decay, beta = np.full(heads, np.nan), np.full(heads, np.nan)
    acc_ids, small_calls, records = set(), [], []

    def copy(src, dst, offset=0):
        assert offset == 0
        dst[:] = src

    def ab(w, xn, acc, tile, first=1):
        acc_ids.add(id(acc))
        assert acc.shape == (32,)
        if first and tile == 0:
            acc[:] = 0
        acc[:] += xn[tile * 64:(tile + 1) * 64] @ w

    def small(sm, a, b, d, be, base=0, active=None):
        assert sm == "small"
        active = heads if active is None else active
        small_calls.append((base, active))
        for acc, w in ((a, weights[0]), (b, weights[1])):
            np.testing.assert_array_equal(acc[:active], w[base:base + active] @ x)
        d[base:base + active] = a[:active]
        be[base:base + active] = b[:active]

    def conv(*args):
        assert len(small_calls) == banks and not np.isnan(decay).any()
        assert args[5] == args[6] == "conv"

    def emit(qk, vt, d, b, record, tile, lane):
        head = tile * 8 + lane
        records.append(head)
        assert d[head] == weights[0][head] @ x
        assert b[head] == weights[1][head] @ x

    ns = dict(DENSE=dense, WIDE_GLUE=wide, AB_BANKS=banks, AB_TILES=tiles, AB_ELEMS=sum(tiles), NHEAD=heads,
              KEY_TILES=4, VALUE_TILES=heads // 8, CONVW_ELEMS=2, CONV_ROWS=3,
              D=Q36.linear(ModelSpec.from_dict(dict(spec27().to_dict(), lin_value_heads=heads))),
              range_=range)
    fn = worker(ns)
    args = [sin, ain, out, acc_a, acc_b, decay, beta, np.zeros(4096), np.zeros(1024),
            np.zeros(2048), ab, small, conv, emit, copy]
    fn(*args, *([xin] if wide else []))
    assert acc_ids == {id(acc_a), id(acc_b)}
    assert small_calls == ([(0, 32), (32, 16)] if wide else [(0, heads)])
    assert records == list(range(heads))
    assert out.count == 3 * conv_tiles + heads
    assert not sin.values and not xin.values and not ain.values
    assert xin.count == (12 if wide else 0)
    assert not any(f.held for f in (sin, xin, ain, out))


def test_ab_bank_count_is_geometry_not_model_name():
    s = spec27()
    for heads, banks in ((16, 1), (32, 1), (48, 2), (64, 2), (80, 3)):
        assert Q36.ab_banks(ModelSpec.from_dict(dict(s.to_dict(), lin_value_heads=heads))) == banks


def test_wide_dispatch_stays_blocked_until_dma_bringup(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    # Keep the independent FFN L1 blocker out of this assertion.
    s = ModelSpec.from_dict(dict(spec27().to_dict(), intermediate=8192))
    with pytest.raises(OpRangeError, match="not implemented.*wide.*DMA"):
        Q35.recipe(s)


def test_small_bank_pointer_arithmetic_in_compiled_cpp(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ required for the actual glue_small_bank header test")
    src = ROOT / "specs/open-engine/tests/fixtures/glue_small_bank_test.cpp"
    binary = tmp_path / "glue-small-bank-test"
    subprocess.run([compiler, "-std=c++17", "-O1", "-fsanitize=undefined,bounds",
                    "-fno-sanitize-recover=all", "-I" + str(ROOT / "open_kernels/designs/dn_glue"),
                    str(src), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
