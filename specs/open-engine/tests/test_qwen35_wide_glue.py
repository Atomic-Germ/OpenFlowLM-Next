"""Execute lx's actual glue worker with checked FIFOs, without pretending to run IRON.

This tests stream order, accumulator reuse and the conv/record continuation, and that
the host's side-channel DMA issues exactly what the worker consumes. Past 32 value
heads the glue runs glue_ab_w.cc's 64-lane tile, and at three xn halves it walks
half-outer (recipes/qwen36moe.py glue_fills). Hardware placement and AIE arithmetic
remain separate validation gates.
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
from recipes.catalogue import LIMITS
from recipes.spec import ModelSpec

ROOT = Path(__file__).resolve().parents[3]
LX = ROOT / "open_kernels/designs/layer_x/lx.py"


def spec27():
    return ModelSpec.from_hf_config(json.loads(
        (Path(__file__).parent / "fixtures/config_qwen38_27b.json").read_text()))


def spec(hidden, heads):
    # intermediate 8192 keeps the 27B's separate FFN width question out of these tests
    return ModelSpec.from_dict(dict(spec27().to_dict(), hidden=hidden,
                                    lin_value_heads=heads, intermediate=8192))


def lx_function(name):
    tree = ast.parse(LX.read_text())
    lx = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "lx")
    return next(n for n in lx.body if isinstance(n, ast.FunctionDef) and n.name == name)


def worker(namespace):
    body = lx_function("glue_body")
    exec(compile(ast.Module(body=[body], type_ignores=[]), str(LX), "exec"), namespace)
    return namespace["glue_body"]


def glue_namespace(d, hidden, dense=True):
    """lx.py's module-level names that glue_body reads, from the recipe's Linear."""
    tiles = [min(2048, hidden - h * 2048) // d.AB_ROWS for h in range(d.XN_SIDE_ELEMS)]
    return dict(DENSE=dense, HALF_OUTER=d.GLUE_HALF_OUTER, AB_TILES=tiles, AB_ELEMS=d.AB_ELEMS,
                NHEAD=d.NHEAD, KEY_TILES=d.VALUE_TILE0, VALUE_TILES=d.NT - d.VALUE_TILE0,
                CONVW_ELEMS=2, CONV_ROWS=3, D=d, range_=range)


@pytest.mark.parametrize("heads,hidden", [(48, 5120), (48, 2560), (32, 4096), (32, 2560), (16, 2048), (16, 1024)])
def test_actual_glue_declarations_preserve_legacy_and_budget_wide_buffers(heads, hidden):
    """Evaluate the real design's type/FIFO declarations with recording constructors.

    This is declared storage only: IRON alignment/placement is not simulated. Past 32
    heads the accumulators are one 64-lane W row and the xn halves still ride the side
    fifo, so the glue core declares the same fifos as every other size.
    """
    s = spec(hidden, heads)
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
        kernels=lambda *args: {}, ln_kernels=lambda *args: {}, LN=Path("ln"), RT=Path("router"),
        LNI_DEPTH=5)
    ns = dict(layout.constants(), np=np, bfloat16=np.uint16, D=d, SPEC=s, X=x,
              ELEM=4096, HID=hidden, NHEAD=heads, TILE=1024, N_CORES=8,
              DENSE=True, AB_WIDE=d.AB_LANES > 32, ObjectFifo=fifo, ExternalFunction=external,
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
                          {"x", "lni", "lno", "side", "gact", "gout", "pin", "pout"})
    assert (fifos["side"].depth, fifos["gact"].depth, fifos["gout"].depth) == (2, 5, 3)
    assert ns["f_small"].name == "glue_small_fn"
    if heads > 32:
        assert d.AB_LANES == 64
        assert size(ns["facc"]) == 64 * 4
        assert size(ns["f32"]) == heads * 4
        assert ns["f_ab"].name == "glue_ab_w"
        fifo_bytes = sum(size(fifos[n].ty) * fifos[n].depth for n in ("side", "gact", "gout"))
        private_bytes = 2 * size(ns["facc"]) + 2 * size(ns["f32"]) + sum(size(ns[n]) for n in ("fqk", "fvt", "fxn"))
        assert fifo_bytes + private_bytes + 0x1800 == 56192 <= Q36.L1_BUDGET
    else:
        assert d.AB_LANES == 32
        assert ns["facc"] == ns["f32"]
        assert ns["f_ab"].name == "glue_ab_e"


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


@pytest.mark.parametrize("hidden,heads,dense", [(5120, 48, True), (2560, 48, True), (4096, 32, True), (2560, 32, True),
                                              (2048, 16, True), (1024, 16, True), (2048, 32, False)])
def test_glue_stream_order_reuses_accumulators_and_continues_to_records(hidden, heads, dense):
    d = Q36.linear(spec(hidden, heads))
    lanes, rows = d.AB_LANES, d.AB_ROWS
    assert (lanes, rows) == ((64, 32) if heads > 32 else (32, 64))
    assert d.GLUE_HALF_OUTER == (hidden > 4096)          # three xn halves
    half_outer = dense and d.GLUE_HALF_OUTER
    ns = glue_namespace(d, hidden, dense)
    tiles = ns["AB_TILES"]
    assert sum(tiles) == d.AB_ELEMS
    rng = np.random.default_rng(81)
    x = rng.integers(-2, 3, hidden).astype(np.float64)
    weights = [rng.integers(-2, 3, (heads, hidden)).astype(np.float64) for _ in range(2)]
    # the packer's [hidden, lanes] W, lanes past `heads` zero; one side element is `rows` rows
    packed = []
    for w in weights:
        p = np.zeros((hidden, lanes))
        p[:, :heads] = w.T
        packed.append(p)

    def half(off, nt):
        chunk = np.zeros(2048)
        chunk[:nt * rows] = x[off:off + nt * rows]
        return chunk

    def tiles_of(p, off, nt):
        return [p[off + t * rows:off + (t + 1) * rows] for t in range(nt)]

    side = []
    if not dense:
        side.append(x.copy())
        for p in packed:
            side += tiles_of(p, 0, sum(tiles))
    elif half_outer:
        off = 0
        for nt in tiles:
            side.append(half(off, nt))
            for p in packed:
                side += tiles_of(p, off, nt)
            off += nt * rows
    else:
        for p in packed:
            off = 0
            for nt in tiles:
                side.append(half(off, nt))
                side += tiles_of(p, off, nt)
                off += nt * rows
    side.append("small")
    # Exactly 4 key tiles and heads/8 value tiles; a marker catches premature conv.
    conv_tiles = d.NT
    assert conv_tiles == 4 + heads // 8
    side.extend(["conv"] * (conv_tiles * 2))
    sin, ain, out = Input(side), Input([None] * (conv_tiles * 5)), Output()
    acc_a, acc_b = np.full(lanes, np.nan), np.full(lanes, np.nan)
    decay, beta = np.full(heads, np.nan), np.full(heads, np.nan)
    acc_ids, records, copies, small_calls = set(), [], [], []

    def copy(src, dst, offset=0):
        assert offset == 0
        copies.append(len(src))
        dst[:len(src)] = src

    def ab(w, xn, acc, tile, first=1):
        acc_ids.add(id(acc))
        assert acc.shape == (lanes,) and w.shape == (rows, lanes)
        if first and tile == 0:
            acc[:] = 0
        acc[:] += xn[tile * rows:(tile + 1) * rows] @ w

    def small(sm, a, b, dd, be):
        assert sm == "small"
        small_calls.append(1)
        for acc, w in ((a, weights[0]), (b, weights[1])):
            np.testing.assert_array_equal(acc[:heads], w @ x)
            assert not acc[heads:].any()                  # the zero lanes stay zero
        dd[:] = a[:heads]
        be[:] = b[:heads]

    def conv(*args):
        assert small_calls and not np.isnan(decay).any()
        assert args[5] == args[6] == "conv"

    def emit(qk, vt, dd, b, record, tile, lane):
        head = tile * 8 + lane
        records.append(head)
        assert dd[head] == weights[0][head] @ x
        assert b[head] == weights[1][head] @ x

    fn = worker(ns)
    fn(sin, ain, out, acc_a, acc_b, decay, beta, np.zeros(4096), np.zeros(1024),
       np.zeros(hidden if not dense else 2048), ab, small, conv, emit, copy)
    assert acc_ids == {id(acc_a), id(acc_b)}
    assert len(small_calls) == 1
    assert records == list(range(heads))
    assert out.count == 3 * conv_tiles + heads
    assert not sin.values and not ain.values
    # half-outer carries each xn half once; accumulator-outer once per accumulator
    assert len(copies) == (1 if not dense else len(tiles) if half_outer else 2 * len(tiles))
    assert not any(f.held for f in (sin, ain, out))


def test_ab_bank_count_is_geometry_not_model_name():
    s = spec27()
    for heads, banks in ((16, 1), (32, 1), (48, 2), (64, 2), (80, 3)):
        assert Q36.ab_banks(ModelSpec.from_dict(dict(s.to_dict(), lin_value_heads=heads))) == banks


def test_the_fused_glue_takes_48_heads_within_the_side_budget(monkeypatch):
    """A banked 48-head glue (#121) needs a third input DMA channel for its xn replay. The
    fused layer instead carries the xn halves on the side channel at 64 lanes, half-outer at
    three halves, so it composes with neither the probe flag nor the unvalidated override,
    and its fills fit the shim."""
    monkeypatch.delenv("OPEN_KERNELS_UNVALIDATED", raising=False)
    monkeypatch.delenv("OPEN_KERNELS_WIDE_GLUE_PROBE", raising=False)
    s = spec(5120, 48)
    d = Q35.recipe(s).linear
    assert (d.NHEAD, d.AB_LANES, d.GLUE_HALF_OUTER) == (48, 64, True)
    assert Q35.glue_side_fills(s) == 11 <= LIMITS["shim_fills"]


def side_fills():
    """The glue's side fills, cut out of lx.py's dense_sequence: from `ps = Pipeline(3)` to
    its last `ps.fill` (the conv taps)."""
    body = lx_function("dense_sequence").body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and
                 isinstance(n.targets[0], ast.Name) and n.targets[0].id == "ps")
    stop = max(i for i, n in enumerate(body) if "ps.fill(" in ast.unparse(n))
    return ast.Module(body=body[start:stop + 1], type_ignores=[])


@pytest.mark.parametrize("hidden,heads", [(5120, 48), (2560, 48), (4096, 32), (2560, 32), (1024, 16)])
def test_side_dma_issues_exactly_what_the_glue_consumes(hidden, heads):
    """Replay the host's side fills as 4 KB elements into the real glue_body. Every xn half
    and weight tile must arrive in the order the worker acquires it, each tile must be the
    one for the half it is multiplied against, and the count must be the recipe's
    `glue_side_fills` -- the number the shim budget is checked with."""
    s = spec(hidden, heads)
    d, layout = Q36.linear(s), Q35.layout(s)
    ns = glue_namespace(d, hidden)
    assert ns["AB_TILES"] == Q35.ab_tiles_per_half(s)
    fills = []

    class Pipe:
        def __init__(self, depth):
            pass

        def fill(self, endpoint, source, tap):
            total, off, size = tap
            assert endpoint == "side" and 0 <= off < off + size <= total and size % 4096 == 0
            fills.append((source, off, size))

    host = dict(layout.constants(), Pipeline=Pipe, AB_TILES=ns["AB_TILES"], HALF_OUTER=d.GLUE_HALF_OUTER,
                ELEM=4096, bt=lambda total, off, size: (total, off, size),
                side_p="side", a_act="act", a_consts="consts")
    exec(compile(side_fills(), str(LX), "exec"), host)
    assert len(fills) == Q35.glue_side_fills(s) <= LIMITS["shim_fills"]

    side = layout.C_SIDE
    stream = []
    for source, off, size in fills:
        if source == "act":
            assert size == 4096
            stream.append(("xn", (off - layout.A_XN) // 4096))
        elif off == side + layout.SIDE_SMALL:
            stream.append("small")
        elif off >= side + layout.SIDE_CONV:
            stream += ["conv"] * (size // 4096)
        else:
            stream += [("w", off - side + i * 4096) for i in range(size // 4096)]

    state = {"half": None}
    used = {}

    def copy(src, dst, offset=0):
        assert src[0] == "xn"
        state["half"] = src[1]

    def ab(w, xn, acc, tile, first=1):
        assert w[0] == "w"
        used.setdefault(id(acc), []).append((state["half"], tile, w[1]))

    def small(sm, *args):
        assert sm == "small"

    def conv(*args):
        assert args[5] == args[6] == "conv"

    acc_a, acc_b = np.zeros(d.AB_LANES), np.zeros(d.AB_LANES)
    sin, ain, out = Input(stream), Input([None] * (d.NT * 5)), Output()
    worker(ns)(sin, ain, out, acc_a, acc_b, np.zeros(heads), np.zeros(heads), None, None, None,
               ab, small, conv, lambda *args: None, copy)
    assert not sin.values and not ain.values
    tiles = ns["AB_TILES"]
    for acc, region in ((acc_a, layout.SIDE_ALPHA), (acc_b, layout.SIDE_BETA)):
        # tile t of half h is element sum(tiles[:h]) + t of that accumulator's region
        assert used[id(acc)] == [(h, t, region + (sum(tiles[:h]) + t) * 4096)
                                 for h, nt in enumerate(tiles) for t in range(nt)]


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
