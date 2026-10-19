"""Execute production worker/schedule bodies with checked streams before NPU builds."""
import ast
from pathlib import Path
import sys
import importlib.util

import numpy as np
import pytest

from test_qwen35_wide_glue import Input, Output

ROOT = Path(__file__).resolve().parents[3]


def nested(path, design, name, ns):
    tree = ast.parse(path.read_text())
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == design)
    fn = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


def test_conv_worker_consumes_ab_and_emits_exactly_48_records():
    path = ROOT / "open_kernels/designs/wide_deltanet/glue.py"
    sin = Input(["ab"] + ["conv"] * 20)
    ain, out = Input([None] * 50), Output()
    decay, beta = np.full(48, np.nan), np.full(48, np.nan)
    seen = []
    def load(ab, d, b):
        assert ab == "ab"
        d[:] = np.arange(48) + 100
        b[:] = np.arange(48) + 200
    def conv(*args):
        assert args[5] == args[6] == "conv"
        assert np.isfinite(decay).all() and np.isfinite(beta).all()
    def emit(qk, vt, d, b, record, tile, lane):
        head = tile * 8 + lane
        seen.append(head)
        assert d[head] == head + 100 and b[head] == head + 200
    ns = dict(range_=range, KEY_TILES=4, VALUE_TILES=6, HEADS_PER_TILE=8)
    fn = nested(path, "wide_glue", "core_body", ns)
    fn(sin, ain, out, decay, beta, np.zeros(4096), np.zeros(1024), load, conv, emit)
    assert seen == list(range(48)) and out.count == 30 + 48
    assert not sin.values and not ain.values and sin.held == ain.held == out.held == 0


@pytest.mark.parametrize("heads,cores", [(16, 8), (32, 8), (48, 8)])
def test_recurrence_two_pass_streams_cover_every_head(heads, cores):
    path = ROOT / "open_kernels/designs/deltanet/dn_step.py"
    # The compile-time head count must determine every BO and worker count.
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dn_step")
    assert "n_heads" in [a.arg for a in fn.args.kwonlyargs]
    assert not any(isinstance(n, ast.Name) and n.id == "HEADS" for n in ast.walk(fn))
    seen = []
    for core in range(cores):
        first = core * (heads // cores)
        sin = Input([h for h in range(first, first + heads // cores) for _ in range(16)])
        vin = Input(list(range(first, first + heads // cores)))
        sout, oout = Output(), Output()
        def p1(s, v, t, k, q, block):
            assert s == v
            if block == 0:
                seen.append(v)
        def p2(s, so, v, t, ob, k, q, d, block):
            assert s == v
        ns = dict(range_=range, heads_per_core=heads // cores, NBLK=8, D=128)
        body = nested(path, "dn_step", "core_body", ns)
        body(sin, vin, sout, oout, None, np.zeros(128), None, None, None, p1, p2)
        assert not sin.values and not vin.values
        assert sout.count == heads // cores * 8 and oout.count == heads // cores
    assert seen == list(range(heads))


def reference_module():
    sys.path.insert(0, str(ROOT / "utilities"))
    import wide_deltanet_reference
    return wide_deltanet_reference


def test_reference_grouping_and_conv_state_shift():
    ref = reference_module()
    from ml_dtypes import bfloat16
    # Last tap alone: each channel has a distinct value, exposing every head mapping.
    qkv = np.linspace(0.1, 0.9, 10240, dtype=np.float32)
    weights = np.zeros((4, 10240), bfloat16)
    weights[3] = 1
    state = np.stack([np.full(10240, n, bfloat16) for n in (1, 2, 3)])
    ab = np.stack([np.zeros(48), np.zeros(48), np.linspace(.5, .9, 48), np.linspace(.1, .8, 48)])
    ns, records = ref.glue_reference(qkv, state, weights, ab)
    np.testing.assert_array_equal(ns[:2], state[1:])
    np.testing.assert_array_equal(ns[2], qkv.astype(bfloat16))
    activated = qkv.astype(np.float64) / (1 + np.exp(-qkv.astype(np.float64)))
    for head in range(48):
        kh = head // 3
        q = activated[kh * 128:(kh + 1) * 128]
        k = activated[2048 + kh * 128:2048 + (kh + 1) * 128]
        np.testing.assert_allclose(records[head, :128], k / np.sqrt(k @ k + 1e-6))
        np.testing.assert_allclose(records[head, 128:256], q / np.sqrt(q @ q + 1e-6))
        np.testing.assert_allclose(records[head, 256:384], activated[4096 + head * 128:4096 + (head + 1) * 128])
    np.testing.assert_array_equal(records[:, 384:386], ab[2:].T)
    assert not records[:, 386:].any()


def test_reference_recurrence_uses_persistent_state_and_correct_orientation():
    ref = reference_module()
    state = np.zeros((48, 128, 128), np.float64)
    vec = np.zeros((48, 512), np.float64)
    vec[:, 2] = 1  # k
    vec[:, 128 + 2] = 1  # q
    vec[:, 256 + 7] = np.arange(1, 49)  # v, deliberately a different index
    vec[:, 384] = .75
    vec[:, 385] = .5
    for factor in (.5, .6875, .7578125):
        state, out = ref.step_reference(state, vec)
        expected = factor * np.arange(1, 49)
        np.testing.assert_allclose(state[:, 2, 7], expected)
        np.testing.assert_allclose(out[:, 7], expected / np.sqrt(128))
        assert np.count_nonzero(state) == np.count_nonzero(out) == 48


def test_metric_rejects_wrong_shapes_nan_and_changed_tail():
    ref = reference_module()
    want = np.ones((48, 512))
    assert ref.metric(want, want, .99999)["passed"]
    for bad in (want[:32], np.full_like(want, np.nan)):
        assert not ref.metric(bad, want, .99999)["passed"]
    wrong = want.copy()
    wrong[47, 0] = 10
    assert not ref.metric(wrong, want, .99999)["passed"]


@pytest.mark.parametrize("heads,cores,act,vec_off,o_off", [
    (47, 8, 0, 0, 0), (48, 0, 0, 0, 0), (48, 9, 0, 0, 0),
    (0, 8, 0, 0, 0), (48, 8, 48 * 512 - 1, 0, 0),
    (48, 8, 0, 1, 0), (48, 8, 30000, -1, 0), (48, 8, 30000, 0, 30000),
])
def test_recurrence_rejects_incomplete_heads_or_out_of_bounds_buffers(heads, cores, act, vec_off, o_off):
    path = ROOT / "open_kernels/designs/deltanet/dn_step.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dn_step")
    guards = []
    for stmt in fn.body:
        if not isinstance(stmt, ast.If):
            break
        guards.append(stmt)
    ns = dict(n_heads=heads, n_cores=cores, act_f32=act, vec_off=vec_off, o_off=o_off, VEC=512, D=128)
    with pytest.raises(ValueError):
        exec(compile(ast.Module(body=guards, type_ignores=[]), str(path), "exec"), ns)


def test_chain_program_carries_device_state_and_never_loads_reference(tmp_path, monkeypatch):
    pytest.importorskip("ml_dtypes")
    sys.path.insert(0, str(ROOT / "utilities"))
    path = ROOT / "utilities/test-wide-deltanet-chain.py"
    spec = importlib.util.spec_from_file_location("wide_chain_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "DESIGN", tmp_path)
    # Byte placeholders are enough to test fixture construction; no NPU claim.
    for name in ("build_ab_h2560", "build_glue", "build_step"):
        folder = tmp_path / name
        folder.mkdir()
        for file in ("final.xclbin", "insts.bin"):
            (folder / file).write_bytes(b"fixture-only")
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test-only")
    out = tmp_path / "fixtures"
    module.prepare(out, 2560, 2)
    commands = (out / "chain.cfg").read_text().splitlines()
    assert not any("ref-" in line for line in commands)
    assert commands.count("run ab abs xn ab") == 4
    assert commands.count("run glue side qkv cs conv vec") == 4
    assert commands.count("run step s vec state output") == 4
    assert commands.count("copy s 0 state 0 3145728") == 4
    assert commands.count("copy cs 0 conv 0 61440") == 4
    assert sum(line.startswith("load s ") for line in commands) == 2
    assert sum(line.startswith("load cs ") for line in commands) == 2
    with pytest.raises(ValueError, match="multiple tokens"):
        module.prepare(out, 2560, 1)


@pytest.mark.parametrize('scale', [1.0, 1e-25, 1e-150, 1e150])
def test_metric_is_scale_invariant_for_nonzero_vectors(scale):
    module = reference_module()
    want = np.array([1., -2., 3.])
    got = want + np.array([1e-5, -2e-5, 1e-5])
    baseline = module.metric(got, want, .9999999)
    scaled = module.metric(got * scale, want * scale, .9999999)
    assert scaled['passed']
    assert scaled['cosine'] == pytest.approx(baseline['cosine'], abs=1e-14)
    assert scaled['maxrel'] == pytest.approx(baseline['maxrel'], rel=1e-10)


def test_metric_does_not_mask_tiny_relative_errors_or_nonzero_against_zero():
    module = reference_module()
    want = np.array([1e-100, 2e-100])
    assert not module.metric(want * 1.01, want, .9999999)['passed']
    assert module.metric(np.zeros(2), np.zeros(2), .9999999)['passed']
    assert not module.metric(np.zeros(2), want, .9999999)['passed']
    assert not module.metric(want, np.zeros(2), .9999999)['passed']
