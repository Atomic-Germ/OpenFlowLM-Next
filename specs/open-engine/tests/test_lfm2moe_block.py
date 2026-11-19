# Traces: OPEN-FAMILY-LFM2MOE (block prefill), OPEN-PREFILL-BATCH
# (canonical spec: specs/open-engine/spec.md)
"""LFM2.5-24B-A2B block prefill (on by default; LFM2MOE_BLOCK=0 drops it): the recipe's emission and the expert kernel's host layouts. Pure
functions over the HF config fixture of test_lfm2moe.py and synthetic arrays -- no model, no hardware (the kernel and the
route are validated on hardware: designs/lfm2moe_batch/make_test.py + compare.py, the model gate in .plans/lfm2moe-native.md)."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from recipes import lfm2moe as F

from test_lfm2moe import spec

D = Path(__file__).resolve().parents[3] / "open_kernels" / "designs" / "lfm2moe_batch"


def _layout():
    # by path under its own name: several designs ship a layout.py
    sp = importlib.util.spec_from_file_location("lfm2moe_batch_layout", D / "layout.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("LFM2MOE_BLOCK", "1")


def test_on_by_default(monkeypatch):
    monkeypatch.delenv("LFM2MOE_BLOCK", raising=False)
    assert F.gemm_route(spec()) is not None


def test_off_leaves_the_sequential_emission_alone(monkeypatch):
    monkeypatch.setenv("LFM2MOE_BLOCK", "0")
    s = spec()
    assert F.gemm_route(s) is None
    assert all("gemm_block" not in lt for lt in F.programs(s)["layer_types"].values())
    assert not any(k.startswith(("gemm_", "mb_")) for k in F.builds(s))
    assert not any(o["tensor"].endswith("ffn_up.weight") and o["dst"] >= F.layout(s).POOL_MIX // 4
                   for lt in F.pack_plan(s)["layer_types"].values() for o in lt["pool"])


def test_the_route_is_one_gemm_block_a_layer_type(on):
    s = spec()
    g = F.programs(s)["layer_types"]
    assert sorted(g) == ["full_attention", "short_conv", "short_conv_dense"]
    conv, att, dense = (g[n]["gemm_block"] for n in ("short_conv", "full_attention", "short_conv_dense"))
    assert [conv["kind"], conv["mixer"], att["mixer"]] == ["lfm", "conv", "att"]
    assert [p["kernel"] for p in conv["program"]] == ["gemm_n6144_k2048", "gemm_n2048_k2048"]
    assert [p["kernel"] for p in att["program"]] == ["gemm_n3072_k2048", "gemm_n2048_k2048"]
    assert [p["kernel"] for p in dense["ffn_program"]] == ["gemm_n23552_k2048", "gemm_n2048_k11776"]
    assert "moe_batch" in conv and "route" in conv and "ffn_program" not in conv
    assert "moe_batch" not in dense and dense["ff"] == 11776
    assert att["rot"] == att["hd"] == 64 and att["qw"] == 2048 and att["kvw"] == 512
    assert conv["taps"] == 3


def test_gemm_weights_are_contiguous_pool_runs(on):
    s = spec()
    plan = F.pack_plan(s)["layer_types"]
    L = F.layout(s)
    gb = F.programs(s)["layer_types"]["short_conv_dense"]["gemm_block"]
    pool = plan["short_conv_dense"]["pool"]
    ops = [pool[i] for i in gb["weights"]["gin_w"]["ops"]]
    assert [o["dst"] for o in ops] == [L.POOL_B, L.POOL_C, L.POOL_U]     # B | C | u are in_proj's three row slices
    up, gate = (pool[i] for i in gb["ffn_weights"]["gffn_ug_w"]["ops"])
    (down,) = (pool[i] for i in gb["ffn_weights"]["gffn_down_w"]["ops"])
    bb = F.band_bytes(2048)
    assert gate["dst"] == up["dst"] + 11776 // 64 * bb                      # [up | gate]: one GEMM of 2 x 11776 rows
    assert down["dst"] == gate["dst"] + 11776 // 64 * bb
    # the copies sit between the 23 sequential slots and the mixer: no overlap with either
    seq_end = F.ffn(s, dense=True).POOL_DOWN + F.ffn(s, dense=True).POOL_SLOTS * F.ffn(s, dense=True).UP_BYTES
    assert seq_end <= up["dst"] and down["dst"] + 2048 // 64 * F.band_bytes(11776) <= L.POOL_MIX


def test_expert_ladder_and_pool_law(on):
    s = spec()
    mb = F.moe_batch(s)
    assert list(mb["moe_batch"]["kernels"]) == ["64", "32", "16", "8"]
    assert mb["moe_batch"]["y_layout"] == "rbg_half_row" and mb["moe_batch"]["nt"] == 8
    b = mb["builds"]["mb_s64"]
    Fm = F.ffn(s, dense=False)
    assert b["design"] == "lfm2moe_batch/mbl.py"
    assert (b["env"]["MBL_HID"], b["env"]["MBL_FF"], b["env"]["MBL_EXPERTS"]) == ("2048", "1536", "64")
    assert int(b["env"]["MBL_POOL_DOWN"]) == Fm.POOL_DOWN == 64 * Fm.EXPERT_UG
    # mbl's taps (designs/lfm2moe_batch/mbl.py) assume the fl_q4 law: gate / up bands 2 BB_H apart, an expert's
    # gate | up block twice its down
    bb_h = 2 * (2048 // 256) * 5120
    assert Fm.EXPERT_UG == 2 * (1536 // 64) * bb_h and Fm.UP_BYTES == (2048 // 64) * 2 * (1536 // 256) * 5120
    assert Fm.EXPERT_UG == 2 * Fm.UP_BYTES
    assert mb["globals"]["mb_y"] == 64 * 2048 * 8 * 4


def test_mbl_l1_budget_and_per_band_silu():
    src = (D / "mbl.py").read_text()
    # six gate / up bands do not fit beside the 24 KB h copy; the kernel runs them per rbg with one band of u / g
    assert "UG_BANDS = 1" in src and "CHUNK = 5120" in src
    # the L1 formula of mbl.py at LFM's dimensions (NT 8): stack, a double-buffered band-k-group, u / g, y, output
    # elements, the B fifo and the whole h
    NT, BAND, FF = 8, 2 * 5120, 1536
    l1 = 0x1000 + 2 * BAND + 2 * (1 * 64 * NT * 4) + 64 * NT * 4 + 2 * (32 * NT * 4) + 5 * (64 * NT * 2) + FF * NT * 2
    assert l1 == 62464 <= 64 << 10
    six = 0x1000 + 2 * BAND + 2 * (6 * 64 * NT * 4) + 64 * NT * 4 + 2 * (32 * NT * 4) + 5 * (64 * NT * 2) + FF * NT * 2
    assert six > 64 << 10


def test_y_layout_is_a_permutation_inverse_at_lfm_dimensions():
    lay = _layout()
    assert (lay.HID, lay.FF) == (2048, 1536)
    S = 2
    dev = np.arange(S * lay.HID * lay.NT, dtype=np.float32)
    y = lay.y_from_dev(dev, lay.HID)
    assert y.shape == (S, lay.NT, lay.HID)
    assert sorted(y.reshape(-1).tolist()) == dev.tolist()
    rbg, half, r = 5, 1, 3
    start = ((rbg * 2 + half) * 4 + r) * 256
    band = 4 * rbg + r
    assert np.isin(dev[start:start + 256], y[0, :, band * 64 + 32 * half:band * 64 + 32 * half + 32]).all()


def test_the_runtime_scatter_matches_the_layout():
    """core.cpp moe_block (y_layout rbg_half_row) written out in numpy against layout.y_from_dev."""
    lay = _layout()
    NT, hid, NG = lay.NT, lay.HID, lay.NT // 8
    rng = np.random.default_rng(0)
    dev = rng.standard_normal(hid * NT).astype(np.float32)
    want = lay.y_from_dev(dev, hid)[0]                           # [token, row]
    for tt in range(NT):
        got = np.zeros(hid, np.float32)
        for band in range(hid // 64):
            rbg, r = band // 4, band % 4
            for half in range(2):
                el = ((rbg * 2 + half) * 4 + r) * 256
                for gg in range(2):
                    for p in range(2):
                        v = dev[el + ((gg * 2 + p) * NG + tt // 8) * 64 + (tt % 8) * 8:][:8]
                        base = band * 64 + (2 * half + gg) * 16 + p
                        got[base + 2 * np.arange(8)] = v
        assert np.array_equal(got, want[tt]), tt
