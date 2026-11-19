# Traces: OPEN-FAMILY-LFM3 (block prefill), OPEN-PREFILL-BATCH
# (canonical spec: specs/open-engine/spec.md)
"""LFM3-1.7B block prefill (on by default; LFM3_BLOCK=0 drops it): the recipe's emission, pure functions over the HF config fixture of test_lfm3.py -- no
model, no hardware (the route is validated on hardware: .plans/lfm3-native.md "block prefill")."""
from __future__ import annotations

import pytest

from recipes import lfm3 as F

from test_lfm3 import LEGACY, legacy, spec  # noqa: F401 (legacy: autouse, every other switch off)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("LFM3_BLOCK", "1")


def test_on_by_default(monkeypatch):
    monkeypatch.delenv("LFM3_BLOCK")
    assert F.gemm_route(spec()) is not None


def test_off_leaves_decode_alone():
    s = spec()
    assert F.gemm_route(s) is None
    assert all("gemm_block" not in lt for lt in F.programs(s)["layer_types"].values())
    assert not any(k.startswith("gemm_") for k in F.builds(s))


def test_the_pack_plan_and_decode_kernels_do_not_move(monkeypatch):
    s = spec()
    plan, progs, builds = F.pack_plan(s), F.programs(s), F.builds(s)
    monkeypatch.setenv("LFM3_BLOCK", "1")
    assert F.pack_plan(s) == plan                      # the route reads the decode pool's bytes as they are
    on_p, on_b = F.programs(s), F.builds(s)
    for k, v in progs["kernels"].items():
        assert on_p["kernels"][k] == v
    for k, v in builds.items():
        assert on_b[k] == v


def test_one_gemm_block_a_layer_type(on):
    s = spec()
    g = F.programs(s)["layer_types"]
    assert sorted(g) == ["dense_local", "full_attention", "short_conv"]
    conv, full, loc = (g[n]["gemm_block"] for n in ("short_conv", "full_attention", "dense_local"))
    assert [conv["kind"], conv["mixer"], full["mixer"], loc["mixer"]] == ["lfm", "conv", "att", "att"]
    assert [p["kernel"] for p in conv["program"]] == ["gemm_n6144_k2048", "gemm_n2048_k2048"]
    assert [p["kernel"] for p in full["program"]] == ["gemm_n5120_k2048", "gemm_n2048_k2048"]   # q | gate | k | v, o
    assert conv["taps"] == 7 and conv["sandwich"] and full["gate"] and loc["gate"]
    assert (full["window"], loc["window"]) == (0, 256)
    assert full["rot"] == full["hd"] == 128 and full["qw"] == 2048 and full["kvw"] == 512


def test_rope_tables_are_the_zero_tail_progressions(on):
    s = spec()
    g = F.programs(s)["layer_types"]
    full, loc = g["full_attention"]["gemm_block"]["rope_inv_freq"], g["dense_local"]["gemm_block"]["rope_inv_freq"]
    assert len(full) == len(loc) == 64
    assert all(f > 0 for f in full[:32]) and all(f == 0.0 for f in full[32:])      # only the first 32 pairs turn
    assert full[0] == 1.0 == loc[0] and full[1] == pytest.approx(1e6 ** (-2 / 128))
    assert all(f > 0 for f in loc) and loc[1] == pytest.approx(1e4 ** (-2 / 128))


def test_the_mlp_reads_the_decode_pool_in_place(on):
    s = spec()
    L, Fm = F.layout(s), F.ffn(s)
    gb = F.programs(s)["layer_types"]["short_conv"]["gemm_block"]
    assert gb["ffn_slots"] == 5 and len(gb["ffn_program"]) == 10 and gb["ff"] == 1536
    assert [p["kernel"] for p in gb["ffn_program"][:2]] == ["gemm_n3072_k2048", "gemm_n2048_k1536"]
    w = gb["ffn_weights"]
    for e in range(5):
        # slot e's gate | up bands: 2 x 24 alternating 64-row bands, one GEMM operand of 3072 rows
        assert w[f"gug{e}_w"]["range"] == [e * Fm.EXPERT_UG, Fm.EXPERT_UG]
        assert w[f"gdn{e}_w"]["range"] == [Fm.POOL_DOWN + e * Fm.UP_BYTES, Fm.UP_BYTES]
    assert Fm.EXPERT_UG == 2 * (1536 // 64) * F.band_bytes(2048) and Fm.UP_BYTES == (2048 // 64) * F.band_bytes(1536)
    assert Fm.POOL_DOWN + 5 * Fm.UP_BYTES == L.POOL_MIX                  # the mixer starts right after the slots


def test_mixer_weights_are_contiguous_pool_ranges(on):
    s = spec()
    L = F.layout(s)
    g = F.programs(s)["layer_types"]
    qb, kb = 2048 // 64 * F.band_bytes(2048), 512 // 64 * F.band_bytes(2048)
    assert g["full_attention"]["gemm_block"]["weights"]["gqkv_w"]["range"] == [L.POOL_Q, 2 * qb + 2 * kb]
    assert L.POOL_GATE == L.POOL_Q + qb and L.POOL_K == L.POOL_GATE + qb and L.POOL_V == L.POOL_K + kb
    assert L.POOL_O == L.POOL_V + kb
    assert g["short_conv"]["gemm_block"]["weights"]["gin_w"]["range"] == [L.POOL_B, 3 * (2048 // 64) * F.band_bytes(2048)]
    assert (L.POOL_C, L.POOL_U) == (L.POOL_B + 2048 // 64 * F.band_bytes(2048), L.POOL_B + 2 * 2048 // 64 * F.band_bytes(2048))
