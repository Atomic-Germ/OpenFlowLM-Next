# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-QWEN35, OPEN-PACK-PLAN (canonical spec: specs/open-engine/spec.md)
"""Qwen3.5 dense: the derivation from the four published shapes, the composed
layout (the MoE recipe's attention half + the dense recipe's FFN half), the
programs, and the two new pack ops.

The config fixtures are the models' own `config.json`, unedited:
`config_qwen35_9b.json` is FLM's flat container config for
`Atomic-Germ/Qwen3.8-Distilled-9B-NPU2`; the 4B / 2B are the matching
`Atomic-Germ` containers; the 0.8B is `Qwen/Qwen3.5-0.8B`, whose shape lives in
a nested `text_config` with `model_type: qwen3_5_text`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from recipes import pack, qwen35 as Q35, qwen36moe as Q36
from recipes.catalogue import LIMITS, OpRangeError
from recipes.families import family_module
from recipes.load import load_spec
from recipes.manifest import manifest
from recipes.spec import FULL, LINEAR, ModelSpec, SpecError, hf_model_types

FIX = Path(__file__).resolve().parent / "fixtures"
SPEC_9B = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs" / "qwen35-9b.json"


def cfg(name: str) -> dict:
    return json.loads((FIX / f"config_qwen35_{name}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def spec9():
    return load_spec(SPEC_9B)


@pytest.fixture(scope="module")
def unvalidated(monkeypatch_module=None):
    return None


# ---------------------------------------------------------------- derivation
def test_the_four_published_shapes_derive():
    """The plan's size table, read off the models' own config.json."""
    want = {
        # name: (hidden, layers, intermediate, heads, kv, lin_value_heads)
        "9b":   (4096, 32, 12288, 16, 4, 32),
        "4b":   (2560, 32,  9216, 16, 4, 32),
        "2b":   (2048, 24,  6144,  8, 2, 16),
        "0p8b": (1024, 24,  3584,  8, 2, 16),
    }
    for name, (hid, nl, ff, nh, kvh, lvh) in want.items():
        s = ModelSpec.from_hf_config(cfg(name))
        assert s.family == "qwen35", name
        assert (s.hidden, s.num_layers, s.intermediate) == (hid, nl, ff), name
        assert (s.num_heads, s.num_kv_heads, s.lin_value_heads) == (nh, kvh, lvh), name
        assert s.num_experts == 0 and s.moe_intermediate == 0 and s.shared_expert_intermediate == 0, name
        # the MoE's layer pattern, head dim, partial RoPE, gate, conv, key heads and vocab
        assert s.layer_types == tuple(FULL if (l + 1) % 4 == 0 else LINEAR for l in range(nl)), name
        assert (s.head_dim, s.rotary_dim, s.rope_theta) == (256, 64, 1e7), name
        assert s.attn_gate is True and s.qk_norm is True, name
        assert (s.lin_key_heads, s.lin_key_dim, s.lin_value_dim, s.conv_kernel) == (16, 128, 128, 4), name
        assert s.vocab == 248320 and s.activation == "silu" and s.norm_eps == 1e-6, name


def test_the_nested_text_config_is_read():
    """Qwen's own repos wrap the text tower in `text_config` (model_type qwen3_5_text);
    FLM's containers flatten it (model_type qwen3_5). Both derive the same tower."""
    stock = cfg("0p8b")
    assert stock["model_type"] == "qwen3_5" and stock["text_config"]["model_type"] == "qwen3_5_text"
    a = ModelSpec.from_hf_config(stock)
    b = ModelSpec.from_hf_config(stock["text_config"])
    da, db = a.to_dict(), b.to_dict()
    da.pop("extra"), db.pop("extra")
    assert da == db
    assert sorted(hf_model_types("qwen35")) == ["qwen3_5", "qwen3_5_text"]


def test_the_moe_model_type_still_derives_to_qwen36moe():
    moe = {"model_type": "qwen3_5_moe", "hidden_size": 2048, "num_hidden_layers": 40,
           "full_attention_interval": 4, "vocab_size": 248320, "head_dim": 256,
           "num_attention_heads": 16, "num_key_value_heads": 2, "rope_theta": 1e7,
           "partial_rotary_factor": 0.25, "attn_output_gate": True,
           "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
           "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "num_experts": 256,
           "num_experts_per_tok": 8, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512}
    assert ModelSpec.from_hf_config(moe).family == "qwen36moe"
    assert family_module("qwen35") is Q35 and family_module("qwen3_5" if False else "qwen36moe") is Q36


def test_a_qwen35_config_without_a_dense_ffn_is_refused():
    bad = dict(cfg("9b"))
    bad.pop("intermediate_size")
    with pytest.raises(SpecError, match="intermediate_size"):
        ModelSpec.from_hf_config(bad)
    moe_ish = dict(cfg("9b"), num_experts=64)
    with pytest.raises(SpecError, match="num_experts"):
        ModelSpec.from_hf_config(moe_ish)


def test_the_checked_in_spec_is_the_9b_config(spec9):
    d = ModelSpec.from_hf_config(cfg("9b"), real_vocab=spec9.real_vocab).to_dict()
    got = spec9.to_dict()
    d.pop("extra"), got.pop("extra")
    assert d == got


# ---------------------------------------------------------- composed layout
SHARED_CONSTANTS = [
    "C_LNW", "C_SIDE", "C_NW", "GLUE_SIDE_BYTES", "SIDE_ALPHA", "SIDE_BETA", "SIDE_SMALL", "SIDE_CONV",
    "A_XN", "A_QKV", "A_Z", "A_VEC", "A_O", "A_OG", "A_OUT",
    "CA_LNW", "CA_POSTLN", "CA_META", "AA_XN", "AA_QG", "AA_KVN", "AA_OG", "AA_OUT",
    "S_ROWS", "S_HEAD_BYTES", "STATE_S_OFF", "STATE_BYTES", "KV_ROW", "PTAB_ROW",
    "LMHEAD_BAND_BYTES", "LMHEAD_BANDS", "LMHEAD_POOL_BYTES", "ELN", "E_A",
]


def _dense_twin(spec, intermediate: int):
    """The same spec with the MoE block swapped for a dense FFN of the given width."""
    d = spec.to_dict()
    d.update(family="qwen35", intermediate=intermediate, num_experts=0, experts_per_tok=0,
             moe_intermediate=0, shared_expert_intermediate=0)
    return ModelSpec.from_dict(d)


def test_the_attention_half_is_the_moe_recipes():
    """Swapping ONLY the FFN moves nothing in the DeltaNet / attention half: the 27B spec
    and a dense twin of it (an FFN narrow enough to keep 10 KB weight elements, so the
    element size is the same variable) give identical shared constants -- and they do
    because `qwen35.layout` IS `qwen36moe.layout(..., ffn='dense')`, not a re-derivation."""
    from recipes.load import default_spec

    moe = default_spec()
    twin = _dense_twin(moe, 4096)
    a, b = Q36.layout(moe), Q35.layout(twin)
    assert Q35.common(twin).CALL_BYTES == Q36.common(moe).CALL_BYTES     # like for like
    diff = {k: (getattr(a, k), getattr(b, k)) for k in SHARED_CONSTANTS
            if getattr(a, k) != getattr(b, k)}
    assert diff == {}


def test_the_composition_calls_the_moe_recipe_not_a_copy(spec9, monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    assert Q35.layout(spec9) == Q36.layout(spec9, 4096, "dense")
    assert Q35.common(spec9) == Q36.common(spec9, "dense")
    R = Q35.recipe(spec9)
    assert R.linear == Q36.linear(spec9) and R.attn == Q36.attn(spec9)   # unbranched by the FFN
    assert R.kind == "dense" and R.ffn == Q36.ffn_geometry(spec9)


def test_the_9b_deltanet_and_attention_constants(spec9, monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    R = Q35.recipe(spec9)
    C, L, D, A = R.common, R.layout, R.linear, R.attn
    # 5 KB weight elements: the 12288-wide down table leaves no room for two 10 KB ones
    assert C.PER_CALL == 1 and C.CALL_BYTES == 5120
    assert C.TAB_BYTES == 2 * 12288 + 12288 // 4 and C.KWIDE == 12288 and C.H_TAB_OFF == 0
    # DeltaNet on 5 KB elements: 10 rows per element, 13 slices, 130 padded rows
    assert (C.DN_ROWS, C.DN_SLICES, C.DN_PAD, C.DN_HEADS_PC, C.DN_DIM) == (10, 13, 130, 4, 128)
    assert L.S_ROWS == 130 and L.S_HEAD_BYTES == 130 * 128 * 4
    assert L.STATE_S_OFF == 3 * 8192 * 2 and L.STATE_BYTES == L.STATE_S_OFF + 32 * L.S_HEAD_BYTES
    # 8 KB norm elements at HID 4096 (R5: ln_fn's five-in / three-out form does not fit)
    assert L.ELN == 8192 and C.HID == 4096
    # the linear layer's GEMV band counts
    assert (D.QKV_PC, D.Z_PC, D.OUT_PC) == (16, 8, 8) and (D.QKV_DIM, D.VW, D.OUT_K) == (8192, 4096, 4096)
    assert (D.NCH, D.NHEAD, D.NT, D.AB_ELEMS, D.NG, D.KEY_WIDTH) == (8192, 32, 8, 64, 4, 2048)
    # the attention layer: 16 q heads over 4 kv heads, HD 256, 2 f32 heads per element, 4 og heads
    assert (A.NH, A.KVH, A.HD, A.ROT) == (16, 4, 256, 64)
    assert (A.QW, A.KVW, A.E_A, A.HPE, A.HPO) == (4096, 1024, 2048, 2, 4)
    assert (A.Q_AIN_ELEMS, A.K_AIN_ELEMS, A.OG_AOUT_ELEMS) == (8, 2, 4)
    assert L.KV_ROW == 4096 and L.PTAB_ROW == 2048


def test_the_ffn_half_is_the_dense_recipes_arithmetic(spec9, monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    R = Q35.recipe(spec9)
    F, L = R.ffn, R.layout
    hid, ff, n = 4096, 12288, 8
    assert F.FF == ff and F.UP_PC == ff // 64 // n and F.DOWN_PC == hid // 64 // n
    assert (F.MS_U, F.MS_G, F.MS_FLOATS) == (0, 64, 128)          # dense.geometry's scratch
    assert (F.XN_ELEMS, F.XM_ELEMS, F.H_ELEMS) == (2, 2, 12)
    assert R.linear.OG_ELEMS == 2 and R.attn.OG_ELEMS == 2   # og: bf16[VW] / bf16[QW]
    # the pool holds q4-sized up | gate | down, and both layer types see them at the same offsets
    up = Q36.q4_bytes(ff, hid)
    assert L.POOL_FFN_UP == 0 and L.POOL_FFN_GATE == up and L.POOL_FFN_DOWN == 2 * up
    assert Q36.q4_bytes(hid, ff) == up
    proj0 = 3 * up
    assert L.POOL_QKV == proj0 and L.POOL_Q == proj0                # the per-type projections follow
    assert L.POOL_BYTES % (1 << 20) == 0 and L.POOL_BYTES >= proj0 + Q36.q4_bytes(hid, 4096) * 2


def test_no_moe_block_and_no_router_anywhere(spec9, monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    L = Q35.layout(spec9)
    assert (L.C_RW, L.C_SGW, L.CA_RW, L.CA_SGW, L.A_ROUT, L.A_HP, L.AA_ROUT, L.AA_HP) == (0,) * 8
    assert (L.POOL_DOWN, L.POOL_SHARE_UP, L.POOL_SHARE_GATE, L.POOL_SHARE_DOWN) == (0, 0, 0, 0)
    m = manifest(spec9)
    assert "moe" not in m["layout"] and "rout_idx_off" not in m["layout"]
    for lt in m["layer_types"].values():
        names = [o["tensor"] for o in lt["pack"]["consts"] if "tensor" in o]
        assert not any("router" in t or "shared_expert" in t for t in names)
        assert all(o["op"] != "moeroute2" for o in lt["program"])


def test_one_instruction_stream_per_layer_type(spec9, monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    p = Q35.programs(spec9)
    lin, full = p["layer_types"][LINEAR], p["layer_types"][FULL]
    assert lin["program"] == [{"op": "run", "kernel": "lx",
                               "args": ["pool", "xres", "consts", "state", "act"]}]
    assert full["program"] == [{"op": "run", "kernel": "ax",
                                "args": ["pool", "xres", "consts", "state", "act", "ptab"]}]
    assert p["kernels"]["ax"]["patch"] == "attnpos" and p["kernels"]["lx"].get("patch", "") == ""
    assert [s["kernel"] for s in p["tail"]] == ["ln", "lm"]
    assert p["contexts"]["lm"] == "lm_head_q8/final.xclbin"        # the q8 head, as the MoE family
    b = Q35.builds(spec9)
    assert b["lx"]["build_dir"] == "layer_x/build_qwen35_lx_h4096"
    assert b["ax"]["build_dir"] == "layer_x/build_qwen35_ax_h4096"
    assert b["lm_head_q8"]["env"]["LMHEAD_K"] == "4096" and b["ln"]["env"]["LN_N"] == "4096"


def test_the_pack_plan_names_the_containers_tensors(spec9, monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    plan = Q35.pack_plan(spec9)
    lin = plan["layer_types"][LINEAR]
    ops = {o["tensor"]: o for o in lin["pool"] + lin["consts"] if "tensor" in o}
    assert all(t.startswith("model.layers.{l}.") for t in ops), sorted(ops)
    # the three tensors the MoE container does not have in this form. ssm_out_proj is q8
    # here and q4_1 in the 35B's container, and the plan is the SAME op either way: the
    # packers re-quantise a q8 source transparently (OPEN-PACK-PLAN).
    out = ops["model.layers.{l}.linear_attn.ssm_out_proj.weight"]
    assert out["op"] == "std_perm" and "chunk_bytes" not in out and out["in_dim"] == 4096
    assert out["nch"] == Q36.q4_chunks(4096, 4096) and out["dst"] == Q35.layout(spec9).C_WOUT
    for kind in ("alpha", "beta"):
        t = ops[f"model.layers.{{l}}.linear_attn.ssm_{kind}_proj.bf16.weight"]
        assert t["op"] == "transpose" and t["rows"] == 32 and t["cols"] == 4096 and t["elem"] == 2
    # the FFN comes through the dense recipe's std_perm
    for kind, in_dim in (("up", 4096), ("gate", 4096), ("down", 12288)):
        t = ops[f"model.layers.{{l}}.mlp.{kind}_proj.weight"]
        assert t["op"] == "std_perm" and t["in_dim"] == in_dim
    full = plan["layer_types"][FULL]
    fops = {(o["tensor"], o.get("chunk0", 0)): o for o in full["pool"]}
    nq = Q36.q4_chunks(4096, 4096)
    assert fops[("model.layers.{l}.self_attn.q_proj.weight", nq)]["nch"] == nq   # the fused q | gate split
    assert plan["lm_head"]["ops"][0]["op"] == "lmhead_q8"


# ------------------------------------------- the remaining sizes: 16 heads, and the 9B's halves
def _linear(name: str, monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    spec = ModelSpec.from_hf_config(cfg(name))
    return spec, Q35.recipe(spec)


@pytest.mark.parametrize("name,heads,nch,value_tiles", [("9b", 32, 8192, 4), ("4b", 32, 8192, 4),
                                                        ("2b", 16, 6144, 2), ("0p8b", 16, 6144, 2)])
def test_the_glue_emits_exactly_one_record_per_value_head(name, heads, nch, value_tiles, monkeypatch):
    """The 2B / 0.8B hang: designs/layer_x/lx.py looped the VALUE conv tiles KEY_TILES times,
    so the glue core emitted 4 x 8 = 32 records where the host drains one per value head. The
    two counts are equal only while there are 32 value heads; at 16 there are 2 value tiles
    against 4 key tiles."""
    spec, R = _linear(name, monkeypatch)
    D = R.linear
    assert (D.NHEAD, D.NCH, D.NT) == (heads, nch, nch // 1024)
    assert D.VALUE_TILE0 == 2 * D.KEY_WIDTH // 1024 == 4          # q then k, always 4 tiles
    assert D.NT - D.VALUE_TILE0 == value_tiles                     # what lx.py's VALUE_TILES is
    assert D.HEADS_PER_TILE == 8
    assert (D.NT - D.VALUE_TILE0) * D.HEADS_PER_TILE == D.NHEAD    # the records the host drains
    assert (D.NT - D.VALUE_TILE0 == D.VALUE_TILE0) == (heads == 32)


@pytest.mark.parametrize("name,heads,hid,ab_elems", [("9b", 32, 4096, 64), ("4b", 32, 2560, 40),
                                                     ("2b", 16, 2048, 32), ("0p8b", 16, 1024, 16)])
def test_the_alpha_beta_projection_is_padded_to_the_accumulator_width(name, heads, hid, ab_elems, monkeypatch):
    """dn_glue's accumulator and a W element's row are 32 lanes whatever the head count, so a
    16-head model's projection is packed [hid, 32] with columns 16..31 zero -- a 4 KB element
    stays 64 rows x 32 bf16. AB_ELEMS therefore counts padded elements: hid / 64."""
    spec, R = _linear(name, monkeypatch)
    D, L = R.linear, R.layout
    assert Q36.ab_lanes(spec) == 32 and D.NHEAD == heads
    assert D.AB_ELEMS == ab_elems == hid * 32 * 2 // 4096
    assert L.SIDE_BETA - L.SIDE_ALPHA == hid * 32 * 2 == D.AB_ELEMS * 4096
    # `small` is [A f32[heads] | dt_bias f32[heads]] -- dn_glue.h reads dt_bias at kNHead,
    # not at a fixed 32, and the two `put` caps are the real head count.
    ops = {o["tensor"]: o for o in Q35.pack_plan(spec)["layer_types"][LINEAR]["consts"] if "tensor" in o}
    pre = "model.layers.{l}.linear_attn."
    assert ops[pre + "ssm_a"]["dst"] == L.C_SIDE + L.SIDE_SMALL
    assert ops[pre + "ssm_dt.bias"]["dst"] == L.C_SIDE + L.SIDE_SMALL + heads * 4
    assert ops[pre + "ssm_a"]["cap"] == ops[pre + "ssm_dt.bias"]["cap"] == heads * 4
    for kind in ("alpha", "beta"):
        t = ops[pre + f"ssm_{kind}_proj.bf16.weight"]
        assert (t["rows"], t["cols"]) == (heads, hid)
        # the key appears ONLY when it changes something: a 32-head plan (and its manifest,
        # and its build key) must be the one the 4B and 9B already have.
        assert t.get("dst_rows") == (32 if heads != 32 else None)


@pytest.mark.parametrize("name,halves", [("9b", [32, 32]), ("4b", [32, 8]), ("2b", [32]), ("0p8b", [16])])
def test_the_projection_is_walked_in_4_kb_halves(name, halves, monkeypatch):
    """The 9B's glue core cannot hold bf16[4096]; the xn is re-streamed per 4 KB element and
    each half runs its own weight tiles. The halves are equal only when HID is a multiple of
    2048 -- at HID 2560 the second holds 512 rows, i.e. 8 tiles of 64."""
    spec, R = _linear(name, monkeypatch)
    assert Q35.xn_side_elems(spec) == len(halves) == R.linear.XN_SIDE_ELEMS
    assert Q35.ab_tiles_per_half(spec) == halves
    assert sum(halves) == R.linear.AB_ELEMS


@pytest.mark.parametrize("name,fills", [("9b", 10), ("4b", 10), ("2b", 6), ("0p8b", 6)])
def test_the_glue_side_fills_stay_inside_the_shim_budget(name, fills, monkeypatch):
    """Per accumulator, per half: the xn half then its weight tiles; then `small` and the conv
    taps. The recipe counts them so a too-wide model is refused here, not by a late IRON
    failure -- LIMITS['shim_fills'] is 13, so HID 6144 (three halves, 14 fills) is the wall."""
    spec, _ = _linear(name, monkeypatch)
    assert Q35.glue_side_fills(spec) == fills <= LIMITS["shim_fills"]
    wide = ModelSpec.from_dict(dict(spec.to_dict(), hidden=6144, intermediate=6144))
    assert Q35.glue_side_fills(wide) == 14
    with pytest.raises(OpRangeError, match="side channel needs 14 fills"):
        Q35.recipe(wide)


def test_the_catalogue_takes_sixteen_value_heads_and_still_refuses_eight(monkeypatch, capsys):
    """16 value heads entered the validated set with the 2B / 0.8B hardware pass (2026-09-07).
    The knob is validated per point, so 8 -- which nobody has built -- is still refused BY NAME,
    and OPEN_KERNELS_UNVALIDATED is what would let it through, saying so on stderr."""
    from recipes import catalogue

    catalogue.require("deltanet", heads=16, dim=128, key_heads=16, conv_kernel=4)
    for name in ("2b", "0p8b"):
        Q35.recipe(ModelSpec.from_hf_config(cfg(name)))
    assert capsys.readouterr().err == ""

    with pytest.raises(OpRangeError, match=r"deltanet: heads=8 is outside the validated set \{16, 32\}"):
        catalogue.require("deltanet", heads=8, dim=128, key_heads=16, conv_kernel=4)
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    monkeypatch.setattr(catalogue, "_WARNED", set())
    catalogue.require("deltanet", heads=8, dim=128, key_heads=16, conv_kernel=4)
    assert "deltanet: heads=8 is outside the validated set {16, 32}" in capsys.readouterr().err


def test_the_padded_transpose_zeroes_the_unused_lanes():
    """recipes/pack.py's `transpose` with dst_rows: [16, hid] -> [hid, 32], columns 16..31
    zero. src/open_qwen36/pools_test.cpp asserts the same law in C++."""
    rng = np.random.default_rng(13)
    src = rng.integers(1, 65536, (16, 512), dtype=np.uint16)
    dst = np.zeros(512 * 32 * 2, np.uint8)
    pack.apply_op({"op": "transpose", "tensor": "w", "dst": 0, "rows": 16, "cols": 512,
                   "elem": 2, "dst_rows": 32}, _Bytes({"w": src.tobytes()}), 0, dst)
    got = dst.view(np.uint16).reshape(512, 32)
    assert np.array_equal(got[:, :16], src.T)
    assert not got[:, 16:].any()
    # without dst_rows it is the plain transpose it always was
    d2 = np.zeros(512 * 16 * 2, np.uint8)
    pack.apply_op({"op": "transpose", "tensor": "w", "dst": 0, "rows": 16, "cols": 512, "elem": 2},
                  _Bytes({"w": src.tobytes()}), 0, d2)
    assert np.array_equal(d2.view(np.uint16).reshape(512, 16), src.T)
    with pytest.raises(ValueError, match="dst_rows"):
        pack.apply_op({"op": "transpose", "tensor": "w", "dst": 0, "rows": 16, "cols": 512,
                       "elem": 2, "dst_rows": 8}, _Bytes({"w": src.tobytes()}), 0, dst)


def test_every_published_size_composes_and_a_neighbour_nobody_built_does_not(spec9):
    """All four sizes ran on hardware on 2026-09-07, so their points -- lm_head_q8 K 1024 /
    4096, gemv_q4 K 3584, the 8/2 attention tuple, 16 value heads -- are in the catalogue and
    each composes with no override. A width nobody has built is still refused by the first
    template that cannot take it."""
    import dataclasses

    Q35.recipe(spec9)
    for name in ("9b", "4b", "2b", "0p8b"):
        Q35.recipe(ModelSpec.from_hf_config(cfg(name)))
    with pytest.raises(OpRangeError, match=r"ln: width=5120 is outside"):
        Q35.recipe(dataclasses.replace(spec9, hidden=5120))


# --------------------------------------------------------------- pack ops
def _q8_chunks(nch: int, rng) -> np.ndarray:
    """Synthetic q8 chunks: 256 bf16 scales then 8192 int8 codes."""
    out = np.zeros((nch, 8704), np.uint8)
    sc = (rng.random((nch, 256), np.float32) * 0.02 + 1e-3).astype(np.float32)
    out[:, :512] = ((sc.view(np.uint32) + 0x7FFF + ((sc.view(np.uint32) >> 16) & 1)) >> 16
                    ).astype(np.uint16).view(np.uint8).reshape(nch, 512)
    out[:, 512:] = rng.integers(-128, 128, (nch, 8192), dtype=np.int8).view(np.uint8)
    return out


def test_requant_q4_1_is_the_optimal_q4_1_of_the_dequantised_q8():
    from q4nx import bf16_to_f32, dq_chunks_q4_1, dq_chunks_q8

    rng = np.random.default_rng(7)
    src = _q8_chunks(12, rng)
    got = pack.requant_q4_1(src)
    assert got.shape == (12, 5120) and got.dtype == np.uint8
    a = dq_chunks_q8(src).astype(np.float64)
    b = dq_chunks_q4_1(got).astype(np.float64)
    # d is the block's stored scale; every value is within d/2 of its q4_1 reading
    d = bf16_to_f32(np.ascontiguousarray(got[:, :512]).view(np.uint16))
    r = np.arange(32)[:, None, None]; bc = np.arange(8)[None, :, None]; i = np.arange(32)[None, None, :]
    dd = d[:, (bc * 32 + r + 0 * i).reshape(-1)].reshape(-1, 32, 8, 32).astype(np.float64)
    assert np.all(np.abs(a - b) <= dd / 2 + 1e-30)
    # and it is a real quantisation, not a copy: 4 bits over a 15-step range
    assert 0.0 < np.abs(a - b).max() and np.abs(a - b).max() < dd.max()
    nib = np.frombuffer(got[:, 1024:].tobytes(), np.uint8)
    assert nib.max() > 0 and (nib >> 4).max() <= 15


def test_transpose_gives_the_bytes_the_moe_packer_copies():
    rng = np.random.default_rng(11)
    src = rng.integers(0, 65536, (32, 4096), dtype=np.uint16)
    dst = np.zeros(4096 * 32 * 2, np.uint8)
    pack.apply_op({"op": "transpose", "tensor": "w", "dst": 0, "rows": 32, "cols": 4096, "elem": 2},
                  _Bytes({"w": src.tobytes()}), 0, dst)
    assert np.array_equal(dst.view(np.uint16).reshape(4096, 32), src.T)


class _Bytes:
    def __init__(self, d):
        self.d = d

    def raw(self, name):
        return self.d[name]


def _shared_q8_vector(nch: int) -> np.ndarray:
    """The vector src/open_qwen36/pools_test.cpp builds, byte for byte: per chunk one LCG
    seeded from the chunk index, 256 bf16 scales with exponent 0x76 (finite and positive,
    so the requantizer sees real ranges) then 8192 int8 codes."""
    M = 0xFFFFFFFF
    out = np.zeros((nch, 8704), np.uint8)
    for c in range(nch):
        s = (0x9E3779B9 * (c + 1)) & M
        sc = np.zeros(256, np.uint16)
        for i in range(256):
            s = (s * 1664525 + 1013904223) & M
            sc[i] = 0x3B00 | (s >> 24)
        out[c, :512] = sc.view(np.uint8)
        for i in range(512, 8704):
            s = (s * 1664525 + 1013904223) & M
            out[c, i] = s >> 24
    return out


def _fnv1a(b) -> int:
    h = 1469598103934665603
    for x in bytes(b):
        h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def test_the_numpy_and_cpp_packers_agree_byte_for_byte():
    """The gate on the two implementations: both build the same input and both assert the
    same FNV-1a of the output. The C++ half is src/open_qwen36/pools_test.cpp (built and run
    by src/open_qwen36/build.cmd); if either side changes, one of the two fails."""
    out = pack.requant_q4_1(_shared_q8_vector(12))
    assert _fnv1a(out.tobytes()) == 0x4AC3BABAD266DD49, "requant_q4_1 changed; update pools_test.cpp too"
    t = np.frombuffer(bytes(((i * 37 + 11) & 0xFF) for i in range(32 * 64 * 2)), np.uint8)
    td = np.ascontiguousarray(t.reshape(32, 64, 2).transpose(1, 0, 2)).reshape(-1)
    assert _fnv1a(td.tobytes()) == 0xB27D0B6F7149FD83, "transpose changed; update pools_test.cpp too"


def test_the_pack_ops_refuse_a_missing_size():
    dst = np.zeros(64, np.uint8)
    m = _Bytes({"w": b"\0" * 64})
    with pytest.raises(ValueError, match="nch"):
        pack.apply_op({"op": "std_perm", "tensor": "w", "dst": 0, "in_dim": 4096}, m, 0, dst)
    with pytest.raises(ValueError, match="rows"):
        pack.apply_op({"op": "transpose", "tensor": "w", "dst": 0, "cols": 4}, m, 0, dst)


# --------------------------------------------------------------- manifest
def test_the_manifest_fixture_is_current(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    from make_fixtures import FIXTURE_Q35, fixture_manifest_q35

    assert FIXTURE_Q35.is_file(), "run make_fixtures.py"
    assert json.loads(FIXTURE_Q35.read_text(encoding="utf-8")) == fixture_manifest_q35(), \
        "fixtures/manifest_qwen35_9b.json is stale: run make_fixtures.py"


def test_the_manifest_carries_what_the_engine_needs(spec9, monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    m = manifest(spec9)
    assert m["family"] == "qwen35" and len(m["layers"]) == 32
    lay = m["layout"]
    assert lay["hidden"] == 4096 and lay["vocab"] == 248320 and lay["chunk_bytes"] == 5120
    assert lay["lmhead_chunk_bytes"] == 8704 and lay["kv_row"] == 4096 and lay["ptab_row"] == 2048
    assert lay["rotary_dim"] == 64 and len(lay["rope_inv_freq"]) == 32
    assert m["hf_config_check"]["model_type"] == ["qwen3_5", "qwen3_5_text"]
    assert m["hf_config_check"]["intermediate_size"] == 12288
    assert "num_experts" not in m["hf_config_check"]
    for lt, d in m["layer_types"].items():
        assert len(d["program"]) == 1 and d["program"][0]["op"] == "run"
        assert d["pack"]["pool"] and d["pack"]["consts"]
    assert m["layer_types"][FULL]["buffers"]["state"] == {"kind": "kv", "row": 4096}
    assert m["layer_types"][LINEAR]["buffers"]["state"]["kind"] == "linear"
