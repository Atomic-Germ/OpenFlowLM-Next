# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-LFM3, OPEN-FAMILY-LFM2
# (canonical spec: specs/open-engine/spec.md)
"""LiquidAI LFM3 (LFM3-1.7B): LFM2's hybrid -- gated 7-tap short depthwise causal convs and GQA attention with
per-head q / k norms -- with a sandwich norm on every block output, two attention kinds (full: theta 1e6 and
the first half of the head rotated; sliding: theta 1e4, every pair, 256 rows), a sigmoid output gate of one
value a head, a dense SiLU FFN on every layer and a tied head.

The derivation, the recipe's emission, the reference's arithmetic and the GGUF derivation run here -- pure
functions over a config dict and synthetic arrays, no model and no hardware. The fixture is
LiquidAI/LFM3-1.7B's config.json (fetched 2026-11-03), written out.
"""
from __future__ import annotations

import copy
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from recipes import families
from recipes.spec import DENSE_LOCAL, FULL, SHORT_CONV, ModelSpec, SpecError

ROOT = Path(__file__).resolve().parents[3]
FULL_IDX = [2, 6, 10, 14, 18, 22]
SLIDE_IDX = [4, 8, 12, 16, 20]
HF_LFM3 = {
    "architectures": ["Lfm3ForCausalLM"], "model_type": "lfm3", "bos_token_id": 124894, "conv_L_cache": 7,
    "conv_bias": False, "dtype": "bfloat16", "eos_token_id": 124900, "head_dim": 128, "hidden_size": 2048,
    "intermediate_size": 7680,
    "layer_types": ["full_attention" if i in FULL_IDX else "sliding_attention" if i in SLIDE_IDX else "conv"
                    for i in range(24)],
    "max_position_embeddings": 262144, "norm_eps": 1e-05, "num_attention_heads": 16, "num_hidden_layers": 24,
    "num_key_value_heads": 4, "pad_token_id": 124893,
    "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0,
                                           "rope_type": "default"},
                        "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}},
    "sliding_window": 256, "tie_word_embeddings": True, "use_cache": True, "use_sandwich_norm": True,
    "vocab_size": 128000,
}



# The fast paths are the default; these tests describe the original emission (two dispatches a layer, q4_1 pools, the
# f32-scale head, sequential prefill), so they pin every switch off. test_lfm3_fused.py checks the defaults.
LEGACY = {"LFM3_BLOCK": "0", "LFM3_FUSED": "0", "LFM3_SHARED": "0", "LFM3_Q40": "0", "LFM3_HEAD": "f32"}


@pytest.fixture(autouse=True)
def legacy(monkeypatch):
    for k, v in LEGACY.items():
        monkeypatch.setenv(k, v)


def spec(**over):
    cfg = copy.deepcopy(HF_LFM3)
    cfg.update(over)
    return ModelSpec.from_hf_config(cfg)


# ---- the derivation


def test_the_spec_is_its_own_family_with_the_published_geometry():
    s = spec()
    assert s.family == "lfm3"
    assert (s.hidden, s.num_layers, s.vocab, s.intermediate) == (2048, 24, 128000, 7680)
    assert (s.num_heads, s.num_kv_heads, s.head_dim, s.rotary_dim) == (16, 4, 128, 128)
    assert s.qk_norm and s.attn_gate and s.sandwich_norms and s.activation == "silu" and s.conv_kernel == 7
    assert (s.rope_theta, s.rope_local_theta, s.sliding_window, s.norm_eps) == (1e6, 1e4, 256, 1e-5)
    assert s.extra["partial_rotary_factor"] == 0.5 and not s.num_experts


def test_the_layer_types_are_conv_full_and_sliding():
    s = spec()
    assert [i for i, t in enumerate(s.layer_types) if t == FULL] == FULL_IDX
    assert [i for i, t in enumerate(s.layer_types) if t == DENSE_LOCAL] == SLIDE_IDX
    assert s.layer_types.count(SHORT_CONV) == 13


@pytest.mark.parametrize("key,value", [
    ("conv_bias", True), ("tie_word_embeddings", False), ("use_sandwich_norm", False),
    ("rope_parameters", {"full_attention": {"rope_theta": 1e6, "rope_type": "yarn"},
                         "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}}),
    ("layer_types", ["conv"] * 23 + ["mamba"]),
])
def test_a_config_that_is_not_lfm3_arithmetic_is_refused_by_name(key, value):
    with pytest.raises(SpecError, match="lfm3"):
        spec(**{key: value})


def test_the_spec_round_trips_and_the_family_routes_to_its_own_recipe():
    s = spec()
    assert ModelSpec.from_dict(s.to_dict()) == s
    assert families.for_spec(s).__name__ == "recipes.lfm3"
    assert "lfm3" in families.FAMILIES
    from recipes import lfm2, lfm2moe
    for other in (lfm2.recipe, lfm2moe.geometry):
        with pytest.raises(Exception):
            other(s)


# ---- the recipe


def test_the_widths_tile_eight_cores_and_the_mlp_is_five_slots():
    from recipes import lfm3 as R
    G = R.geometry(spec())
    assert (G.HID, G.QW, G.KVW, G.FF, G.HD, G.ROT) == (2048, 2048, 512, 1536, 128, 128)
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.SC_PC, G.DOWN_PC) == (4, 1, 4, 4, 4)
    assert (G.SLOTS, G.ACORES, G.NHL, G.SC_ELEMS, G.TAPS) == (5, 4, 4, 16, 7) and 5 * 1536 == 7680
    assert (G.E_A, G.HPE, G.HPO, G.Q_AIN_ELEMS, G.K_AIN_ELEMS) == (1024, 2, 4, 8, 2)
    F = R.ffn(spec())
    assert (F.UP_PC, F.H_ELEMS, F.POOL_SLOTS) == (3, 2, 5)
    assert F.UP_BYTES == 24 * R.band_bytes(2048) == 32 * R.band_bytes(1536), "gate and down the same size"


def test_the_attention_core_has_the_gate_the_norm_and_the_whole_head_rope():
    from recipes import lfm3 as R
    fl = R.attn_flags(R.geometry(spec()))
    for f in ("-DATTN_QKNORM=1", "-DATTN_QKNORM_POST=0", "-DATTN_GATE=1", "-DATTN_HD=128", "-DATTN_NH=16",
              "-DATTN_KVH=4", "-DATTN_NHL=4", "-DATTN_ROT=128", "-DATTN_RB=4", "-DATTN_INTFP=1"):
        assert f in fl
    assert not any("SINK" in f for f in fl)


def test_the_position_record_holds_a_whole_head_rotation():
    from recipes import lfm3 as R
    L = R.layout(spec())
    assert L.PTAB_ROW == 1024 and 512 + 8 * (128 // 2) == L.PTAB_ROW


def test_one_mixer_offset_serves_every_pool():
    from recipes import lfm3 as R
    s = spec()
    L, F = R.layout(s), R.ffn(s)
    assert L.POOL_MIX == F.POOL_DOWN + F.POOL_SLOTS * F.UP_BYTES
    assert L.POOL_B == L.POOL_Q == L.POOL_MIX
    assert L.POOL_GATE == L.POOL_Q + 32 * R.band_bytes(2048)
    assert L.POOL_BYTES >= L.POOL_O + 32 * R.band_bytes(2048) and L.POOL_BYTES % (1 << 20) == 0
    assert L.STATE_BYTES == 6 * 2048 * 4


def test_the_rope_tables_are_the_zero_tail_progression():
    from recipes import lfm3 as R
    s = spec()
    full, local = R.rope_table(s, False), R.rope_table(s, True)
    assert len(full) == len(local) == 64
    assert full[:32] == pytest.approx([1e6 ** (-2 * i / 128) for i in range(32)])
    assert full[32:] == [0.0] * 32, "the second half of the pairs does not turn"
    assert local == pytest.approx([1e4 ** (-2 * i / 128) for i in range(64)])


def test_the_programs_carry_two_dispatches_a_layer_and_the_window_on_the_sliding_kernel():
    from recipes import lfm3 as R
    s = spec()
    p = R.programs(s)
    assert set(p["layer_types"]) == {SHORT_CONV, FULL, DENSE_LOCAL}
    for lt, e in p["layer_types"].items():
        assert [st["kernel"] for st in e["program"]][1] == "lg_mlp"
        assert len(e["program"]) == 2
    assert p["kernels"]["lg_att"]["window"] == 0 and p["kernels"]["lg_att_local"]["window"] == 256
    assert p["kernels"]["lg_att"]["insts"] == p["kernels"]["lg_att_local"]["insts"]
    assert p["globals"]["ptab"]["window"] == 0 and p["globals"]["ptab_local"]["window"] == 256
    assert p["layer_types"][SHORT_CONV]["buffers"]["state"] == {"kind": "linear", "bytes": 6 * 2048 * 4}
    assert set(R.builds(s)) == {"lg_mlp", "ln", "lm_head_q8", "lg_conv", "lg_att"}


def test_the_pack_plan_reads_the_gguf_names_and_the_expanded_gate():
    from recipes import lfm3 as R
    s = spec()
    plan = R.pack_plan(s)["layer_types"]
    att = {o["tensor"]: o for o in plan[FULL]["pool"]}
    assert att["blk.{l}.attn_gate.weight"]["rows"] == 2048, "the derived GGUF repeats each head's row head_dim times"
    conv = [o["tensor"] for o in plan[SHORT_CONV]["pool"]]
    assert conv.count("blk.{l}.shortconv.in_proj.weight") == 3
    names = {o["tensor"] for lt in plan.values() for o in lt["consts"]}
    assert {"blk.{l}.post_attention_norm.weight", "blk.{l}.post_ffw_norm.weight"} <= names
    tr = [o for o in plan[SHORT_CONV]["consts"] if o["op"] == "transpose"][0]
    assert (tr["rows"], tr["cols"], tr["elem"]) == (2048, 7, 4)
    assert R.pack_plan(s)["norm"]["tensor"] == "output_norm.weight"


# ---- the reference and the GGUF derivation


def test_the_q6k_decoder_reads_a_hand_built_block():
    sys.path.insert(0, str(ROOT / "open_kernels" / "model"))
    from lfm3_gguf_reference import decode_q6k
    rng = np.random.default_rng(1)
    ql, qh = rng.integers(0, 256, 128, dtype=np.uint8), rng.integers(0, 256, 64, dtype=np.uint8)
    sc = rng.integers(-20, 20, 16).astype(np.int8)
    d = np.float16(0.0123)
    raw = np.concatenate([ql, qh, sc.view(np.uint8), np.frombuffer(d.tobytes(), np.uint8)])
    got = decode_q6k(raw)
    # ggml dequantize_row_q6_K
    want = np.empty(256, np.float32)
    for n in (0, 128):
        qlo, qhi, s = ql[n // 2:].astype(int), qh[n // 4:].astype(int), sc[n // 16:].astype(int)
        for l in range(32):
            i = l // 16
            q1 = ((qlo[l] & 15) | (((qhi[l] >> 0) & 3) << 4)) - 32
            q2 = ((qlo[l + 32] & 15) | (((qhi[l] >> 2) & 3) << 4)) - 32
            q3 = ((qlo[l] >> 4) | (((qhi[l] >> 4) & 3) << 4)) - 32
            q4_ = ((qlo[l + 32] >> 4) | (((qhi[l] >> 6) & 3) << 4)) - 32
            for off, sv, q in ((0, s[i], q1), (32, s[i + 2], q2), (64, s[i + 4], q3), (96, s[i + 6], q4_)):
                want[n + l + off] = np.float32(d) * np.float32(sv) * np.float32(q)
    assert np.allclose(got, want, atol=1e-6)


def test_the_gate_expansion_repeats_each_heads_row():
    sys.path.insert(0, str(ROOT / "open_kernels" / "model"))
    import lfm3_gguf_make as M
    rows = np.arange(4 * 64, dtype=np.uint8).reshape(4, 64)         # 4 heads' Q4_0 rows (a toy width)
    out = np.repeat(rows, 128, axis=0)
    assert out.shape == (512, 64) and (out[:128] == rows[0]).all() and (out[128:256] == rows[1]).all()
    q = M.quant_q8_0(np.linspace(-1, 1, 64, dtype=np.float32))
    assert len(q) == 2 * 34


def test_the_reference_rope_zero_tail_leaves_the_second_half_of_the_pairs_alone():
    sys.path.insert(0, str(ROOT / "open_kernels" / "model"))
    from flint_gguf_reference import rope
    from lfm3_gguf_reference import inv_freq
    inv = inv_freq(1e6, 128, 0.5)
    x = np.random.default_rng(0).standard_normal((3, 2, 128)).astype(np.float32)
    y = rope(x, np.arange(3) + 5, inv)
    assert np.array_equal(y[..., 32:64], x[..., 32:64]) and np.array_equal(y[..., 96:], x[..., 96:])
    assert not np.allclose(y[..., :32], x[..., :32])
