# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-PHI3 (canonical spec: specs/open-engine/spec.md)
"""Phi-3 / Phi-4-mini on the dense recipe: spec derivation, the partial rotation (96 of
128 head dims), the longrope tables (a short and a long factor list over the same
theta, and one attention scale on cos / sin), the layout at the 4B's size, and the
manifest carrying the scale."""
from __future__ import annotations

import math

import numpy as np
import pytest

from recipes import dense as DR
from recipes import pack
from recipes.manifest import manifest
from recipes.spec import DENSE, ModelSpec, SpecError

from test_pack_plan import _fnv1a

# FNV-1a 64 of pack.ptab(6, 8, ..., inv_freq=[0.5, 0.25, 0.125, 0.0625], scale=1.19).
# src/open_qwen36/pools_test.cpp builds the same table through pools::build_ptab and
# asserts the same number, so a Python/C++ mismatch on the scaled path cannot reach
# hardware validation unnoticed.
PTAB_SCALE_FNV1A = 0x091CD4029A9681A8
# FNV-1a 64 of a 10-row table switching from a short to a long table at row 4, scaled --
# the actual longrope mechanism, both knobs together. pools_test.cpp asserts the same
# number through pools::build_ptab.
PTAB_SWITCH_FNV1A = 0x7000741BF5FFA40B

# FastFlowLM/Phi4-mini-Instruct-NPU2 config.json, the fields the derivation reads.
SHORT = [1.0] * 48
LONG = [1, 1.118320672, 1.250641126, 1.398617824, 1.564103225, 1.74916897, 1.956131817, 2.187582649,
        2.446418898, 2.735880826, 3.059592084, 3.421605075, 3.826451687, 4.279200023, 4.785517845,
        5.351743533, 5.984965424, 6.693110555, 7.485043894, 8.370679318, 9.36110372, 10.4687158,
        11.70738129, 13.09260651, 14.64173252, 16.37415215, 18.31155283, 20.47818807, 22.90118105,
        25.61086418, 28.64115884, 32.03, 32.1, 32.13, 32.23, 32.6, 32.61, 32.64, 32.66, 32.7, 32.71,
        32.93, 32.97, 33.28, 33.49, 33.5, 44.16, 47.77]
HF_PHI4_MINI = {
    "model_type": "phi3", "hidden_size": 3072, "intermediate_size": 8192, "num_hidden_layers": 32,
    "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128, "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0, "vocab_size": 200064, "tie_word_embeddings": True,
    "partial_rotary_factor": 0.75, "original_max_position_embeddings": 4096,
    "max_position_embeddings": 131072,
    "rope_scaling": {"type": "longrope", "short_factor": SHORT, "long_factor": LONG},
}


def hf_longrope(theta, dim, factors, factor, orig):
    """transformers' _compute_longrope_parameters, verbatim in NumPy: the inverse
    frequencies divided by the factor list, and the attention scale."""
    inv = 1.0 / (np.asarray(factors, np.float64) * theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    return inv, math.sqrt(1 + math.log(factor) / math.log(orig))


@pytest.fixture
def unvalidated(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


def test_spec_derives_the_partial_rotation_and_the_longrope_tables():
    s = ModelSpec.from_hf_config(HF_PHI4_MINI, real_vocab=200064)
    assert s.family == "phi3" and s.layer_types == tuple([DENSE] * 32)
    assert (s.num_heads, s.num_kv_heads, s.head_dim, s.rotary_dim) == (24, 8, 128, 96)
    assert s.qk_norm is False and s.attn_gate is False and s.activation == "silu" and s.norm_eps == 1e-5
    assert s.rope_scaling["rope_type"] == "longrope"
    assert s.rope_scaling["factor"] == 32.0 and s.rope_scaling["original_max_position_embeddings"] == 4096
    assert len(s.rope_scaling["short_factor"]) == 48 == len(s.rope_scaling["long_factor"])


def test_the_tables_match_transformers_and_the_context_picks_the_list():
    """Short factors below and at the original context, long ones above it (HF switches
    when the sequence passes original_max_position_embeddings). The attention scale
    applies either way."""
    s = ModelSpec.from_hf_config(HF_PHI4_MINI)
    want_short, scale = hf_longrope(10000.0, 96, SHORT, 32.0, 4096)
    want_long, _ = hf_longrope(10000.0, 96, LONG, 32.0, 4096)
    assert np.allclose(s.rope_inv_freq(), want_short, rtol=1e-12, atol=0)
    assert np.allclose(s.rope_inv_freq(ctx=4096), want_short, rtol=1e-12, atol=0)
    assert np.allclose(s.rope_inv_freq(ctx=4097), want_long, rtol=1e-12, atol=0)
    assert np.allclose(s.rope_inv_freq(ctx=32768), want_long, rtol=1e-12, atol=0)
    assert s.rope_scale() == pytest.approx(scale, rel=1e-12) and scale == pytest.approx(1.1902380714, rel=1e-9)
    assert len(s.rope_inv_freq()) == 48


def test_other_families_are_unchanged():
    from test_llama3 import HF_LLAMA31_8B
    s = ModelSpec.from_hf_config(HF_LLAMA31_8B)
    assert s.rope_scale() == 1.0
    assert s.rope_inv_freq() == s.rope_inv_freq(ctx=100000)


def test_refusals_and_defaults():
    bad = dict(HF_PHI4_MINI, rope_scaling={"type": "yarn", "factor": 4.0})
    with pytest.raises(SpecError, match="yarn"):
        ModelSpec.from_hf_config(bad)
    plain = dict(HF_PHI4_MINI)
    del plain["rope_scaling"]
    del plain["partial_rotary_factor"]
    s = ModelSpec.from_hf_config(plain)
    assert s.rotary_dim == 128 and s.rope_scaling is None and s.rope_scale() == 1.0
    # every compatibility key is emitted even for a config that omitted it; what the absent
    # key MEANS is in hf_config_defaults, so the check is two-way (see test below)
    d = DR.hf_config_check(s)
    assert d["partial_rotary_factor"] == 1.0 and d["rope_scaling"] is None and d["rope_theta"] == 10000.0
    assert "original_max_position_embeddings" not in d and "max_position_embeddings" not in d
    with pytest.raises(SpecError, match="hidden_act"):
        ModelSpec.from_hf_config(dict(HF_PHI4_MINI, hidden_act="gelu"))


def test_layout_and_manifest(unvalidated):
    """Llama 3.2 3B's widths (3072 / 8192, 24 over 8 heads) with a 48-pair rotation and
    a 200064-row head; the ptab global carries the attention scale."""
    s = ModelSpec.from_hf_config(HF_PHI4_MINI, real_vocab=200064)
    R = DR.recipe(s)
    L, G = R.layout, R.geo
    assert G.ROT == 96
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (6, 2, 6, 16, 6)
    assert (G.HPE, G.HPO, G.Q_AIN_ELEMS, G.K_AIN_ELEMS, G.OG_AOUT_ELEMS) == (4, 8, 6, 2, 3)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (6144, 2048, 4096, 2048)
    assert G.PER_CALL == 2 and G.TAB_BYTES == 18432 and G.KWIDE == 8192
    assert (L.CD_BYTES, L.AD_BYTES) == (16384, 143360)
    assert L.LMHEAD_BANDS == 3126 and L.LMHEAD_BAND_BYTES == 122880 and DR.lm_rows(s) == 200064
    m = manifest(s, 4096)
    assert m["family"] == "phi3" and m["builds"]["dx"]["build_dir"] == "dense/build_phi3_h3072"
    assert m["builds"]["lm_head_q4"]["env"] == {"LMHEAD_N": "200064", "LMHEAD_K": "3072", "LMHEAD_CORES": "8"}
    assert m["layout"]["rotary_dim"] == 96 and len(m["layout"]["rope_inv_freq"]) == 48
    g = m["globals"]["ptab"]
    assert g["scale"] == pytest.approx(s.rope_scale(), rel=1e-12) and len(g["inv_freq"]) == 48
    assert m["hf_config_check"]["partial_rotary_factor"] == 0.75 and m["hf_config_check"]["head_dim"] == 128
    assert m["hf_config_check"]["model_type"] == ["phi3"]
    # the compatibility check names enough of the RoPE configuration that a same-shaped
    # container with a different theta or a different longrope table would be refused at
    # load, not silently accepted and run with the wrong baked position table
    check = m["hf_config_check"]
    assert check["rope_theta"] == 10000.0
    assert check["rope_scaling"] == {"type": "longrope", "short_factor": SHORT, "long_factor": LONG}
    assert check["original_max_position_embeddings"] == 4096
    # both tables are baked, not just the export's own max_ctx: row r takes long_inv_freq once
    # r >= switch_row, matching HF's per-call seq_len = pos + 1 > original rule applied per row
    # (OPEN-FAMILY-PHI3) -- max_ctx no longer decides which table ships, only how many rows do
    assert np.allclose(g["inv_freq"], s.rope_inv_freq())            # the short table, ctx <= 4096
    assert np.allclose(g["long_inv_freq"], s.rope_inv_freq(ctx=4097))
    assert g["switch_row"] == 4096
    assert manifest(s, 8192)["globals"]["ptab"]["long_inv_freq"] == g["long_inv_freq"]      # max_ctx-independent


def test_the_compatibility_check_is_two_way_through_the_defaults():
    """Manifest::check_model compares an ABSENT config key against hf_config_defaults, so
    a kernel set built from a full-rotation config refuses a 0.75 container and one built
    from a 0.75 config refuses a container that omits the field -- neither direction is a
    silent accept -- while a config that omits head_dim is accepted through HF's own
    hidden / heads fallback. The C++ side of this is manifest_test.cpp's phi3 block."""
    s = ModelSpec.from_hf_config(HF_PHI4_MINI)
    d, dflt = DR.hf_config_check(s), DR.hf_config_defaults(s)
    assert dflt == {"head_dim": 128, "partial_rotary_factor": 1.0, "rope_scaling": None,
                    "original_max_position_embeddings": None}
    assert d["partial_rotary_factor"] == 0.75 != dflt["partial_rotary_factor"]      # absent -> refused
    assert d["head_dim"] == dflt["head_dim"]                                        # absent -> accepted
    assert d["rope_scaling"] is not None and dflt["rope_scaling"] is None           # absent -> refused
    # max_position_embeddings set the attention scale (rope_scaling carried no factor)
    assert d["max_position_embeddings"] == 131072 and "max_position_embeddings" not in dflt
    assert d["original_max_position_embeddings"] == 4096
    # a config carrying the factor itself does not depend on max_position_embeddings
    with_factor = dict(HF_PHI4_MINI, rope_scaling={**HF_PHI4_MINI["rope_scaling"], "factor": 32.0})
    assert "max_position_embeddings" not in DR.hf_config_check(ModelSpec.from_hf_config(with_factor))
    # nested original_max_position_embeddings: the default is what the sub-object says
    nested = dict(HF_PHI4_MINI, rope_scaling={**HF_PHI4_MINI["rope_scaling"], "original_max_position_embeddings": 4096})
    del nested["original_max_position_embeddings"]
    assert DR.hf_config_defaults(ModelSpec.from_hf_config(nested))["original_max_position_embeddings"] == 4096
    # every other family: no defaults, its check is unchanged
    from test_llama3 import HF_LLAMA32_3B
    assert DR.hf_config_defaults(ModelSpec.from_hf_config(HF_LLAMA32_3B)) == {}


def test_a_spec_loaded_from_json_still_emits_the_full_check():
    """The checked-in recipes/specs/phi4-mini-4b.json goes through export --spec too; its
    checks must not depend on the derivation's raw metadata being present."""
    from recipes.load import load_spec
    from pathlib import Path
    s = load_spec(Path(DR.__file__).resolve().parent / "specs" / "phi4-mini-4b.json")
    d = DR.hf_config_check(s)
    assert d["partial_rotary_factor"] == 0.75 and d["rope_scaling"]["type"] == "longrope"
    assert len(d["rope_scaling"]["long_factor"]) == 48 and d["max_position_embeddings"] == 131072
    bare = ModelSpec.from_dict({**s.to_dict(), "extra": {"model_type": "phi3"}})   # no raw sub-object
    assert DR.hf_config_check(bare)["rope_scaling"] == d["rope_scaling"]


def test_the_compatibility_check_catches_a_different_longrope_table():
    """Two containers agreeing on every shape field but disagreeing on the factor lists
    (a real scenario: two longrope fine-tunes extended to different context lengths) must
    NOT produce the same hf_config_check -- that is what OPEN-FAMILY-PHI3's load-time
    refusal (Manifest::check_model) keys off."""
    other = dict(HF_PHI4_MINI, rope_scaling={"type": "longrope", "short_factor": SHORT,
                                             "long_factor": [x * 2 for x in LONG]})
    a = DR.hf_config_check(ModelSpec.from_hf_config(HF_PHI4_MINI))
    b = DR.hf_config_check(ModelSpec.from_hf_config(other))
    assert a["rope_scaling"] != b["rope_scaling"]
    other_theta = dict(HF_PHI4_MINI, rope_theta=500000.0)
    assert DR.hf_config_check(ModelSpec.from_hf_config(other_theta))["rope_theta"] != a["rope_theta"]


def test_a_family_without_a_scale_writes_no_scale_key(unvalidated):
    from test_llama3 import HF_LLAMA32_3B
    m = manifest(ModelSpec.from_hf_config(HF_LLAMA32_3B))
    assert "scale" not in m["globals"]["ptab"]


def test_ptab_applies_the_scale_to_cos_and_sin():
    inv = [0.5, 0.25]
    t = pack.ptab(3, 4, 10000.0, 1024, inv_freq=inv, scale=1.25).reshape(3, 1024)
    cs = t[:, 512:528].copy().view(np.float32).reshape(3, 4)          # [cos, cos, sin, sin]
    p = np.arange(3)[:, None]
    ang = p * np.asarray(inv)[None, :]
    assert np.allclose(cs[:, :2], 1.25 * np.cos(ang), rtol=1e-6)
    assert np.allclose(cs[:, 2:], 1.25 * np.sin(ang), rtol=1e-6)
    assert np.allclose(pack.ptab(3, 4, 10000.0, 1024, inv_freq=inv).reshape(3, 1024)[:, 512:528].copy()
                       .view(np.float32).reshape(3, 4)[:, :2], np.cos(ang), rtol=1e-6)


def test_ptab_switches_table_per_row_at_the_threshold():
    """Row r < switch_row reads `inv_freq`; row r >= switch_row reads `long_inv_freq`. Both
    tables live in the SAME call (`pack.ptab` builds a whole max_ctx x row table in one shot),
    so this is the actual mechanism OPEN-FAMILY-PHI3 relies on, not just two separate calls
    that happen to agree."""
    short, long_ = [0.5, 0.25], [2.0, 4.0]
    t = pack.ptab(6, 4, 10000.0, 1024, inv_freq=short, long_inv_freq=long_, switch_row=3).reshape(6, 1024)
    p = np.arange(6)[:, None]
    want_short = np.cos(p * np.asarray(short))
    want_long = np.cos(p * np.asarray(long_))
    for row in range(6):
        cos = t[row, 512:520].copy().view(np.float32)
        want = want_long[row] if row >= 3 else want_short[row]
        assert np.allclose(cos, want, rtol=1e-6), row
    with pytest.raises(ValueError, match="long_inv_freq and switch_row"):
        pack.ptab(6, 4, 10000.0, 1024, inv_freq=short, long_inv_freq=long_)
    with pytest.raises(ValueError, match="long_inv_freq and switch_row"):
        pack.ptab(6, 4, 10000.0, 1024, inv_freq=short, switch_row=3)


def test_dense_decode_picks_the_table_from_pos_plus_one():
    """`replica_dense.dense_decode` reads `spec.rope_inv_freq(ctx=pos + 1)` -- HF's own
    `seq_len = pos + 1` rule, applied per token since this engine computes one row per token.
    Position original - 1 (seq_len == original, still short) and position original (seq_len
    == original + 1, long) must straddle the switch exactly there, matching the packer's
    `switch_row = original` in `recipes/dense.py::programs`."""
    s = ModelSpec.from_hf_config(HF_PHI4_MINI)
    orig = s.rope_scaling["original_max_position_embeddings"]
    short, long_ = np.asarray(s.rope_inv_freq()), np.asarray(s.rope_inv_freq(ctx=orig + 1))
    assert not np.allclose(short, long_)
    assert np.array_equal(s.rope_inv_freq(ctx=(orig - 1) + 1), short)
    assert np.array_equal(s.rope_inv_freq(ctx=orig + 1), long_)


def test_numpy_and_cpp_agree_on_a_scaled_ptab():
    """The whole-table builder, not just one row: `pools::build_ptab` (C++) and
    `pack.ptab` (NumPy) on the same rows/inv_freq/scale must be byte-identical, or a
    C++-only regression in the scaled path would pass every Python test and still ship."""
    t = pack.ptab(6, 8, 10000.0, 1024, inv_freq=[0.5, 0.25, 0.125, 0.0625], scale=1.19)
    got = _fnv1a(t)
    print(f"\nptab scale fnv1a = 0x{got:016x}")
    assert got == PTAB_SCALE_FNV1A


def test_numpy_and_cpp_agree_on_a_switched_and_scaled_ptab():
    """The full longrope mechanism together -- a per-row table switch plus the attention
    scale -- must be byte-identical between the two packers, or a C++-only regression in
    the switch could pass every Python test and still ship wrong to hardware."""
    t = pack.ptab(10, 8, 10000.0, 1024, inv_freq=[0.5, 0.25, 0.125, 0.0625],
                 long_inv_freq=[3.0, 1.5, 0.75, 0.375], switch_row=4, scale=1.19)
    got = _fnv1a(t)
    print(f"\nptab switch fnv1a = 0x{got:016x}")
    assert got == PTAB_SWITCH_FNV1A


def test_the_replica_rotates_the_first_rot_dims_only_and_scales_them():
    import replica_dense as RD
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 128))
    inv = np.asarray(ModelSpec.from_hf_config(HF_PHI4_MINI).rope_inv_freq())
    y = RD.rope(x, 5, 96, 10000.0, inv, scale=1.19)
    assert np.array_equal(y[:, 96:], x[:, 96:])
    ang = 5 * inv
    c, s = np.cos(ang), np.sin(ang)
    assert np.allclose(y[:, :48], 1.19 * (x[:, :48] * c - x[:, 48:96] * s))
    assert np.allclose(y[:, 48:96], 1.19 * (x[:, 48:96] * c + x[:, :48] * s))
    assert np.array_equal(RD.rope(x, 5, 96, 10000.0, inv), RD.rope(x, 5, 96, 10000.0, inv, scale=1.0))
