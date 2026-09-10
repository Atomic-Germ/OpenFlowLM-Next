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
    # the export's context picks the table the manifest bakes
    assert np.allclose(manifest(s, 8192)["globals"]["ptab"]["inv_freq"], s.rope_inv_freq(ctx=8192))
    assert np.allclose(g["inv_freq"], s.rope_inv_freq(ctx=4096))


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
