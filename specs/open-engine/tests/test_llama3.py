# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-LLAMA3 (canonical spec: specs/open-engine/spec.md)
"""Llama 3 on the dense recipe: spec derivation (HF and GGUF), the llama3 RoPE
frequency scaling, the 8B's layout (8 KB norm elements, one chunk per weight
element because of the 32 KB table), the two Llama 3.2 shapes (tied heads that
the container materialises; head_dim 64 on the 1B), and the manifest."""
from __future__ import annotations

import math

import numpy as np
import pytest

from recipes import dense as DR
from recipes import pack
from recipes.manifest import manifest
from recipes.spec import DENSE, ModelSpec, SpecError

HF_LLAMA31_8B = {
    "model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0, "vocab_size": 128256, "tie_word_embeddings": False,
    "rope_scaling": {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                     "original_max_position_embeddings": 8192, "rope_type": "llama3"},
}
GGUF_LLAMA31_8B = {
    "general.architecture": "llama", "llama.embedding_length": 4096, "llama.block_count": 32,
    "llama.vocab_size": 128256, "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
    "llama.attention.key_length": 128, "llama.rope.freq_base": 500000.0, "llama.feed_forward_length": 14336,
    "llama.attention.layer_norm_rms_epsilon": 1e-05, "llama.rope.scaling.type": "llama3",
    "llama.rope.scaling.factor": 8.0, "llama.rope.scaling.low_freq_factor": 1.0,
    "llama.rope.scaling.high_freq_factor": 4.0, "llama.rope.scaling.original_context_length": 8192,
}
# OpenFlowLM/Llama-3.2-{3,1}B-NPU2 config.json, verbatim apart from OFLM's addr_* keys.
# Both set tie_word_embeddings; both containers carry lm_head.weight as its own q4
# tensor (I8 [48096, 5120] / [32064, 5120] = the whole 128256-row head).
HF_LLAMA32_3B = {
    "model_type": "llama", "hidden_size": 3072, "intermediate_size": 8192, "num_hidden_layers": 28,
    "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128, "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0, "vocab_size": 128256, "tie_word_embeddings": True,
    "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                     "original_max_position_embeddings": 8192, "rope_type": "llama3"},
}
HF_LLAMA32_1B = {
    "model_type": "llama", "hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 16,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 64, "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0, "vocab_size": 128256, "tie_word_embeddings": True,
    "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                     "original_max_position_embeddings": 8192, "rope_type": "llama3"},
}
# FastFlowLM/Nanbeige4.1-3B-NPU2 config.json: `model_type: llama` with its own head count
# (20 over 4 kv heads), a 166144-row head and theta 7e7, no scaling.
HF_NANBEIGE41_3B = {
    "model_type": "llama", "hidden_size": 2560, "intermediate_size": 10752, "num_hidden_layers": 32,
    "num_attention_heads": 20, "num_key_value_heads": 4, "head_dim": 128, "rms_norm_eps": 1e-05,
    "rope_theta": 70000000, "vocab_size": 166144, "tie_word_embeddings": False, "rope_scaling": None,
    "hidden_act": "silu", "attention_bias": False,
}


def hf_llama3_inv_freq(theta, dim, factor, lo, hi, old):
    """transformers' _compute_llama3_parameters, verbatim in NumPy."""
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    low_wl, high_wl = old / lo, old / hi
    wl = 2 * math.pi / inv
    out = np.where(wl > low_wl, inv / factor, inv)
    smooth = (old / wl - lo) / (hi - lo)
    smoothed = (1 - smooth) * out / factor + smooth * out
    mid = (wl <= low_wl) & (wl >= high_wl)
    return np.where(mid, smoothed, out)


def test_spec_from_hf_and_gguf_agree():
    a = ModelSpec.from_hf_config(HF_LLAMA31_8B, real_vocab=128256)
    b = ModelSpec.from_gguf_metadata(GGUF_LLAMA31_8B)
    assert a.family == "llama3" and a.layer_types == tuple([DENSE] * 32)
    assert a.qk_norm is False and a.attn_gate is False and a.rotary_dim == 128 and a.norm_eps == 1e-5
    assert a.rope_scaling == {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                              "original_max_position_embeddings": 8192}
    da, db = a.to_dict(), b.to_dict()
    for d in (da, db):
        d.pop("extra")
    assert da == db


def test_llama3_rope_scaling_matches_transformers():
    s = ModelSpec.from_hf_config(HF_LLAMA31_8B)
    got = np.array(s.rope_inv_freq())
    want = hf_llama3_inv_freq(500000.0, 128, 8.0, 1.0, 4.0, 8192)
    assert got.shape == (64,) and np.allclose(got, want, rtol=1e-12)
    assert got[0] == 1.0 and got[-1] < want[0] / 8 * 1.0001    # the lowest frequency is divided by the factor
    plain = ModelSpec.from_hf_config(dict(HF_LLAMA31_8B, rope_scaling=None)).rope_inv_freq()
    assert np.allclose(plain, 500000.0 ** (-np.arange(64) / 64))


def test_unsupported_scaling_is_refused():
    with pytest.raises(SpecError, match="rope_scaling type 'yarn'"):
        ModelSpec.from_hf_config(dict(HF_LLAMA31_8B, rope_scaling={"rope_type": "yarn", "factor": 2}))


def test_tied_embeddings_are_accepted_and_the_head_is_still_its_own_tensor():
    """Llama 3.2 ties the head to the embedding table; the containers we pack from
    materialise `lm_head.weight` anyway, so the derivation accepts the flag and the
    packer -- not the spec builder -- is what refuses a container that really is tied."""
    tied = ModelSpec.from_hf_config(dict(HF_LLAMA31_8B, tie_word_embeddings=True))
    untied = ModelSpec.from_hf_config(HF_LLAMA31_8B)
    assert tied.to_dict() == untied.to_dict()          # the flag is not a spec field
    plan = DR.pack_plan(tied)
    assert plan["lm_head"]["ops"][0]["tensor"] == "lm_head.weight"
    assert plan["embed"]["tensor"] == "model.embed_tokens.weight"

    class NoHead:
        def raw(self, name):
            raise KeyError(name)

    op = plan["lm_head"]["ops"][0]
    with pytest.raises(KeyError, match="no tensor 'lm_head.weight'"):
        pack.apply_op(op, NoHead(), 0, np.zeros(op["nch"] * 5120, dtype=np.uint8))


def test_llama32_3b_layout(monkeypatch):
    """3072 hidden / 8192 FF / 24 query heads: a new norm width, two new GEMV K, a new
    lm_head K and -- the one the catalogue catches by itself -- a 24-head attention point."""
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    spec = ModelSpec.from_hf_config(HF_LLAMA32_3B)
    assert (spec.hidden, spec.num_layers, spec.intermediate) == (3072, 28, 8192)
    assert (spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.rotary_dim) == (24, 8, 128, 128)
    assert spec.attn_q_width == 3072 and spec.attn_kv_width == 1024 and spec.qk_norm is False
    assert spec.rope_scaling["factor"] == 32.0 and spec.norm_eps == 1e-5
    R = DR.recipe(spec)
    L, G = R.layout, R.geo
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (6, 2, 6, 16, 6)
    assert (G.HPE, G.HPO, G.Q_AIN_ELEMS, G.K_AIN_ELEMS, G.OG_AOUT_ELEMS) == (4, 8, 6, 2, 3)
    assert (G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS, G.H_ELEMS) == (2, 2, 2, 8)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (6144, 2048, 4096, 2048)
    assert G.PER_CALL == 2 and G.CALL_BYTES == 10240 and G.TAB_BYTES == 18432 and G.KWIDE == 8192
    assert (L.CD_BYTES, L.AD_BYTES) == (16384, 143360)
    assert L.LMHEAD_BANDS == 2004 and L.LMHEAD_BAND_BYTES == 122880
    assert DR.lm_rows(spec) == 128256 == spec.vocab
    m = manifest(spec)
    assert m["family"] == "llama3" and m["layers"] == [DENSE] * 28
    assert m["builds"]["dx"]["build_dir"] == "dense/build_llama3_h3072"
    assert m["builds"]["ln"]["env"]["LN_N"] == "3072" and m["builds"]["lm_head_q4"]["env"]["LMHEAD_K"] == "3072"
    assert len(m["layout"]["rope_inv_freq"]) == 64


def test_llama32_1b_layout(monkeypatch):
    """head_dim 64 -- half the smallest head the attention design has ever run. The
    element sizes halve with it (E_A 1024, one KV band per core) and the RoPE record
    still fits the 1 KB position row."""
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    spec = ModelSpec.from_hf_config(HF_LLAMA32_1B)
    assert (spec.hidden, spec.num_layers, spec.intermediate) == (2048, 16, 8192)
    assert (spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.rotary_dim) == (32, 8, 64, 64)
    assert spec.attn_q_width == 2048 and spec.attn_kv_width == 512
    R = DR.recipe(spec)
    L, G = R.layout, R.geo
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (4, 1, 4, 16, 4)
    assert (G.HPE, G.HPO, G.Q_AIN_ELEMS, G.K_AIN_ELEMS, G.OG_AOUT_ELEMS) == (4, 8, 8, 2, 4)
    assert (G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS, G.H_ELEMS) == (1, 1, 1, 8)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (4096, 1024, 2048, 1024)
    assert 512 + 4 * spec.rotary_dim <= L.PTAB_ROW and 2 * spec.head_dim * 2 <= L.E_A
    assert G.PER_CALL == 2 and G.TAB_BYTES == 18432 and G.KWIDE == 8192
    assert (L.CD_BYTES, L.AD_BYTES) == (12288, 102400)
    assert L.LMHEAD_BANDS == 2004 and L.LMHEAD_BAND_BYTES == 81920
    assert len(spec.rope_inv_freq()) == 32           # rotary_dim / 2
    m = manifest(spec)
    assert m["builds"]["dx"]["build_dir"] == "dense/build_llama3_h2048"
    assert m["builds"]["ln"]["env"] == {"LN_N": "2048", "LN_EPS": "1e-05"}
    assert m["globals"]["ptab"]["per_row"] == 1024 and len(m["globals"]["ptab"]["inv_freq"]) == 32
    assert "head_dim" not in m["hf_config_check"]    # Llama configs may omit it


def test_8b_layout_and_manifest(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    spec = ModelSpec.from_hf_config(HF_LLAMA31_8B)
    R = DR.recipe(spec)
    L, G = R.layout, R.geo
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (8, 2, 8, 28, 8)
    assert (G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS, G.H_ELEMS) == (2, 2, 2, 14)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (8192, 2048, 4096, 2048)
    assert G.PER_CALL == 1 and G.CALL_BYTES == 5120 and G.TAB_BYTES == 32256   # the K = 14336 table
    assert G.QKNORM is False and G.EPS == 1e-5
    assert L.LMHEAD_BANDS == 2004
    m = manifest(spec)
    assert m["family"] == "llama3" and len(m["layout"]["rope_inv_freq"]) == 64
    assert "head_dim" not in m["hf_config_check"] and m["hf_config_check"]["intermediate_size"] == 14336
    consts = m["layer_types"][DENSE]["pack"]["consts"]
    assert [o["tensor"].split(".")[-2] for o in consts] == ["input_layernorm", "post_attention_layernorm"]
    assert m["builds"]["dx"]["build_dir"] == "dense/build_llama3_h4096"


def test_qwen3_4b_keeps_two_chunks_per_element():
    from test_qwen3_dense import HF_QWEN3_4B
    assert DR.per_call(ModelSpec.from_hf_config(HF_QWEN3_4B)) == 2


def test_ptab_uses_the_inverse_frequencies():
    inv = [0.5, 0.25]
    t = pack.ptab(3, 4, 10.0, 1024, inv).reshape(3, 1024)
    cos = t[2, 512:520].view(np.float32)
    sin = t[2, 520:528].view(np.float32)
    assert np.allclose(cos, np.cos([1.0, 0.5])) and np.allclose(sin, np.sin([1.0, 0.5]))


def test_nanbeige41_3b_derives_and_lays_out_on_the_llama_recipe(monkeypatch):
    """Nanbeige4.1-3B declares llama and is one: GQA 20 over 4 at head_dim 128 (a new
    attention tuple), an FFN of 10752 (a new GEMV K), theta 7e7 unscaled, its own vocab."""
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    spec = ModelSpec.from_hf_config(HF_NANBEIGE41_3B, real_vocab=166144)
    assert spec.family == "llama3" and spec.layer_types == tuple([DENSE] * 32)
    assert (spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.rotary_dim) == (20, 4, 128, 128)
    assert spec.attn_q_width == 2560 == spec.hidden and spec.attn_kv_width == 512
    assert spec.rope_theta == 7e7 and spec.rope_scaling is None and spec.qk_norm is False
    assert spec.rope_inv_freq() == [7e7 ** (-i / 64) for i in range(64)]
    R = DR.recipe(spec)
    L, G = R.layout, R.geo
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (5, 1, 5, 21, 5)
    assert (G.HPE, G.HPO, G.Q_AIN_ELEMS, G.K_AIN_ELEMS, G.OG_AOUT_ELEMS) == (2, 4, 10, 2, 5)
    assert (G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS, G.H_ELEMS) == (2, 2, 2, 11)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (5120, 1024, 2048, 1024)
    assert G.PER_CALL == 2 and G.TAB_BYTES == 24192 and G.KWIDE == 10752     # 60032 of the 61440-byte L1
    assert L.CD_BYTES == 12288
    assert L.LMHEAD_BANDS == 2596 and L.LMHEAD_BAND_BYTES == 102400 and DR.lm_rows(spec) == 166144
    m = manifest(spec)
    assert m["family"] == "llama3" and m["builds"]["dx"]["build_dir"] == "dense/build_llama3_h2560"
    assert m["builds"]["lm_head_q4"]["env"]["LMHEAD_N"] == "166144"
    assert m["hf_config_check"]["num_attention_heads"] == 20 and "head_dim" not in m["hf_config_check"]
