# Traces: OPEN-SPEC-DERIVE, OPEN-PACK-PLAN, OPEN-FAMILY-QWEN3 (canonical spec: specs/open-engine/spec.md)
"""The Qwen3 dense family: spec derivation, the general pool-order law, and the
recipe's manifest for the 4B and for the three shapes waiting on a hardware pass
(8B, 1.7B, 0.6B). The kernel points they need are validated on the NPU per the
spec's procedure; the recipe's own arithmetic is checked here."""
from __future__ import annotations

import numpy as np
import pytest

import legacy_pools as LEG
from recipes import pack
from recipes import dense as QR
from recipes.load import load_spec, DEFAULT_SPEC
from recipes.manifest import manifest
from recipes.spec import DENSE, ModelSpec

HF_QWEN3_4B = {
    "model_type": "qwen3", "vocab_size": 151936, "hidden_size": 2560, "intermediate_size": 9728,
    "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
    "rms_norm_eps": 1e-06, "rope_theta": 1000000, "use_sliding_window": False,
}
GGUF_QWEN3_4B = {
    "general.architecture": "qwen3", "qwen3.embedding_length": 2560, "qwen3.block_count": 36,
    "qwen3.vocab_size": 151936, "qwen3.attention.head_count": 32, "qwen3.attention.head_count_kv": 8,
    "qwen3.attention.key_length": 128, "qwen3.rope.freq_base": 1000000.0, "qwen3.feed_forward_length": 9728,
    "qwen3.attention.layer_norm_rms_epsilon": 1e-06,
}
# FastFlowLM/Qwen3-{8,1.7,0.6}B-NPU2 config.json, verbatim apart from FLM's addr_* keys.
# DynaGuard-8B-NPU2 and DeepSeek-R1-0528-Qwen3-8B-NPU2 publish the 8B's shape byte for byte.
HF_QWEN3_8B = {
    "model_type": "qwen3", "vocab_size": 151936, "hidden_size": 4096, "intermediate_size": 12288,
    "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
    "rms_norm_eps": 1e-06, "rope_theta": 1000000, "use_sliding_window": False, "tie_word_embeddings": False,
}
HF_QWEN3_1_7B = {
    "model_type": "qwen3", "vocab_size": 151936, "hidden_size": 2048, "intermediate_size": 6144,
    "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
    "rms_norm_eps": 1e-06, "rope_theta": 1000000, "use_sliding_window": False, "tie_word_embeddings": True,
}
HF_QWEN3_0_6B = {
    "model_type": "qwen3", "vocab_size": 151936, "hidden_size": 1024, "intermediate_size": 3072,
    "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
    "rms_norm_eps": 1e-06, "rope_theta": 1000000, "use_sliding_window": False, "tie_word_embeddings": True,
}


@pytest.fixture
def unvalidated(monkeypatch):
    """The dense points are validated on hardware (OPEN-FAMILY-QWEN3); the recipe's arithmetic is checked here."""
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


def test_spec_from_hf_and_gguf_agree():
    a = ModelSpec.from_hf_config(HF_QWEN3_4B, real_vocab=151669)
    b = ModelSpec.from_gguf_metadata(GGUF_QWEN3_4B)
    assert a.family == "qwen3" and a.layer_types == tuple([DENSE] * 36)
    assert a.rotary_dim == 128 and a.attn_gate is False and a.intermediate == 9728 and a.real_vocab == 151669
    da, db = a.to_dict(), b.to_dict()
    for d in (da, db):
        d.pop("extra"); d.pop("real_vocab")
    assert da == db
    assert a.spec_hash() != load_spec(DEFAULT_SPEC).spec_hash()


def test_general_pool_law_equals_the_verified_one_where_it_was_verified():
    for nch, out_dim, in_dim in ((2048, 8192, 2048), (1024, 4096, 2048), (128, 512, 2048), (128, 2048, 512),
                                 (1024, 2048, 4096)):
        assert np.array_equal(pack.std_perm(nch, in_dim), LEG.std_perm(nch, out_dim, in_dim)), (nch, in_dim)


def test_general_pool_law_is_a_permutation_at_the_dense_widths():
    for rows, in_dim in ((4096, 2560), (2560, 9728), (9728, 2560), (151936, 2560)):
        nch = rows * in_dim // 8192
        p = pack.std_perm(nch, in_dim)
        assert sorted(p.tolist()) == list(range(nch)), (rows, in_dim)


def test_dense_layout_for_the_4b(unvalidated):
    spec = ModelSpec.from_hf_config(HF_QWEN3_4B, real_vocab=151669)
    R = QR.recipe(spec)
    L, G = R.layout, R.geo
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (8, 2, 5, 19, 5)
    assert (G.HPE, G.HPO, G.Q_AIN_ELEMS, G.K_AIN_ELEMS, G.OG_AOUT_ELEMS) == (4, 8, 8, 2, 4)
    assert (G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS, G.H_ELEMS) == (2, 2, 2, 10)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (5120, 2048, 4096, 2048)
    assert L.POOL_DOWN + 2560 * 9728 // 8192 * 5120 <= L.POOL_BYTES == 61 * 2 ** 20   # 12320 chunks, 1 MB-rounded
    assert L.LMHEAD_BANDS == 2374 and L.LMHEAD_POOL_BYTES == 233 * 2 ** 20   # 2376 bands x 102400 B, 1 MB-rounded
    assert L.CD_META == 2 * 5120 and L.AD_BYTES % 4096 == 0
    m = manifest(spec)
    assert m["family"] == "qwen3" and m["layers"] == [DENSE] * 36
    assert "moe" not in m["layout"] and "layer_types" not in m["hf_config_check"]
    d = m["layer_types"][DENSE]
    assert d["program"] == [{"op": "run", "kernel": "dx", "args": ["pool", "xres", "consts", "state", "act", "ptab"]}]
    assert m["kernels"]["dx"]["patch"] == "attnpos" and m["kernels"]["lm"]["build"] == "lm_head_q4"
    assert [o["op"] for o in d["pack"]["pool"]] == ["std_perm"] * 7 and len(d["pack"]["consts"]) == 4
    assert m["pack"]["lm_head"]["ops"][0] == {"op": "std_perm", "tensor": "lm_head.weight", "dst": 0, "nch": 47480, "in_dim": 2560}
    assert m["globals"]["normw"] == 5120 and m["globals"]["ptab"]["per_row"] == 2048 and m["globals"]["ptab"]["window"] == 0
    assert len(m["globals"]["ptab"]["inv_freq"]) == 64


def test_qwen3_8b_layout(unvalidated):
    """4096 hidden / 12288 FF: every point but the K = 12288 GEMV is already validated
    (the norm width, the attention tuple and the K = 4096 head are Llama 3.1 8B's).
    The 12288-wide table is what pushes the weight stream to one chunk per element."""
    spec = ModelSpec.from_hf_config(HF_QWEN3_8B, real_vocab=151669)
    assert (spec.hidden, spec.num_layers, spec.intermediate) == (4096, 36, 12288)
    assert (spec.num_heads, spec.num_kv_heads, spec.head_dim) == (32, 8, 128)
    assert spec.attn_q_width == 4096 and spec.attn_kv_width == 1024 and spec.qk_norm is True
    R = QR.recipe(spec)
    L, G = R.layout, R.geo
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (8, 2, 8, 24, 8)
    assert (G.HPE, G.HPO, G.Q_AIN_ELEMS, G.K_AIN_ELEMS, G.OG_AOUT_ELEMS) == (4, 8, 8, 2, 4)
    assert (G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS, G.H_ELEMS) == (2, 2, 2, 12)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (8192, 2048, 4096, 2048)
    assert G.PER_CALL == 1 and G.CALL_BYTES == 5120 and G.TAB_BYTES == 27648 and G.KWIDE == 12288
    assert (L.CD_BYTES, L.AD_BYTES) == (20480, 188416)
    assert L.POOL_BYTES == 115 * 2 ** 20
    assert L.LMHEAD_BANDS == 2374 and L.LMHEAD_BAND_BYTES == 163840 and L.LMHEAD_POOL_BYTES == 372 * 2 ** 20
    m = manifest(spec)
    assert m["family"] == "qwen3" and m["layers"] == [DENSE] * 36
    assert m["builds"]["dx"]["build_dir"] == "dense/build_qwen3_h4096"
    assert m["builds"]["ln"]["env"] == {"LN_N": "4096", "LN_EPS": "1e-06"}
    assert m["pack"]["lm_head"]["ops"][0]["nch"] == 75968 and m["pack"]["lm_head"]["ops"][0]["in_dim"] == 4096
    assert len(m["layer_types"][DENSE]["pack"]["consts"]) == 4      # in / post-attention ln + q/k norms


def test_qwen3_1_7b_layout(unvalidated):
    """2048 hidden / 6144 FF with 16 query heads over 8 kv heads: a GQA group of 2 at
    head dim 128, which no validated point covers even though 16, 8 and 128 are each in
    the attn sets on their own (the catalogue checks parameters, not tuples)."""
    spec = ModelSpec.from_hf_config(HF_QWEN3_1_7B, real_vocab=151669)
    assert (spec.hidden, spec.num_layers, spec.intermediate) == (2048, 28, 6144)
    assert (spec.num_heads, spec.num_kv_heads, spec.head_dim) == (16, 8, 128)
    assert spec.num_heads // spec.num_kv_heads == 2
    assert spec.attn_q_width == 2048 and spec.attn_kv_width == 1024
    R = QR.recipe(spec)
    L, G = R.layout, R.geo
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (4, 2, 4, 12, 4)
    assert (G.HPE, G.HPO, G.Q_AIN_ELEMS, G.K_AIN_ELEMS, G.OG_AOUT_ELEMS) == (4, 8, 4, 2, 2)
    assert (G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS, G.H_ELEMS) == (1, 1, 1, 6)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (4096, 2048, 4096, 2048)
    assert G.PER_CALL == 2 and G.CALL_BYTES == 10240 and G.TAB_BYTES == 13824 and G.KWIDE == 6144
    assert (L.CD_BYTES, L.AD_BYTES) == (12288, 98304)
    assert L.POOL_BYTES == 30 * 2 ** 20
    assert L.LMHEAD_BANDS == 2374 and L.LMHEAD_BAND_BYTES == 81920 and L.LMHEAD_POOL_BYTES == 186 * 2 ** 20
    m = manifest(spec)
    assert m["builds"]["dx"]["build_dir"] == "dense/build_qwen3_h2048"
    assert m["builds"]["lm_head_q4"]["env"] == {"LMHEAD_N": "151936", "LMHEAD_K": "2048", "LMHEAD_CORES": "8"}
    assert m["pack"]["lm_head"]["ops"][0]["nch"] == 37984
    # tied in config.json, but the head is packed from its own tensor either way
    assert m["pack"]["lm_head"]["ops"][0]["tensor"] == "lm_head.weight"


def test_qwen3_0_6b_layout(unvalidated):
    """1024 hidden: the narrowest shape the dense recipe has produced. The norm element
    (2048 B) is now HALF an x-stream element, and the q width (2048) is twice the hidden,
    so the o-projection GEMV is the widest K in the layer after the FFN's."""
    spec = ModelSpec.from_hf_config(HF_QWEN3_0_6B, real_vocab=151669)
    assert (spec.hidden, spec.num_layers, spec.intermediate) == (1024, 28, 3072)
    assert spec.attn_q_width == 2048 == 2 * spec.hidden and spec.attn_kv_width == 1024
    R = QR.recipe(spec)
    L, G = R.layout, R.geo
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (4, 2, 2, 6, 2)
    assert (G.HPE, G.HPO, G.Q_AIN_ELEMS, G.K_AIN_ELEMS, G.OG_AOUT_ELEMS) == (4, 8, 4, 2, 2)
    assert (G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS, G.H_ELEMS) == (1, 1, 1, 3)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (2048, 2048, 4096, 2048)
    assert L.ELN * 2 == 4096                       # the norm element is half an x element
    assert G.PER_CALL == 2 and G.TAB_BYTES == 6912 and G.KWIDE == 3072
    assert (L.CD_BYTES, L.AD_BYTES) == (8192, 65536)
    assert L.POOL_BYTES == 10 * 2 ** 20
    assert L.LMHEAD_BANDS == 2374 and L.LMHEAD_BAND_BYTES == 40960 and L.LMHEAD_POOL_BYTES == 93 * 2 ** 20
    m = manifest(spec)
    assert m["builds"]["ln"]["env"] == {"LN_N": "1024", "LN_EPS": "1e-06"}
    assert m["builds"]["lm_head_q4"]["env"]["LMHEAD_K"] == "1024"
    assert m["globals"]["normw"] == 2048 and m["globals"]["xres"] == 4096


def test_the_new_shapes_need_the_points_the_handoff_lists(unvalidated):
    """Which catalogue points each waiting shape asks for -- the list the hardware pass
    works through (.claude/plans/k-new-points-handoff.md)."""
    want = {
        "8B": (HF_QWEN3_8B, {4096, 12288}, (128, 32, 8)),
        "1.7B": (HF_QWEN3_1_7B, {2048, 6144}, (128, 16, 8)),
        "0.6B": (HF_QWEN3_0_6B, {1024, 2048, 3072}, (128, 16, 8)),
    }
    for name, (cfg, gemv_k, attn) in want.items():
        s = ModelSpec.from_hf_config(cfg, real_vocab=151669)
        QR.recipe(s)                                  # composes without a physical-limit failure
        assert {s.hidden, s.attn_q_width, s.intermediate} == gemv_k, name
        assert (s.head_dim, s.num_heads, s.num_kv_heads) == attn, name
        # the ln and the q4 head are always built at the hidden width
        assert QR.builds(s)["ln"]["env"]["LN_N"] == str(s.hidden), name
        assert QR.builds(s)["lm_head_q4"]["env"]["LMHEAD_K"] == str(s.hidden), name


def test_the_catalogue_refuses_the_dense_points_until_validated(monkeypatch):
    monkeypatch.delenv("OPEN_KERNELS_UNVALIDATED", raising=False)
    from recipes.catalogue import CATALOGUE, OpRangeError
    spec = ModelSpec.from_hf_config(HF_QWEN3_4B)
    if 2560 in (CATALOGUE["ln"].params["width"].values or ()):
        pytest.skip("the dense points are in the catalogue now")
    with pytest.raises(OpRangeError):
        QR.recipe(spec)
