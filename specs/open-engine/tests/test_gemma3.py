# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-GEMMA3 (canonical spec: specs/open-engine/spec.md)
"""Gemma 3 on the dense recipe: spec derivation, the two layer types and their
position tables, the sliding window's row counts, the sandwich layout."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from recipes import dense as DR
from recipes import pack
from recipes.manifest import manifest
from recipes.spec import DENSE, DENSE_LOCAL, ModelSpec, SpecError

HF_GEMMA3_4B = {
    "model_type": "gemma3_text", "hidden_size": 2560, "intermediate_size": 10240, "num_hidden_layers": 34,
    "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256, "rope_theta": 1000000.0,
    "rope_local_base_freq": 10000.0, "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
    "sliding_window": 1024, "sliding_window_pattern": 6, "vocab_size": 262208, "query_pre_attn_scalar": 256,
    "hidden_activation": "gelu_pytorch_tanh", "final_logit_softcapping": None, "attn_logit_softcapping": None,
    "rms_norm_eps": 1e-06,
}
GGUF_GEMMA3_4B = {
    "general.architecture": "gemma3", "gemma3.embedding_length": 2560, "gemma3.block_count": 34,
    "gemma3.vocab_size": 262208, "gemma3.attention.head_count": 8, "gemma3.attention.head_count_kv": 4,
    "gemma3.attention.key_length": 256, "gemma3.rope.freq_base": 1000000.0, "gemma3.rope.scaling.factor": 8.0,
    "gemma3.rope.local_freq_base": 10000.0, "gemma3.attention.sliding_window": 1024,
    "gemma3.attention.sliding_window_pattern": 6, "gemma3.feed_forward_length": 10240,
    "gemma3.attention.layer_norm_rms_epsilon": 1e-06,
}


@pytest.fixture
def unvalidated(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


def test_spec_from_hf_and_gguf_agree():
    a = ModelSpec.from_hf_config(HF_GEMMA3_4B, real_vocab=262145)
    b = ModelSpec.from_gguf_metadata(GGUF_GEMMA3_4B)
    assert a.family == "gemma3" and a.layer_types.count(DENSE) == 5 and a.layer_types[5] == DENSE and a.layer_types[0] == DENSE_LOCAL
    assert a.activation == "gelu_tanh" and a.sandwich_norms and a.qk_norm and not a.attn_gate
    assert a.sliding_window == 1024 and a.rope_local_theta == 1e4
    da, db = a.to_dict(), b.to_dict()
    for d in (da, db):
        d.pop("extra"); d.pop("real_vocab")
    assert da == db


def test_two_rope_tables():
    s = ModelSpec.from_hf_config(HF_GEMMA3_4B)
    g, l = np.array(s.rope_inv_freq()), np.array(s.rope_inv_freq(local=True))
    assert g.shape == l.shape == (128,)
    assert np.allclose(g, 1e6 ** (-np.arange(128) / 128) / 8) and np.allclose(l, 1e4 ** (-np.arange(128) / 128))


def test_window_rows():
    valid, nf = pack.window_rows(np.array([0, 1, 1023, 1024, 1500]), 1024)
    assert valid.tolist() == [0, 1, 1023, 1023, 1023] and nf.tolist() == [1, 1, 1023, 1023, 1023]
    valid, nf = pack.window_rows(np.array([0, 1, 5000]), 0)
    assert valid.tolist() == [0, 1, 5000] and nf.tolist() == [1, 1, 5000]
    t = pack.ptab(1500, 256, 1e4, 2048, None, 1024).reshape(1500, 2048)
    assert t[1500 - 1, :8].view(np.int32).tolist() == [1023, 1023] and t[0, :8].view(np.int32).tolist() == [0, 1]


def test_unsupported_configs_are_refused():
    with pytest.raises(SpecError, match="hidden_activation"):
        ModelSpec.from_hf_config(dict(HF_GEMMA3_4B, hidden_activation="silu"))
    with pytest.raises(SpecError, match="softcapping"):
        ModelSpec.from_hf_config(dict(HF_GEMMA3_4B, final_logit_softcapping=30.0))
    with pytest.raises(SpecError, match="query_pre_attn_scalar"):
        ModelSpec.from_hf_config(dict(HF_GEMMA3_4B, query_pre_attn_scalar=128))


def test_4b_layout_and_manifest(unvalidated):
    spec = ModelSpec.from_hf_config(HF_GEMMA3_4B, real_vocab=262145)
    R = DR.recipe(spec)
    L, G = R.layout, R.geo
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (4, 2, 5, 20, 5)
    assert (G.HPE, G.HPO, G.OG_ELEMS, G.H_ELEMS, G.PER_CALL) == (2, 4, 1, 10, 2)
    assert G.ACT == "gelu_tanh" and G.SANDWICH and G.WINDOW == 1024
    assert L.CD_PREFFN == 2 * 5120 + 2048 and L.CD_POSTFFN == L.CD_PREFFN + 5120 and L.CD_BYTES == 24576
    assert L.AD_T2 == L.AD_T + 2560 * 4 and L.AD_BYTES % 4096 == 0
    m = manifest(spec)
    assert sorted(m["layer_types"]) == [DENSE, DENSE_LOCAL]
    assert m["kernels"]["dx"]["window"] == 0 and m["kernels"]["dx_local"]["window"] == 1024
    assert m["kernels"]["dx"]["insts"] == m["kernels"]["dx_local"]["insts"]
    assert m["layer_types"][DENSE_LOCAL]["program"][0]["args"][-1] == "ptab_local"
    assert m["globals"]["ptab_local"]["window"] == 1024 and m["globals"]["ptab"]["window"] == 0
    assert m["globals"]["ptab_local"]["inv_freq"][0] == 1.0 and m["globals"]["ptab"]["inv_freq"][0] == 0.125
    consts = [o["tensor"].split(".")[-2] for o in m["layer_types"][DENSE]["pack"]["consts"]]
    assert consts == ["input_layernorm", "post_attention_layernorm", "q_norm", "k_norm",
                      "pre_feedforward_layernorm", "post_feedforward_layernorm"]
    assert m["hf_config_check"]["sliding_window"] == 1024 and m["layout"]["lmhead_pool_bytes"] > 0
    # the dense block prefill route (OPEN-PREFILL-BATCH): both layer types carry it, each
    # naming its OWN attention kernel and table -- dxB / ptab for dense, dxB_local / ptab_local
    # for the sliding-window layers -- sharing one instruction stream and one GEMM context.
    dg, lg = m["layer_types"][DENSE]["gemm_block"], m["layer_types"][DENSE_LOCAL]["gemm_block"]
    assert dg["t"] == 256 and dg["kind"] == "dense" and lg["t"] == 256 and lg["kind"] == "dense"
    assert dg["attn_kernel"] == "dxB" and dg["attn_args"][-1] == "ptab"
    assert lg["attn_kernel"] == "dxB_local" and lg["attn_args"][-1] == "ptab_local"
    assert dg["sandwich"] and lg["sandwich"] and dg["act"] == "gelu_tanh" and lg["act"] == "gelu_tanh"
    assert m["kernels"]["dxB"]["window"] == 0 and m["kernels"]["dxB_local"]["window"] == 1024
    assert m["kernels"]["dxB"]["insts"] == m["kernels"]["dxB_local"]["insts"]
    assert m["kernels"]["dxB"]["context"] == m["kernels"]["dxB_local"]["context"] == "dxa"
    assert sorted(m["contexts"]) == ["dx", "dxa", "gemm", "lm", "ln"]


def test_a_mismatched_sandwich_and_activation_combo_gets_no_route(unvalidated):
    """The block route's host chain (core.cpp step_gemm_block_layer) only implements two
    pairings: plain residuals with SiLU, and Gemma 3's sandwich norms with GeGLU-tanh. A spec
    combining sandwich norms with a different activation, or the reverse, must get no route --
    the sequential path stays correct where a wrongly-served route would silently compute the
    residual chain or the gate wrong."""
    spec = ModelSpec.from_hf_config(HF_GEMMA3_4B, real_vocab=262145)
    assert DR.gemm_route(dataclasses.replace(spec, activation="silu")) is None
    assert DR.gemm_route(dataclasses.replace(spec, sandwich_norms=False)) is None
    assert DR.gemm_route(spec) is not None, "the real combination still gets a route"
