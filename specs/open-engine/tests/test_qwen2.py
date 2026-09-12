# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-QWEN2, OPEN-ATTN-QKV-BIAS (canonical spec: specs/open-engine/spec.md)
"""Qwen2.5 dense: GQA without q/k norms, full RoPE, silu-gated FFN, and a per-channel bias
on q, k and v that the other dense families do not have. The bias is a family property, not
a spec field -- every Qwen2 has one -- so it lives in the recipe: three slots at the end of
`consts` and a second stream into the attention core, one bias element per projection
element.

Older Qwen2 configs omit `head_dim`; it is hidden_size / num_attention_heads."""
from __future__ import annotations

import pytest

from recipes import dense as DR
from recipes import families
from recipes.spec import DENSE, ModelSpec, SpecError

# Qwen/Qwen2.5-3B-Instruct, the fields the derivation reads.
HF_QWEN25_3B = {
    "model_type": "qwen2",
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-06,
}


def test_spec_derives_without_qk_norm_or_gate():
    spec = ModelSpec.from_hf_config(HF_QWEN25_3B)
    assert spec.family == "qwen2"
    assert not spec.qk_norm, "Qwen2 has no q/k RMSNorm (Qwen3 added it)"
    assert not spec.attn_gate
    assert spec.layer_types == tuple([DENSE] * 36)
    assert spec.activation == "silu"
    assert spec.intermediate == 11008


def test_head_dim_falls_back_to_hidden_over_heads():
    """Qwen2.5 configs predate the head_dim key."""
    spec = ModelSpec.from_hf_config(HF_QWEN25_3B)
    assert spec.head_dim == 2048 // 16 == 128
    assert spec.rotary_dim == spec.head_dim


def test_an_explicit_head_dim_wins_over_the_fallback():
    spec = ModelSpec.from_hf_config({**HF_QWEN25_3B, "head_dim": 64})
    assert spec.head_dim == 64 and spec.rotary_dim == 64


def test_a_hidden_size_that_is_not_a_multiple_of_the_heads_is_refused():
    with pytest.raises(SpecError, match="head_dim"):
        ModelSpec.from_hf_config({**HF_QWEN25_3B, "hidden_size": 2050})


def test_qwen2_routes_to_the_dense_recipe():
    assert "qwen2" in families.FAMILIES
    assert families.family_module("qwen2") is DR


def test_the_bias_is_a_family_property_and_only_qwen2_has_one():
    spec = ModelSpec.from_hf_config(HF_QWEN25_3B)
    assert DR.qkv_bias(spec)
    assert not any(f in DR.QKV_BIAS_FAMILIES for f in ("qwen3", "llama3", "gemma3", "hunyuan",
                                                       "granite", "phi3"))
    assert "qkv_bias" not in spec.to_dict(), "a spec field would move every shipped model's hash"


def test_the_three_bias_vectors_get_their_own_slots_in_consts():
    spec = ModelSpec.from_hf_config(HF_QWEN25_3B)
    L, G = DR.layout(spec), DR.geometry(spec)
    assert (L.CD_QB, L.CD_KB, L.CD_VB) == (8704, 8704 + G.QW * 2, 8704 + (G.QW + G.KVW) * 2)
    assert L.CD_BYTES >= L.CD_VB + G.KVW * 2
    ops = {o["tensor"]: o for o in DR.pack_plan(spec)["layer_types"]["dense"]["consts"]}
    for name, cap in (("q_proj", G.QW * 2), ("k_proj", G.KVW * 2), ("v_proj", G.KVW * 2)):
        op = ops["model.layers.{l}.self_attn." + name + ".bias"]
        assert op["op"] == "put" and op["cap"] == cap


def test_a_family_without_a_bias_keeps_the_consts_layout_it_had():
    """-1, not 0: 0 is the input-norm's own offset, so a stray read would land on a
    real tensor instead of failing."""
    from test_qwen3_dense import HF_QWEN3_4B
    L = DR.layout(ModelSpec.from_hf_config(HF_QWEN3_4B))
    assert (L.CD_QB, L.CD_KB, L.CD_VB) == (-1, -1, -1)


def test_the_bias_stream_runs_in_step_with_the_projection_stream():
    """One bias element per q / k / v element, which is what lets the two fifos stay in
    lockstep without interleaving the fills. A projection element is KVH/2 heads of f32
    (E_A bytes); the same heads of a bf16 bias are half that."""
    spec = ModelSpec.from_hf_config(HF_QWEN25_3B)
    L, G = DR.layout(spec), DR.geometry(spec)
    assert G.QW * 2 % (L.E_A // 2) == 0 and G.KVW * 2 % (L.E_A // 2) == 0
    assert G.QW * 2 // (L.E_A // 2) == G.Q_AIN_ELEMS
    assert G.KVW * 2 // (L.E_A // 2) == G.K_AIN_ELEMS


def test_the_position_record_is_two_attention_elements_here():
    """Every dense family before this one had a record exactly one element wide, so the
    design read cos / sin at a fixed offset into the single element it acquired. Two kv
    heads at head dim 128 give a 512-byte element and a 1024-byte record; acquiring one
    would leave the other half in the stream to be read as q."""
    spec = ModelSpec.from_hf_config(HF_QWEN25_3B)
    L, G = DR.layout(spec), DR.geometry(spec)
    assert (L.E_A, L.PTAB_ROW) == (512, 1024)
    assert (G.PTAB_ELEMS, G.PTAB_CS_ELEM) == (2, 1)
    from test_qwen3_dense import HF_QWEN3_4B
    G3 = DR.geometry(ModelSpec.from_hf_config(HF_QWEN3_4B))
    assert (G3.PTAB_ELEMS, G3.PTAB_CS_ELEM) == (1, 0), "one element, as every family before"


def test_a_record_that_does_not_tile_the_element_is_refused():
    """The limits of attn.h's ATTN_PTAB_SPLIT, named rather than discovered on hardware."""
    from recipes.catalogue import OpRangeError
    # 3 kv heads: a 768-byte element, which 1024 is not a multiple of
    with pytest.raises(OpRangeError, match="whole number of"):
        DR.geometry(ModelSpec.from_hf_config({**HF_QWEN25_3B, "head_dim": 128,
                                              "num_key_value_heads": 3}))
    # 2 kv heads at head dim 64: a 256-byte element, so the record is four of them
    with pytest.raises(OpRangeError, match="spans 4"):
        DR.geometry(ModelSpec.from_hf_config({**HF_QWEN25_3B, "head_dim": 64}))


def test_the_geometry_is_not_in_the_catalogue_until_hardware_has_run_it():
    """OPEN-OP-RANGE: the kernels exist, the point has not been compared on the NPU, and
    the recipe says so rather than emitting a bundle nobody has checked."""
    from recipes.catalogue import OpRangeError
    with pytest.raises(OpRangeError, match=r"128, 16, 2, 128, False, False, False, True"):
        DR.recipe(ModelSpec.from_hf_config(HF_QWEN25_3B))


def test_other_families_still_route():
    for fam in ("qwen3", "llama3", "gemma3", "qwen35", "qwen36moe"):
        assert families.family_module(fam) is not None


def test_a_missing_key_is_named():
    broken = {k: v for k, v in HF_QWEN25_3B.items() if k != "intermediate_size"}
    with pytest.raises(SpecError, match="intermediate_size"):
        ModelSpec.from_hf_config(broken)


def test_the_spec_round_trips_through_json():
    spec = ModelSpec.from_hf_config(HF_QWEN25_3B)
    assert ModelSpec.from_json(spec.to_json()).spec_hash() == spec.spec_hash()
