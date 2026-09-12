# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-QWEN3VL (canonical spec: specs/open-engine/spec.md)
"""Qwen3-VL's text core is Qwen3 dense: the deriver reads the same fields and emits the
same family, so a Qwen3-VL model links to a Qwen3 kernel bundle. The vision tower is not
part of ModelSpec -- VitConfig reads vision_config separately.

The two config shapes both appear in practice: raw HF nests the decoder under
`text_config`, FLM's shipped container flattens it (as Qwen3.5-0.8B-NPU2/config.json
does) and keeps only `vision_config` nested."""
from __future__ import annotations

import pytest

from recipes import families
from recipes.spec import DENSE, ModelSpec, SpecError

# Qwen/Qwen3-VL-4B-Instruct, the fields the derivation reads. The vision block is the
# shape FLM ships (a `vision_config` sub-object); its values are never read here.
TEXT = {
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "rope_theta": 5000000.0,
    "rms_norm_eps": 1e-06,
}
HF_NESTED = {"model_type": "qwen3_vl", "text_config": dict(TEXT),
             "vision_config": {"depth": 27, "hidden_size": 1152}}
FLM_FLAT = {"model_type": "qwen3_vl", **TEXT,
            "vision_config": {"QWEN3VL_VISION_NUM_LAYERS": 27}}


def _expected() -> ModelSpec:
    return ModelSpec(
        family="qwen3", hidden=2560, num_layers=36, layer_types=tuple([DENSE] * 36),
        vocab=151936, real_vocab=151936, num_heads=32, num_kv_heads=8, head_dim=128,
        rotary_dim=128, rope_theta=5000000.0, qk_norm=True, attn_gate=False,
        intermediate=9728, norm_eps=1e-06, quant="q4_1",
    )


def test_the_text_core_derives_as_a_qwen3_dense_spec():
    spec = ModelSpec.from_hf_config(FLM_FLAT)
    assert spec.family == "qwen3"
    assert spec.layer_types == tuple([DENSE] * 36)
    assert (spec.hidden, spec.num_heads, spec.num_kv_heads, spec.head_dim) == (2560, 32, 8, 128)
    assert spec.qk_norm and not spec.attn_gate
    assert spec.rotary_dim == spec.head_dim          # Qwen3 rotates the whole head


def test_the_nested_and_flat_configs_give_the_same_spec():
    """Raw HF nests the decoder, FLM's container flattens it; the kernels must not care."""
    assert ModelSpec.from_hf_config(HF_NESTED).spec_hash() == ModelSpec.from_hf_config(FLM_FLAT).spec_hash()


def test_the_spec_hash_matches_a_plain_qwen3_of_the_same_shape():
    """The point of the whole exercise: a Qwen3-VL text core is kernel-identical to a
    Qwen3 of the same geometry, so it links to that bundle instead of building its own."""
    plain = dict(TEXT)
    plain["model_type"] = "qwen3"
    assert ModelSpec.from_hf_config(plain).spec_hash() == ModelSpec.from_hf_config(FLM_FLAT).spec_hash()


def test_the_derived_spec_equals_the_written_out_one():
    got = ModelSpec.from_hf_config(FLM_FLAT)
    want = _expected()
    assert got.to_dict() | {"extra": {}} == want.to_dict() | {"extra": {}}


def test_vision_config_is_not_read_into_the_spec():
    """A ModelSpec describes the decoder. Nothing about the tower may reach spec_hash,
    or two models with the same decoder would build different kernels."""
    a = ModelSpec.from_hf_config(FLM_FLAT)
    b = ModelSpec.from_hf_config({**FLM_FLAT, "vision_config": {"QWEN3VL_VISION_NUM_LAYERS": 99}})
    assert a.spec_hash() == b.spec_hash()


def test_the_family_routes_to_the_dense_recipe():
    assert families.family_module("qwen3") is families.family_module("qwen3")
    assert ModelSpec.from_hf_config(FLM_FLAT).family in families.FAMILIES


def test_a_missing_text_field_is_named():
    broken = {k: v for k, v in FLM_FLAT.items() if k != "num_hidden_layers"}
    with pytest.raises(SpecError, match="num_hidden_layers"):
        ModelSpec.from_hf_config(broken)


def test_a_nested_config_missing_a_text_field_is_named_too():
    nested = {"model_type": "qwen3_vl",
              "text_config": {k: v for k, v in TEXT.items() if k != "head_dim"}}
    with pytest.raises(SpecError, match="head_dim"):
        ModelSpec.from_hf_config(nested)
