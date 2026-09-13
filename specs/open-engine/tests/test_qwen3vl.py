# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-QWEN3VL (canonical spec: specs/open-engine/spec.md)
"""Qwen3-VL's text core is Qwen3 dense: the deriver reads the same fields and emits the
same family, so a Qwen3-VL model links to a Qwen3 kernel bundle. The vision tower is not
part of ModelSpec -- VitConfig reads vision_config separately.

The two config shapes both appear in practice: raw HF nests the decoder under
`text_config`, OFLM's shipped container flattens it. The last three tests read the two
real files -- Qwen3-VL-4B-Instruct-NPU2's container config and Qwen/Qwen3-VL-4B-Instruct's
-- rather than a fixture written from memory, because the container turns out not to look
the way one would guess."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes import families
from recipes.spec import DENSE, ModelSpec, SpecError

FIX = Path(__file__).resolve().parent / "fixtures"


def fixture(name: str) -> dict:
    return json.loads((FIX / f"config_{name}.json").read_text(encoding="utf-8"))

# Qwen/Qwen3-VL-4B-Instruct, the fields the derivation reads. The vision block here is a
# stand-in used only to show that nothing inside it reaches the spec.
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
FLAT = {"model_type": "qwen3_vl", **TEXT, "vision_config": {"depth": 27, "hidden_size": 1024}}


def _expected() -> ModelSpec:
    return ModelSpec(
        family="qwen3", hidden=2560, num_layers=36, layer_types=tuple([DENSE] * 36),
        vocab=151936, real_vocab=151936, num_heads=32, num_kv_heads=8, head_dim=128,
        rotary_dim=128, rope_theta=5000000.0, qk_norm=True, attn_gate=False,
        intermediate=9728, norm_eps=1e-06, quant="q4_1",
    )


def test_the_text_core_derives_as_a_qwen3_dense_spec():
    spec = ModelSpec.from_hf_config(FLAT)
    assert spec.family == "qwen3"
    assert spec.layer_types == tuple([DENSE] * 36)
    assert (spec.hidden, spec.num_heads, spec.num_kv_heads, spec.head_dim) == (2560, 32, 8, 128)
    assert spec.qk_norm and not spec.attn_gate
    assert spec.rotary_dim == spec.head_dim          # Qwen3 rotates the whole head


def test_the_nested_and_flat_configs_give_the_same_spec():
    """Raw HF nests the decoder, the container flattens it; the kernels must not care."""
    assert ModelSpec.from_hf_config(HF_NESTED).spec_hash() == ModelSpec.from_hf_config(FLAT).spec_hash()


def test_the_spec_hash_matches_a_plain_qwen3_of_the_same_shape():
    """The point of the whole exercise: a Qwen3-VL text core is kernel-identical to a
    Qwen3 of the same geometry, so it links to that bundle instead of building its own."""
    plain = dict(TEXT)
    plain["model_type"] = "qwen3"
    assert ModelSpec.from_hf_config(plain).spec_hash() == ModelSpec.from_hf_config(FLAT).spec_hash()


def test_the_derived_spec_equals_the_written_out_one():
    got = ModelSpec.from_hf_config(FLAT)
    want = _expected()
    assert got.to_dict() | {"extra": {}} == want.to_dict() | {"extra": {}}


def test_vision_config_is_not_read_into_the_spec():
    """A ModelSpec describes the decoder. Nothing about the tower may reach spec_hash,
    or two models with the same decoder would build different kernels."""
    a = ModelSpec.from_hf_config(FLAT)
    b = ModelSpec.from_hf_config({**FLAT, "vision_config": {"depth": 99, "hidden_size": 99}})
    assert a.spec_hash() == b.spec_hash()


def test_the_family_routes_to_the_dense_recipe():
    assert families.family_module("qwen3") is families.family_module("qwen3")
    assert ModelSpec.from_hf_config(FLAT).family in families.FAMILIES


def test_a_missing_text_field_is_named():
    broken = {k: v for k, v in FLAT.items() if k != "num_hidden_layers"}
    with pytest.raises(SpecError, match="num_hidden_layers"):
        ModelSpec.from_hf_config(broken)


def test_a_nested_config_missing_a_text_field_is_named_too():
    nested = {"model_type": "qwen3_vl",
              "text_config": {k: v for k, v in TEXT.items() if k != "head_dim"}}
    with pytest.raises(SpecError, match="head_dim"):
        ModelSpec.from_hf_config(nested)


# ---- the two config.json files that actually ship, verbatim


def test_the_shipped_container_links_to_the_qwen3_4b_bundle():
    """config_qwen3vl_4b.json is Qwen3-VL-4B-Instruct-NPU2's own config.json and
    config_qwen3_4b.json is Qwen3-4B-NPU2's. Same text stack, same hash, so oflm-add
    links the VL model to the bundle Qwen3-4B already has instead of building one.

    This is the hash over config.json alone; oflm-add derives through
    `recipes.load.spec_from_model_dir`, which also folds in the tokenizer's real vocab and
    the container's per-role weight formats. Those need the files, so the last word on
    whether the two link to one bundle is `oflm-add` on the pulled model."""
    vl = ModelSpec.from_hf_config(fixture("qwen3vl_4b"))
    plain = ModelSpec.from_hf_config(fixture("qwen3_4b"))
    assert vl.family == "qwen3"
    assert (vl.hidden, vl.num_layers, vl.num_heads, vl.num_kv_heads, vl.head_dim) == (2560, 36, 32, 8, 128)
    assert vl.spec_hash() == plain.spec_hash()


def test_the_container_is_tagged_plain_qwen3_not_qwen3_vl():
    """A trap for anyone testing this with a hand-written fixture: OFLM's container does
    not say `qwen3_vl` anywhere. It is flattened, tagged `qwen3`, and carries the tower
    only as file names -- so `_qwen3vl_hf` never runs for the model people actually pull,
    `_qwen3_hf` does."""
    cfg = fixture("qwen3vl_4b")
    assert cfg["model_type"] == "qwen3"
    assert "text_config" not in cfg and "vision_config" not in cfg
    assert cfg["vision_model_weight"] == "vision_weight.q4nx"


def test_the_upstream_config_derives_the_same_geometry_at_a_different_rope_theta():
    """Qwen/Qwen3-VL-4B-Instruct nests its decoder under `text_config`; every field the
    spec reads matches the container except rope_theta, which upstream puts at 5e6 and
    the container at 1e6. Same geometry, different hash -- and a model built from the
    container rotates at the container's theta, right or wrong."""
    hf = ModelSpec.from_hf_config(fixture("qwen3vl_4b_hf"))
    container = ModelSpec.from_hf_config(fixture("qwen3vl_4b"))
    assert hf.family == "qwen3"
    assert hf.to_dict() | {"rope_theta": 0.0, "extra": {}} == container.to_dict() | {"rope_theta": 0.0, "extra": {}}
    assert (hf.rope_theta, container.rope_theta) == (5_000_000.0, 1_000_000.0)
    assert hf.spec_hash() != container.spec_hash()
