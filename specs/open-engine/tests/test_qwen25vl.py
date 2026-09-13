# Traces: OPEN-FAMILY-QWEN25VL (canonical spec: specs/open-engine/spec.md)
"""Qwen2.5-VL's decoder derives as Qwen2.5, so it links to that bundle instead of building.

The claim worth testing is not that the two configs look alike -- it is that the hash
`oflm-add` matches on is the SAME one, and that hash is not the config's alone: it folds in
the tokenizer's id count and the per-role weight formats read out of the container. Those
two models do not even share a weight format (Qwen2.5-3B ships the signed 4-bit quantiser,
Qwen2.5-VL ships real q4_1), which is exactly the sort of difference that would break the
link if the format reached the spec. It does not, and the test below says so against the
real containers when they are installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes import families
from recipes.spec import ModelSpec, SpecError

HERE = Path(__file__).parent
MODELS = Path.home() / ".flm" / "models"
VL, TEXT = "Qwen2.5-VL-3B-Instruct-NPU2", "Qwen2.5-3B-Instruct-NPU2"


def cfg() -> dict:
    return json.loads((HERE / "fixtures" / "config_qwen25vl_3b.json").read_text(encoding="utf-8"))


def test_the_decoder_derives_as_a_qwen2_dense_spec():
    s = ModelSpec.from_hf_config(cfg())
    assert s.family == "qwen2"
    assert (s.num_layers, s.hidden, s.intermediate) == (36, 2048, 11264)
    assert (s.num_heads, s.num_kv_heads, s.head_dim) == (16, 2, 128)
    assert s.rotary_dim == s.head_dim and s.rope_theta == 1e6
    assert not s.qk_norm and not s.attn_gate
    assert set(s.layer_types) == {"dense"}
    # the q/k/v bias is a property of the family, never a spec field -- one would move every
    # shipped model's hash (OPEN-FAMILY-QWEN2)
    from recipes import dense as DR
    assert DR.qkv_bias(s)
    assert "qkv_bias" not in s.to_dict()


def test_the_vision_block_does_not_reach_the_spec():
    """The tower is VitConfig's; M-RoPE only changes the position records the engine builds.
    If either leaked into the spec the hash would move and the bundle would not link."""
    c = cfg()
    assert "vision_config" in c and "rope_scaling" in c
    bare = {k: v for k, v in c.items() if k not in ("vision_config", "rope_scaling")}
    assert ModelSpec.from_hf_config(c).spec_hash() == ModelSpec.from_hf_config(bare).spec_hash()


def test_the_decoder_is_accepted_nested_and_flat():
    """Raw HF nests it under text_config; the container OFLM ships flattens it."""
    c = cfg()
    inner = {k: v for k, v in c.items() if k != "vision_config"}
    nested = {"model_type": "qwen2_5_vl", "vision_config": c["vision_config"], "text_config": inner}
    assert ModelSpec.from_hf_config(nested).spec_hash() == ModelSpec.from_hf_config(c).spec_hash()


def test_the_family_routes_to_the_dense_recipe():
    assert families.for_spec(ModelSpec.from_hf_config(cfg())) is families.family_module("qwen2")


def test_a_model_type_the_derivation_does_not_know_is_refused_by_name():
    with pytest.raises(SpecError, match="qwen2_5_omni"):
        ModelSpec.from_hf_config({**cfg(), "model_type": "qwen2_5_omni"})


@pytest.mark.skipif(not (MODELS / VL / "model.q4nx").exists() or not (MODELS / TEXT / "model.q4nx").exists(),
                    reason="needs both shipped containers installed")
def test_both_containers_derive_the_one_hash_oflm_add_links_on():
    """`spec_from_model_dir`, not `from_hf_config`: it reads the tokenizer and the container's
    per-role quant too, and either of those moves the hash on its own. This is the check that
    says one kernel set serves both models."""
    from recipes.load import spec_from_model_dir
    a, b = spec_from_model_dir(MODELS / VL), spec_from_model_dir(MODELS / TEXT)
    assert a.spec_hash() == b.spec_hash() == "sha256:e32bfd7e950ccd7b304aa530cb87d2fe903a41c4e3df9e634588354dfd4a8953"


@pytest.mark.skipif(not (MODELS / VL / "model.q4nx").exists() or not (MODELS / TEXT / "model.q4nx").exists(),
                    reason="needs both shipped containers installed")
def test_the_two_containers_do_not_share_a_weight_format():
    """Qwen2.5-3B stores the signed quantiser and Qwen2.5-VL stores real q4_1 (OPEN-PACK-Q4-0).
    The packer detects that per tensor, which is why one kernel set can serve both -- and had
    only one of them ever been packed, nothing would have exercised that."""
    import numpy as np
    from q4nx import Q4NX
    signed = {}
    for name in (VL, TEXT):
        m = Q4NX(str(MODELS / name / "model.q4nx"))
        t = "model.layers.0.self_attn.q_proj.weight"
        first = np.frombuffer(m.raw(t), np.uint8).reshape(-1, 5120)[0]
        signed[name] = bool((first[512:1024].view(np.uint16) == 0).all())
    assert signed[TEXT] is True, "Qwen2.5-3B ships the signed quantiser"
    assert signed[VL] is False, "Qwen2.5-VL ships real q4_1"
