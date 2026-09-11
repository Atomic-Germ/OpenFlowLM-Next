# Traces: OPEN-SPEC-DERIVE (canonical spec: specs/open-engine/spec.md)
"""ModelSpec from an HF config.json and from GGUF metadata; unknown / missing keys are named."""
from __future__ import annotations

import pytest

from recipes.load import default_spec
from recipes.spec import FULL, LINEAR, ModelSpec, SpecError, hf_model_types

# the fields of ~/.oflm/models/Qwen3.6-35B-A3B-NPU2/config.json the derivation reads
HF_QWEN36 = {
    "model_type": "qwen3_5_moe", "hidden_size": 2048, "num_hidden_layers": 40, "full_attention_interval": 4,
    "head_dim": 256, "num_attention_heads": 16, "num_key_value_heads": 2, "attn_output_gate": True,
    "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 32, "linear_value_head_dim": 128,
    "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512, "rms_norm_eps": 1e-06, "vocab_size": 248320,
}

# what llama.cpp's converter writes for the same model (arch qwen35moe / qwen3next keys)
GGUF_QWEN36 = {
    "general.architecture": "qwen35moe",
    "qwen35moe.embedding_length": 2048, "qwen35moe.block_count": 40, "qwen35moe.vocab_size": 248320,
    "qwen35moe.attention.head_count": 16, "qwen35moe.attention.head_count_kv": 2,
    "qwen35moe.attention.key_length": 256, "qwen35moe.rope.dimension_count": 64,
    "qwen35moe.rope.freq_base": 10000000.0, "qwen35moe.full_attention_interval": 4,
    "qwen35moe.ssm.conv_kernel": 4, "qwen35moe.ssm.inner_size": 4096, "qwen35moe.ssm.state_size": 128,
    "qwen35moe.ssm.group_count": 16, "qwen35moe.ssm.time_step_rank": 32,
    "qwen35moe.expert_count": 256, "qwen35moe.expert_used_count": 8,
    "qwen35moe.expert_feed_forward_length": 512, "qwen35moe.expert_shared_feed_forward_length": 512,
    "qwen35moe.attention.layer_norm_rms_epsilon": 1e-06,
}


def strip(s: ModelSpec) -> dict:
    d = s.to_dict()
    d.pop("extra")
    return d


def test_hf_config_gives_the_checked_in_27b_spec():
    s = ModelSpec.from_hf_config(HF_QWEN36, real_vocab=248070)
    assert strip(s) == strip(default_spec())
    assert s.layer_types[3] == FULL and s.layer_types[0] == LINEAR and s.layer_types.count(FULL) == 10
    assert s.rotary_dim == 64 and s.lin_qkv_dim == 8192 and s.attn_q_width == 4096


def test_the_text_only_moe_model_type_derives_the_same_spec():
    """Ornith-1.0-35B-A3B is `Qwen3_5MoeForCausalLM` / `qwen3_5_moe_text`: the VLM
    wrapper's text tower shipped on its own. Every field the recipe and the config check
    read is the VLM's, so it derives through the same builder and lands on the same
    kernels -- the alias `gemma3_text` and `qwen3_5_text` already have."""
    s = ModelSpec.from_hf_config(dict(HF_QWEN36, model_type="qwen3_5_moe_text"), real_vocab=248070)
    ref = ModelSpec.from_hf_config(HF_QWEN36, real_vocab=248070)
    assert strip(s) == strip(ref) and s.family == "qwen36moe"
    assert hf_model_types("qwen36moe") == ["qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"]


def test_hf_layer_types_list_wins_over_the_interval():
    cfg = dict(HF_QWEN36, layer_types=[LINEAR] * 39 + [FULL])
    s = ModelSpec.from_hf_config(cfg)
    assert s.layer_types.count(FULL) == 1


def test_gguf_metadata_gives_the_same_hyperparameters():
    s = ModelSpec.from_gguf_metadata(GGUF_QWEN36)
    ref = strip(default_spec())
    got = strip(s)
    assert got.pop("real_vocab") == 248320      # GGUF metadata has no tokenizer-side count
    ref.pop("real_vocab")
    assert got == ref


def test_unknown_family_names_the_key():
    with pytest.raises(SpecError, match="model_type 'mamba'"):
        ModelSpec.from_hf_config(dict(HF_QWEN36, model_type="mamba"))
    with pytest.raises(SpecError, match="general.architecture 'mamba'"):
        ModelSpec.from_gguf_metadata({"general.architecture": "mamba"})


def test_missing_key_is_named():
    cfg = dict(HF_QWEN36)
    del cfg["linear_num_value_heads"]
    with pytest.raises(SpecError, match="'linear_num_value_heads'"):
        ModelSpec.from_hf_config(cfg)
    md = dict(GGUF_QWEN36)
    del md["qwen35moe.expert_count"]
    with pytest.raises(SpecError, match="'qwen35moe.expert_count'"):
        ModelSpec.from_gguf_metadata(md)


def test_json_round_trip_and_hash():
    s = default_spec()
    again = ModelSpec.from_json(s.to_json())
    assert again == s and again.spec_hash() == s.spec_hash()
    with pytest.raises(SpecError, match="unknown ModelSpec field"):
        ModelSpec.from_dict(dict(s.to_dict(), bogus=1))
