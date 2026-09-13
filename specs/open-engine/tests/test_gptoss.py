# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-GPTOSS, OPEN-ROPE-YARN, OPEN-ATTN-SINK
# (canonical spec: specs/open-engine/spec.md)
"""GPT-OSS: dense GQA attention over an MoE FFN, a 128-row sliding window on every other
layer, YaRN RoPE, a per-head attention sink, and a bias on nearly everything.

Only the derivation runs here -- a pure function over a config dict, no model and no
hardware. `families.family_module("gptoss")` deliberately raises: the recipe would have to
emit kernels for the clamped-SwiGLU experts and the sinks, and emitting dense kernels that
silently drop either is worse than refusing.

The fixture is gpt-oss-20b's published geometry (24 layers, 32 experts, hidden 2880, 64
heads over 8 kv heads at head dim 64) written in the key names a container's `config.json`
uses. It is not a downloaded file: the field values are cross-checked against
`transformers.models.gpt_oss.configuration_gpt_oss.GptOssConfig` below, where torch is
installed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from recipes import families
from recipes.spec import DENSE, DENSE_LOCAL, ModelSpec, SpecError

HF_GPTOSS_20B = {
    "model_type": "gpt_oss",
    "hidden_size": 2880,
    "intermediate_size": 2880,
    "num_hidden_layers": 24,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "num_local_experts": 32,
    "num_experts_per_tok": 4,
    "vocab_size": 201088,
    "sliding_window": 128,
    "layer_types": ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(24)],
    "rope_theta": 150000.0,
    "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                     "truncate": False, "original_max_position_embeddings": 4096},
    "rms_norm_eps": 1e-05,
    "hidden_act": "silu",
    "attention_bias": True,
    "swiglu_limit": 7.0,
    "max_position_embeddings": 131072,
    "tie_word_embeddings": False,
}


# ---- the shape


def test_the_spec_is_dense_attention_over_an_moe_ffn():
    spec = ModelSpec.from_hf_config(HF_GPTOSS_20B)
    assert spec.family == "gptoss"
    assert (spec.hidden, spec.num_layers, spec.vocab) == (2880, 24, 201088)
    assert (spec.num_heads, spec.num_kv_heads, spec.head_dim) == (64, 8, 64)
    assert spec.rotary_dim == 64, "GPT-OSS rotates the whole head"
    assert (spec.num_experts, spec.experts_per_tok) == (32, 4)
    assert spec.moe_intermediate == 2880, "intermediate_size is the EXPERT width; there is no dense FFN"
    assert spec.intermediate == 0
    assert spec.shared_expert_intermediate == 0, "GPT-OSS has no shared expert"
    assert not spec.qk_norm and not spec.attn_gate
    assert not spec.has_linear and not spec.has_full, "no Gated DeltaNet layers; every layer is attention"
    assert spec.norm_eps == 1e-05


def test_every_other_layer_is_a_sliding_window_layer():
    """HF's own alternation, and its own key names. `full_attention` in a GPT-OSS config
    means a plain dense layer, NOT the spec's FULL -- that is Qwen3.6's linear/full split,
    which GPT-OSS has nothing to do with."""
    spec = ModelSpec.from_hf_config(HF_GPTOSS_20B)
    assert spec.layer_types == tuple([DENSE_LOCAL, DENSE] * 12)
    assert spec.sliding_window == 128
    assert spec.has_local and spec.has_dense


def test_the_layer_types_default_to_the_alternation_when_the_config_omits_them():
    cfg = {k: v for k, v in HF_GPTOSS_20B.items() if k != "layer_types"}
    assert ModelSpec.from_hf_config(cfg).layer_types == tuple([DENSE_LOCAL, DENSE] * 12)


def test_an_unknown_layer_type_is_refused_by_name():
    bad = {**HF_GPTOSS_20B, "layer_types": ["chunked_attention"] + HF_GPTOSS_20B["layer_types"][1:]}
    with pytest.raises(SpecError, match="chunked_attention"):
        ModelSpec.from_hf_config(bad)


def test_a_sliding_layer_without_a_window_is_refused():
    cfg = {k: v for k, v in HF_GPTOSS_20B.items() if k != "sliding_window"}
    with pytest.raises(SpecError, match="sliding_window"):
        ModelSpec.from_hf_config(cfg)


def test_the_head_dim_key_wins_over_the_fallback():
    """GPT-OSS's head dim is 64 while hidden / heads is 45, so the key is not decoration --
    a deriver that fell back here would build the wrong attention."""
    assert ModelSpec.from_hf_config(HF_GPTOSS_20B).head_dim == 64
    cfg = {k: v for k, v in HF_GPTOSS_20B.items() if k != "head_dim"}
    assert ModelSpec.from_hf_config(cfg).head_dim == 2880 // 64 == 45


def test_a_hidden_size_that_is_not_a_multiple_of_the_heads_is_refused():
    cfg = {k: v for k, v in HF_GPTOSS_20B.items() if k != "head_dim"}
    with pytest.raises(SpecError, match="head_dim"):
        ModelSpec.from_hf_config({**cfg, "hidden_size": 2890})


def test_the_clamped_swiglu_is_named_rather_than_passed_off_as_silu():
    """GPT-OSS's expert FFN is not silu(gate) * up: gate is clamped above at 7, up is clamped
    both ways, the sigmoid takes alpha * gate, and up is offset by one. `hidden_act` says
    silu and is wrong about it, so the spec records what the model actually computes."""
    spec = ModelSpec.from_hf_config(HF_GPTOSS_20B)
    assert spec.activation == "clamped_swiglu"
    assert "alpha" not in spec.to_dict() and "swiglu_limit" not in spec.to_dict(), \
        "alpha 1.702 and the limit 7.0 are family constants; a spec field moves every hash"


def test_a_missing_key_is_named():
    for key in ("hidden_size", "num_hidden_layers", "num_local_experts", "num_experts_per_tok",
                "vocab_size", "rope_theta"):
        broken = {k: v for k, v in HF_GPTOSS_20B.items() if k != key}
        with pytest.raises(SpecError, match=key):
            ModelSpec.from_hf_config(broken)


def test_the_newer_rope_parameters_form_derives_the_same_spec():
    """transformers 5 moved rope_theta and the scaling into one `rope_parameters` object;
    the containers shipped so far carry the older pair."""
    rp = {**HF_GPTOSS_20B["rope_scaling"], "rope_theta": 150000.0}
    cfg = {k: v for k, v in HF_GPTOSS_20B.items() if k not in ("rope_theta", "rope_scaling")}
    assert ModelSpec.from_hf_config({**cfg, "rope_parameters": rp}).spec_hash() == \
        ModelSpec.from_hf_config(HF_GPTOSS_20B).spec_hash()


def test_the_spec_round_trips_through_json():
    spec = ModelSpec.from_hf_config(HF_GPTOSS_20B)
    assert ModelSpec.from_json(spec.to_json()).spec_hash() == spec.spec_hash()


# ---- YaRN


def test_yarn_inverse_frequencies_match_the_published_table():
    """The values transformers' own `_compute_yarn_parameters` produces for this config,
    read off it in fp32 (torch's dtype there); the derivation works in fp64, so the
    comparison is at fp32's own precision."""
    inv = ModelSpec.from_hf_config(HF_GPTOSS_20B).rope_inv_freq()
    assert len(inv) == 32
    # the low pairs are pure extrapolation: the ramp clamps to 0 there, so no 1/32
    assert inv[0] == pytest.approx(1.0, rel=1e-12)
    assert inv[1] == pytest.approx(0.6890442967414856, rel=1e-6)
    assert inv[15] == pytest.approx(0.00105260219424963, rel=1e-6)
    assert inv[16] == pytest.approx(0.0004564839182421565, rel=1e-6)
    assert inv[17] == pytest.approx(0.00012931869423482567, rel=1e-6)
    assert inv[31] == pytest.approx(3.023511396804679e-07, rel=1e-6)


def test_yarn_interpolates_the_high_pairs_by_the_full_factor():
    """Above the correction range the ramp is 1 and the frequency is divided by `factor`, so
    the last pair is the unscaled one over 32."""
    spec = ModelSpec.from_hf_config(HF_GPTOSS_20B)
    inv = spec.rope_inv_freq()
    assert inv[31] == pytest.approx(150000.0 ** (-31 / 32) / 32.0, rel=1e-9)
    assert inv[0] == pytest.approx(1.0, rel=1e-12), "and below it the ramp is 0: unscaled"


def test_the_yarn_attention_factor_scales_cos_and_sin():
    spec = ModelSpec.from_hf_config(HF_GPTOSS_20B)
    assert spec.rope_scale() == pytest.approx(0.1 * math.log(32.0) + 1.0, rel=1e-15)
    assert spec.rope_scale() == pytest.approx(1.3465735902799727, rel=1e-12)


def test_a_yarn_factor_of_one_or_less_leaves_cos_and_sin_alone():
    sc = {**HF_GPTOSS_20B["rope_scaling"], "factor": 1.0}
    assert ModelSpec.from_hf_config({**HF_GPTOSS_20B, "rope_scaling": sc}).rope_scale() == 1.0


def test_yarn_does_not_disturb_any_other_scaling_type():
    """The branch is keyed on rope_type, so llama3, linear and longrope come out unchanged."""
    from test_llama3 import HF_LLAMA31_8B
    a = ModelSpec.from_hf_config(HF_LLAMA31_8B)
    assert a.rope_scale() == 1.0 and len(a.rope_inv_freq()) == a.rotary_dim // 2


def test_an_explicit_attention_factor_wins_over_the_derived_one():
    sc = {**HF_GPTOSS_20B["rope_scaling"], "attention_factor": 1.0}
    assert ModelSpec.from_hf_config({**HF_GPTOSS_20B, "rope_scaling": sc}).rope_scale() == 1.0


# ---- the recipe refusal (the standing rule)


def test_the_family_has_no_recipe_and_says_what_is_missing():
    """Routing a GPT-OSS spec to the dense or MoE recipe would emit kernels that drop the
    sink and compute the wrong FFN, and report parity while doing it."""
    with pytest.raises(NotImplementedError) as e:
        families.family_module("gptoss")
    msg = str(e.value)
    for want in ("gptoss", "sink", "clamped", "bias"):
        assert want in msg.lower(), msg


def test_gptoss_is_not_in_the_recipe_list():
    """FAMILIES is the families that HAVE a recipe -- cache.py and the geometry tests walk
    it and call family_module on every entry."""
    assert "gptoss" not in families.FAMILIES
    from recipes import dense as DR
    assert "gptoss" not in DR.DENSE_FAMILIES


def test_the_quant_role_table_knows_gpt_oss_tensor_names():
    """The names q4nx-build writes (utilities/q4nx-build/configs/gpt-oss.json), so a
    container's header derives a role map instead of an error about an unknown family."""
    from recipes.spec import quant_map_from_chunk_sizes
    names = {f"model.layers.3.self_attn.{p}_proj.weight": 5120 for p in "qkvo"}
    names.update({f"model.layers.3.ffn_{p}_exps.weight": 5120 for p in ("up", "gate", "down")})
    assert quant_map_from_chunk_sizes("gptoss", names) == {}
    names["model.layers.3.self_attn.q_proj.weight"] = 8704
    with pytest.raises(SpecError, match="one format across the model"):
        quant_map_from_chunk_sizes("gptoss", names)


# ---- the fixture itself


def test_the_fixture_agrees_with_transformers_own_config_class():
    """Not a download: GptOssConfig's defaults plus gpt-oss-20b's published geometry. Skipped
    without torch."""
    pytest.importorskip("torch")
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    cfg = GptOssConfig(num_hidden_layers=24, num_local_experts=32, hidden_size=2880,
                       intermediate_size=2880, head_dim=64, num_attention_heads=64,
                       num_key_value_heads=8, vocab_size=201088, sliding_window=128,
                       rms_norm_eps=1e-05)
    assert cfg.layer_types == HF_GPTOSS_20B["layer_types"]
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "num_experts_per_tok", "vocab_size",
                "sliding_window", "attention_bias", "rms_norm_eps"):
        assert getattr(cfg, key) == HF_GPTOSS_20B[key], key
    assert cfg.num_local_experts == HF_GPTOSS_20B["num_local_experts"]
    rp = cfg.rope_parameters
    assert rp["rope_type"] == "yarn" and rp["rope_theta"] == HF_GPTOSS_20B["rope_theta"]
    for key in ("factor", "beta_fast", "beta_slow", "truncate", "original_max_position_embeddings"):
        assert rp[key] == HF_GPTOSS_20B["rope_scaling"][key], key


def test_the_sink_tensor_is_one_scalar_per_attention_head():
    """What the kernel design rests on: `self_attn.sinks` is NH values for the whole layer,
    not a per-channel vector like the q/k/v bias. 64 of them is 128 bytes as bf16, which is
    why they ride in the meta element rather than on a fifo of their own."""
    spec = ModelSpec.from_hf_config(HF_GPTOSS_20B)
    e_a = spec.attn_kv_width * 2                    # one attention fifo element
    assert e_a == 1024
    assert 4 * spec.head_dim + 2 * spec.num_heads <= e_a, "qn | kn | sinks fit one meta element"
    assert 2 * spec.num_heads == 128
    pytest.importorskip("torch")
    import torch
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention
    cfg = GptOssConfig(num_hidden_layers=24, num_attention_heads=64, num_key_value_heads=8,
                       head_dim=64, hidden_size=2880, intermediate_size=2880,
                       num_local_experts=32, vocab_size=201088)
    with torch.device("meta"):
        attn = GptOssAttention(cfg, layer_idx=0)
    assert tuple(attn.sinks.shape) == (spec.num_heads,)
    assert attn.q_proj.bias is not None and attn.o_proj.bias is not None, \
        "attention_bias covers o_proj too, which OPEN-ATTN-QKV-BIAS does not"


def test_the_reference_sink_softmax_is_reachable_from_the_replica():
    from replica_dense import sink_softmax
    assert sink_softmax([0.0], 0.0) == pytest.approx([0.5], abs=1e-15)
    assert np.asarray(sink_softmax([0.0, 0.0], 0.0)).sum() < 1.0
