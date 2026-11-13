# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-GPTOSS, OPEN-ROPE-YARN, OPEN-ATTN-SINK, OPEN-PREFILL-GPTOSS
# (canonical spec: specs/open-engine/spec.md)
"""GPT-OSS: dense GQA attention over an MoE FFN, a 128-row sliding window on every other
layer, YaRN RoPE, a per-head attention sink, and a bias on nearly everything.

The derivation and the recipe's emission run here -- pure functions over a config dict, no
model and no hardware: the decode route (gx_att / gptoss_route / gx_moe, the experts as mx4
half-tiles by default) and the block prefill route (OPEN-PREFILL-GPTOSS).

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


# ---- the recipe (recipes/gptoss.py, OPEN-FAMILY-GPTOSS)


def _recipe():
    from recipes import gptoss
    return gptoss, ModelSpec.from_hf_config(HF_GPTOSS_20B)


def test_the_family_routes_to_its_own_recipe_not_the_dense_one():
    """Routing a GPT-OSS spec to the dense or MoE recipe would drop the sink and compute the
    wrong FFN; the family has its own recipe, and dense still refuses it."""
    from recipes import dense as DR, gptoss
    assert families.family_module("gptoss") is gptoss
    assert "gptoss" in families.FAMILIES and "gptoss" not in DR.DENSE_FAMILIES
    with pytest.raises(Exception, match="dense recipe given"):
        DR._check(ModelSpec.from_hf_config(HF_GPTOSS_20B))


def test_the_widths_pad_to_whole_k_tiles_over_eight_cores():
    R, spec = _recipe()
    G, L = R.geometry(spec), R.layout(spec)
    assert (G.HID, G.HR, G.FF, G.FR) == (3072, 2880, 3072, 2880)
    assert L.UP_BYTES == L.EXPERT_UG // 2 and L.POOL_DOWN == spec.num_experts * L.EXPERT_UG
    assert L.POOL_ATTN == L.POOL_DOWN + spec.num_experts * L.UP_BYTES < L.POOL_BYTES < 1 << 32
    assert L.AD_W == L.AD_XM + 2 * G.HID and L.AD_W + 32 <= L.AD_XM + G.XM_ELEMS * 4096


def test_the_attention_core_is_built_with_the_sink_and_no_qk_norm():
    R, spec = _recipe()
    flags = R.attn_flags(R.geometry(spec))
    assert "-DATTN_SINK=1" in flags and "-DATTN_QKNORM=0" in flags and "-DATTN_GATE=0" in flags


def test_both_position_tables_turn_at_the_yarn_frequencies():
    """GPT-OSS's sliding layers share the full layers' rotation; `rope_inv_freq(local=True)` is
    Gemma's separate unscaled table, and using it moved the sliding layers' attention by 4%."""
    R, spec = _recipe()
    g = R.programs(spec)["globals"]
    for tab in ("ptab", "ptab_local"):
        assert g[tab]["inv_freq"] == spec.rope_inv_freq() and g[tab]["scale"] == spec.rope_scale()
    assert g["ptab_local"]["window"] == 128 and g["ptab"]["window"] == 0


def test_a_layer_is_two_dispatches_with_the_router_on_the_host_between():
    R, spec = _recipe()
    for lt in (DENSE, DENSE_LOCAL):
        prog = R.programs(spec)["layer_types"][lt]["program"]
        assert [s["op"] for s in prog] == ["run", "gptoss_route", "run"]
        assert prog[1]["kernel"] == prog[2]["kernel"] == "gx_moe" and prog[1]["hidden_real"] == 2880


def test_the_experts_stream_as_mx4_half_tiles_unless_asked_for_q8(monkeypatch):
    """GX_MX4 (default on): the pool's experts are the GGUF's MXFP4 nibbles (2368 B a half-tile, gx_ug_mx4 /
    gx_down_mx4), the attention stays q8; GX_MX4=0 keeps the q8 experts the first route shipped, at 5120 B."""
    R, spec = _recipe()
    L = R.layout(spec)
    ops = R.pack_plan(spec)["layer_types"][DENSE]["pool"]
    assert [o["op"] for o in ops[:2]] == ["gx_ug_mx4", "gx_down_mx4"]
    assert L.UP_BYTES == (3072 // 64) * (3072 // 64) * R.MX_CH and L.POOL_BYTES < 600 << 20
    b = R.builds(spec)
    assert b["gx_moe"]["env"]["GX_MX4"] == "1" and b["gx_moe"]["build_dir"].endswith("_mx4")
    monkeypatch.setenv("GX_MX4", "0")
    assert R.pack_plan(spec)["layer_types"][DENSE]["pool"][0]["op"] == "gx_ug"
    assert R.layout(spec).UP_BYTES == (3072 // 64) * (3072 // 64) * R.CH


def test_the_block_prefill_route_is_four_gemms_two_contexts_and_the_host():
    R, spec = _recipe()
    prog = R.programs(spec)
    g = prog["gptoss_prefill"]
    assert g["t"] == 256 and (g["x"], g["y"]) == ("gp_x", "gp_y")
    # q|k|v and o on gemm_bfp16's context, the experts on the MXFP4 variant's, re-pointed per expert
    k = prog["kernels"]
    assert {k[n]["context"] for n in ("gpq_qkv", "gpq_o")} == {"gpgemm"}
    assert {k[n]["context"] for n in ("gpx_glu", "gpx_down")} == {"gpmx4"}
    assert k["gpx_glu"]["patch"] == k["gpx_down"]["patch"] == "gxexpert" and "patch" not in k["gpq_qkv"]
    # shapes: hi | lo stacked weights for q|k|v and o, gate | up stacked for the experts
    assert (g["qkv"]["n"], g["qkv"]["k"], g["qkv"].get("split")) == (2 * 5120, 2880, True)
    assert (g["o"]["n"], g["o"]["k"], g["o"].get("split")) == (2 * 3072, 4096, True)
    assert (g["glu"]["n"], g["glu"]["k"], g["down"]["n"], g["down"]["k"]) == (6144, 2880, 3072, 2880)
    assert g["glu"]["expert_bytes"] == 48 * 45 * 4352 and g["down"]["expert_bytes"] == 24 * 45 * 4352
    assert prog["globals"]["gp_x"] == 256 * 4096 // 8 * 9 and prog["globals"]["gp_y"] == 256 * 10240 * 4
    assert g["window"] == {DENSE: 0, DENSE_LOCAL: 128} and g["rope_scale"] == spec.rope_scale()
    b = R.builds(spec)
    assert b["gp_glu"]["env"]["GB_B"] == "mx4" and "GB_B" not in b["gp_qkv"]["env"]
    # the experts again at 128 rows (32-row tiles, a context of its own) for the chunks that half fill 256
    assert g["t_small"] == 128 and g["glu_small"]["w"] == "gpw_glu" and g["down_small"]["w"] == "gpw_down"
    assert {k[n]["context"] for n in ("gpx_glu_s", "gpx_down_s")} == {"gpmx4s"} and k["gpx_glu_s"]["patch"] == "gxexpert"
    assert (b["gp_glu_s"]["env"]["GB_M"], b["gp_glu_s"]["env"]["GB_TM"]) == ("128", "32") and g["min_prompt"] == 12


def test_the_attention_weights_go_in_as_hi_and_lo_halves_of_one_operand(monkeypatch):
    """bfp16 keeps 8 bits of each block of eight; Q8_0's codes are 8 bits of a block of 32, and rounding them to
    bfp16 was the route's largest error (OPEN-PREFILL-GPTOSS). The lo copy of every op sits N / 2 rows on."""
    R, spec = _recipe()
    ops = R.programs(spec)["gptoss_prefill"]["weights"]["gpw_qkv"]
    assert [(o["tensor"].split(".")[-2], o["split"], o["row0"]) for o in ops] == [
        ("attn_q", "hi", 0), ("attn_q", "lo", 5120), ("attn_k", "hi", 4096), ("attn_k", "lo", 9216),
        ("attn_v", "hi", 4608), ("attn_v", "lo", 9728)]
    monkeypatch.setenv("GX_PREFILL_HILO", "0")
    g = R.programs(spec)["gptoss_prefill"]
    assert g["qkv"]["n"] == 5120 and "split" not in g["qkv"] and len(g["weights"]["gpw_qkv"]) == 3
    monkeypatch.setenv("GX_PREFILL", "0")
    assert "gptoss_prefill" not in R.programs(spec) and "gp_glu" not in R.builds(spec)


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


def test_the_role_table_knows_the_container_the_converter_actually_writes():
    """The converter deletes ffn_{up,gate,down}_exps.weight and writes ONE fused
    ffn_gate_up_down_exps.weight, so the role table has to name that."""
    from recipes.spec import ROLE_TENSORS
    assert "ffn_gate_up_down_exps.weight" in ROLE_TENSORS["gptoss"]
    assert ROLE_TENSORS["gptoss"]["ffn_gate_up_down_exps.weight"] == "experts"


def test_the_real_container_derives_mxfp4_experts_not_the_q4_1_default():
    """The shipped container is 2560-byte chunks throughout, q4_1 for the attention
    projections and the head (dtype I8) and MXFP4 for the fused expert tensor (dtype U8).

    Before this landed the caller got {} -- the everything-is-q4_1 default -- for a
    container whose experts are 4-bit FLOAT. A wrong answer with no error is the failure
    shape this repo keeps relearning."""
    from recipes.spec import quant_map_from_chunk_sizes
    real = {"model.layers.3.ffn_gate_up_down_exps.weight": 2560,
            "model.layers.3.self_attn.q_proj.weight": 2560,
            "lm_head.weight": 2560}
    dt = {"model.layers.3.ffn_gate_up_down_exps.weight": "U8",
          "model.layers.3.self_attn.q_proj.weight": "I8",
          "lm_head.weight": "I8"}
    assert quant_map_from_chunk_sizes("gptoss", real, dt) == {"experts": "mxfp4"}


def test_2560_without_dtypes_is_refused_rather_than_guessed():
    """The byte count does not name the format at 2560 - GPT-OSS ships q4_1 projections and
    MXFP4 experts at the same size. A caller with no dtypes gets a refusal naming both
    readings, not the q4_1 default it would otherwise collapse to."""
    from recipes.spec import quant_map_from_chunk_sizes
    with pytest.raises(SpecError, match="does not name the format"):
        quant_map_from_chunk_sizes("gptoss", {"model.layers.3.self_attn.q_proj.weight": 2560})


def test_a_known_role_at_an_unknown_chunk_size_is_refused_by_name():
    """The silent wrong this replaces: a role the packer places, at a chunk size it cannot
    read, used to `continue` and leave the role at the q4_1 default."""
    from recipes.spec import quant_map_from_chunk_sizes
    with pytest.raises(SpecError, match="1280"):
        quant_map_from_chunk_sizes("gptoss", {"model.layers.3.self_attn.q_proj.weight": 1280})


def test_a_tensor_with_no_role_at_an_unknown_size_is_still_ignored():
    """The refusal must not fire on tensors the packer never places - norms, biases and
    anything else a container carries - or every model stops loading."""
    from recipes.spec import quant_map_from_chunk_sizes
    assert quant_map_from_chunk_sizes("gptoss", {"model.layers.3.input_layernorm.weight": 1280}) == {}


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
