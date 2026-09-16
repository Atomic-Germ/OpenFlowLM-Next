# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-LFM2, OPEN-SHORT-CONV-REF (canonical spec: specs/open-engine/spec.md)
"""LFM2: ten of its sixteen layers replace attention with a three-tap depthwise causal
convolution, so the layer type is a new member of LAYER_TYPES rather than a parameter on
an existing one. The other six are plain GQA attention with q/k RMSNorm, and every layer
carries the same silu-gated FFN, so an attention layer is the dense recipe's block.

The block, per token: in_proj gives 3 x hidden rows; the first third is a gate B, the
second a gate C, the third the value u. B*u goes through the depthwise conv over the last
three tokens, C multiplies the result, out_proj brings it back to hidden. No bias
anywhere, no normalisation inside the block.

The fixture is Q4NX container LFM2-1.2B-NPU2's own config.json, verbatim.
"""
from __future__ import annotations

import numpy as np
import pytest

import replica_lfm2 as RL
from recipes import dense as DR
from recipes import families, qwen35
from recipes.catalogue import OpRangeError
from recipes.spec import (FULL, LAYER_TYPES, SHORT_CONV, ModelSpec, SpecError,
                          quant_map_from_chunk_sizes)

# C:\Users\josha\.flm\models\LFM2-1.2B-NPU2\config.json, verbatim.
HF_LFM2_1_2B = {
    "architectures": ["Lfm2ForCausalLM"],
    "block_auto_adjust_ff_dim": True,
    "block_dim": 2048,
    "block_ff_dim": 12288,
    "block_ffn_dim_multiplier": 1.0,
    "block_mlp_init_scale": 1.0,
    "block_multiple_of": 256,
    "block_norm_eps": 1e-05,
    "block_out_init_scale": 1.0,
    "block_use_swiglu": True,
    "block_use_xavier_init": True,
    "bos_token_id": 1,
    "conv_L_cache": 3,
    "conv_bias": False,
    "conv_dim": 2048,
    "conv_dim_out": 2048,
    "conv_use_xavier_init": True,
    "eos_token_id": 7,
    "intermediate_size": 8192,
    "full_attn_idxs": [2, 5, 8, 10, 12, 14],
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "max_position_embeddings": 128000,
    "model_type": "lfm2",
    "norm_eps": 1e-05,
    "num_attention_heads": 32,
    "head_dim": 64,
    "num_heads": 32,
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "pad_token_id": 0,
    "rope_theta": 1000000.0,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "use_pos_enc": True,
    "vocab_size": 65536,
}

CONV_LAYERS = (0, 1, 3, 4, 6, 7, 9, 11, 13, 15)
ATTN_LAYERS = (2, 5, 8, 10, 12, 14)


# ---- the spec derivation
def test_the_layer_schedule_comes_from_full_attn_idxs():
    """Six attention layers at 2, 5, 8, 10, 12, 14; the other ten are short-conv. Read off
    the container, which holds `self_attn.*` for exactly those six and `shortconv.*` for the
    other ten."""
    spec = ModelSpec.from_hf_config(HF_LFM2_1_2B)
    assert spec.family == "lfm2"
    assert spec.num_layers == 16 and len(spec.layer_types) == 16
    assert tuple(i for i, t in enumerate(spec.layer_types) if t == FULL) == ATTN_LAYERS
    assert tuple(i for i, t in enumerate(spec.layer_types) if t == SHORT_CONV) == CONV_LAYERS
    assert spec.has_short_conv and spec.has_full
    assert not spec.has_dense and not spec.has_linear


def test_the_geometry_matches_the_container():
    spec = ModelSpec.from_hf_config(HF_LFM2_1_2B)
    assert (spec.hidden, spec.vocab, spec.intermediate) == (2048, 65536, 8192)
    assert (spec.num_heads, spec.num_kv_heads, spec.head_dim) == (32, 8, 64)
    assert spec.rotary_dim == spec.head_dim, "LFM2 rotates the whole head"
    assert spec.qk_norm and not spec.attn_gate
    assert spec.conv_kernel == 3
    assert spec.activation == "silu"
    assert spec.rope_theta == 1000000.0
    assert spec.norm_eps == 1e-05
    assert spec.rope_scaling is None and spec.sliding_window == 0
    assert spec.num_experts == 0


def test_the_ffn_width_follows_transformers_own_block_ff_dim_rule():
    """`Lfm2Config` lets `block_ff_dim` override `intermediate_size`, and `Lfm2MLP` then
    takes two thirds of it and rounds up to `block_multiple_of`: 12288 -> 8192, which is
    what the container's gate / up projections actually hold (8192 rows)."""
    assert ModelSpec.from_hf_config(HF_LFM2_1_2B).intermediate == 8192
    # a config carrying only the unadjusted width derives the same 8192
    only_block = {k: v for k, v in HF_LFM2_1_2B.items() if k != "intermediate_size"}
    assert ModelSpec.from_hf_config(only_block).intermediate == 8192
    # the adjustment off means the width is taken as written
    off = {**HF_LFM2_1_2B, "block_auto_adjust_ff_dim": False}
    assert ModelSpec.from_hf_config(off).intermediate == 12288


def test_an_explicit_layer_types_list_is_accepted_and_agrees():
    """`Lfm2Config` writes `layer_types` with "conv" / "full_attention"; a container that
    ships the list instead of the index list must derive the same spec."""
    lt = ["full_attention" if i in ATTN_LAYERS else "conv" for i in range(16)]
    listed = {k: v for k, v in HF_LFM2_1_2B.items() if k != "full_attn_idxs"}
    listed["layer_types"] = lt
    assert (ModelSpec.from_hf_config(listed).spec_hash()
            == ModelSpec.from_hf_config(HF_LFM2_1_2B).spec_hash())


def test_a_layer_type_name_the_family_does_not_have_is_refused():
    bad = {k: v for k, v in HF_LFM2_1_2B.items() if k != "full_attn_idxs"}
    bad["layer_types"] = ["conv"] * 15 + ["linear_attention"]
    with pytest.raises(SpecError, match="linear_attention"):
        ModelSpec.from_hf_config(bad)


def test_a_conv_width_that_is_not_the_hidden_size_is_refused_by_name():
    """`conv_dim` is not a ModelSpec field -- a field would move every shipped model's
    hash -- so the deriver refuses the only case where it would need to be one."""
    with pytest.raises(SpecError, match="conv_dim"):
        ModelSpec.from_hf_config({**HF_LFM2_1_2B, "conv_dim": 1024})
    with pytest.raises(SpecError, match="conv_dim_out"):
        ModelSpec.from_hf_config({**HF_LFM2_1_2B, "conv_dim_out": 1024})


def test_a_conv_bias_is_refused_by_name():
    """The installed container has none (`shortconv.conv` is the only conv tensor), and the
    block the kernel implements has no place to add one."""
    with pytest.raises(SpecError, match="conv_bias"):
        ModelSpec.from_hf_config({**HF_LFM2_1_2B, "conv_bias": True})


def test_a_missing_key_is_named():
    for key in ("hidden_size", "num_hidden_layers", "num_key_value_heads", "rope_theta"):
        broken = {k: v for k, v in HF_LFM2_1_2B.items() if k != key}
        with pytest.raises(SpecError, match=key):
            ModelSpec.from_hf_config(broken)


def test_the_spec_round_trips_through_json():
    spec = ModelSpec.from_hf_config(HF_LFM2_1_2B)
    again = ModelSpec.from_json(spec.to_json())
    assert again == spec and again.spec_hash() == spec.spec_hash()
    assert again.layer_types[0] == SHORT_CONV


# ---- the layer type is a VALUE, not a field
def test_short_conv_is_a_layer_type_value():
    assert SHORT_CONV in LAYER_TYPES
    assert set(LAYER_TYPES) >= {"linear_attention", "full_attention", "dense", "dense_local",
                                "short_conv"}


def test_no_shipped_models_hash_moved():
    """`spec_hash()` hashes every dataclass FIELD, so a new field would move all of these.
    Adding a member to the `layer_types` tuple cannot: no shipped spec uses it. These are
    the hashes at the commit before the short-conv layer type existed. qwen25-3b joined
    after, when Qwen2.5 was measured onto the fast attention path; its hash is the one the
    engine logs for the installed container, and lfm2-1.2b joined the same way. The lfm2
    entry is also the one spec here whose own family owns the new layer type, so it is
    what would catch a `short_conv` change that moved the hash. minicpm5-2b arrived from
    main in #85 and is pinned at the value it hashes to here, which is the point: the
    eleven that predate this branch did not move, so `qkv_bias` is a kernel knob and not
    a ModelSpec field."""
    import pathlib

    from recipes.load import load_spec
    frozen = {
        "gemma3-12b.json": "sha256:df7005de2191d524c50ed90b8bb41f92ff222d2699a532652689a5fbdce21c46",
        "gemma3-4b.json": "sha256:3496169f71a4b2d6679065fb35888493ccf695084625797ce8f200e6f0d84681",
        "granite42-3b.json": "sha256:695134de825eb93e39a7b31008bb5791d2e27c8744c0980c2ed8edbe083b5656",
        "hy-mt2-7b.json": "sha256:857e09843a229d259ec52d5174560e429af3597b446142eab140977a571cd1d7",
        "llama31-8b.json": "sha256:e4baa37e429635bc133b50b9bdac750abb02e6ce58c0d10efcb06bc2bd29a4bb",
        "phi4-mini-4b.json": "sha256:76d8c86eaad5a6e5f5853a47a293cf196dc971455df8d9406d8fbcdbda3f4860",
        "qwen25-3b.json": "sha256:e32bfd7e950ccd7b304aa530cb87d2fe903a41c4e3df9e634588354dfd4a8953",
        "qwen3-4b.json": "sha256:602fa1836b218cfd17b8a11628cde954587cd53ad3345a04ef1d998d23951dfd",
        "qwen35-9b.json": "sha256:4105149d2111c0c7e208e1a6c6f8273064394bfb2b5f010fa0fe5c0dfbc6711c",
        "qwen36-35b-a3b.json": "sha256:32e980528551df6ae76741cce159c2e79a7a7daa6b0d01d78f665a1164b9f780",
        "lfm2-1.2b.json": "sha256:fd500fa0be3851a42ecd72a97346ef21f5df2c63866d0f6196a4bf3ed6aeabb8",
        "minicpm5-2b.json": "sha256:3297b81a61cb0beb690fd0c8c34515be2bc08c089de26441ab463cccae994974",
    }
    specs = pathlib.Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"
    assert {p.name for p in specs.glob("*.json")} == set(frozen), "a new shipped spec wants a hash here"
    for name, want in frozen.items():
        assert load_spec(specs / name).spec_hash() == want, name


def test_the_conv_geometry_rides_on_fields_that_already_exist():
    """`conv_kernel` is DeltaNet's field; LFM2 reuses it and its conv width is `hidden`.
    Nothing about the short conv is a new field."""
    d = ModelSpec.from_hf_config(HF_LFM2_1_2B).to_dict()
    for key in ("conv_dim", "conv_channels", "short_conv", "full_attn_idxs", "conv_bias"):
        assert key not in d, key


# ---- routing
def test_lfm2_routes_to_its_own_recipe_not_the_dense_one():
    """Routing it to `dense` would build a layer with attention where the conv belongs and
    then report parity against a replica making the same mistake."""
    from recipes import lfm2 as LR
    assert families.family_module("lfm2") is LR


def test_a_family_with_no_recipe_still_says_so_by_name():
    """The NOT_IMPLEMENTED table is the standing rule's mechanism; lfm2 left it when the
    design landed, and gptoss is what it holds now."""
    with pytest.raises(NotImplementedError) as e:
        families.family_module("gptoss")
    assert "gptoss" in str(e.value)


def test_lfm2_is_in_the_implemented_family_list():
    """It joined when designs/short_conv landed; the catalogue points followed with the
    hardware pass on 2026-09-13 (OPEN-SHORT-CONV-KERNEL)."""
    assert "lfm2" in families.FAMILIES


def test_for_spec_routes_an_lfm2_spec_to_its_own_recipe():
    from recipes import lfm2 as LR
    spec = ModelSpec.from_hf_config(HF_LFM2_1_2B)
    assert families.for_spec(spec) is LR


def test_the_dense_and_qwen35_recipes_refuse_an_lfm2_spec():
    spec = ModelSpec.from_hf_config(HF_LFM2_1_2B)
    with pytest.raises(OpRangeError, match="lfm2"):
        DR.recipe(spec)
    with pytest.raises(OpRangeError, match="lfm2"):
        qwen35.recipe(spec)


def test_the_attention_geometry_entered_the_catalogue_as_a_whole_tuple():
    """qk_norm at head dim 64 over 32 heads and 8 kv heads passed on 2026-09-13. Llama 3.2
    1B's tuple is the same shape WITHOUT the norms, the near-miss OPEN-OP-RANGE exists to
    catch, so the tuple is accepted with the norms and a gate on top is still refused."""
    from recipes.catalogue import require
    require("attn", head_dim=64, num_heads=32, num_kv_heads=8, rotary_dim=64,
            rope_theta=1e6, qk_norm=True, attn_gate=False)
    with pytest.raises(OpRangeError, match=r"\(64, 32, 8, 64, True, True, False, False\)"):
        require("attn", head_dim=64, num_heads=32, num_kv_heads=8, rotary_dim=64,
                rope_theta=1e6, qk_norm=True, attn_gate=True)


# ---- the per-role weight format
def test_the_short_conv_projections_reuse_the_linear_roles():
    """The sequence-mixing block's fused input projection and its output projection are the
    same two roles a DeltaNet layer has, so no new quant role is needed -- and a new role
    would appear in every spec's quant map."""
    pre = "model.layers.3."
    all_q4 = {pre + n: 5120 for n in ("shortconv.in_proj.weight", "shortconv.out_proj.weight",
                                      "mlp.up_proj.weight", "mlp.gate_proj.weight",
                                      "mlp.down_proj.weight")}
    all_q4["model.layers.2.self_attn.q_proj.weight"] = 5120
    assert quant_map_from_chunk_sizes("lfm2", all_q4) == {}, "the installed container is all 4-bit"
    q8_out = {**all_q4, pre + "shortconv.out_proj.weight": 8704}
    assert quant_map_from_chunk_sizes("lfm2", q8_out) == {"linear_out": "q8"}
    q8_in = {**all_q4, pre + "shortconv.in_proj.weight": 8704}
    assert quant_map_from_chunk_sizes("lfm2", q8_in) == {"linear": "q8"}


# ---- the fp64 reference for the short-conv block
#
# Three taps, two channels, hand-computed. in_proj rows are [B | C | u]:
#   B = (x0, x1)      C = (2*x0, 3*x1)      u = (x0 + x1, x0 - x1)
# conv weight w[c, k]; k = 2 is the current token's tap.
HID, L = 2, 3
W_IN = np.array([[1.0, 0.0], [0.0, 1.0],          # B
                 [2.0, 0.0], [0.0, 3.0],          # C
                 [1.0, 1.0], [1.0, -1.0]])        # u
W_CONV = np.array([[0.5, 0.25, 2.0],
                   [1.0, -1.0, 0.5]])
W_OUT = np.array([[1.0, 1.0], [0.0, 2.0]])
TOKENS = [np.array([1.0, 2.0]), np.array([0.0, 1.0]), np.array([2.0, 0.0])]
# Bx per token: [3, -2], [0, -1], [4, 0]
WANT_OUT = [np.array([6.0, -12.0]), np.array([4.5, 9.0]), np.array([38.0, 0.0])]
WANT_STATE = [np.array([[0.0, 0.0], [3.0, -2.0]]),
              np.array([[3.0, -2.0], [0.0, -1.0]]),
              np.array([[0.0, -1.0], [4.0, 0.0]])]


def test_the_block_reproduces_three_hand_computed_tokens():
    state = np.zeros((L - 1, HID))
    for t, (x, want, want_state) in enumerate(zip(TOKENS, WANT_OUT, WANT_STATE)):
        out, state = RL.short_conv_step(x, W_IN, W_CONV, W_OUT, state)
        assert out.tolist() == want.tolist(), f"token {t}"
        assert state.tolist() == want_state.tolist(), f"token {t} state"


def test_the_cache_holds_the_gated_product_not_the_layer_input():
    """What rides in the conv state is B*u, the product the convolution actually reduces
    over -- transformers caches the same thing. Caching the block input instead would give
    the right answer only at position 0."""
    out, state = RL.short_conv_step(TOKENS[0], W_IN, W_CONV, W_OUT, np.zeros((L - 1, HID)))
    assert state[-1].tolist() == [3.0, -2.0]        # B*u, not x (1, 2) and not B (1, 2)


def test_the_newest_token_pairs_with_the_last_tap():
    """The one orientation error a transposed weight would make. With only the FIRST tap
    non-zero the very first token sees nothing, because its own tap is the LAST one."""
    first_tap = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out, _ = RL.short_conv_step(TOKENS[0], W_IN, first_tap, W_OUT, np.zeros((L - 1, HID)))
    assert out.tolist() == [0.0, 0.0]
    last_tap = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    out, _ = RL.short_conv_step(TOKENS[0], W_IN, last_tap, W_OUT, np.zeros((L - 1, HID)))
    # conv_out = Bx = [3, -2]; y = C * conv_out = [2*3, 6*-2] = [6, -12]; W_OUT @ y
    assert out.tolist() == [-6.0, -24.0]


def test_decoding_from_a_zero_state_is_exact_prefill():
    """The padded-convolution form transformers uses for a whole prompt, against the
    one-token-at-a-time form the NPU runs. They must agree to the bit."""
    xs = np.array(TOKENS)
    seq = RL.short_conv_sequence(xs, W_IN, W_CONV, W_OUT)
    state = np.zeros((L - 1, HID))
    for t in range(len(TOKENS)):
        out, state = RL.short_conv_step(xs[t], W_IN, W_CONV, W_OUT, state)
        assert seq[t].tolist() == out.tolist(), t


def test_the_convolution_is_causal():
    """Appending a token cannot change an earlier output. The receptive field is three
    tokens back and no further, which is what makes the state two rows wide."""
    xs = np.array(TOKENS)
    a = RL.short_conv_sequence(xs, W_IN, W_CONV, W_OUT)
    b = RL.short_conv_sequence(np.vstack([xs, [[9.0, -9.0]]]), W_IN, W_CONV, W_OUT)
    assert b[:len(xs)].tolist() == a.tolist()
    # and a change four tokens back is out of reach by position 3
    xs2 = xs.copy()
    xs2[0] = [-5.0, 7.0]
    c = RL.short_conv_sequence(np.vstack([xs2, [[9.0, -9.0]]]), W_IN, W_CONV, W_OUT)
    assert c[3].tolist() == b[3].tolist()
    assert c[0].tolist() != a[0].tolist()


def test_a_conv_state_of_the_wrong_depth_is_refused():
    with pytest.raises(ValueError, match="conv state"):
        RL.short_conv_step(TOKENS[0], W_IN, W_CONV, W_OUT, np.zeros((L, HID)))


def test_an_in_projection_that_is_not_three_times_hidden_is_refused():
    with pytest.raises(ValueError, match="3 x"):
        RL.short_conv_step(TOKENS[0], W_IN[:4], W_CONV, W_OUT, np.zeros((L - 1, HID)))


def test_the_reference_computes_in_float64_whatever_it_is_given():
    state = np.zeros((L - 1, HID), np.float32)
    out, new = RL.short_conv_step(TOKENS[0].astype(np.float32), W_IN.astype(np.float32),
                                  W_CONV.astype(np.float32), W_OUT.astype(np.float32), state)
    assert out.dtype == np.float64 and new.dtype == np.float64
    assert out.tolist() == WANT_OUT[0].tolist()


# ---- which half of the replica a layer goes through
def test_the_replica_dispatches_by_layer_type():
    spec = ModelSpec.from_hf_config(HF_LFM2_1_2B)
    assert [RL.layer_kind(spec, l) for l in range(16)] == [
        "attn" if l in ATTN_LAYERS else "short_conv" for l in range(16)]


def test_an_attention_layer_is_the_dense_replica_unchanged():
    """LFM2's attention layer is norm -> GQA with q/k RMSNorm -> residual -> norm -> silu
    FFN -> residual, which is exactly what `replica_dense.dense_decode` computes, on tensor
    names the container already uses. Reuse, not a second copy of the same math."""
    from replica_dense import dense_decode
    assert RL.attn_decode is dense_decode


def test_the_conv_state_shape_the_engine_has_to_carry():
    spec = ModelSpec.from_hf_config(HF_LFM2_1_2B)
    assert RL.conv_state_shape(spec) == (2, 2048)
    assert RL.conv_state_bytes(spec) == 2 * 2048 * 4
