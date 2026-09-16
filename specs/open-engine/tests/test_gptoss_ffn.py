# Traces: OPEN-GPTOSS-FFN-REF, OPEN-FAMILY-GPTOSS, OPEN-ATTN-SINK
# (canonical spec: specs/open-engine/spec.md)
"""The GPT-OSS pieces that are not the sink: the widths a build has to survive, the clamped
SwiGLU in the experts, the biases on the router and on all three expert projections, the
fused gate/up row order, and the whole layer composed with a sliding window.

`replica_gptoss` is the fp64 reference. Every check here runs it against transformers' own
`GptOssExperts`, `GptOssTopKRouter`, `GptOssMLP` or `GptOssDecoderLayer` with the same
parameters -- an independent source, not our dequantiser -- so a reference that agrees with
itself and with nothing else cannot pass. The arithmetic-only tests need neither torch nor a
model; the comparisons skip without torch.
"""
from __future__ import annotations

import numpy as np
import pytest

from replica_gptoss import (ALPHA, LIMIT, clamped_swiglu, expert_ffn, fuse_gate_up,
                            gptoss_layer_step, moe_block, route, sigmoid, sink_attention,
                            split_gate_up, window_start)

RNG = np.random.default_rng(20260913)

# A toy GPT-OSS: the same shape rules as the 20B (head_dim from its own key, GQA 4:1, an
# MoE with no dense FFN, a sliding window on the even layers) at a size a CPU test can run.
TINY = dict(hidden_size=32, intermediate_size=16, num_hidden_layers=2, num_attention_heads=8,
            num_key_value_heads=2, head_dim=4, num_local_experts=6, num_experts_per_tok=3,
            vocab_size=64, sliding_window=3, rms_norm_eps=1e-5, attention_bias=True)


def _tiny_config(**over):
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    cfg = GptOssConfig(**{**TINY, **over})
    cfg._attn_implementation = "eager"
    return cfg


# ---- the widths


def test_gpt_oss_20b_gets_one_core_at_its_own_hidden_size():
    """The thing that blocks a build, and it is not the arithmetic. 2880 is 45 bands of 64
    and shares no factor above 1 with the q width's 64 bands or the kv width's 8, so
    `cores_for` falls all the way through to a single core -- an eighth of the array. Gemma 3
    12B's answer to a non-dividing hidden was four cores, which measured nearly free; one
    core is a different thing, so this family needs its widths padded instead."""
    import dataclasses

    from recipes.dense import cores_for
    from recipes.spec import ModelSpec
    from test_gptoss import HF_GPTOSS_20B
    spec = ModelSpec.from_hf_config(HF_GPTOSS_20B)
    assert (spec.hidden, spec.attn_q_width, spec.attn_kv_width) == (2880, 4096, 512)
    assert cores_for(spec) == 1
    padded = dataclasses.replace(spec, hidden=3072, moe_intermediate=3072)
    assert cores_for(padded) == 8, "3072 is the smallest width that gives the whole array"
    # and the same pad makes the q4_1 chunk's 256 columns tile: 2880 is 11.25 of them
    assert 2880 % 256 != 0 and 3072 % 256 == 0


# ---- the activation


def test_the_clamped_swiglu_is_not_silu_times_up():
    """The three differences from every other family's FFN, each isolated. A test that only
    compared the whole expression would pass with alpha folded into the clamp."""
    g, u = 0.5, 0.25
    assert clamped_swiglu(g, u) == pytest.approx((u + 1) * g * sigmoid(ALPHA * g), rel=1e-15)
    # the offset: an up of exactly -1 kills the channel, an up of 0 passes the gate through
    assert clamped_swiglu(g, -1.0) == pytest.approx(0.0, abs=1e-15)
    assert clamped_swiglu(g, 0.0) == pytest.approx(g * sigmoid(ALPHA * g), rel=1e-15)
    # alpha is inside the sigmoid only, so plain silu is a different number here
    assert clamped_swiglu(g, 0.0) != pytest.approx(g * sigmoid(g), rel=1e-6)


def test_gate_is_clipped_above_only_and_up_is_clipped_both_ways():
    """Asymmetric on purpose: a very negative gate still shuts the channel (which a symmetric
    clamp would floor at -7 and leak), while up saturates in both directions."""
    assert clamped_swiglu(LIMIT + 5.0, 0.0) == pytest.approx(clamped_swiglu(LIMIT, 0.0), rel=1e-15)
    assert clamped_swiglu(-100.0, 0.0) == pytest.approx(-100.0 * sigmoid(-170.2), abs=1e-30)
    assert clamped_swiglu(-100.0, 0.0) != pytest.approx(clamped_swiglu(-LIMIT, 0.0), rel=1e-6)
    assert clamped_swiglu(1.0, LIMIT + 5.0) == pytest.approx(clamped_swiglu(1.0, LIMIT), rel=1e-15)
    assert clamped_swiglu(1.0, -LIMIT - 5.0) == pytest.approx(clamped_swiglu(1.0, -LIMIT),
                                                              rel=1e-15)


def test_the_activation_is_vectorised_elementwise():
    g, u = RNG.normal(0, 6, 64), RNG.normal(0, 6, 64)
    got = clamped_swiglu(g, u)
    assert got.shape == (64,)
    for i in (0, 7, 63):
        assert got[i] == pytest.approx(clamped_swiglu(g[i], u[i]), rel=1e-15)


def test_the_activation_matches_transformers_apply_gate():
    """The authority. `_apply_gate` takes the FUSED row order, so the reference's split pair
    has to be interleaved back before the comparison -- which pins the row order too."""
    pytest.importorskip("torch")
    import torch
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    cfg = _tiny_config()
    with torch.device("meta"):
        ex = GptOssExperts(cfg)
    g, u = RNG.normal(0, 5, 16), RNG.normal(0, 5, 16)
    want = ex._apply_gate(torch.tensor(fuse_gate_up(g, u), dtype=torch.float64))
    assert clamped_swiglu(g, u) == pytest.approx(want.numpy(), rel=1e-12, abs=1e-14)
    assert (ex.alpha, ex.limit) == (ALPHA, LIMIT), "alpha and the limit are family constants"


# ---- the fused row order


def test_gate_is_the_even_rows_and_up_the_odd():
    gu = np.arange(12.0)
    g, u = split_gate_up(gu)
    assert list(g) == [0, 2, 4, 6, 8, 10] and list(u) == [1, 3, 5, 7, 9, 11]
    assert list(fuse_gate_up(g, u)) == list(gu)


def test_the_split_works_down_a_stacked_expert_axis():
    """What a packer actually holds: [experts, 2 * moe_intermediate, hidden] with the fused
    axis in the middle, not last."""
    gu = RNG.normal(size=(3, 8, 5))
    g, u = split_gate_up(gu, axis=1)
    assert g.shape == u.shape == (3, 4, 5)
    assert np.array_equal(g[1, 2], gu[1, 4]) and np.array_equal(u[1, 2], gu[1, 5])
    assert np.array_equal(fuse_gate_up(g, u, axis=1), gu)


def test_fusing_a_mismatched_pair_is_refused():
    with pytest.raises(ValueError, match="must match"):
        fuse_gate_up(np.zeros(4), np.zeros(5))


# ---- one expert


def test_one_expert_matches_transformers_with_the_fused_weight_split_apart():
    """The reference takes gate and up as two matrices, which is what a GGUF source and so
    `q4nx-build/configs/gpt-oss.json` carry; transformers keeps one fused interleaved
    tensor. Same numbers or the de-interleave rule is wrong."""
    torch = pytest.importorskip("torch")
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    cfg = _tiny_config()
    ex = GptOssExperts(cfg).to(torch.float64)
    with torch.no_grad():
        for p in (ex.gate_up_proj, ex.gate_up_proj_bias, ex.down_proj, ex.down_proj_bias):
            p.copy_(torch.tensor(RNG.normal(0, 0.7, tuple(p.shape))))
    e = 2
    x = RNG.normal(0, 1.0, cfg.hidden_size)
    # HF stores gate_up_proj as [E, hidden, 2 * inter] -- the transpose of the linear
    # convention -- so the split runs down its last axis and each half transposes.
    gu = ex.gate_up_proj[e].detach().numpy()
    Wg, Wu = (h.T for h in split_gate_up(gu, axis=-1))
    bg, bu = split_gate_up(ex.gate_up_proj_bias[e].detach().numpy())
    Wd = ex.down_proj[e].detach().numpy().T
    bd = ex.down_proj_bias[e].detach().numpy()
    got = expert_ffn(x, Wg, bg, Wu, bu, Wd, bd)
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float64)
        gate_up = xt @ ex.gate_up_proj[e] + ex.gate_up_proj_bias[e]
        want = ex._apply_gate(gate_up) @ ex.down_proj[e] + ex.down_proj_bias[e]
    assert got == pytest.approx(want.numpy(), rel=1e-11, abs=1e-13)


def test_the_expert_biases_actually_reach_the_output():
    """All three of them. Dropping any one is a silent wrong answer the whole-block test
    would also catch, but not name."""
    H, F = 6, 4
    Wg, Wu, Wd = np.zeros((F, H)), np.zeros((F, H)), np.eye(H, F)
    x = np.zeros(H)
    base = expert_ffn(x, Wg, np.zeros(F), Wu, np.zeros(F), Wd, np.zeros(H))
    assert base == pytest.approx(np.zeros(H), abs=1e-15), "a zero gate gives a zero output"
    bg = np.full(F, 2.0)
    assert expert_ffn(x, Wg, bg, Wu, np.zeros(F), Wd, np.zeros(H))[0] == \
        pytest.approx(clamped_swiglu(2.0, 0.0), rel=1e-14)
    bu = np.full(F, 3.0)
    assert expert_ffn(x, Wg, bg, Wu, bu, Wd, np.zeros(H))[0] == \
        pytest.approx(clamped_swiglu(2.0, 3.0), rel=1e-14)
    bd = np.arange(float(H))
    no_bd = expert_ffn(x, Wg, bg, Wu, bu, Wd, np.zeros(H))
    assert expert_ffn(x, Wg, bg, Wu, bu, Wd, bd) == pytest.approx(no_bd + bd, rel=1e-14)


# ---- the router


def test_the_router_bias_moves_the_choice_and_the_weights():
    """Not decoration: a bias big enough on one expert takes a slot that the weights alone
    would not have given it."""
    Wr = np.zeros((4, 3))
    Wr[0] = [1.0, 0, 0]
    x = np.array([1.0, 0, 0])
    idx, w = route(x, Wr, np.zeros(4), 2)
    assert idx[0] == 0
    idx2, w2 = route(x, Wr, np.array([0.0, 5.0, 0, 0]), 2)
    assert list(idx2[:2]) == [1, 0]
    assert w2[0] > w[0]


def test_the_router_matches_transformers_top_k_router():
    """Including the tie-break: HF's topk and the reference's stable argsort have to pick the
    same experts, or a comparison against the device is comparing different arithmetic."""
    torch = pytest.importorskip("torch")
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter
    cfg = _tiny_config()
    r = GptOssTopKRouter(cfg).to(torch.float64)
    with torch.no_grad():
        r.weight.copy_(torch.tensor(RNG.normal(0, 1.0, tuple(r.weight.shape))))
        r.bias.copy_(torch.tensor(RNG.normal(0, 1.0, tuple(r.bias.shape))))
    x = RNG.normal(0, 1.0, cfg.hidden_size)
    with torch.no_grad():
        _, scores, indices = r(torch.tensor(x, dtype=torch.float64)[None])
    idx, w = route(x, r.weight.detach().numpy(), r.bias.detach().numpy(), cfg.num_experts_per_tok)
    assert list(idx) == indices[0].tolist()
    assert w == pytest.approx(scores[0].numpy(), rel=1e-12)


def test_softmaxing_the_top_k_equals_renormalising_the_full_softmax():
    """Why the existing router core's shape still fits: GPT-OSS softmaxes the top-k logits
    where Qwen3.6 softmaxes all of them and renormalises the top-k. Same number."""
    lg = RNG.normal(0, 3, 32)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    top = np.argsort(-lg, kind="stable")[:4]
    idx, w = route(np.array([1.0]), lg[:, None], np.zeros(32), 4)
    assert list(idx) == list(top)
    assert w == pytest.approx(p[top] / p[top].sum(), rel=1e-13)


# ---- the whole MoE block


def test_the_moe_block_matches_transformers_mlp():
    torch = pytest.importorskip("torch")
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
    cfg = _tiny_config()
    mlp = GptOssMLP(cfg).to(torch.float64)
    with torch.no_grad():
        for p in mlp.parameters():
            p.copy_(torch.tensor(RNG.normal(0, 0.6, tuple(p.shape))))
    x = RNG.normal(0, 1.0, cfg.hidden_size)
    with torch.no_grad():
        want, _ = mlp(torch.tensor(x, dtype=torch.float64)[None, None])
    got, idx, w = moe_block(x, *_mlp_arrays(mlp), cfg.num_experts_per_tok)
    assert got == pytest.approx(want[0, 0].numpy(), rel=1e-10, abs=1e-12)
    assert len(idx) == cfg.num_experts_per_tok and w.sum() == pytest.approx(1.0, rel=1e-14)


def test_forcing_the_expert_choice_keeps_the_reference_weights():
    """`top` exists so a near-tie on the device can be compared like for like; it must not
    also replace the routing weights with something invented."""
    cfg_experts = 6
    Wr = RNG.normal(0, 1, (cfg_experts, 4))
    br = RNG.normal(0, 1, cfg_experts)
    Wg = RNG.normal(0, .5, (cfg_experts, 3, 4))
    Wu = RNG.normal(0, .5, (cfg_experts, 3, 4))
    Wd = RNG.normal(0, .5, (cfg_experts, 4, 3))
    z3, z4 = np.zeros((cfg_experts, 3)), np.zeros((cfg_experts, 4))
    x = RNG.normal(0, 1, 4)
    _, idx, w = moe_block(x, Wr, br, Wg, z3, Wu, z3, Wd, z4, 2)
    out2, idx2, w2 = moe_block(x, Wr, br, Wg, z3, Wu, z3, Wd, z4, 2, top=idx)
    assert list(idx2) == list(idx) and w2 == pytest.approx(w, rel=1e-15)
    swapped, _, sw = moe_block(x, Wr, br, Wg, z3, Wu, z3, Wd, z4, 2, top=idx[::-1])
    assert sw == pytest.approx(w[::-1], rel=1e-14), "the weights follow the forced order"
    assert swapped == pytest.approx(out2, rel=1e-13)


def _mlp_arrays(mlp):
    """(router_w, router_b, gate_w, gate_b, up_w, up_b, down_w, down_b) in the linear
    convention, out of a transformers GptOssMLP."""
    ex = mlp.experts
    gu = ex.gate_up_proj.detach().numpy()                      # [E, hidden, 2 * inter]
    g, u = split_gate_up(gu, axis=-1)
    bg, bu = split_gate_up(ex.gate_up_proj_bias.detach().numpy(), axis=-1)
    return (mlp.router.weight.detach().numpy(), mlp.router.bias.detach().numpy(),
            g.transpose(0, 2, 1), bg, u.transpose(0, 2, 1), bu,
            ex.down_proj.detach().numpy().transpose(0, 2, 1), ex.down_proj_bias.detach().numpy())


# ---- the sliding window


def test_the_window_keeps_this_token_and_the_ones_before_it():
    """HF's rule is `kv_idx > q_idx - sliding_window`, so a window of 128 admits 128 rows,
    not 129. Off by one here is a whole extra cached row on every sliding layer."""
    assert window_start(0, 128) == 0
    assert window_start(127, 128) == 0
    assert window_start(128, 128) == 1
    assert window_start(1000, 128) == 873
    assert 1000 - window_start(1000, 128) + 1 == 128
    assert window_start(1000, None) == 0 and window_start(1000, 0) == 0


def test_the_window_agrees_with_transformers_own_mask_function():
    torch = pytest.importorskip("torch")
    from transformers.masking_utils import sliding_window_causal_mask_function
    fn = sliding_window_causal_mask_function(5)
    for pos in (0, 3, 4, 5, 12):
        q = torch.tensor(pos)
        allowed = [t for t in range(pos + 1) if fn(q, q, q, torch.tensor(t))]
        assert allowed == list(range(window_start(pos, 5), pos + 1)), pos


# ---- attention with the sink


def test_a_sink_far_below_the_scores_gives_plain_gqa_attention():
    K, V = RNG.normal(size=(9, 2, 4)), RNG.normal(size=(9, 2, 4))
    q = RNG.normal(size=(8, 4))
    got = sink_attention(q, K, V, np.full(8, -1e30))
    for h in range(8):
        s = (K[:, h // 4] @ q[h]) / 2.0
        a = np.exp(s - s.max())
        assert got[h] == pytest.approx((a / a.sum()) @ V[:, h // 4], rel=1e-12)


def test_the_window_cuts_the_rows_the_head_can_see():
    K, V = RNG.normal(size=(9, 2, 4)), RNG.normal(size=(9, 2, 4))
    q = RNG.normal(size=(8, 4))
    win = sink_attention(q, K, V, np.zeros(8), sliding_window=3, pos=8)
    cut = sink_attention(q, K[6:], V[6:], np.zeros(8), pos=2)
    assert win == pytest.approx(cut, rel=1e-14)


def test_attention_matches_transformers_eager_path_with_the_sinks():
    """The whole GQA loop, sink included, against `eager_attention_forward` -- the function
    the model actually runs. The sink is the raw stored scalar there, so a reference that
    divided it by sqrt(head_dim) fails here."""
    torch = pytest.importorskip("torch")
    from transformers.models.gpt_oss.modeling_gpt_oss import (GptOssAttention,
                                                              eager_attention_forward)
    cfg = _tiny_config()
    attn = GptOssAttention(cfg, layer_idx=1).to(torch.float64)
    sinks = RNG.normal(0, 1.5, cfg.num_attention_heads)
    with torch.no_grad():
        attn.sinks.copy_(torch.tensor(sinks))
    T, nh, kvh, hd = 7, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    K, V = RNG.normal(size=(T, kvh, hd)), RNG.normal(size=(T, kvh, hd))
    q = RNG.normal(size=(nh, hd))
    qt = torch.tensor(q, dtype=torch.float64).reshape(1, 1, nh, hd).transpose(1, 2)
    kt = torch.tensor(K, dtype=torch.float64)[None].transpose(1, 2)
    vt = torch.tensor(V, dtype=torch.float64)[None].transpose(1, 2)
    with torch.no_grad():
        want, _ = eager_attention_forward(attn, qt, kt, vt, None, scaling=hd ** -0.5)
    assert sink_attention(q, K, V, sinks) == pytest.approx(want[0, 0].numpy(), rel=1e-11,
                                                           abs=1e-13)


def test_a_positive_sink_holds_back_some_of_the_weight():
    """The property the whole thing exists for: the output is the plain-softmax answer scaled
    down, by the same factor on every channel of the head."""
    K, V = RNG.normal(size=(5, 1, 4)), RNG.normal(size=(5, 1, 4))
    q = RNG.normal(size=(1, 4))
    plain = sink_attention(q, K, V, np.array([-1e30]))
    damped = sink_attention(q, K, V, np.array([0.4]))
    ratio = damped[0] / plain[0]
    assert ratio == pytest.approx(np.full(4, ratio[0]), rel=1e-12)
    assert 0.0 < ratio[0] < 1.0


# ---- the whole layer


def _layer_vs_reference(monkeypatch, fp64_norm):
    """Decode one token at a time through `gptoss_layer_step` and through transformers'
    `GptOssDecoderLayer`, and hand back (ours, theirs) for the last token. Layer 0 slides, so
    this composes the sink, the o_proj bias, the MoE with its three expert biases and the
    router bias, the window and YaRN's cos/sin scaling in one go."""
    import torch
    from transformers.models.gpt_oss import modeling_gpt_oss as M
    if fp64_norm:
        monkeypatch.setattr(M.GptOssRMSNorm, "forward", lambda self, h: self.weight * (
            h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)))
    cfg = _tiny_config(rope_parameters={"rope_type": "yarn", "rope_theta": 150000.0,
                                        "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                                        "truncate": False,
                                        "original_max_position_embeddings": 4096})
    layer = M.GptOssDecoderLayer(cfg, layer_idx=0).to(torch.float64)    # even: sliding
    assert layer.self_attn.sliding_window == cfg.sliding_window, "layer 0 slides"
    rng = np.random.default_rng(6180339)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.tensor(rng.normal(0, 0.35, tuple(p.shape))))
        for nm in ("input_layernorm", "post_attention_layernorm"):
            getattr(layer, nm).weight.copy_(torch.tensor(rng.normal(1.0, 0.1, cfg.hidden_size)))

    T = 6
    hs = torch.tensor(rng.normal(0, 1.0, (1, T, cfg.hidden_size)), dtype=torch.float64)
    rot = M.GptOssRotaryEmbedding(cfg).to(torch.float64)
    inv = rot.inv_freq.numpy().astype(np.float64)
    # GptOssRotaryEmbedding forces float32 for its own table; hand it the fp64 one instead so
    # this compares the layer rather than RoPE's rounding.
    ang = np.arange(T)[:, None] * inv[None, :]
    cos = torch.tensor(np.cos(ang) * float(rot.attention_scaling), dtype=torch.float64)[None]
    sin = torch.tensor(np.sin(ang) * float(rot.attention_scaling), dtype=torch.float64)[None]
    mask = torch.full((1, 1, T, T), float("-inf"), dtype=torch.float64)
    for i in range(T):
        for j in range(window_start(i, cfg.sliding_window), i + 1):
            mask[0, 0, i, j] = 0.0
    with torch.no_grad():
        want = layer(hs, attention_mask=mask, position_ids=torch.arange(T)[None],
                     position_embeddings=(cos, sin))
    want = want[0] if isinstance(want, tuple) else want

    w, geom = _layer_arrays(layer), _geom(cfg, inv, rot)
    K = V = np.zeros((0, cfg.num_key_value_heads, cfg.head_dim))
    got = None
    for t in range(T):
        got, K, V = gptoss_layer_step(w, hs[0, t].numpy(), K, V, t, **geom)
    return got, want[0, -1].numpy(), (w, geom, hs[0, 0].numpy())


def test_a_whole_layer_matches_the_transformers_decoder_layer(monkeypatch):
    pytest.importorskip("torch")
    got, ref, (w, geom, x0) = _layer_vs_reference(monkeypatch, fp64_norm=False)
    assert np.abs(got - ref).max() < 1e-5 * np.abs(ref).max()
    # and the tolerance discriminates: losing any one of the new pieces is orders coarser
    z = np.zeros((0, geom["num_kv_heads"], geom["head_dim"]))
    good, _, _ = gptoss_layer_step(w, x0, z, z, 0, **geom)
    for broken in ({"sinks": np.full(geom["num_heads"], -1e30)},
                   {"bo": np.zeros_like(w["bo"])},
                   {"router_b": np.zeros_like(w["router_b"])},
                   {"gate_b": np.zeros_like(w["gate_b"])},
                   {"up_b": np.zeros_like(w["up_b"])},
                   {"down_b": np.zeros_like(w["down_b"])}):
        bad, _, _ = gptoss_layer_step({**w, **broken}, x0, z, z, 0, **geom)
        assert np.abs(bad - good).max() > 1e-3 * np.abs(good).max(), list(broken)


def test_the_residual_gap_is_transformers_fp32_rms_norm_and_nothing_else(monkeypatch):
    """`GptOssRMSNorm` computes its variance in fp32 whatever the parameter dtype, and that
    is the whole of the 1e-6 above. Give it an fp64 variance and the two agree to fp64."""
    pytest.importorskip("torch")
    got, ref, _ = _layer_vs_reference(monkeypatch, fp64_norm=True)
    assert np.abs(got - ref).max() < 1e-12 * np.abs(ref).max()


def _geom(cfg, inv, rot):
    return dict(num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim, top_k=cfg.num_experts_per_tok, inv_freq=inv,
                rope_scale=float(rot.attention_scaling), norm_eps=cfg.rms_norm_eps,
                sliding_window=cfg.sliding_window)


def _layer_arrays(layer):
    a, mlp = layer.self_attn, layer.mlp
    (rw, rb, gw, gb, uw, ub, dw, db) = _mlp_arrays(mlp)
    return dict(ln1=layer.input_layernorm.weight.detach().numpy(),
                ln2=layer.post_attention_layernorm.weight.detach().numpy(),
                wq=a.q_proj.weight.detach().numpy(), bq=a.q_proj.bias.detach().numpy(),
                wk=a.k_proj.weight.detach().numpy(), bk=a.k_proj.bias.detach().numpy(),
                wv=a.v_proj.weight.detach().numpy(), bv=a.v_proj.bias.detach().numpy(),
                wo=a.o_proj.weight.detach().numpy(), bo=a.o_proj.bias.detach().numpy(),
                sinks=a.sinks.detach().numpy(), router_w=rw, router_b=rb,
                gate_w=gw, gate_b=gb, up_w=uw, up_b=ub, down_w=dw, down_b=db)
