# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-LFM2MOE, OPEN-FAMILY-LFM2
# (canonical spec: specs/open-engine/spec.md)
"""LiquidAI LFM2-MoE (LFM2.5-24B-A2B): LFM2's hybrid -- short depthwise causal conv layers and GQA attention
layers with per-head q / k norms and a NEOX RoPE -- with the FFN of every layer past the first two replaced by
64 routed SiLU experts of 1536 (sigmoid router, a selection-only expert bias, top 4, renormalised, no shared
expert) and a tied head.

The derivation, the recipe's emission and the reference's arithmetic run here -- pure functions over a config
dict and synthetic arrays, no model and no hardware. The fixture is LiquidAI/LFM2.5-24B-A2B's config.json @
3fcfdbfa (fetched 2026-10-31), written out.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from recipes import families
from recipes.spec import FULL, SHORT_CONV, ModelSpec, SpecError

ROOT = Path(__file__).resolve().parents[3]
ATT = list(range(2, 40, 4))
HF_LFM2MOE = {
    "architectures": ["Lfm2MoeForCausalLM"], "model_type": "lfm2_moe", "bos_token_id": 124894,
    "conv_L_cache": 3, "conv_bias": False, "dtype": "bfloat16", "eos_token_id": 124900, "hidden_size": 2048,
    "initializer_range": 0.02, "intermediate_size": 11776,
    "layer_types": ["full_attention" if i in ATT else "conv" for i in range(40)],
    "max_position_embeddings": 131072, "moe_intermediate_size": 1536, "norm_eps": 1e-05, "norm_topk_prob": True,
    "num_attention_heads": 32, "num_dense_layers": 2, "num_experts": 64, "num_experts_per_tok": 4,
    "num_hidden_layers": 40, "num_key_value_heads": 8, "pad_token_id": 124893,
    "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}, "routed_scaling_factor": 1.0,
    "tie_word_embeddings": True, "transformers_version": "5.12.1", "use_cache": True, "use_expert_bias": True,
    "vocab_size": 128000,
}


def spec(**over):
    cfg = copy.deepcopy(HF_LFM2MOE)
    cfg.update(over)
    return ModelSpec.from_hf_config(cfg)


# ---- the derivation


def test_the_spec_is_its_own_family_with_the_published_geometry():
    s = spec()
    assert s.family == "lfm2moe"
    assert (s.hidden, s.num_layers, s.vocab) == (2048, 40, 128000)
    assert (s.num_heads, s.num_kv_heads, s.head_dim, s.rotary_dim) == (32, 8, 64, 64)
    assert (s.num_experts, s.experts_per_tok, s.moe_intermediate) == (64, 4, 1536)
    assert (s.shared_expert_intermediate, s.intermediate, s.dense_lead) == (0, 11776, 2)
    assert s.qk_norm and not s.attn_gate and s.activation == "silu" and s.conv_kernel == 3
    assert s.norm_eps == 1e-5 and s.rope_theta == 1e7 and s.sliding_window == 0
    assert s.extra["router_scale"] == 1.0


def test_the_layer_types_are_lfm2s_names():
    s = spec()
    assert [i for i, t in enumerate(s.layer_types) if t == FULL] == ATT
    assert s.layer_types.count(SHORT_CONV) == 30


@pytest.mark.parametrize("key,value", [
    ("conv_bias", True), ("use_expert_bias", False), ("norm_topk_prob", False), ("tie_word_embeddings", False),
    ("rope_parameters", {"rope_theta": 1e7, "rope_type": "yarn", "factor": 4.0}),
])
def test_a_config_that_is_not_lfm2moe_arithmetic_is_refused_by_name(key, value):
    with pytest.raises(SpecError, match=key.split("_")[0]):
        spec(**{key: value})


def test_the_spec_round_trips_and_the_family_routes_to_its_own_recipe():
    s = spec()
    assert ModelSpec.from_dict(s.to_dict()) == s
    assert families.for_spec(s).__name__ == "recipes.lfm2moe"
    assert "lfm2moe" in families.FAMILIES
    from recipes import flint, hyv3, lfm2
    for other in (lfm2.recipe, flint.geometry, hyv3.geometry):
        with pytest.raises(Exception):
            other(s)


def test_lfm2_still_derives_its_own_family():
    s = ModelSpec.from_hf_config({**HF_LFM2MOE, "model_type": "lfm2", "rope_theta": 1e6})
    assert s.family == "lfm2"


# ---- the recipe


def test_the_widths_need_no_padding_and_tile_eight_cores():
    from recipes import lfm2moe as R
    s = spec()
    G = R.geometry(s)
    assert (G.HID, G.QW, G.KVW, G.FF, G.HD) == (2048, 2048, 512, 1536, 64)
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.SC_PC, G.DOWN_PC) == (4, 1, 4, 4, 4)
    assert (G.SLOTS, G.DENSE_SLOTS, G.ACORES, G.NHL, G.SC_ELEMS) == (4, 23, 4, 8, 4)
    fm, fd = R.ffn(s, False), R.ffn(s, True)
    assert (fm.UP_PC, fm.H_ELEMS, fm.POOL_SLOTS) == (3, 2, 64), "1536 = 24 bands = 3 a core"
    assert fm.UP_BYTES == 24 * R.band_bytes(2048) == 32 * R.band_bytes(1536), "moe2: gate and down the same size"
    assert (fd.FF, fd.UP_PC, fd.SLOTS) == (512, 1, 23) and 23 * 512 == 11776


def test_the_attention_core_has_no_sink_and_the_qk_norm():
    from recipes import lfm2moe as R
    fl = R.attn_flags(R.geometry(spec()))
    for f in ("-DATTN_QKNORM=1", "-DATTN_QKNORM_POST=0", "-DATTN_GATE=0", "-DATTN_HD=64", "-DATTN_NH=32",
              "-DATTN_KVH=8", "-DATTN_NHL=8", "-DATTN_ROT=64", "-DATTN_RB=4"):
        assert f in fl
    assert not any("SINK" in f for f in fl)


def test_one_mixer_offset_serves_every_pool():
    from recipes import lfm2moe as R
    s = spec()
    L, fm, fd = R.layout(s), R.ffn(s, False), R.ffn(s, True)
    assert L.POOL_MIX == fm.POOL_DOWN + 64 * fm.UP_BYTES
    assert fd.POOL_DOWN + 23 * fd.UP_BYTES <= L.POOL_MIX
    assert L.POOL_B == L.POOL_Q == L.POOL_MIX
    assert L.POOL_OUT + 32 * R.band_bytes(2048) <= L.POOL_BYTES < 1 << 32
    plan = R.pack_plan(s)["layer_types"]
    ip = [o for o in plan["short_conv"]["pool"] if "in_proj" in o["tensor"]]
    assert [(o["dst"], o["row0"], o["rows"]) for o in ip] == [(L.POOL_B, 0, 2048), (L.POOL_C, 2048, 2048),
                                                              (L.POOL_U, 4096, 2048)], "B | C | u by rows"
    conv = [o for o in plan["short_conv_dense"]["consts"] if o["op"] == "transpose"]
    assert conv == [{"op": "transpose", "tensor": "blk.{l}.shortconv.conv.weight", "dst": L.CD_CONV,
                     "rows": 2048, "cols": 3, "elem": 4}], "the F32 taps go tap-major, kept f32"
    dn = [o for o in plan["short_conv_dense"]["pool"] if o["tensor"].endswith("ffn_down.weight")]
    assert dn[0]["cols"] == 512 and dn[0]["experts"] == 23, "the dense down's slots are its column slices"


def test_a_moe_layer_is_mixer_route_moe_and_a_dense_layer_has_no_route():
    from recipes import lfm2moe as R
    s = spec()
    P = R.programs(s)
    assert R.layer_type_names(s)[:4] == ["short_conv_dense", "short_conv_dense", "full_attention", "short_conv"]
    c = P["layer_types"]["short_conv"]
    assert [(st["op"], st["kernel"]) for st in c["program"]] == [("run", "lm_conv"), ("hy_route", "lm_moe"),
                                                                 ("run", "lm_moe")]
    assert c["buffers"]["state"] == {"kind": "linear", "bytes": 2 * 2048 * 4}
    r = c["program"][1]
    assert (r["experts"], r["topk"], r["slots"], r["shared"], r["scale"]) == (64, 4, 4, [], 1.0)
    assert r["bias"] == "blk.{l}.exp_probs_b.bias" and "dense" not in r
    a = P["layer_types"]["full_attention"]
    assert a["program"][0]["kernel"] == "lm_att" and a["buffers"]["state"]["kind"] == "kv"
    assert P["kernels"]["lm_att"]["patch"] == "attnpos" and P["kernels"]["lm_moe"]["patch"] == "moeroute2"
    assert "patch" not in P["kernels"]["lm_conv"] and "patch" not in P["kernels"]["lm_dense"]
    d = P["layer_types"]["short_conv_dense"]["program"]
    assert [(st["op"], st["kernel"]) for st in d] == [("run", "lm_conv"), ("run", "lm_dense")]
    assert R.moe_geometry(s)["experts"] == 64 and R.moe_geometry(s)["topk"] == 4
    assert R.moe_geometry(s)["share_up"] == R.layout(s).POOL_MIX
    assert P["globals"]["ptab"]["inv_freq"] == s.rope_inv_freq()


def test_the_manifest_is_its_own_gguf_view_with_a_tied_q8_head():
    from recipes.manifest import manifest
    m = manifest(spec())
    assert m["layers"][:3] == ["short_conv_dense", "short_conv_dense", "full_attention"]
    assert m["gguf"]["layers"] == m["layers"] and m["gguf"]["pack"] == m["pack"] and "gguf" not in m["gguf"]
    assert m["pack"]["lm_head"]["ops"][0]["op"] == "gx_lmhead"
    assert m["pack"]["lm_head"]["ops"][0]["tensor"] == m["pack"]["embed"]["tensor"] == "token_embd.weight"
    assert m["pack"]["norm"]["tensor"] == "token_embd_norm.weight"
    assert m["builds"]["lm_head_q8"]["env"]["LMHEAD_SCALES_F32"] == "1"


def test_the_routing_weights_sit_past_xm_in_its_last_element():
    from recipes import lfm2moe as R
    s = spec()
    L, G = R.layout(s), R.geometry(s)
    assert L.AD_W >= L.AD_XM + G.HID * 2 and L.AD_W + 4 * R.WV <= L.AD_XM + G.XM_ELEMS * 4096


def test_the_act_regions_do_not_overlap():
    from recipes import lfm2moe as R
    L = R.layout(spec())
    names = ["AD_XN", "AD_Q", "AD_KVN", "AD_OG", "AD_B", "AD_C", "AD_U", "AD_Y", "AD_OUT", "AD_RES", "AD_XM",
             "AD_H", "AD_OUT2", "AD_JUNK"]
    offs = [getattr(L, n) for n in names]
    assert offs == sorted(offs) and len(set(offs)) == len(offs) and L.AD_JUNK < L.AD_BYTES


# ---- the reference's arithmetic


def _ref():
    sys.path.insert(0, str(ROOT / "open_kernels" / "model"))
    import lfm2moe_gguf_reference as M
    return M


def test_the_reference_router_selects_with_the_bias_and_weighs_without_it():
    M = _ref()
    rng = np.random.default_rng(0)
    wr = rng.standard_normal((64, 32)).astype(np.float32)
    x = rng.standard_normal((3, 32)).astype(np.float32)
    bias = (rng.standard_normal(64) * 0.5).astype(np.float32)
    top, w = M.route(x, wr, bias, 4, 1.0)
    s = 1 / (1 + np.exp(-(x.astype(np.float64) @ wr.T.astype(np.float64))))
    for t in range(3):
        want = np.argsort(-(s[t] + bias), kind="stable")[:4]
        assert list(top[t]) == list(want)
        assert np.allclose(w[t], s[t][want] / s[t][want].sum(), rtol=1e-6)
    assert not np.array_equal(top, M.route(x, wr, np.zeros(64, np.float32), 4, 1.0)[0]), "the bias changes picks"


def test_the_reference_conv_pairs_the_last_tap_with_this_token():
    M = _ref()
    rng = np.random.default_rng(2)
    T, hid = 5, 8
    h = rng.standard_normal((T, hid)).astype(np.float32)
    w_in = rng.standard_normal((3 * hid, hid)).astype(np.float32)
    w_conv = rng.standard_normal((hid, 3)).astype(np.float32)
    eye = np.eye(hid, dtype=np.float32)
    d = {}
    y = M.short_conv(h, w_in, w_conv, eye, lambda a: a, d)
    bx = d["b"] * d["u"]
    for t in range(T):
        conv = w_conv[:, 2] * bx[t] + (w_conv[:, 1] * bx[t - 1] if t >= 1 else 0) + \
            (w_conv[:, 0] * bx[t - 2] if t >= 2 else 0)
        assert np.allclose(y[t], d["c"][t] * conv, atol=1e-5)
    assert np.allclose(d["b"], (h @ w_in.T)[:, :hid]) and np.allclose(d["u"], (h @ w_in.T)[:, 2 * hid:])
