# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-K2 (Stage 1.8: sprint-REPORT §4 + the observer's
# 2026-09-27-stage1.8-k2-plumbing brief)
"""K2-Horizon-3.7B (and, at the end, the 7B's GroupRMSNorm(4)) on the dense recipe: the `_k2_hf` derivation (the frozen config),
the family routing (k2 -> dense, model_type k2_horizon), the norm_groups field (a spec
field that must not move any shipped model's hash), the LN_GROUPS plumbing into the ln
build and the catalogue's grouped point, the manifest's hf_config_check (the nested
rope_parameters), and the qwen3 regression (unchanged at norm_groups=1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes import dense as DR
from recipes.families import FAMILIES, family_module, for_spec
from recipes.manifest import manifest
from recipes.spec import DENSE, ModelSpec, SpecError, hf_model_types

# IFM/K2-Horizon-3.7B config.json, verbatim (agent/artifacts/k2/config.json, SHA256SUMS'd).
HF_K2_3_7B = {
    "model_type": "k2_horizon", "hidden_size": 2560, "intermediate_size": 10240,
    "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
    "head_dim": 128, "rope_head_dim": 128, "rms_norm_eps": 1e-06,
    "layernorm_num_groups": 2, "vocab_size": 250624, "tie_word_embeddings": False,
    "query_key_norm": False, "attention_bias": False, "attention_gate_func": None,
    "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
    "hidden_act": "silu", "use_sliding_window": False, "sliding_window": None,
}
# the tokenizer's own id count (Task A: the last id is 250019, the second eos);
# 250624 is the padded lm_head row count, a multiple of 64.
K2_REAL_VOCAB = 250620
# the checked-in spec serialises exactly this derivation
SPEC_FILE = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs" / "k2-horizon-3.7b.json"
# qwen3-4b.json's hash on main (eb656007), before norm_groups existed -- computed with
# main's pristine recipes/spec.py, byte-identical under the patched one: the field must
# not move a single shipped model's hash while it is 1 (to_dict omits it).
QWEN3_4B_HASH_ON_MAIN = "sha256:602fa1836b218cfd17b8a11628cde954587cd53ad3345a04ef1d998d23951dfd"


def derive(cfg=HF_K2_3_7B, real_vocab=K2_REAL_VOCAB) -> ModelSpec:
    return ModelSpec.from_hf_config(cfg, real_vocab=real_vocab)


def test_spec_derivation_matches_the_frozen_config():
    s = derive()
    assert s.family == "k2"
    assert s.hidden == 2560 and s.num_layers == 36 and s.intermediate == 10240
    assert s.num_heads == 32 and s.num_kv_heads == 8 and s.head_dim == 128
    assert s.rotary_dim == 128 and s.rope_theta == 10000000.0 and s.rope_scaling is None
    assert s.qk_norm is False and s.attn_gate is False
    assert s.norm_eps == 1e-6 and s.norm_groups == 2
    assert s.vocab == 250624 and s.real_vocab == K2_REAL_VOCAB
    assert s.layer_types == tuple([DENSE] * 36)
    assert s.sliding_window == 0 and s.activation == "silu"
    assert s.quant == "q4_1"
    assert s.extra["model_type"] == "k2_horizon"


def test_family_routes_to_dense_and_back():
    s = derive()
    assert family_module("k2") is DR and for_spec(s) is DR
    assert "k2" in FAMILIES and "k2" in DR.DENSE_FAMILIES
    assert hf_model_types("k2") == ["k2_horizon"]
    assert DR.qkv_bias(s) is False          # a qwen2 property, not K2's


def test_checked_in_spec_round_trips():
    s = ModelSpec.from_json(SPEC_FILE.read_text())
    # everything but extra.model, which is the export destination (recipes/load.py stamps
    # it from the model directory) and is not derivable from a config.json
    d = derive()
    assert s.extra["model"] == "K2-Horizon-3.7B-NPU2"
    s.extra.pop("model")
    assert s == d
    assert json.loads(s.to_json())["norm_groups"] == 2
    with pytest.raises(SpecError):        # an unknown field stays a refusal
        ModelSpec.from_dict({**json.loads(SPEC_FILE.read_text()), "bogus": 1})


def test_norm_groups_does_not_move_shipped_hashes():
    # the field is omitted from the canonical dict while it is 1: every existing
    # spec serialises and hashes exactly as it did before the field existed
    qwen3 = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs" / "qwen3-4b.json"
    s = ModelSpec.from_json(qwen3.read_text())
    assert s.norm_groups == 1
    assert "norm_groups" not in s.to_dict()
    assert s.spec_hash() == QWEN3_4B_HASH_ON_MAIN
    # and a grouped spec carries it
    assert derive().to_dict()["norm_groups"] == 2
    with pytest.raises(SpecError):
        ModelSpec.from_dict({**derive().to_dict(), "norm_groups": 3})


def test_qwen3_spec_unaffected_by_the_field():
    hf_qwen3 = {"model_type": "qwen3", "hidden_size": 2560, "intermediate_size": 9728,
                "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
                "head_dim": 128, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
                "vocab_size": 151936, "tie_word_embeddings": True}
    s = ModelSpec.from_hf_config(hf_qwen3)
    assert s.family == "qwen3" and s.qk_norm is True and s.norm_groups == 1
    assert family_module("qwen3") is DR


@pytest.mark.parametrize("bad,why", [
    ({"query_key_norm": True}, "qk_norm is a qwen3 property"),
    ({"attention_gate_func": "sigmoid"}, "a gate is a MoE-family feature"),
    ({"attention_bias": True}, "a qkv bias is a qwen2 property"),
    ({"use_sliding_window": True, "sliding_window": 4096}, "a window is gemma3's"),
    ({"hidden_act": "gelu_tanh"}, "K2's FFN is silu"),
    ({"rope_head_dim": 64}, "a partial rotation is a different kernel"),
    ({"layernorm_num_groups": 3}, "only 1, 2 or 4 exist"),
    ({"rope_parameters": {"rope_theta": 1e7, "rope_type": "yarn"}}, "no rope scaling"),
    ({"rope_parameters": None}, "the nested layout is where the base lives"),
])
def test_config_disagreements_are_refused(bad, why):
    with pytest.raises(SpecError):
        derive({**HF_K2_3_7B, **bad})


def test_missing_head_dim_is_refused_not_inferred():
    # hidden/heads would infer 80 and silently mis-shape the attention elements
    cfg = {k: v for k, v in HF_K2_3_7B.items() if k != "head_dim"}
    with pytest.raises(SpecError):
        derive(cfg)


def test_ln_groups_reaches_the_ln_build():
    b = DR.builds(derive())
    assert b["ln"]["build_dir"] == "ln/build_2560_1e-06_g2"
    assert b["ln"]["env"] == {"LN_N": "2560", "LN_EPS": "1e-06", "LN_GROUPS": "2"}
    assert b["dx"]["build_dir"] == "dense/build_k2_h2560" and b["dx"]["env"] == {}
    # and a norm_groups=1 model keeps the build line it always had
    q3 = ModelSpec.from_json((Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs" /
                              "qwen3-4b.json").read_text())
    b3 = DR.builds(q3)
    assert b3["ln"]["build_dir"] == "ln/build_2560_1e-06"
    assert b3["ln"]["env"] == {"LN_N": "2560", "LN_EPS": "1e-06"}


def test_catalogue_groups_point():
    from recipes.catalogue import OpRangeError, require
    require("ln", width=2560, groups=2)             # E3's hardware-validated point
    require("ln", width=2560)                        # groups defaults to 1 (the old call sites)
    with pytest.raises(OpRangeError):
        require("ln", width=2048, groups=2)          # the fused kernel has one reduction


def test_hf_config_check_emits_the_semantic_fields():
    d = DR.hf_config_check(derive())
    assert d["head_dim"] == 128
    assert d["rope_parameters"] == {"rope_theta": 10000000.0, "rope_type": "default"}
    assert d["layernorm_num_groups"] == 2
    assert d["hidden_size"] == 2560 and d["intermediate_size"] == 10240
    assert "model_type" not in d and "vocab_size" in d


def test_manifest_carries_the_check():
    m = manifest(derive(), 4096)
    chk = m["hf_config_check"]
    assert chk["model_type"] == ["k2_horizon"]
    assert chk["rope_parameters"] == {"rope_theta": 10000000.0, "rope_type": "default"}
    assert chk["layernorm_num_groups"] == 2
    assert m["family"] == "k2"
    # the whole-layer recipe and layout derive (the catalogue require()s all pass)
    assert m["layers"] == ["dense"] * 36
    assert m["layout"]["hidden"] == 2560 and m["layout"]["rotary_dim"] == 128


def test_dense_layout_derives():
    r = DR.recipe(derive(), 4096)
    L, G = r.layout, r.geo
    assert G.HID == 2560 and G.NH == 32 and G.KVH == 8 and G.HD == 128
    assert G.QKNORM is False and G.GATE is False
    assert L.ELN == 5120                      # the ln core's element: HID*2 bytes, groups-agnostic
    r3 = DR.recipe(ModelSpec.from_json(
        (Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs" / "qwen3-4b.json").read_text()), 4096)
    assert r3.layout.ELN == L.ELN            # the I/O contract did not move with the groups


# IFM/K2-Horizon-7B config.json: the 3.7B's family at Qwen3-8B's widths, four norm groups.
HF_K2_7B = {**HF_K2_3_7B, "hidden_size": 4096, "intermediate_size": 12288, "layernorm_num_groups": 4}
SPEC_FILE_7B = SPEC_FILE.with_name("k2-horizon-7b.json")


def test_7b_derives_four_groups_and_round_trips():
    s = derive(HF_K2_7B)
    assert s.hidden == 4096 and s.intermediate == 12288 and s.num_layers == 36
    assert s.norm_groups == 4 and s.vocab == 250624 and s.real_vocab == K2_REAL_VOCAB
    c = ModelSpec.from_json(SPEC_FILE_7B.read_text())
    assert c.extra.pop("model") == "K2-Horizon-7B-NPU2"
    assert c == s


def test_7b_ln_build_and_host_norm_groups():
    s = derive(HF_K2_7B)
    b = DR.builds(s)
    assert b["ln"]["build_dir"] == "ln/build_4096_1e-06_g4"
    assert b["ln"]["env"] == {"LN_N": "4096", "LN_EPS": "1e-06", "LN_GROUPS": "4"}
    assert DR.hf_config_check(s)["layernorm_num_groups"] == 4
