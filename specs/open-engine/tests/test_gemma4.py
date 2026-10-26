# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-GEMMA4 (canonical spec: specs/open-engine/spec.md)
"""Gemma 4 12B on two dense geometries: spec derivation, the proportional RoPE table, the
global layers' geometry and pack plan, the per-kernel rows in the manifest, and that the
catalogue does not mistake a Gemma 4 layer for the Gemma 3 point it shares a shape with."""
from __future__ import annotations

import numpy as np
import pytest

from recipes import gemma4 as G4
from recipes.catalogue import OpRangeError
from recipes.manifest import manifest
from recipes.spec import DENSE, DENSE_LOCAL, ModelSpec, SpecError

# google/gemma-4-12B-it config.json, the text tower's fields that matter (transformers 5.15)
TC = {
    "model_type": "gemma4_unified_text", "hidden_size": 3840, "intermediate_size": 15360,
    "num_hidden_layers": 48, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256,
    "global_head_dim": 512, "num_global_key_value_heads": 1, "attention_k_eq_v": True,
    "layer_types": ["full_attention" if (l + 1) % 6 == 0 else "sliding_attention" for l in range(48)],
    "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                           "rope_type": "proportional"},
                        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
    "sliding_window": 1024, "vocab_size": 262144, "final_logit_softcapping": 30.0,
    "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-06, "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0, "use_double_wide_mlp": False, "enable_moe_block": False,
    "attention_bias": False, "tie_word_embeddings": True,
}
HF_12B = {"model_type": "gemma4_unified", "text_config": TC}


@pytest.fixture
def unvalidated(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


def spec():
    return ModelSpec.from_hf_config(HF_12B)


def test_spec_from_hf():
    s = spec()
    assert s.family == "gemma4" and s.layer_types.count(DENSE) == 8 and s.layer_types[5] == DENSE
    assert s.layer_types[0] == DENSE_LOCAL and (s.head_dim, s.num_kv_heads, s.rotary_dim) == (256, 8, 256)
    assert (s.global_head_dim, s.global_kv_heads, s.global_rotary_dim) == (512, 1, 128)
    assert s.attn_k_eq_v and s.v_norm and s.attn_scale == 1.0 and s.final_softcap == 30.0
    assert (s.rope_theta, s.rope_local_theta, s.sliding_window) == (1e6, 1e4, 1024)


def test_the_new_fields_leave_every_other_spec_alone():
    d = ModelSpec.from_dict({k: v for k, v in spec().to_dict().items()}).to_dict()
    assert d["global_head_dim"] == 512
    g3 = dict(spec().to_dict(), family="gemma3")
    for k in ("global_head_dim", "global_kv_heads", "global_rotary_dim", "attn_k_eq_v", "v_norm",
              "attn_scale", "final_softcap"):
        g3.pop(k)
    assert not any(k in ModelSpec.from_dict(g3).to_dict() for k in ("v_norm", "final_softcap"))


def test_the_e_models_and_unbuilt_features_are_refused():
    with pytest.raises(SpecError, match="E2B / E4B"):
        ModelSpec.from_hf_config({"model_type": "gemma4", "text_config": dict(TC, model_type="gemma4_text")})
    with pytest.raises(SpecError, match="num_kv_shared_layers"):
        ModelSpec.from_hf_config({"model_type": "gemma4_unified", "text_config": dict(TC, num_kv_shared_layers=20)})
    rp = {"full_attention": {"rope_theta": 1e6, "rope_type": "default"}, "sliding_attention": TC["rope_parameters"]["sliding_attention"]}
    with pytest.raises(SpecError, match="proportional"):
        ModelSpec.from_hf_config({"model_type": "gemma4_unified", "text_config": dict(TC, rope_parameters=rp)})


def test_proportional_rope_table():
    """HF _compute_proportional_rope_parameters: 64 turning pairs at theta^(-2i/512); the
    other 192 are zero there, and the kernel's pair distance (256) carries them."""
    s = spec()
    g, l = np.array(s.rope_inv_freq()), np.array(s.rope_inv_freq(local=True))
    assert g.shape == (64,) and np.allclose(g, 1e6 ** (-2 * np.arange(64) / 512))
    assert l.shape == (128,) and np.allclose(l, 1e4 ** (-2 * np.arange(128) / 256))


def test_two_geometries(unvalidated):
    s = spec()
    gl, gg = G4.geometry(s, "local"), G4.geometry(s, "global")
    assert (gl.HD, gl.KVH, gl.ROT, gl.ROPE_PAIR) == (256, 8, 256, 0)
    assert (gg.HD, gg.KVH, gg.ROT, gg.ROPE_PAIR) == (512, 2, 128, 256)
    for g in (gl, gg):
        assert g.VNORM == 1 and g.SCALE == 1.0 and g.QLOCAL == 1 and g.N_CORES == 4 and g.NHL == 4
    assert G4.layout(s, kind="global").PTAB_ROW == 2048 and G4.layout(s, kind="local").PTAB_ROW == 4096


def test_global_pack_plan_duplicates_k_into_both_slots(unvalidated):
    s = spec()
    pool = G4.pack_plan(s)["layer_types"][DENSE]["pool"]
    names = [o["tensor"].rsplit(".", 2)[-2] for o in pool]
    assert names == ["q_proj", "k_proj", "k_proj", "k_proj", "k_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    L = G4.layout(s, kind="global")
    ks = [o["dst"] for o in pool if "k_proj" in o["tensor"]]
    assert ks[0] == L.POOL_K and ks[2] == L.POOL_V and ks[1] - ks[0] == ks[3] - ks[2] == (L.POOL_V - L.POOL_K) // 2
    consts = G4.pack_plan(s)["layer_types"][DENSE_LOCAL]["consts"]
    run = [o["tensor"].rsplit(".", 2)[-2] for o in consts if "div_running" in o]
    assert run == ["post_attention_layernorm", "post_feedforward_layernorm"]


def test_manifest_rows_are_per_kernel(unvalidated):
    m = manifest(spec())["gguf"]
    k = m["kernels"]
    assert (k["dx_local"]["context"], k["dx"]["context"]) == ("dx", "dxg")
    assert (k["dx_local"]["kv_row"], k["dx_local"]["ptab_row"]) == (8192, 4096)
    assert (k["dx"]["kv_row"], k["dx"]["ptab_row"]) == (4096, 2048)
    assert m["globals"]["ptab"]["rotary_dim"] == 128 and len(m["globals"]["ptab"]["inv_freq"]) == 64
    assert m["globals"]["ptab_local"]["rotary_dim"] == 256 and len(m["globals"]["ptab_local"]["inv_freq"]) == 128
    assert m["layout"]["final_logit_softcapping"] == 30.0
    assert m["layer_types"][DENSE]["buffers"]["state"]["row"] == 4096
    # manifest.cpp's own check: the layout's frequencies match the layout's rotary dim, and
    # they are the local table (the layout's rows are the local layers')
    for mm in (m, manifest(spec())):
        lay = mm["layout"]
        assert len(lay["rope_inv_freq"]) == lay["rotary_dim"] // 2
        assert lay["rope_inv_freq"] == mm["globals"]["ptab_local"]["inv_freq"]


def test_the_catalogue_does_not_take_gemma3s_point_for_a_gemma4_layer(monkeypatch):
    """The sliding layers have Gemma 3 12B's shape, which IS validated; with a V norm and a
    score scale of 1 it is another kernel, and it is refused until it has run."""
    monkeypatch.delenv("OPEN_KERNELS_UNVALIDATED", raising=False)
    with pytest.raises(OpRangeError):
        G4._require(spec(), "local", G4.geometry(spec(), "local"))
