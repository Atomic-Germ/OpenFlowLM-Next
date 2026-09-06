# Traces: OPEN-SPEC-DERIVE, OPEN-FAMILY-GRANITE (canonical spec: specs/open-engine/spec.md)
"""IBM Granite 4.2 3B on the dense recipe: spec derivation (HF and GGUF), the
four scalar multipliers and why the recipe may ignore them, the 3B's layout
(head_dim 64 at hidden 2560 -- the point every shipped design refused), and the
manifest."""
from __future__ import annotations

import numpy as np
import pytest

from recipes import dense as DR
from recipes.manifest import manifest
from recipes.spec import DENSE, ModelSpec, SpecError

# The FOLDED container's config.json (vegahyo/Granite-4.2-3B-NPU2). q4nx-build
# folds Granite's four multipliers into the weights, so attention_multiplier
# reads head_dim**-0.5 = 0.125 here and the originals are kept for the record
# under q4nx_folded_multipliers (attention_multiplier was 0.015625).
HF_GRANITE_42_3B = {
    "model_type": "granite", "hidden_size": 2560, "intermediate_size": 8192, "num_hidden_layers": 40,
    "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 64, "rms_norm_eps": 1e-05,
    "rope_theta": 10000000.0, "vocab_size": 100352, "tie_word_embeddings": False, "rope_scaling": None,
    "attention_multiplier": 0.125, "embedding_multiplier": 1.0, "residual_multiplier": 1.0,
    "logits_scaling": 1.0,
}
GGUF_GRANITE_42_3B = {
    "general.architecture": "granite", "granite.embedding_length": 2560, "granite.block_count": 40,
    "granite.vocab_size": 100352, "granite.attention.head_count": 40, "granite.attention.head_count_kv": 8,
    "granite.attention.key_length": 64, "granite.rope.freq_base": 10000000.0,
    "granite.feed_forward_length": 8192, "granite.attention.layer_norm_rms_epsilon": 1e-05,
    "granite.attention.scale": 0.125, "granite.embedding_scale": 1.0, "granite.residual_scale": 1.0,
    "granite.logit_scale": 1.0,
}


def test_spec_from_hf_and_gguf_agree():
    a = ModelSpec.from_hf_config(HF_GRANITE_42_3B, real_vocab=100352)
    b = ModelSpec.from_gguf_metadata(GGUF_GRANITE_42_3B)
    assert a.family == "granite" and a.layer_types == tuple([DENSE] * 40)
    assert a.qk_norm is False and a.attn_gate is False
    assert a.head_dim == 64 and a.rotary_dim == 64 and a.norm_eps == 1e-5
    assert a.rope_scaling is None and a.rope_theta == 1e7
    da, db = a.to_dict(), b.to_dict()
    for d in (da, db):
        d.pop("extra")
    assert da == db


def test_unfolded_multipliers_are_refused_by_name():
    """The whole reason Granite fits: attn.h hard-codes 1/sqrt(HD), so the
    recipe accepts a FOLDED container and must refuse an unfolded one loudly
    rather than run it and return plausible garbage."""
    with pytest.raises(SpecError, match=r"attention_multiplier 0\.015625 != head_dim\*\*-0\.5"):
        ModelSpec.from_hf_config(dict(HF_GRANITE_42_3B, attention_multiplier=0.015625))
    with pytest.raises(SpecError, match="UNFOLDED"):
        ModelSpec.from_gguf_metadata(dict(GGUF_GRANITE_42_3B, **{"granite.attention.scale": 0.015625}))
    for name in ("embedding_multiplier", "residual_multiplier", "logits_scaling"):
        with pytest.raises(SpecError, match=f"{name} 2.0 != 1.0"):
            ModelSpec.from_hf_config(dict(HF_GRANITE_42_3B, **{name: 2.0}))


def test_scaling_and_tied_embeddings_are_refused():
    with pytest.raises(SpecError, match="rope_scaling is not supported"):
        ModelSpec.from_hf_config(dict(HF_GRANITE_42_3B, rope_scaling={"rope_type": "linear", "factor": 2}))
    with pytest.raises(SpecError, match="tied embeddings"):
        ModelSpec.from_hf_config(dict(HF_GRANITE_42_3B, tie_word_embeddings=True))


def test_the_3b_is_inside_every_validated_set():
    """head_dim 64, num_heads 40 and gemv_q4 K 8192 entered the catalogue with
    OPEN-FAMILY-GRANITE on 2026-09-06, so the recipe now builds with no escape
    hatch. Before that run it refused, by name, on all three."""
    DR.recipe(ModelSpec.from_hf_config(HF_GRANITE_42_3B))


def test_3b_layout_and_manifest():
    spec = ModelSpec.from_hf_config(HF_GRANITE_42_3B)
    R = DR.recipe(spec)
    L, G = R.layout, R.geo
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (5, 1, 5, 16, 5)
    assert (G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS, G.H_ELEMS) == (2, 2, 2, 8)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (5120, 1024, 2048, 1024)
    # Two chunks per weight element: PER_CALL must divide every matrix's k-tile
    # count, and the K = 8192 table costs 18 KB of the 60 KB L1 budget.
    assert G.PER_CALL == 2 and G.CALL_BYTES == 10240 and G.TAB_BYTES == 18432
    assert G.QKNORM is False and G.EPS == 1e-5
    assert (G.NH, G.KVH, G.HD, G.ROT) == (40, 8, 64, 64)
    assert L.LMHEAD_BANDS == 1568                              # 100352 / 64
    m = manifest(spec)
    assert m["family"] == "granite" and len(m["layout"]["rope_inv_freq"]) == 32
    chk = m["hf_config_check"]
    assert chk["head_dim"] == 64 and chk["intermediate_size"] == 8192
    # The engine refuses an unfolded container at LOAD too, not just here.
    assert chk["attention_multiplier"] == 0.125
    consts = m["layer_types"][DENSE]["pack"]["consts"]
    assert [o["tensor"].split(".")[-2] for o in consts] == ["input_layernorm", "post_attention_layernorm"]
    assert m["builds"]["dx"]["build_dir"] == "dense/build_granite_h2560"


def test_one_dispatch_per_layer():
    """Our own tuned Granite kernels ran a layer in four dispatches at
    1744.7 us; the dense recipe's program is one step per layer plus a
    per-token ln + lm_head tail."""
    prog = DR.programs(ModelSpec.from_hf_config(HF_GRANITE_42_3B))
    assert [s["kernel"] for s in prog["layer_types"][DENSE]["program"]] == ["dx"]
    assert [s["kernel"] for s in prog["tail"]] == ["ln", "lm"]


def test_rope_is_the_plain_unscaled_table():
    s = ModelSpec.from_hf_config(HF_GRANITE_42_3B)
    got = np.array(s.rope_inv_freq())
    assert got.shape == (32,)
    assert np.allclose(got, 1e7 ** (-np.arange(32) / 32))
