# Traces: OPEN-SPEC-DERIVE, OPEN-OP-RANGE, OPEN-FAMILY-HUNYUAN (canonical spec: specs/open-engine/spec.md)
"""HunYuan dense on the dense recipe: spec derivation (HF and GGUF), the NTK-alpha
RoPE base both sources agree on, the post-RoPE q/k norm order, the unpadded
vocabulary the head rounds up, and Hy-MT2-7B's layout (Llama 3.1 8B's geometry)."""
from __future__ import annotations

import numpy as np
import pytest

from recipes import dense as DR
from recipes.catalogue import OpRangeError
from recipes.load import load_spec
from recipes.manifest import manifest
from recipes.spec import DENSE, ModelSpec, SpecError

HF_HY_MT2_7B = {
    "model_type": "hunyuan_v1_dense", "hidden_size": 4096, "intermediate_size": 14336,
    "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
    "head_dim": 128, "attention_head_dim": 128, "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
    "vocab_size": 128167, "org_vocab_size": 128167, "tie_word_embeddings": True,
    "use_qk_norm": True, "use_cla": False, "cla_share_factor": 2, "norm_type": "rms",
    "attention_bias": False, "mlp_bias": False, "hidden_act": "silu",
    "rope_scaling": {"alpha": 1000.0, "beta_fast": 32, "beta_slow": 1, "factor": 1.0,
                     "mscale": 1.0, "mscale_all_dim": 1.0, "type": "dynamic"},
}
# llama.cpp folds the alpha into rope.freq_base and writes scaling type NONE
# (conversion/hunyuan.py: `scaled_base = base * (alpha ** (dim / (dim - 2)))`).
SCALED_BASE = 10000.0 * (1000.0 ** (128 / 126))
GGUF_HY_MT2_7B = {
    "general.architecture": "hunyuan-dense", "hunyuan-dense.embedding_length": 4096,
    "hunyuan-dense.block_count": 32, "hunyuan-dense.vocab_size": 128167,
    "hunyuan-dense.attention.head_count": 32, "hunyuan-dense.attention.head_count_kv": 8,
    "hunyuan-dense.attention.key_length": 128, "hunyuan-dense.feed_forward_length": 14336,
    "hunyuan-dense.rope.freq_base": SCALED_BASE, "hunyuan-dense.rope.scaling.factor": 1.0,
    "hunyuan-dense.attention.layer_norm_rms_epsilon": 1e-05,
}


@pytest.fixture
def unvalidated(monkeypatch):
    """A no-op since OPEN-FAMILY-HUNYUAN's compare passed (2026-09-06) and
    qk_norm_post_rope=True entered the catalogue; kept so these tests stay
    honest about which points they depend on."""
    return None


def test_spec_from_hf_and_gguf_agree():
    a = ModelSpec.from_hf_config(HF_HY_MT2_7B)
    b = ModelSpec.from_gguf_metadata(GGUF_HY_MT2_7B)
    assert a.family == "hunyuan" and a.layer_types == tuple([DENSE] * 32)
    assert a.qk_norm is True and a.attn_gate is False and a.rotary_dim == 128 and a.norm_eps == 1e-5
    assert a.rope_scaling is None and a.intermediate == 14336 and a.vocab == 128167
    da, db = a.to_dict(), b.to_dict()
    for d in (da, db):
        d.pop("extra")
    assert da == db


def test_ntk_alpha_folds_into_one_static_base():
    """The dynamic/alpha scaling is a one-time base stretch, not a per-length schedule:
    the frequencies are the plain theta^(-2i/d) of the stretched base."""
    s = ModelSpec.from_hf_config(HF_HY_MT2_7B)
    assert s.rope_theta == pytest.approx(SCALED_BASE, rel=1e-15)
    assert s.rope_theta == pytest.approx(11158839.925, rel=1e-9)
    got = np.array(s.rope_inv_freq())
    assert got.shape == (64,)
    assert np.allclose(got, SCALED_BASE ** (-np.arange(64) / 64), rtol=1e-12)
    plain = ModelSpec.from_hf_config(dict(HF_HY_MT2_7B, rope_scaling=None))
    assert plain.rope_theta == 10000.0


def test_unsupported_rope_and_attention_variants_are_refused():
    def bad(**kw):
        return dict(HF_HY_MT2_7B, **kw)

    with pytest.raises(SpecError, match="rope_scaling type 'yarn'"):
        ModelSpec.from_hf_config(bad(rope_scaling={"type": "yarn", "factor": 2.0}))
    with pytest.raises(SpecError, match="factor 4.0 is not 1"):
        ModelSpec.from_hf_config(bad(rope_scaling=dict(HF_HY_MT2_7B["rope_scaling"], factor=4.0)))
    with pytest.raises(SpecError, match="mscale"):
        ModelSpec.from_hf_config(bad(rope_scaling=dict(HF_HY_MT2_7B["rope_scaling"], mscale=1.5)))
    with pytest.raises(SpecError, match="use_cla"):
        ModelSpec.from_hf_config(bad(use_cla=True))
    with pytest.raises(SpecError, match="attention_bias"):
        ModelSpec.from_hf_config(bad(attention_bias=True))
    with pytest.raises(SpecError, match="MoE variants"):
        ModelSpec.from_hf_config(bad(num_experts=8))
    with pytest.raises(SpecError, match="rope.scaling.factor"):
        ModelSpec.from_gguf_metadata(dict(GGUF_HY_MT2_7B, **{"hunyuan-dense.rope.scaling.factor": 8.0}))


def test_qk_norm_is_a_post_rope_family(unvalidated):
    """The order is a family property of the recipe, not a ModelSpec field."""
    from test_llama3 import HF_LLAMA31_8B
    from test_qwen3_dense import HF_QWEN3_4B

    hy = DR.geometry(ModelSpec.from_hf_config(HF_HY_MT2_7B))
    assert hy.QKNORM is True and hy.QKNORM_POST is True
    for cfg in (HF_QWEN3_4B, HF_LLAMA31_8B):
        g = DR.geometry(ModelSpec.from_hf_config(cfg))
        assert g.QKNORM_POST is False


def test_the_post_rope_point_is_in_the_catalogue():
    """It entered the set when OPEN-FAMILY-HUNYUAN's compare passed on hardware
    (2026-09-06); an unvalidated neighbour is still refused by name."""
    DR.recipe(ModelSpec.from_hf_config(HF_HY_MT2_7B))              # no env override needed
    with pytest.raises(OpRangeError, match="head_dim=64 is outside the validated set"):
        DR.recipe(ModelSpec.from_hf_config(dict(HF_HY_MT2_7B, head_dim=64, attention_head_dim=64)))


def test_the_head_rounds_an_unpadded_vocabulary_up_to_whole_bands(unvalidated):
    spec = ModelSpec.from_hf_config(HF_HY_MT2_7B, real_vocab=128166)
    assert spec.vocab == 128167 and DR.lm_rows(spec) == 128192
    m = manifest(spec)
    # the model's own count is what config.json is checked against; the head is the padded one
    assert m["hf_config_check"]["vocab_size"] == 128167
    assert m["layout"]["vocab"] == 128192 and m["layout"]["real_vocab"] == 128166
    assert m["globals"]["logits"] == 128192 * 4
    assert m["builds"]["lm_head_q4"]["env"]["LMHEAD_N"] == "128192"
    from recipes.qwen36moe import q4_chunks
    assert m["pack"]["lm_head"]["ops"][0]["nch"] == q4_chunks(128192, 4096) == 64096


def test_7b_layout_matches_the_8b_geometry(unvalidated):
    """Same widths as Llama 3.1 8B, so the same bands, elements and 32 KB table."""
    from test_llama3 import HF_LLAMA31_8B

    R = DR.recipe(ModelSpec.from_hf_config(HF_HY_MT2_7B))
    L, G = R.layout, R.geo
    ref = DR.recipe(ModelSpec.from_hf_config(HF_LLAMA31_8B))
    assert (G.Q_PC, G.KV_PC, G.O_PC, G.UP_PC, G.DOWN_PC) == (8, 2, 8, 28, 8)
    assert (L.ELN, L.E_A, L.KV_ROW, L.PTAB_ROW) == (8192, 2048, 4096, 2048)
    assert G.PER_CALL == 1 and G.TAB_BYTES == 32256 and G.EPS == 1e-5
    assert L.POOL_BYTES == ref.layout.POOL_BYTES and L.CD_BYTES == ref.layout.CD_BYTES
    assert L.LMHEAD_BANDS == 2003


def test_manifest_names_the_family_and_its_own_build_dirs(unvalidated):
    m = manifest(ModelSpec.from_hf_config(HF_HY_MT2_7B, real_vocab=128166))
    assert m["family"] == "hunyuan" and m["hf_config_check"]["model_type"] == ["hunyuan_v1_dense"]
    assert m["hf_config_check"]["head_dim"] == 128
    assert m["builds"]["dx"]["build_dir"] == "dense/build_hunyuan_h4096"
    consts = m["layer_types"][DENSE]["pack"]["consts"]
    assert [o["tensor"].split(".")[-2] for o in consts] == [
        "input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"]


def test_checked_in_spec_is_the_derivation(unvalidated):
    """recipes/specs/hy-mt2-7b.json is what Hy-MT2-7B's own config.json derives."""
    from pathlib import Path

    p = Path(DR.__file__).resolve().parent / "specs" / "hy-mt2-7b.json"
    got = load_spec(p).to_dict()
    want = ModelSpec.from_hf_config(HF_HY_MT2_7B, real_vocab=128166).to_dict()
    got.pop("extra"), want.pop("extra")
    assert got == want


def test_replica_norms_after_rope():
    """The fp64 oracle implements the family's order, so the whole-layer compare
    tests it rather than agreeing with the kernel by construction."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(DR.__file__).resolve().parents[1] / "model"))
    import replica_dense as RD

    hid, nh, kvh, hd, ff = 8, 2, 2, 4, 16
    rng = np.random.default_rng(0)
    W = {}

    class Stub:
        """Just enough of the q4nx container: dense fp64 tensors by name."""

        def bf16(self, name):
            return W[name]

        def matmul_w(self, name, rows, cols):
            assert W[name].shape == (rows, cols)
            return W[name]

    def spec_for(family):
        return ModelSpec(family=family, hidden=hid, num_layers=1, layer_types=(DENSE,), vocab=64,
                         real_vocab=64, num_heads=nh, num_kv_heads=kvh, head_dim=hd, rotary_dim=hd,
                         rope_theta=10000.0, qk_norm=True, attn_gate=False, intermediate=ff, norm_eps=1e-5)

    pre = "model.layers.0."
    for n, shape in ((pre + "input_layernorm.weight", (hid,)), (pre + "post_attention_layernorm.weight", (hid,)),
                     (pre + "self_attn.q_norm.weight", (hd,)), (pre + "self_attn.k_norm.weight", (hd,))):
        W[n] = rng.normal(size=shape)
    for n, shape in ((pre + "self_attn.q_proj.weight", (nh * hd, hid)), (pre + "self_attn.k_proj.weight", (kvh * hd, hid)),
                     (pre + "self_attn.v_proj.weight", (kvh * hd, hid)), (pre + "self_attn.o_proj.weight", (hid, nh * hd)),
                     (pre + "mlp.up_proj.weight", (ff, hid)), (pre + "mlp.gate_proj.weight", (ff, hid)),
                     (pre + "mlp.down_proj.weight", (hid, ff))):
        W[n] = rng.normal(size=shape) * 0.1

    m, x = Stub(), rng.normal(size=hid)
    # two cached rows, so the attention weights actually depend on q . k
    K0, V0 = rng.normal(size=(2, kvh, hd)), rng.normal(size=(2, kvh, hd))
    args = dict(layer=0, K=K0, V=V0, pos=3)
    hy, hy_K, _ = RD.dense_decode(m, spec_for("hunyuan"), x_res=x, **args)
    q3, _, _ = RD.dense_decode(m, spec_for("qwen3"), x_res=x, **args)
    assert not np.allclose(hy, q3), "the two orders must differ, or the test proves nothing"

    # hand-rolled: rms -> rope -> weight, for the cached k of the only position
    xn = (RD.rms(x, 1e-5) * W[pre + "input_layernorm.weight"]).astype(np.float32)
    kk = RD.rms((xn @ W[pre + "self_attn.k_proj.weight"].T).reshape(kvh, hd), 1e-5)
    want = RD.rope(kk, 3, hd, 10000.0, spec_for("hunyuan").rope_inv_freq()) * W[pre + "self_attn.k_norm.weight"]
    assert np.allclose(hy_K[-1], want, rtol=1e-12)
