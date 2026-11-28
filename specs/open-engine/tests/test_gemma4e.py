# Traces: OPEN-FAMILY-GEMMA4, OPEN-SPEC-DERIVE, OPEN-GGUF-BF16-POOL (canonical spec: specs/open-engine/spec.md)
"""Gemma 4 E2B / E4B (text), family gemma4e: the spec from Google's GGUF and from transformers' config, the
KV-sharing layout, the tensor names the plan reads against the real file's, and the layer kinds' attention
geometry. The 12B is test_gemma4.py's."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes import dense as DR
from recipes import gemma4e as G4
from recipes.manifest import manifest
from recipes.spec import DENSE, DENSE_LOCAL, DENSE_LOCAL_SHARED, DENSE_SHARED, ModelSpec, SpecError

FIX = Path(__file__).resolve().parent / "fixtures"
HDR = json.loads((FIX / "gguf_gemma4_e4b_header.json").read_text(encoding="utf-8"))
HDR_E2B = json.loads((FIX / "gguf_gemma4_e2b_header.json").read_text(encoding="utf-8"))
CFG_E4B = json.loads((FIX / "config_gemma4_e4b.json").read_text(encoding="utf-8"))
CFG_E2B = json.loads((FIX / "config_gemma4_e2b.json").read_text(encoding="utf-8"))
SPECS = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"


def gguf_md(with_rope_freqs=True, hdr=HDR):
    md = dict(hdr["metadata"])
    md["tokenizer.ggml.tokens"] = [""] * hdr["vocab"]
    if with_rope_freqs:
        md["gemma4.rope_freqs"] = hdr["rope_freqs"]
    return md


@pytest.fixture
def unvalidated(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


@pytest.fixture
def e4b():
    return ModelSpec.from_gguf_metadata(gguf_md())


# ---- spec derivation
def test_gguf_and_hf_config_derive_the_same_spec(e4b):
    hf = ModelSpec.from_hf_config(CFG_E4B)
    assert e4b.spec_hash() == hf.spec_hash()


def test_e4b_shape(e4b):
    assert (e4b.hidden, e4b.num_layers, e4b.num_heads, e4b.num_kv_heads) == (2560, 42, 8, 2)
    assert (e4b.head_dim, e4b.global_head_dim, e4b.rotary_dim, e4b.global_rotary_dim) == (256, 512, 256, 128)
    assert (e4b.intermediate, e4b.intermediate_shared, e4b.ple_dim, e4b.final_softcap) == (10240, 0, 256, 30.0)
    assert (e4b.family, e4b.global_kv_heads, e4b.attn_k_eq_v, e4b.v_norm, e4b.attn_scale) == ("gemma4e", 2, False, True, 1.0)
    assert (e4b.rope_theta, e4b.rope_local_theta, e4b.sliding_window, e4b.norm_eps) == (1e6, 1e4, 512, 1e-6)


def test_the_global_rotation_is_read_from_rope_freqs():
    # llama.cpp writes transformers' partial_rotary_factor as 64 ones then 1e30 (pairs that never turn)
    assert sum(1 for f in HDR["rope_freqs"] if f == 1.0) == 64
    assert ModelSpec.from_gguf_metadata(gguf_md()).global_rotary_dim == 128
    assert ModelSpec.from_gguf_metadata(gguf_md(with_rope_freqs=False)).global_rotary_dim == 128


def test_layer_kinds_and_the_kv_shared_tail(e4b):
    lt = e4b.layer_types
    assert [i for i, t in enumerate(lt) if t in (DENSE, DENSE_SHARED)] == [5, 11, 17, 23, 29, 35, 41]
    assert all(t in (DENSE_LOCAL_SHARED, DENSE_SHARED) for t in lt[24:])
    assert not any(t in (DENSE_LOCAL_SHARED, DENSE_SHARED) for t in lt[:24])


def test_kv_sources_are_the_last_own_layer_of_each_kind(e4b):
    src = G4.kv_sources(e4b)
    assert src[:24] == [-1] * 24
    assert all(src[l] == (23 if e4b.layer_types[l] == DENSE_SHARED else 22) for l in range(24, 42))


def test_e2b_doubles_the_shared_layers_ffn_and_has_one_kv_head():
    s = ModelSpec.from_hf_config(CFG_E2B)
    assert (s.num_layers, s.num_kv_heads, s.intermediate, s.intermediate_shared) == (35, 1, 6144, 12288)
    assert G4.kv_sources(s)[15:] == [13 if t == DENSE_LOCAL_SHARED else 14 for t in s.layer_types[15:]]


def test_e2b_gguf_and_hf_config_derive_the_same_spec():
    # the GGUF lists feed_forward_length per layer (6144 x 15, then 12288 x 20)
    g = ModelSpec.from_gguf_metadata(gguf_md(hdr=HDR_E2B))
    assert g.spec_hash() == ModelSpec.from_hf_config(CFG_E2B).spec_hash()
    assert (g.hidden, g.global_head_dim, g.global_rotary_dim, g.ple_dim) == (1536, 512, 128, 256)


def test_the_shipped_e2b_spec_is_the_derived_one():
    from recipes.load import load_spec
    shipped = load_spec(Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs" / "gemma4-e2b.json")
    assert shipped.spec_hash() == ModelSpec.from_gguf_metadata(gguf_md(hdr=HDR_E2B)).spec_hash()


@pytest.mark.parametrize("key,val", [("enable_moe_block", True), ("use_bidirectional_attention", "all")])
def test_the_moe_and_bidirectional_variants_are_refused(key, val):
    cfg = json.loads(json.dumps(CFG_E4B))
    cfg["text_config"][key] = val
    with pytest.raises(SpecError):
        ModelSpec.from_hf_config(cfg)


def test_vision_only_bidirectional_attention_leaves_text_causal():
    cfg = json.loads(json.dumps(CFG_E4B))
    cfg["text_config"]["use_bidirectional_attention"] = "vision"
    assert ModelSpec.from_hf_config(cfg).spec_hash() == ModelSpec.from_hf_config(CFG_E4B).spec_hash()


def test_no_other_spec_hash_moved_by_the_new_fields():
    # every new field serialises only when set, so a spec without them hashes as before
    d = ModelSpec.from_gguf_metadata(gguf_md()).to_dict()
    assert {"global_head_dim", "global_rotary_dim", "ple_dim", "final_softcap"} <= set(d)
    assert not {"intermediate_shared", "attn_k_eq_v"} & set(d)
    from recipes.load import load_spec
    q = load_spec(Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs" / "qwen3-4b.json")
    assert not {"global_head_dim", "global_rotary_dim", "intermediate_shared", "ple_dim", "final_softcap",
                "global_kv_heads", "attn_k_eq_v"} & set(q.to_dict())


# ---- the tensor names, against the real file's table
def plan_names(spec):
    p = G4.pack_plan(spec)
    by_kind = {k: [o["tensor"] for o in v["pool"] + v["consts"]] for k, v in p["layer_types"].items()}
    glob = [o["tensor"] for o in p["lm_head"]["ops"]] + [p["embed"]["tensor"], p["norm"]["tensor"]]
    hs = G4.host_stages(spec)
    st = hs["ple"]
    return by_kind, glob + [st["embed"], st["model_proj"], st["model_norm"]], \
        [st["gate"], st["proj"], st["norm"], hs["layer_scalar"]]


def test_every_tensor_the_plan_reads_is_in_the_file(unvalidated, e4b):
    have = set(HDR["tensors"])
    by_kind, glob, per_layer = plan_names(e4b)
    assert set(glob) <= have
    for l, kind in enumerate(e4b.layer_types):
        names = [n.replace("{l}", str(l)) for n in by_kind[kind] + per_layer]
        assert set(names) <= have, sorted(set(names) - have)


def test_the_shared_layers_read_no_k_or_v_and_the_file_has_none(unvalidated, e4b):
    have = set(HDR["tensors"])
    by_kind, _, _ = plan_names(e4b)
    for kind in (DENSE_LOCAL_SHARED, DENSE_SHARED):
        assert not any(t in n for n in by_kind[kind] for t in ("attn_k.", "attn_v.", "attn_k_norm"))
    for l in range(24, 42):
        assert not any(f"blk.{l}.{t}" in have for t in ("attn_k.weight", "attn_v.weight", "attn_k_norm.weight"))


def test_projection_widths_match_the_file(unvalidated, e4b):
    # GGUF dims are [in, out]: the plan's nch * 8192 values must be the tensor's whole size
    p = G4.pack_plan(e4b)
    for kind, l in ((DENSE_LOCAL, 0), (DENSE, 5), (DENSE_LOCAL_SHARED, 24), (DENSE_SHARED, 29)):
        for op in p["layer_types"][kind]["pool"]:
            dims, ty = HDR["tensors"][op["tensor"].replace("{l}", str(l))]
            assert ty == "Q4_0" and op["op"] == "std_perm_gguf"
            assert op["in_dim"] == dims[0] and op["nch"] * 8192 == dims[0] * dims[1]


# ---- the four kinds' attention geometry
@pytest.mark.parametrize("part,hd,rot,pair,cs_off,elems,shared", [
    ("local", 256, 256, 0, 1024, 2, False), ("global", 512, 128, 256, 512, 1, False),
    ("local_shared", 256, 256, 0, 1024, 2, True), ("global_shared", 512, 128, 256, 512, 1, True)])
def test_part_geometry(unvalidated, monkeypatch, e4b, part, hd, rot, pair, cs_off, elems, shared):
    monkeypatch.setenv("G4_PART", part)
    R = G4.recipe(e4b)
    G, L = R.geo, R.layout
    assert (G.HD, G.ROT, G.ROPE_PAIR, G.PTAB_CS_OFF, G.PTAB_ELEMS, G.KV_SHARED) == (hd, rot, pair, cs_off, elems, shared)
    assert G.SCALE == 1.0 and G.VNORM and G.VEXP and G.ACORES == 4 and G.RB == 2 and not G.QLOCAL
    assert L.PTAB_ROW == 2048 and L.KV_ROW == 2 * 2 * hd * 2     # [K | V] rows of 2 kv heads, bf16
    if shared:                       # no k / v projection in the pool
        assert L.POOL_K == L.POOL_V == L.POOL_O


# ---- E2B: one kv head, and a twice-wider FFN in the KV-shared layers
@pytest.fixture
def e2b():
    return ModelSpec.from_gguf_metadata(gguf_md(hdr=HDR_E2B))


def test_e2b_plan_reads_tensors_the_file_has_at_their_widths(unvalidated, e2b):
    have = HDR_E2B["tensors"]
    by_kind, glob, per_layer = plan_names(e2b)
    assert set(glob) <= set(have)
    for l, kind in enumerate(e2b.layer_types):
        names = [n.replace("{l}", str(l)) for n in by_kind[kind] + per_layer]
        assert set(names) <= set(have), sorted(set(names) - set(have))
    p = G4.pack_plan(e2b)
    for kind, l in ((DENSE_LOCAL, 0), (DENSE, 4), (DENSE_LOCAL_SHARED, 15), (DENSE_SHARED, 19)):
        for op in p["layer_types"][kind]["pool"]:
            dims, ty = have[op["tensor"].replace("{l}", str(l))]
            assert ty == "Q4_0" and op["op"] == "std_perm_gguf"
            assert op["in_dim"] == dims[0] and op["nch"] * 8192 == dims[0] * dims[1]
    assert have["blk.15.ffn_up.weight"][0] == [1536, 12288] and have["blk.14.ffn_up.weight"][0] == [1536, 6144]
    assert not any(f"blk.{l}.attn_k.weight" in have for l in range(15, 35))


@pytest.mark.parametrize("part,hd,rot,ff,n,e_a,elems,shared", [
    ("local", 256, 256, 6144, 4, 1024, 2, False), ("global", 512, 128, 6144, 8, 2048, 1, False),
    ("local_shared", 256, 256, 12288, 4, 1024, 2, True), ("global_shared", 512, 128, 12288, 8, 2048, 1, True)])
def test_e2b_part_geometry(unvalidated, monkeypatch, e2b, part, hd, rot, ff, n, e_a, elems, shared):
    """One kv head: the element is the whole cache row; the 256-row kv projection splits over 4 cores."""
    monkeypatch.setenv("G4_PART", part)
    R = G4.recipe(e2b)
    G, L = R.geo, R.layout
    assert (G.HD, G.ROT, G.FF, G.N_CORES, G.KVH, G.KV_SHARED) == (hd, rot, ff, n, 1, shared)
    assert (L.E_A, L.KV_ROW, L.PTAB_ROW, G.PTAB_ELEMS, G.PTAB_CS_ELEM, G.PTAB_CS_OFF) == (e_a, e_a, 2048, elems, 1, 512)
    assert G.PER_CALL == 1


def test_only_e2b_caps_the_chunks_per_element(e4b, e2b):
    # the 12288-wide FFN table leaves room for one chunk per element; the generated TUs are shared by every kind
    assert G4.per_call_cap(e4b) is None
    assert G4.per_call_cap(e2b) == 1


def test_e2b_manifest_rows_and_sources(unvalidated, e2b):
    m = manifest(e2b)
    assert m["kv_source"][15:] == [13 if t == DENSE_LOCAL_SHARED else 14 for t in e2b.layer_types[15:]]
    assert {k: v["kv_row"] for k, v in m["kernels"].items() if "kv_row" in v} == {
        "dx_local": 1024, "dx": 2048, "dx_local_shared": 1024, "dx_shared": 2048}
    assert (m["globals"]["ptab_local"]["per_row"], m["globals"]["ptab_local"]["cs_off"]) == (2048, 512)


def test_other_families_keep_their_attention_defaults():
    from recipes.load import load_spec
    base = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"
    for name in ("gemma3-4b", "qwen3-4b", "qwen25-3b", "phi4-mini-4b"):
        G = DR.geometry(load_spec(base / f"{name}.json"))
        assert (G.SCALE, G.VNORM, G.ROPE_PAIR, G.PTAB_CS_OFF, G.KV_SHARED) == (0.0, 0, 0, 512, False)


# ---- the manifest
def test_manifest_carries_the_host_stages_and_one_gguf_view(unvalidated, e4b):
    m = manifest(e4b)
    assert m["kv_source"] == G4.kv_sources(e4b) and m["layout"]["final_logit_softcapping"] == 30.0
    assert m["family"] == "gemma4e" and "logit_softcap" not in m
    assert m["ple"]["dim"] == 256 and m["ple"]["embed_scale"] == 16.0
    assert m["gguf"]["kernels"] == m["kernels"] and m["gguf"]["pack"]["chunk_bytes"] == 5120
    assert {k: v["kv_row"] for k, v in m["kernels"].items() if "kv_row" in v} == {
        "dx_local": 2048, "dx": 4096, "dx_local_shared": 2048, "dx_shared": 4096}
    g = m["globals"]["ptab"]
    assert g["rotary_dim"] == 128 and len(g["inv_freq"]) == 64
    assert g["inv_freq"][1] == pytest.approx(1e6 ** (-2 / 512))       # proportional: over the head, not the rotation
    assert m["globals"]["ptab_local"]["cs_off"] == 1024


# ---- the per-layer input inside the layer program (gx.py)
def _one_ctx(monkeypatch, on):
    """The one-context image with the per-layer input in it (the default)."""
    monkeypatch.setenv("OPEN_G4_PLE_NPU", "1")


@pytest.mark.parametrize("which", ["e4b", "e2b"])
def test_per_layer_input_regions_fit_and_do_not_overlap(unvalidated, monkeypatch, which, e4b, e2b):
    """A region that ran into the kind's own layout or into another would corrupt a stage silently."""
    _one_ctx(monkeypatch, True)
    spec = {"e4b": e4b, "e2b": e2b}[which]
    hid, d = spec.hidden, spec.ple_dim
    for k in G4.kinds(spec):
        L, _ = G4.image_geometry(spec, k)
        P = G4.ple_layout(spec, k)
        pool = [(P.POOL_PG, P.POOL_PP), (P.POOL_PP, P.POOL_PP + hid // 64 * DR.band_bytes(d, "q4_1"))]
        assert pool[0][0] >= L.POOL_BYTES and pool[0][1] - pool[0][0] == d // 64 * DR.band_bytes(hid, "q4_1")
        assert pool[1][1] <= P.POOL_BYTES <= G4.pool_bytes(spec)
        assert L.CD_BYTES <= P.CD_PLEN and P.CD_PLES >= P.CD_PLEN + hid * 2 and P.CD_PLES + L.ELN <= P.CD_BYTES
        # act: what each stage reads or writes there, in bytes
        act = sorted([(P.AD_Y, P.AD_Y + max(hid * 4, P.Y_ELEMS * 4096)), (P.AD_G, P.AD_G + max(L.ELN, 2 * d * 4)),
                      (P.AD_Z, P.AD_Z + max(L.ELN, 4096)), (P.AD_P, P.AD_P + hid * 4), (P.AD_R, P.AD_R + hid * 4)])
        assert act[0][0] >= L.AD_BYTES and act[-1][1] <= P.AD_BYTES
        assert all(a[1] <= b[0] for a, b in zip(act, act[1:]))
        assert P.AD_IN == P.AD_G + d * 4                              # ple_in sits right after the gate's output
        assert P.PG_BANDS * 64 == d and P.PP_PC * 64 * G4.image_cores(spec) == hid


def test_per_layer_input_pack_ops_and_manifest(unvalidated, monkeypatch, e2b):
    _one_ctx(monkeypatch, True)
    m = manifest(e2b)["gguf"]
    for k in G4.kinds(e2b):
        lt, P = m["layer_types"][k], G4.ple_layout(e2b, k)
        assert lt["ple_in"] == P.AD_IN and lt["buffers"]["act"] == P.AD_BYTES and lt["buffers"]["consts"] == P.CD_BYTES
        ops = {o["tensor"]: o for o in lt["pack"]["pool"] + lt["pack"]["consts"]}
        g, p = ops["blk.{l}.inp_gate.weight"], ops["blk.{l}.proj.weight"]
        assert (g["op"], g["dst"], g["nch"], g["in_dim"]) == ("std_perm_gguf", P.POOL_PG, 256 * 1536 // 8192, 1536)
        assert (p["op"], p["dst"], p["nch"], p["in_dim"]) == ("std_perm_gguf", P.POOL_PP, 1536 * 256 // 8192, 256)
        assert ops["blk.{l}.post_norm.weight"]["dst"] == P.CD_PLEN
        s = ops["blk.{l}.layer_output_scale.weight"]
        assert (s["op"], s["dst"], s["cap"]) == ("put_f32", P.CD_PLES, 4)   # the scalar stays f32
    assert m["pack"]["pool_bytes"] == G4.pool_bytes(e2b)


def test_the_rollback_keeps_the_per_layer_input_on_the_host(unvalidated, monkeypatch, e2b):
    monkeypatch.setenv("OPEN_G4_PLE_NPU", "0")
    m = manifest(e2b)["gguf"]
    for k in G4.kinds(e2b):
        lt = m["layer_types"][k]
        assert "ple_in" not in lt
        assert not {o["tensor"] for o in lt["pack"]["pool"] + lt["pack"]["consts"]} & {
            "blk.{l}.inp_gate.weight", "blk.{l}.proj.weight", "blk.{l}.post_norm.weight"}


def test_the_per_layer_input_flavour_is_in_the_build_key(unvalidated, monkeypatch, e2b):
    from recipes.cache import build_key
    monkeypatch.setenv("OPEN_G4_PLE_NPU", "0")
    off = build_key(e2b)
    monkeypatch.delenv("OPEN_G4_PLE_NPU", raising=False)
    on = build_key(e2b)
    monkeypatch.setenv("OPEN_G4_PLE_NPU", "1")
    assert off != on == build_key(e2b)
