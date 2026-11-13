# Traces: OPEN-PREFILL-G4, OPEN-PREFILL-ATTN-WIDE (canonical spec: specs/open-engine/spec.md)
"""Gemma 4 E2B / E4B block prefill route: what the recipe emits per layer kind, that each weight is one contiguous
run of pool ops of exactly its GEMM's size, and that the route adds to the decode manifest without changing it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipes import gemma4e as G4
from recipes.manifest import manifest
from recipes.spec import DENSE_LOCAL, DENSE_LOCAL_SHARED, DENSE_SHARED, ModelSpec

FIX = Path(__file__).resolve().parent / "fixtures"
HDRS = {"e4b": json.loads((FIX / "gguf_gemma4_e4b_header.json").read_text(encoding="utf-8")),
        "e2b": json.loads((FIX / "gguf_gemma4_e2b_header.json").read_text(encoding="utf-8"))}
CHUNK = 5120


def spec_of(name):
    hdr = HDRS[name]
    md = dict(hdr["metadata"])
    md["tokenizer.ggml.tokens"] = [""] * hdr["vocab"]
    md["gemma4.rope_freqs"] = hdr["rope_freqs"]
    return ModelSpec.from_gguf_metadata(md)


@pytest.fixture(autouse=True)
def unvalidated(monkeypatch):
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")


@pytest.fixture(params=["e2b", "e4b"])
def built(request):
    spec = spec_of(request.param)
    return spec, manifest(spec, 4096)


def pool_run_bytes(layer_type, ops):
    """(first byte, total bytes) of a weight's pool ops, which must lie end to end."""
    pool = layer_type["pack"]["pool"]
    off = pool[ops[0]]["dst"]
    total = 0
    for i in ops:
        assert pool[i]["dst"] == off + total, "a route weight is one contiguous run of the pool"
        total += pool[i]["nch"] * CHUNK
    return off, total


def test_every_kind_carries_seven_gemma4_steps(built):
    spec, m = built
    for kind in G4.kinds(spec):
        gb = m["layer_types"][kind]["gemm_block"]
        assert gb["kind"] == "gemma4" and gb["t"] == 256 and len(gb["program"]) == 7
        shared = kind in (DENSE_LOCAL_SHARED, DENSE_SHARED)
        assert (gb["kvw"] == 0) == shared
        assert gb["program"][0]["args"][0] == ("gq_w" if shared else "gqkv3_w")
        assert gb["ple_dim"] == spec.ple_dim and gb["sandwich"] and gb["act"] == "gelu_tanh"


def test_steps_have_the_gemm_shapes_of_the_layer_kind(built):
    spec, m = built
    for kind in G4.kinds(spec):
        s = G4.sub_spec(spec, kind)
        gb = m["layer_types"][kind]["gemm_block"]
        qw, kvw = s.num_heads * s.head_dim, 0 if kind in G4.OWN else s.num_kv_heads * s.head_dim
        want = [(qw + 2 * kvw, spec.hidden), (spec.hidden, qw), (s.intermediate, spec.hidden),
                (s.intermediate, spec.hidden), (spec.hidden, s.intermediate), (spec.ple_dim, spec.hidden),
                (spec.hidden, spec.ple_dim)]
        got = [tuple(int(x[1:]) for x in step["kernel"][5:].split("_")) for step in gb["program"]]
        assert got == want
        assert (gb["hd"], gb["kvh"], gb["rot"]) == (s.head_dim, s.num_kv_heads, s.rotary_dim)
        assert gb["window"] == (spec.sliding_window if kind in (DENSE_LOCAL, DENSE_LOCAL_SHARED) else 0)


def test_each_weight_is_a_contiguous_pool_run_of_its_gemm_size(built):
    spec, m = built
    for kind in G4.kinds(spec):
        lt = m["layer_types"][kind]
        for step in lt["gemm_block"]["program"]:
            n, k = (int(x[1:]) for x in step["kernel"][5:].split("_"))
            w = lt["gemm_block"]["weights"][step["args"][0]]
            assert w["from"] == "pool"
            off, total = pool_run_bytes(lt, w["ops"])
            assert total == n * k * 5 // 8, f"{kind} {step['args'][0]}: {total} bytes for [{n}, {k}]"
        assert max(sum(1 for _ in w["ops"]) for w in lt["gemm_block"]["weights"].values()) <= 3


def test_shapes_tile_and_every_kernel_has_a_build(built):
    spec, m = built
    for kind in G4.kinds(spec):
        for step in m["layer_types"][kind]["gemm_block"]["program"]:
            name = step["kernel"]
            n, k = (int(x[1:]) for x in name[5:].split("_"))
            assert n % 256 == 0 and k % 256 == 0
            assert m["kernels"][name] == {"context": "gemm", "insts": f"{name}/insts.bin", "build": name}
            assert m["builds"][name]["env"] == {"GQP_N": str(n), "GQP_K": str(k), "GQP_T": "256"}
            assert m["globals"][f"gemm_x_k{k}"] == k * 256 * 2 and m["globals"][f"gemm_y_n{n}"] == n * 256 * 4
    assert m["contexts"]["gemm"].startswith("gemm_n")


def test_the_cache_row_holds_k_and_v_of_every_head(built):
    spec, m = built
    for kind in G4.kinds(spec):
        gb = m["layer_types"][kind]["gemm_block"]
        assert m["layer_types"][kind]["buffers"]["state"]["row"] >= 4 * gb["kvh"] * gb["hd"]
        assert gb["ptab"] in m["globals"] and len(m["globals"][gb["ptab"]]["inv_freq"]) == gb["rot"] // 2


def test_the_gguf_view_has_the_same_route(built):
    _, m = built
    for kind, lt in m["layer_types"].items():
        assert m["gguf"]["layer_types"][kind]["gemm_block"] == lt["gemm_block"]


def test_the_route_adds_to_the_decode_manifest_and_changes_none_of_it(built, monkeypatch):
    spec, with_route = built
    monkeypatch.setattr(G4, "gemm_route", lambda spec, max_ctx=4096: None)
    base = manifest(spec, 4096)
    added = {n for n in with_route["kernels"] if n.startswith("gemm_")}
    assert added and not any(n.startswith("gemm_") for n in base["kernels"])

    def strip(m):
        m = json.loads(json.dumps(m))
        route = ("gemm_", "fw")      # the GEMMs and OPEN-PREFILL-ATTN-WIDE's fa_wide streams
        for view in (m, m["gguf"]):
            for n in [n for n in view["kernels"] if n.startswith(route)]:
                del view["kernels"][n]
            for n in [n for n in view["builds"] if n.startswith(route)]:
                del view["builds"][n]
            for n in [n for n in view["globals"] if n.startswith(route)]:
                del view["globals"][n]
            for n in [n for n in view["contexts"] if n == "gemm" or n.startswith("fwa")]:
                del view["contexts"][n]
            for lt in view["layer_types"].values():
                lt.pop("gemm_block", None)
            view.pop("build_key", None)
        return m
    assert strip(with_route) == strip(base)


def test_no_route_when_the_per_layer_input_is_not_in_the_pool(monkeypatch):
    monkeypatch.setenv("OPEN_G4_PLE_NPU", "0")
    m = manifest(spec_of("e2b"), 4096)
    assert all("gemm_block" not in lt for lt in m["layer_types"].values())


def test_attention_on_fa_wide_per_kind(built):
    # OPEN-PREFILL-ATTN-WIDE: every kind's block attention is one fa_wide dispatch at its head dim, window and cache row;
    # a shared kind takes its owner's stream; scores unscaled (FW_SCALE 1.0)
    spec, m = built
    hds = set()
    for k, lt in m["layer_types"].items():
        gb = lt["gemm_block"]
        aw = gb["attn_wide"]
        assert (aw["kernel"], aw["hd"], aw["window"], aw["q"], aw["o"]) == (f"fw{gb['hd']}", gb["hd"], gb["window"], "fwq", "fwo")
        hds.add(gb["hd"])
        kd = m["kernels"][aw["kernel"]]
        assert (kd["patch"], kd["hd"], kd["passes"]) == ("fawpos", gb["hd"], 256 // (256 if gb["hd"] == 256 else 128))
        assert kd["kv_ld"] == 2 * gb["kvh"] * gb["hd"]
    for hd in hds:
        env = m["builds"][f"fw{hd}"]["env"]
        assert env["FW_SCALE"] == "1.0" and env["FW_D"] == str(hd) and env["FW_L"] == "256"
    assert m["globals"]["fwq"] == m["globals"]["fwo"] == 256 * max(lt["gemm_block"]["qw"] for lt in m["layer_types"].values()) * 2


def test_no_fa_wide_with_g4_attn_wide_off(monkeypatch):
    monkeypatch.setenv("G4_ATTN_WIDE", "0")
    m = manifest(spec_of("e4b"), 4096)
    assert not any("attn_wide" in (lt.get("gemm_block") or {}) for lt in m["layer_types"].values())
