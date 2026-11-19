# Traces: OPEN-FAMILY-LFM3
"""LFM3 decode with one dispatch a layer (on by default; LFM3_FUSED=0 for two): the mixer's program then the MLP's over the same fifos."""
from __future__ import annotations

import json

import pytest

from recipes.spec import DENSE_LOCAL, FULL, SHORT_CONV, ModelSpec
from test_lfm3 import HF_LFM3, LEGACY, legacy  # noqa: F401 (legacy: autouse, every switch off)


def spec():
    return ModelSpec.from_hf_config(dict(HF_LFM3))


def test_the_defaults_are_the_fast_set(monkeypatch):
    """Unset, every switch is on: one shared image for both mixers, block prefill, Q4_0-native pools, the bf16 head."""
    from recipes import lfm3 as R
    for k in LEGACY:
        monkeypatch.delenv(k)
    s = spec()
    p, b, plan = R.programs(s), R.builds(s), R.pack_plan(s)
    assert {v["context"] for k, v in p["kernels"].items() if k.startswith("lg_")} == {"lg_shared"}
    assert all(lt["gemm_block"]["kind"] == "lfm" for lt in p["layer_types"].values())
    assert plan["chunk_bytes"] == 4608 and plan["lm_head"]["ops"][0]["op"] == "gx_lmhead_bf16"
    assert b["lg_shared_conv"]["env"]["LFM3_Q40"] == "1" and "LMHEAD_SCALES_F32" not in b["lm_head_q8"]["env"]
    monkeypatch.setenv("LFM3_FUSED", "0")            # unfused: no shared image either
    assert "lg_shared" not in R.programs(s)["contexts"] and "lg_mlp" in R.builds(s)


def test_off_is_the_unfused_set():
    from recipes import lfm3 as R
    s = spec()
    assert json.dumps(R.programs(s), sort_keys=True) == json.dumps(R._programs(s), sort_keys=True)
    assert json.dumps(R.builds(s), sort_keys=True) == json.dumps(R._builds(s), sort_keys=True)
    assert "lg_mlp" in R.builds(s) and len(R.programs(s)["layer_types"][SHORT_CONV]["program"]) == 2


def test_on_is_one_dispatch_a_layer_and_the_pack_plan_is_unchanged(monkeypatch):
    from recipes import lfm3 as R
    s = spec()
    plan = json.dumps(R.pack_plan(s), sort_keys=True)
    monkeypatch.setenv("LFM3_FUSED", "1")
    p = R.programs(s)
    assert json.dumps(R.pack_plan(s), sort_keys=True) == plan
    for lt in (SHORT_CONV, FULL, DENSE_LOCAL):
        assert len(p["layer_types"][lt]["program"]) == 1
    assert p["layer_types"][SHORT_CONV]["program"][0]["kernel"] == "lg_convmlp"
    assert p["layer_types"][DENSE_LOCAL]["program"][0]["kernel"] == "lg_attmlp_local"
    assert p["kernels"]["lg_attmlp_local"]["window"] == 256 and p["kernels"]["lg_attmlp"]["window"] == 0
    assert p["kernels"]["lg_attmlp"]["insts"] == p["kernels"]["lg_attmlp_local"]["insts"] == "lg_attmlp/insts.bin"
    assert "lg_mlp" not in p["kernels"] and "lg_mlp" not in p["contexts"]
    assert set(R.builds(s)) == {"ln", "lm_head_q8", "lg_convmlp", "lg_attmlp"}
    assert R.builds(s)["lg_attmlp"]["env"] == {"LG_PART": "attmlp"}
    for k in p["kernels"].values():          # every kernel's context and build exist
        assert k["context"] in p["contexts"] and k["build"] in R.builds(s)


def test_fused_and_block_together_make_a_consistent_manifest(monkeypatch):
    from recipes import lfm3 as R
    monkeypatch.setenv("LFM3_FUSED", "1")
    monkeypatch.setenv("LFM3_BLOCK", "1")
    s = spec()
    p, b = R.programs(s), R.builds(s)
    assert all(lt["gemm_block"]["kind"] == "lfm" for lt in p["layer_types"].values())
    assert {"gemm_n6144_k2048", "lg_convmlp", "lg_attmlp"} <= set(b)
    for k in p["kernels"].values():
        assert k["build"] in b


def test_shared_is_one_context_two_streams_and_off_is_unchanged(monkeypatch):
    from recipes import lfm3 as R
    s = spec()
    monkeypatch.setenv("LFM3_FUSED", "1")
    fused = json.dumps([R.programs(s), R.builds(s)], sort_keys=True)
    plan = json.dumps(R.pack_plan(s), sort_keys=True)
    monkeypatch.delenv("LFM3_FUSED")              # unset is on, which the shared image needs
    monkeypatch.setenv("LFM3_SHARED", "1")
    p, b = R.programs(s), R.builds(s)
    assert json.dumps(R.pack_plan(s), sort_keys=True) == plan
    mixers = {k: v for k, v in p["kernels"].items() if k.startswith("lg_")}
    assert {v["context"] for v in mixers.values()} == {"lg_shared"}
    assert p["contexts"]["lg_shared"] == "lg_shared_conv/final.xclbin"
    assert mixers["lg_convmlp"]["insts"] == "lg_shared_conv/insts.bin"
    assert mixers["lg_attmlp"]["insts"] == mixers["lg_attmlp_local"]["insts"] == "lg_shared_att/insts.bin"
    assert {n: b[n]["env"] for n in ("lg_shared_conv", "lg_shared_att")} == {
        "lg_shared_conv": {"LG_PART": "shared", "LG_STREAM": "conv"},
        "lg_shared_att": {"LG_PART": "shared", "LG_STREAM": "att"}}
    assert "lg_convmlp" not in b and "lg_attmlp" not in b
    # one buffer signature: the conv layer passes the ptab it never reads
    assert p["layer_types"][SHORT_CONV]["program"][0]["args"] == ["pool", "xres", "consts", "state", "act", "ptab"]
    assert len(p["layer_types"][FULL]["program"][0]["args"]) == 6
    for k in p["kernels"].values():
        assert k["context"] in p["contexts"] and k["build"] in b
    monkeypatch.setenv("LFM3_SHARED", "0")
    assert json.dumps([R.programs(s), R.builds(s)], sort_keys=True) == fused


def test_shared_with_block_prefill(monkeypatch):
    from recipes import lfm3 as R
    monkeypatch.setenv("LFM3_FUSED", "1")
    monkeypatch.setenv("LFM3_SHARED", "1")
    monkeypatch.setenv("LFM3_BLOCK", "1")
    s = spec()
    p, b = R.programs(s), R.builds(s)
    assert all(lt["gemm_block"]["kind"] == "lfm" for lt in p["layer_types"].values())
    for k in p["kernels"].values():
        assert k["build"] in b


def test_q40_pools_and_the_bf16_head_and_their_off_switches(monkeypatch):
    from recipes import lfm3 as R
    s = spec()
    off = json.dumps([R.programs(s), R.builds(s), R.pack_plan(s), R.manifest_layout(s, 4096)], sort_keys=True)
    assert R.pack_plan(s)["chunk_bytes"] == 5120 and R.pack_plan(s)["lm_head"]["ops"][0]["op"] == "gx_lmhead"
    monkeypatch.setenv("LFM3_Q40", "1")
    monkeypatch.setenv("LFM3_HEAD", "bf16")
    G, L, plan, b = R.geometry(s), R.layout(s), R.pack_plan(s), R.builds(s)
    assert plan["chunk_bytes"] == 4608 and G.CALL_BYTES == 2 * 4608 and R.band_bytes(2048) == 16 * 4608
    assert plan["lm_head"]["ops"][0]["op"] == "gx_lmhead_bf16" and R.manifest_layout(s, 4096)["lmhead_chunk_bytes"] == 8704
    assert L.LMHEAD_BAND_BYTES == 4 * 8 * 8704
    assert all(o["op"] == "fl_q4" for lt in plan["layer_types"].values() for o in lt["pool"])
    assert "LMHEAD_SCALES_F32" not in b["lm_head_q8"]["env"] and "bf16s" in b["lm_head_q8"]["build_dir"]
    for n, e in b.items():
        if e["design"] == "lfm3/lg.py":
            assert e["env"]["LFM3_Q40"] == "1" and e["build_dir"].endswith("_q40")
    monkeypatch.setenv("LFM3_BLOCK", "1")
    gemms = {n: e for n, e in R.builds(s).items() if e["design"].startswith("gemm_q4_prefill")}
    assert gemms and all(e["env"]["GQP_FMT"] == "q4_0" for e in gemms.values())
    for k in ("LFM3_Q40", "LFM3_HEAD", "LFM3_BLOCK"):
        monkeypatch.setenv(k, LEGACY[k])
    assert json.dumps([R.programs(s), R.builds(s), R.pack_plan(s), R.manifest_layout(s, 4096)], sort_keys=True) == off
