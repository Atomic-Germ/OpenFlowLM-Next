# Traces: OPEN-MANIFEST, OPEN-LAYOUT-FREEZE (canonical spec: specs/open-engine/spec.md)
"""The qwen36moe recipe reproduces, from the ModelSpec alone, every constant
that was hand-written in designs/layer_x/layout.py and xcommon.py on
2026-09-05 -- the numbers the shipped 27B kernels were built with. A change
that moves an offset fails here before it reaches a build."""
from __future__ import annotations

import pytest

from recipes import qwen36moe as Q
from recipes.load import default_spec
from recipes.manifest import manifest

# designs/layer_x/layout.py, verbatim, before it read the recipe
LAYOUT_27B = dict(
    C_LNW=0, C_SIDE=4096, C_NW=335872, C_POSTLN=339968, C_RW=344064, C_SGW=1392640, C_WOUT=1396736,
    C_BYTES=1396736 + 10_485_760, GLUE_SIDE_BYTES=331776,
    CA_LNW=0, CA_POSTLN=4096, CA_META=8192, CA_RW=10240, CA_SGW=1058816, CA_BYTES=1058816 + 4096,
    A_XN=0, A_QKV=4096, A_Z=36864, A_VEC=53248, A_O=118784, A_OG=135168, A_OUT=143360,
    A_RES=151552, A_XM=172032, A_ROUT=176128, A_HP=186368, A_BYTES=186368 + 4096,
    AA_XN=0, AA_QG=4096, AA_KVN=36864, AA_OG=40960, AA_OUT=49152,
    AA_RES=59392, AA_XM=79872, AA_ROUT=83968, AA_HP=94208, AA_BYTES=94208 + 4096,
    S_ROWS=140, S_HEAD_BYTES=140 * 512, STATE_S_OFF=49152, STATE_BYTES=49152 + 32 * 140 * 512,
    POOL_QKV=505_282_560, POOL_Z=515_768_320,
    POOL_Q=505_282_560, POOL_K=510_525_440, POOL_V=511_180_800, POOL_GATE=511_836_160, POOL_O=517_079_040,
    POOL_DOWN=335_544_320, POOL_SHARE_UP=503_316_480, POOL_SHARE_GATE=503_971_840, POOL_SHARE_DOWN=504_627_200,
    POOL_BYTES=536_870_912,
    KV_ROW=2048, PTAB_ROW=1024, MAX_CTX=4096, KV_BYTES=4096 * 2048, PTAB_BYTES=4096 * 1024,
    LMHEAD_POOL_BYTES=542_113_792,
)
# designs/layer_x/xcommon.py + lx.py + ax.py, verbatim
COMMON_27B = dict(NE=8, NX=9, HID=2048, FF=512, TILE=5120, PER_CALL=2, CALL_BYTES=10240, STRIPE=32 * 5120,
                  HALF=16 * 5120, PAIR=2 * 5120, DOWN_BAND=8 * 5120, UP_BYTES=4 * 32 * 5120, N_CORES=8,
                  DOWN_PER_CORE=2, BAND_ROWS=64, BAND16=16 * 5120, BAND32=32 * 5120, N_HDR=3, MS_FLOATS=928,
                  DS_FLOATS=1280, TAB_BYTES=2 * 4096 + 4096 // 4, H_TAB_OFF=4608, W_ELEMS=2048 * 256 * 2 // 4096,
                  DN_ROWS=20, DN_SLICES=7, DN_HEADS_PC=4, KWIDE=4096,
                  MS_RW=0, MS_XR=32, MS_ACC=288, MS_U=544, MS_G=608, MS_YD=672)
LINEAR_27B = dict(QKV_PC=16, Z_PC=8, OUT_PC=4, NCH=8192, NHEAD=32, TILE=1024, NT=8, AB_ELEMS=32, G=1024, NG=4,
                  VW=4096, OUT_K=4096, RECORD_BYTES=2048, O_HEAD_BYTES=512, HEADS_PER_TILE=8, VALUE_TILE0=4)
ATTN_27B = dict(NH=16, KVH=2, HD=256, ROT=64, Q_PC=8, KV_PC=1, O_PC=4, QW=4096, KVW=512, O_K=4096,
                META_BYTES=1024, HEAD_BYTES=1024)


@pytest.fixture(scope="module")
def R():
    """The 35B's recipe at the q4_1 constant set the designs were written against.

    `qwen36-35b-a3b.json` now records the q8 roles its container holds, and a q8 role
    doubles that projection's pool region -- so these hand-written constants are the
    q4_1 ones, taken verbatim from designs/layer_x. The q8 layout is pinned in
    test_quant_q8.test_a_q8_role_doubles_its_pool_region_and_the_27b_stays_at_512_MB.
    """
    import dataclasses

    return Q.recipe(dataclasses.replace(default_spec(), quant="q4_1"))


def test_layout_matches_the_hand_written_constants(R):
    got = R.layout.constants()
    diff = {k: (got[k], v) for k, v in LAYOUT_27B.items() if got[k] != v}
    assert diff == {}


def test_common_geometry_matches(R):
    diff = {k: (getattr(R.common, k), v) for k, v in COMMON_27B.items() if getattr(R.common, k) != v}
    assert diff == {}


def test_linear_and_attention_geometry_match(R):
    assert {k: getattr(R.linear, k) for k in LINEAR_27B} == LINEAR_27B
    assert {k: getattr(R.attn, k) for k in ATTN_27B} == ATTN_27B


def test_manifest_names_what_the_driver_needs(R):
    m = manifest(R.spec)
    assert m["manifest_version"] == 1 and m["family"] == "qwen36moe"
    assert m["layers"] == list(R.spec.layer_types) and len(m["layers"]) == 40
    for lt, d in m["layer_types"].items():
        for step in d["program"]:
            assert step["kernel"] in m["kernels"], (lt, step)
            if step["op"] == "run":
                assert 1 <= len(step["args"]) <= 8
        assert d["pack"]["pool"] and d["pack"]["consts"]
    for k, d in m["kernels"].items():
        assert d["context"] in m["contexts"], k
    assert [s["kernel"] for s in m["tail"]] == ["ln", "lm"]
    lay = m["layout"]
    assert lay["kv_row"] == 2048 and lay["ptab_row"] == 1024 and lay["rout_idx_off"] == 1024
    assert lay["moe"] == {"experts": 256, "topk": 8, "stripe": 163840, "up_bytes": 655360, "down_core": 81920,
                          "pool_down": 335544320, "share_up": 503316480, "share_gate": 503971840,
                          "share_down": 504627200}
    lin, full = m["layer_types"]["linear_attention"], m["layer_types"]["full_attention"]
    assert lin["buffers"] == {"consts": 11882496, "act": 190464, "state": {"kind": "linear", "bytes": 2342912}}
    assert full["buffers"] == {"consts": 1062912, "act": 98304, "state": {"kind": "kv", "row": 2048}}
    assert lin["program"][1] == {"op": "moeroute2", "kernel": "lx1", "act_off": 176128}
    assert full["program"][1] == {"op": "moeroute2", "kernel": "ax1", "act_off": 83968}
    assert m["kernels"]["ax0"]["patch"] == "attnpos" and m["kernels"]["lx1"]["patch"] == "moeroute2"
    assert m["hf_config_check"]["layer_types"] == m["layers"]
