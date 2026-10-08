# Traces: OPEN-DECODE-ROWS, OPEN-UNO-LORA (canonical spec: specs/open-engine/spec.md)
"""recipes/dxl.py's tables: one that disagrees with its stream hangs the array, one over the wrong k-tiles quietly computes garbage."""
from __future__ import annotations

from pathlib import Path

import pytest

from recipes import dense as DR
from recipes import dxl as DXR
from recipes.catalogue import OpRangeError
from recipes.load import load_spec

SPECS = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"
K2_7B = load_spec(SPECS / "k2-horizon-7b.json")


def test_act_regions_do_not_overlap():
    X = DXR.layout(K2_7B, 4)
    regions = sorted((getattr(X, f), f) for f in vars(X) if f.startswith("AD_") and f != "AD_BYTES")
    sizes = {"AD_XN": 4 * X.XN_ROW, "AD_Q": 4 * X.Q_ROW, "AD_K": 4 * X.KV_ROW_ACT, "AD_V": 4 * X.KV_ROW_ACT,
             "AD_OG": 4 * X.OG_ROW, "AD_OUT": 4 * X.HID_ROW, "AD_RES": 4 * X.HID_ROW, "AD_XM": 4 * X.XM_ROW,
             "AD_H": 4 * X.H_ROW, "AD_OUT2": 4 * X.HID_ROW, "AD_ZQKV": 4 * DXR.Z_ROW, "AD_ZO": 4 * DXR.Z_ROW,
             "AD_ZGU": 4 * DXR.Z_ROW, "AD_ZD": 4 * DXR.Z_ROW, "AD_ZERO": DXR.XE, "AD_JUNK": K2_7B.hidden * 2}
    for (a, fa), (b, _) in zip(regions, regions[1:]):
        assert a + sizes[fa] <= b, fa
    assert regions[-1][0] + sizes[regions[-1][1]] <= X.AD_BYTES


def test_the_main_cores_fit_their_memory():
    for L in (4, 8):
        X = DXR.layout(K2_7B, L)
        assert DXR._main_l1(L, X.BT, X.BP, 2) <= DXR.L1_BUDGET


def test_verify_jobs_cover_every_band_once():
    G = DR.geometry(K2_7B)
    js = DXR.jobs(K2_7B, 4, draft=False)
    assert len(js) == 13
    bands = {}
    for j in js:
        assert j.wbuf == "pool" and j.core["S0"] == 0
        bands.setdefault(j.woff, 0)
        bands[j.woff] += j.core["BT"]
        assert j.core["KT"] * 256 == j.x[3] and j.core["S"] * j.core["KS"] == j.x[3]   # K
    L0 = DR.layout(K2_7B)
    assert bands == {L0.POOL_Q: G.Q_PC, L0.POOL_K: G.KV_PC, L0.POOL_V: G.KV_PC, L0.POOL_O: G.O_PC,
                     L0.POOL_UP: G.UP_PC, L0.POOL_GATE: G.UP_PC, L0.POOL_DOWN: G.DOWN_PC}


def test_draft_jobs_append_one_lora_k_tile_per_band():
    js = DXR.jobs(K2_7B, 4, draft=True)
    assert len(js) == 30 and len(js) <= DXR.JMAX
    a = [j for j in js if j.wbuf == "lora" and j.x[0] == "act"]
    assert [j.woff for j in a] == [DXR.layout(K2_7B, 4).LORA[n][0] for n in ("a_qkv", "a_o", "a_gu", "a_d")]
    for i, j in enumerate(js):
        if j.wbuf == "pool":
            K = j.x[3]
            lj = js[i + 1]
            assert lj.x[0] == "z" and lj.core["S"] == 1 and lj.core["KS"] == DXR.LORA_K
            assert lj.core["S0"] == K // 256 and lj.core["KT"] == j.core["KT"] == K // 256 + 1
            assert lj.core["B0"] == j.core["B0"] and lj.core["BT"] == j.core["BT"]
            assert j.core["NDRAIN"] == 0          # the band drains after its LoRA tile, not before
    # the seed row's mask: v reads the second 256-wide window of z, every other tile the first
    zcols = {(j.woff, j.core["OFF"]) for j in js if j.x[0] == "z"}
    X = DXR.layout(K2_7B, 4)
    assert (X.LORA["b_v"][0], DXR.LORA_K) in zcols and (X.LORA["b_q"][0], 0) in zcols


def test_job_table_holds_both_modes():
    t = DXR.job_table(K2_7B, 4)
    assert t.shape == (2, 1 + DXR.JMAX * DXR.NF)
    assert t[0, 0] == 13 and t[1, 0] == 30
    first = dict(zip(DXR.FIELDS, t[0, 1:1 + DXR.NF]))
    assert first == DXR.jobs(K2_7B, 4, draft=False)[0].core


def test_lora_pack_plan_is_contiguous_and_whole_chunks():
    X = DXR.layout(K2_7B, 4)
    ops = DXR.lora_pack_plan(K2_7B, 4)
    assert [o["tensor"].split(".")[-2] for o in ops] == list(DXR.LORA_TENSORS)
    end = 0
    for o in ops:
        assert o["op"] == "std_perm" and o["dst"] == end
        end += o["nch"] * DXR.CHUNK
    assert end <= X.LORA_BYTES


def test_l_must_be_a_multiple_of_four():
    with pytest.raises(OpRangeError):
        DXR.layout(K2_7B, 6)


def test_only_k2_ships_the_rows_route():
    assert DR.rows_route(K2_7B)["rows"]["lora_kernel"] == "dxl_lora"
    assert DR.rows_route(load_spec(SPECS / "qwen3-4b.json")) is None
