# Traces: OPEN-DECODE-ROWS, OPEN-UNO-LORA (canonical spec: specs/open-engine/spec.md)
"""recipes/dxl.py's tables: one that disagrees with its stream hangs the array, one over the wrong k-tiles quietly computes garbage."""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest

from recipes import dense as DR
from recipes import dxl as DXR
from recipes.catalogue import OpRangeError
from recipes.load import load_spec

SPECS = Path(__file__).resolve().parents[3] / "open_kernels" / "recipes" / "specs"
K2_7B = load_spec(SPECS / "k2-horizon-7b.json")
K2_37B = load_spec(SPECS / "k2-horizon-3.7b.json")      # hidden 2560: half slices on q and the MLP
GRANITE = load_spec(SPECS / "granite42-3b.json")         # 2560 everywhere, 5 bands a core
WIDE = (K2_7B, K2_37B, GRANITE)


def test_act_regions_do_not_overlap():
    X = DXR.layout(K2_7B, 4)
    regions = sorted((getattr(X, f), f) for f in vars(X) if f.startswith("AD_") and f != "AD_BYTES")
    sizes = {"AD_XN": 4 * X.XN_ROW, "AD_Q": 4 * X.Q_ROW, "AD_K": 4 * X.KV_ROW_ACT, "AD_V": 4 * X.KV_ROW_ACT,
             "AD_OG": 4 * X.OG_ROW, "AD_OUT": 4 * X.HID_ROW, "AD_RES": 4 * X.HID_ROW, "AD_XM": 4 * X.XM_ROW,
             "AD_H": 4 * X.H_ROW, "AD_OUT2": 4 * X.HID_ROW, "AD_ZQKV": 4 * DXR.Z_ROW, "AD_ZO": 4 * DXR.Z_ROW,
             "AD_ZGU": 4 * DXR.Z_ROW, "AD_ZD": 4 * DXR.Z_ROW, "AD_ZERO": DXR.XE, "AD_MODE": 3 * DXR.XE, "AD_JUNK": K2_7B.hidden * 2}
    for (a, fa), (b, _) in zip(regions, regions[1:]):
        assert a + sizes[fa] <= b, fa
    assert regions[-1][0] + sizes[regions[-1][1]] <= X.AD_BYTES


@pytest.mark.parametrize("spec", WIDE, ids=lambda s: f"h{s.hidden}_{s.family}")
def test_the_main_cores_fit_their_memory(spec):
    for L in (4, 8):
        X = DXR.layout(spec, L)
        assert DXR._main_l1(L, max(X.BTQ, X.BTO, X.BTD), X.BP, 2, X.TAB_K) <= DXR.L1_BUDGET


@pytest.mark.parametrize("spec", WIDE, ids=lambda s: f"h{s.hidden}_{s.family}")
def test_verify_jobs_cover_every_band_once(spec):
    G = DR.geometry(spec)
    js = DXR.jobs(spec, 4, draft=False)
    assert len(js) == 13
    bands = {}
    for j in js:
        assert j.wbuf == "pool" and j.core["S0"] == 0
        bands.setdefault(j.woff, 0)
        bands[j.woff] += j.core["BT"]
        assert j.core["KT"] * 256 == j.x[3] and j.core["S"] * j.core["KS"] == j.x[3]   # K
    L0 = DR.layout(spec)
    assert bands == {L0.POOL_Q: G.Q_PC, L0.POOL_K: G.KV_PC, L0.POOL_V: G.KV_PC, L0.POOL_O: G.O_PC,
                     L0.POOL_UP: G.UP_PC, L0.POOL_GATE: G.UP_PC, L0.POOL_DOWN: G.DOWN_PC}


def test_k2_7b_tables_are_unchanged_by_the_wider_widths():
    t = DXR.job_table(K2_7B, 4)
    nf = DXR.NF - 2                                  # the old twelve fields; NPREP and REP are the next two
    rows = t[:, 1:].reshape(2, DXR.JMAX, DXR.NF)
    assert DXR.FIELDS[-2:] == ("NPREP", "REP") and set(rows[:, :, nf].ravel()) <= {0, 4} and set(rows[:, :, nf + 1].ravel()) <= {0, 1}
    old = np.concatenate([np.concatenate([[t[m, 0]], rows[m, :, :nf].ravel()]) for m in range(2)]).astype(np.int32)
    assert hashlib.sha256(old.tobytes()).hexdigest() == "f69676338a3a437a507f3e73d2690a91497321762a794cc2b66ae577c674c77b"
    assert DXR.layout(K2_7B, 4).TABS == 1 and all(j.core["NPREP"] == 4 and not j.reuse for j in DXR.jobs(K2_7B, 4, False))
    assert DR.rows_route(K2_7B)["rows"]["head_act_bytes"] == 4 * K2_7B.hidden * 2


def test_granite_keeps_five_slice_tables_and_its_jobs_on_one_activation_prepare_it_once():
    # Traces: OPEN-DXL-TABS
    X = DXR.layout(GRANITE, 4)
    assert X.TABS == 5 and DXR._main_l1(4, 5, 4, 1, X.TAB_K, X.TABS) <= DXR.L1_REUSE
    for draft, reused, blocks in ((False, 9, 1984), (True, 12, 3424)):
        js = DXR.jobs(GRANITE, 4, draft)
        assert sum(j.reuse for j in js) == reused
        assert sum(j.core["NPREP"] * j.core["S"] * j.core["KS"] // 32 for j in js) == blocks
        assert all((j.core["NPREP"] == 0) == j.reuse for j in js)
        assert not js[0].reuse and not any(j.reuse for j in js if j.x[0] == "z" or j.core["MODE"] == DXR.PREP_F32)


def test_granite_head_is_28_tiles_of_7_bands_on_the_layers_cores():
    # Traces: OPEN-DXL-HEAD
    p = DXR.head_plan(GRANITE, 4)
    assert (p.nb, p.bt, p.nt, p.bb) == (196, 7, 28, 102400) and p.out_floats == 4 * 100352
    first, rest = p.rows
    assert (first["NPREP"], first["REP"], rest["NPREP"], rest["REP"]) == (4, 1, 0, 27)   # prepared once, the other tiles reuse it
    assert first["S"] * first["KS"] == GRANITE.hidden and first["BT"] <= DXR.layout(GRANITE, 4).YB + DXR.layout(GRANITE, 4).BP
    t = DXR.head_table(GRANITE, 4)
    assert t.shape == (1 + DXR.HEAD_JMAX * DXR.NF,) and t[0] == 2
    assert DXR.head_plan(K2_7B, 4) is None and DXR.head_plan(K2_37B, 4) is None        # no table cache, no folded head


def test_an_odd_multiple_of_512_takes_half_slices():
    X = DXR.layout(GRANITE, 4)
    assert (X.KS_H, X.KS_Q, X.TAB_K) == (512, 512, 512) and (X.BTQ, X.BTO, X.BTD, X.BP) == (5, 5, 5, 4)
    q = DXR.jobs(GRANITE, 4, draft=False)[0]
    assert (q.core["S"], q.core["KS"], q.core["CPS"], q.core["BT"]) == (5, 512, 4, 5)
    X = DXR.layout(K2_37B, 4)
    assert (X.KS_H, X.KS_Q, X.TAB_K) == (512, 1024, 1024)


@pytest.mark.parametrize("spec", WIDE, ids=lambda s: f"h{s.hidden}_{s.family}")
def test_every_x_element_reads_inside_act(spec):
    X = DXR.layout(spec, 4)
    for draft in (False, True):
        for j in DXR.jobs(spec, 4, draft):
            if j.x[0] != "act":
                continue
            _, off, row, K, KS, f32 = j.x
            xb = KS * (4 if f32 else 2)
            assert off + (K // KS - 1) * xb + 3 * row + DXR.XE <= X.AD_BYTES


def test_a_width_off_the_512_grid_is_refused():
    with pytest.raises(OpRangeError):
        DXR.layout(dataclasses.replace(GRANITE, hidden=2304), 4)


@pytest.mark.parametrize("spec", WIDE, ids=lambda s: f"h{s.hidden}_{s.family}")
def test_the_head_act_holds_every_rows_whole_elements(spec):
    ne = -(-spec.hidden // 1024)
    assert 3 * spec.hidden * 2 + ne * DXR.XE <= DXR.head_act_bytes(spec, 4)


@pytest.mark.parametrize("spec", WIDE, ids=lambda s: f"h{s.hidden}_{s.family}")
def test_draft_jobs_append_one_lora_k_tile_per_band(spec):
    js = DXR.jobs(spec, 4, draft=True)
    assert len(js) == 30 and len(js) <= DXR.JMAX
    a = [j for j in js if j.wbuf == "lora" and j.x[0] == "act"]
    assert [j.woff for j in a] == [DXR.layout(spec, 4).LORA[n][0] for n in ("a_qkv", "a_o", "a_gu", "a_d")]
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
    X = DXR.layout(spec, 4)
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


def test_only_k2_and_granite_ship_the_rows_route(monkeypatch):
    monkeypatch.setenv("DXL_HEAD_FOLD", "1")
    assert DR.rows_route(K2_7B)["rows"]["lora_kernel"] == "dxl_lora"
    assert DR.rows_route(K2_37B)["rows"]["head_act_bytes"] == 4 * 2560 * 2 + 1024
    g = DR.rows_route(GRANITE)
    monkeypatch.delenv("DXL_HEAD_FOLD")
    assert DR.rows_route(GRANITE)["rows"]["head"] == "lmhl" and "head_fold" not in DR.rows_route(GRANITE)["rows"]
    monkeypatch.setenv("DXL_HEAD_FOLD", "1")
    assert g["rows"]["head_out_floats"] == 4 * 100352 and g["rows"]["head_fold"] and g["rows"]["lora_pool_bytes"] > 0
    assert g["rows"]["head"] == "dxl_head" and g["kernels"]["dxl_head"]["context"] == "dxl" and "lmhl" not in g["builds"]
    assert "head_fold" not in DR.rows_route(K2_7B)["rows"]
    for spec in (K2_7B, K2_37B, GRANITE):            # the mode element: three x elements in act, after the z seed row
        X, r = DXR.layout(spec, 4), DR.rows_route(spec)["rows"]
        assert (r["mode_off"], r["mode_elem"]) == (X.AD_MODE, DXR.XE) and X.AD_MODE == X.AD_ZERO + DXR.XE
        assert X.AD_MODE + 3 * DXR.XE <= X.AD_JUNK
    assert DR.rows_route(load_spec(SPECS / "qwen3-4b.json")) is None
    heads = {DR.rows_route(s)["builds"]["lmhl"]["build_dir"] for s in (K2_7B, K2_37B)}
    assert len(heads) == 2                 # one vocab, two widths: two head builds
