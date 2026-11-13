r"""gx: every Gemma 4 E-series layer kind on ONE xclbin, so the decode layer walk runs in one `xrt::hw_context`
instead of switching between four (OPEN-DECODE-ONE-CONTEXT; ux.py does the same for the 35B). The per-layer
input runs inside the layer program too (OPEN-FAMILY-GEMMA4), so nothing runs on the host between layers;
OPEN_G4_PLE_NPU=0 at export leaves it on the host.

One CompileTime `part` per kind, hence one instruction stream per kind over one image. The manifest names
them dx_local / dx / dx_local_shared / dx_shared and points all of them at one context.

- The main cores (Tile(c, 2)) run one program for every kind. The counts that differ between kinds --
  q|k|v bands, og elements and width, up | gate bands, h elements and width, and each width's
  group / chunk counts -- are RTP words the stream writes before its first fill (ux.py's plumbing;
  recipes/gemma4e.py rtp_words). Each band is the same GEMV call on the same table as in dx.py.
- Every kind uses the same main cores. E2B's sliding kinds (a 4-band kv projection) ran on 4 in dx.py;
  here cores 0-3 take one k band each and cores 4-7 one v band each (recipes/dense.py qkv_segments).
- One attention core set per kind (dxkit.Attn), each compiled exactly as dx.py compiles that kind, its
  symbols prefixed so the sets do not collide. A set parks on its `ain` while other kinds run. Head
  width, rotation and the shared row are compile-time in attn.h, and the attention cores are within
  2-3 KB of their 16 KB, so one core cannot carry two kinds.
- Shim budget (16 fills, 16 drains): four sets of four og fifos would take 16 drains alone, so each
  set's og elements are joined in a memory tile and drained as one. Fills: 8 w, lni, x, one ain per
  set. Drains: 8 y, lno, one og per set, one aout per own-cache set.
- The per-layer input, after the FFN residual y: the gate GEMV (ple_dim rows, one band per core from
  core 0) on y, z = gelu_tanh(g) * ple_in on the norm helper (the host writes ple_in into act before
  the walk), the projection GEMV over every main core, r = post_norm(p) and xres = (y + r) * scalar on
  the norm helper.

The norm helper and the host sequence are dxkit's: dx.py's layer pieces as they stood when the E-series
was validated (2026-10-20), kept apart so dx.py's later changes do not move these kernels.

Build: G4_PART=<local|global|local_shared|global_shared> python build_design.py designs/dense/gx.py <dir>
(the export does it for the gemma4e family).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
ATTN = HERE.parent / "attn"
LN = HERE.parent / "ln"
LINL = HERE.parent / "lin_layer"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes import dense as D, gemma4e as G4  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM  # noqa: E402
import dxkit as DK  # noqa: E402
from dxkit import bt  # noqa: E402

SPEC = current_spec()
if SPEC.family != "gemma4e":
    sys.exit(f"gx.py: the merged layer image is Gemma 4 E-series'; {SPEC.family} uses dx.py")
if SPEC.q8_roles or SPEC.quant == "q4_1_f32":
    sys.exit("gx.py: one q4_1 GEMV entry per main core (no q8 role, no f32-scale twin)")
KINDS = G4.kinds(SPEC)
PART = os.environ.get("G4_PART") or G4.PART[KINDS[0]]
KIND_OF = {v: k for k, v in G4.PART.items()}
if PART not in KIND_OF or KIND_OF[PART] not in KINDS:
    sys.exit(f"gx.py: G4_PART={PART!r}, the spec's kinds are {[G4.PART[k] for k in KINDS]}")
PART_IX = KINDS.index(KIND_OF[PART])
HID, PLE_D = SPEC.hidden, SPEC.ple_dim
N = G4.image_cores(SPEC)
GEOS = [G4.image_geometry(SPEC, k) for k in KINDS]            # (layout, geometry) per kind
PLE = G4.ple_on_npu()                                         # the per-layer input in the program (OPEN_G4_PLE_NPU=0: host)
FULL_PLE = PLE and PLE_D > 0                                  # else (the 12B) only the layer's output scale
PLES = [G4.ple_layout(SPEC, k) for k in KINDS]
G0 = GEOS[0][1]
assert len({g.CALL_BYTES for _, g in GEOS}) == 1, "gx.py: the kinds disagree on chunks per weight element"
CALL_BYTES = G0.CALL_BYTES
TAB_BYTES = max(g.TAB_BYTES for _, g in GEOS)
CH = D.chunk_bytes("q4_1")


def per_band(K):
    return D.band_bytes(K, "q4_1") // CH


def n_groups(K):
    return D.band_bytes(K, "q4_1") // CALL_BYTES


NG_H, PB_H = n_groups(HID), per_band(HID)
NG_P, PB_P = (n_groups(PLE_D), per_band(PLE_D)) if PLE_D else (0, 0)
XN_ELEMS, XM_ELEMS, O_PC, DOWN_PC = G0.XN_ELEMS, G0.XM_ELEMS, G0.O_PC, G0.DOWN_PC
Y_ELEMS, PG_BANDS, PP_PC = PLES[0].Y_ELEMS, PLES[0].PG_BANDS, PLES[0].PP_PC
assert all((g.XN_ELEMS, g.XM_ELEMS, g.O_PC, g.DOWN_PC, g.MS_U, g.MS_G, g.MS_FLOATS) ==
           (XN_ELEMS, XM_ELEMS, O_PC, DOWN_PC, G0.MS_U, G0.MS_G, G0.MS_FLOATS) for _, g in GEOS)
assert all((p.Y_ELEMS, p.PG_BANDS, p.PP_PC) == (Y_ELEMS, PG_BANDS, PP_PC) for p in PLES)
W = {name: i for i, name in enumerate(G4.RTP)}

# attention core tiles and shim columns, one slot per kind: (tiles, ain fill column, og drain column)
SLOTS = [([Tile(2 + c, 3) for c in range(4)], 2, 2),
         ([Tile(4 + c, 4) for c in range(4)], 4, 4),
         ([Tile(c, 4) for c in range(4)], 3, 3),
         ([Tile(c, 5) for c in range(4)], 5, 5)]
assert len(KINDS) <= len(SLOTS) and all(g.ACORES <= 4 for _, g in GEOS)
OWN_IX = [i for i, k in enumerate(KINDS) if k not in G4.OWN]           # kinds with an aout drain, in order
AOUT_COLS = [1, 6]
assert len(OWN_IX) <= len(AOUT_COLS)
# Join each set's og drains in a memory tile only when they would not fit the shim's 16 drains as they are
# (E2B / E4B: four sets); the 12B's two sets fit, and its global cores emit two og elements each.
JOIN = N + 1 + len(OWN_IX) + sum(g.ACORES for _, g in GEOS) > 16
OG_COLS = [[2, 3, 4, 5], [4, 5, 6, 7]]                            # unjoined og drains, per slot
assert JOIN or (len(KINDS) <= len(OG_COLS) and N <= 4)

_SZ = PLES if PLE else [L for L, _ in GEOS]
TOT = {"pool": max(p.POOL_BYTES for p in _SZ), "act": max(p.AD_BYTES for p in _SZ),
       "consts": max(p.CD_BYTES for p in _SZ), "kv": max(L.KV_BYTES for L, _ in GEOS),
       "ptab": max(L.PTAB_BYTES for L, _ in GEOS)}
OS = ["-Os"]


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gx(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, *, part: CompileTime[int] = 0,
       srchash: CompileTime[int] = 0):
    elem = np.ndarray[(CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(TAB_BYTES,), np.dtype[np.uint8]]
    ms_ty = np.ndarray[(G0.MS_FLOATS,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(G4.RTP_WORDS,), np.dtype[np.int32]]
    i32 = np.int32
    inc = include_dirs() + [str(GEMV), str(ATTN), str(LN), str(LINL)]

    def ef(sym, src, args, flags=OS):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    f_gy = ef("gemv_q4_gy", HERE / "gemv_q4_gy.cc", [elem, tab_ty, y_ty, i32, i32, i32])
    f_gms = ef("gemv_q4_gms", HERE / "gemv_q4_gms.cc", [elem, tab_ty, ms_ty, i32, i32, i32])
    f_act = ef("dense_act", HERE / "dense_act.cc", [ms_ty, y_ty])
    f_prep = ef("dense_prep", HERE / "dense_prep.cc", [x_ty, tab_ty, i32, i32])
    f_prepf = ef("dense_prep_f32", HERE / "dense_prep_f32.cc", [x_ty, tab_ty, i32, i32])
    L0 = GEOS[0][0]
    ln_flags = DK.ln_flags(G0, SPEC.norm_groups)
    lnk = DK.ln_kernels(L0, G0, inc, ln_flags)
    plek = DK.ple_kernels(L0, inc, ln_flags, PLE_D) if PLE else None
    attns = []
    for i, (L, G) in enumerate(GEOS):
        a = DK.Attn(G, L, DK.attn_flags(G, D.probe_env()), inc, prefix=f"g4{i}", sfx=f"_{i}")
        a.kernels()
        attns.append(a)

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)
    of_lni, of_lno = DK.ln_fifos(L0)
    for a in attns:
        a.fifos(join=JOIN)
    # Zeros in the image: the counts belong to the instruction stream (ux.py's rtp).
    rtp = [Buffer(rtp_ty, name=f"rtp{c}", initial_value=np.zeros(G4.RTP_WORDS, dtype=np.int32), use_write_rtp=True)
           for c in range(N)]

    def bands(win, yout, tab, n, ngroups, pb):
        for _ in range_(n):
            ye = yout.acquire(1)
            for g in range_(ngroups):
                we = win.acquire(1)
                f_gy(we, tab, ye, g, pb, 2)
                win.release(1)
            yout.release(1)

    def make_main_body(c):
        npg = 1 if c < PG_BANDS else 0                         # the per-layer gate's bands, one per core

        def main_body(win, xin, yout, tab, ms, w, f_gy, f_gms, f_act, f_prep, f_prepf):
            # The acquire comes FIRST and orders the reads: the stream writes the words before this fill.
            xe = DK.acq(xin, XN_ELEMS)
            n_a, n_og, k_o, ng_o, pb_o = w[W["qkv_bands"]], w[W["og_elems"]], w[W["o_k"]], w[W["o_groups"]], w[W["o_chunks"]]
            n_up, n_h, k_f = w[W["up_bands"]], w[W["h_elems"]], w[W["down_k"]]
            ng_f, pb_f = w[W["down_groups"]], w[W["down_chunks"]]
            for i in range(XN_ELEMS):
                f_prep(xe[i], tab, HID, i)
            bands(win, yout, tab, n_a, NG_H, PB_H)             # q | k | v against xn
            xin.release(XN_ELEMS)
            for i in range_(n_og):                             # o against og (K = q width)
                oe = xin.acquire(1)
                f_prep(oe, tab, k_o, i)
                xin.release(1)
            bands(win, yout, tab, O_PC, ng_o, pb_o)
            me = DK.acq(xin, XM_ELEMS)                         # up | gate per band against xm, act -> h band
            for i in range(XM_ELEMS):
                f_prep(me[i], tab, HID, i)
            for _ in range_(n_up):
                for g in range_(NG_H):
                    we = win.acquire(1)
                    f_gms(we, tab, ms, g, PB_H, G0.MS_U)
                    win.release(1)
                for g in range_(NG_H):
                    we = win.acquire(1)
                    f_gms(we, tab, ms, g, PB_H, G0.MS_G)
                    win.release(1)
                ye = yout.acquire(1)
                f_act(ms, ye)
                yout.release(1)
            xin.release(XM_ELEMS)
            for i in range_(n_h):                              # down against h (K = FFN width)
                he = xin.acquire(1)
                f_prepf(he, tab, k_f, i)
                xin.release(1)
            bands(win, yout, tab, DOWN_PC, ng_f, pb_f)
            if not FULL_PLE:
                return
            for i in range(Y_ELEMS):                           # the per-layer gate against y (f32)
                ye_ = xin.acquire(1)
                f_prepf(ye_, tab, HID, i)
                xin.release(1)
            bands(win, yout, tab, npg, NG_H, PB_H)
            ze = xin.acquire(1)                                # the projection against z (f32, K = ple_dim)
            f_prepf(ze, tab, PLE_D, 0)
            xin.release(1)
            bands(win, yout, tab, PP_PC, NG_P, PB_P)
        return main_body

    workers = [DK.ln_worker(G0, 99, of_lni, of_lno, lnk, ple=plek)]
    for c in range(N):
        workers.append(Worker(make_main_body(c), fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                           Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                                                           rtp[c], f_gy, f_gms, f_act, f_prep, f_prepf],
                               tile=Tile(c, 2), stack_size=0x1800))
    for a, slot in zip(attns, SLOTS):
        workers += a.workers(slot[0])

    def ple_sequence(P, L, a_pool, c_xres, a_consts, a_act, lni, lno, w_prods, x_prod, y_conss):
        A_B, P_B, C_B, ELN, YB = TOT["act"], TOT["pool"], TOT["consts"], L.ELN, BAND_ROWS * 4
        bb_h, bb_p = D.band_bytes(HID, "q4_1"), D.band_bytes(PLE_D, "q4_1")
        pw, py = Pipeline(3), Pipeline(3)
        tg_x = TaskGroup()
        # gate: g = inp_gate . y, one 64-row band on each of cores 0 ..
        for c in range(PG_BANDS):
            pw.fill(w_prods[c], a_pool, bt(P_B, P.POOL_PG + c * bb_h, bb_h))
            py.drain(y_conss[c], a_act, bt(A_B, P.AD_G + c * YB, YB))
        x_prod.fill(a_act, tap=bt(A_B, P.AD_Y, Y_ELEMS * ELEM), wait=True, group=tg_x)
        py.finish()                                                   # g is in DDR, beside ple_in
        tg_z = TaskGroup()
        lni.fill(a_act, tap=bt(A_B, P.AD_G, ELN), wait=True, group=tg_z)
        lno.drain(a_act, tap=bt(A_B, P.AD_Z, ELN), wait=True, group=tg_z)
        for c in range(N):                                            # the projection's weights meanwhile
            pw.fill(w_prods[c], a_pool, bt(P_B, P.POOL_PP + c * PP_PC * bb_p, PP_PC * bb_p))
        tg_z.finish()                                                 # z is in DDR
        for c in range(N):
            py.drain(y_conss[c], a_act, bt(A_B, P.AD_P + c * PP_PC * YB, PP_PC * YB))
        x_prod.fill(a_act, tap=bt(A_B, P.AD_Z, ELEM), wait=True, group=tg_x)
        py.finish()                                                   # p is in DDR
        tg_r = TaskGroup()
        lni.fill(a_act, tap=bt(A_B, P.AD_P, HID * 4), wait=True, group=tg_r)
        lni.fill(a_consts, tap=bt(C_B, P.CD_PLEN, ELN), wait=True, group=tg_r)
        lno.drain(a_act, tap=bt(A_B, P.AD_R, HID * 4), wait=True, group=tg_r)
        tg_r.finish()                                                 # r is in DDR
        tg_o = TaskGroup()
        lni.fill(a_act, tap=bt(A_B, P.AD_Y, HID * 4), wait=True, group=tg_o)
        lni.fill(a_consts, tap=bt(C_B, P.CD_PLES, ELN), wait=True, group=tg_o)
        lni.fill(a_act, tap=bt(A_B, P.AD_R, HID * 4), wait=True, group=tg_o)
        lno.drain(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_o)
        tg_o.finish()
        pw.finish()
        tg_x.finish()

    def sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_ps, aout_cs, og_cs):
        L, G = GEOS[part]
        P = PLES[part]
        words = G4.rtp_words(SPEC, KINDS[part])
        for c in range(N):
            for name, j in W.items():
                rtp[c][j] = words[name]
        bb = DK.Bands(D.band_bytes(HID, "q4_1"), D.band_bytes(G.QW, "q4_1"), D.band_bytes(G.FF, "q4_1"),
                      D.band_bytes(HID, "q4_1"))
        aout = aout_cs[OWN_IX.index(part)] if part in OWN_IX else None
        if JOIN:
            og = og_cs[part]
        else:                                                     # og_cs is every set's drains, set after set
            o0 = sum(g.ACORES for _, g in GEOS[:part])
            og = og_cs[o0:o0 + G.ACORES]
        DK.layer_sequence(G, L, 99, bb, a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod,
                          y_conss, ain_ps[part], aout, og, None, totals=TOT, og_joined=JOIN,
                          y_at=P.AD_Y if PLE else None)
        if FULL_PLE:
            ple_sequence(P, L, a_pool, c_xres, a_consts, a_act, lni, lno, w_prods, x_prod, y_conss)
        elif PLE:                                                 # xres = y * scalar on the norm helper
            tg_s = TaskGroup()
            lni.fill(a_act, tap=bt(TOT["act"], P.AD_Y, HID * 4), wait=True, group=tg_s)
            lni.fill(a_consts, tap=bt(TOT["consts"], P.CD_PLES, L.ELN), wait=True, group=tg_s)
            lno.drain(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_s)
            tg_s.finish()

    pool_ty = np.ndarray[(TOT["pool"],), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(TOT["consts"],), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(TOT["kv"],), np.dtype[np.uint8]]
    act_ty = np.ndarray[(TOT["act"],), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(TOT["ptab"],), np.dtype[np.uint8]]
    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, kv_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N)],
                            [a.of_ain.prod(tile=Tile(slot[1], 0)) for a, slot in zip(attns, SLOTS)],
                            [attns[i].of_aout.cons(tile=Tile(col, 0)) for i, col in zip(OWN_IX, AOUT_COLS)],
                            [a.of_ogj.cons(tile=Tile(slot[2], 0)) for a, slot in zip(attns, SLOTS)] if JOIN else
                            [a.of_og[c].cons(tile=Tile(cols[c], 0)) for a, cols in zip(attns, OG_COLS) for c in range(a.ACORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = gx
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + sorted(f.read_bytes() for f in (HERE.parent.parent / "recipes").glob("*.py"))
                + [(LN / "ln.h").read_bytes(), (LN / "ln_y.cc").read_bytes(), (LN / "ln_xn.cc").read_bytes(), (LN / "ln_nr32.cc").read_bytes(),
                   (LN / "ple_ln.cc").read_bytes(), (LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(), (HERE.parent.parent / "include" / "scalar_fp.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART_IX, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
