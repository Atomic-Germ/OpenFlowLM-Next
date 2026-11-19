r"""lg: one LFM3 layer (LFM3-1.7B) as two dispatches over fl.py's fabric (recipes/lfm3.py).

  LG_PART=conv  ln -> B | C | u (in_proj's three row slices) -> the conv core (7 taps) -> out over y
                -> t = post_operator_norm(out); res = x + t; xm = ffn_norm(res)
                args: pool, xres f32[HID] (in), consts [lnw | postop | ffnn | postffn | - | taps f32 [7][HID]],
                state, act
  LG_PART=att   ln -> q | gate | k | v -> per-head RMSNorm of q, k -> RoPE (the position record) -> causal
                attention (the window is the driver's attnpos) -> * sigmoid(gate) -> o -> the same close
                args: pool, xres f32[HID] (in), consts [... | qn kn], kv, act, ptab
  LG_PART=mlp   per slot e: gate | up -> silu(gate) * up -> h_e; down over the slots into each core's
                accumulator; t = post_ffn_norm(down); xres = res + t. Five slots of 1536 (3 bands a core).
                args: pool, xres f32[HID] (out), consts, act

Both mixers read their weights at POOL_MIX, the same offset in every layer's pool, so one conv build and one
attention build serve every layer; the sliding and the full attention layers share the attention build (the
window and the position table differ, host side).

fl.py's fabric: 8 main cores (Tile(c, 2)) on w (10 KB elements: two q4_1 chunks) / x (4 KB broadcast) / y
(64 floats) streams and the norm helper (Tile(0, 3)); for attention 4 cores of 4 heads (Tile(2 + c, 3)) on one
broadcast `ain`; for the conv the one conv core at Tile(2, 3). The TUs come from gen_kernels.py.

Build: OPEN_KERNELS_SPEC=<lfm3 spec> LG_PART=conv|att|mlp python build_design.py designs/lfm3/lg.py <out>
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
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
ATTN = HERE.parent / "attn"
LN = HERE.parent / "ln"
LINL = HERE.parent / "lin_layer"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes import lfm3 as FR  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM  # noqa: E402

SPEC = current_spec()
R = FR.recipe(SPEC)
L, G = R.layout, R.geo
PART = os.environ.get("LG_PART", "conv")
assert PART in ("conv", "att", "mlp", "convmlp", "attmlp", "shared"), f"LG_PART={PART!r} (conv | att | mlp | convmlp | attmlp | shared)"
F = FR.ffn(SPEC)
POOL_BYTES = L.POOL_BYTES
HID, QW, KVW, HD = G.HID, G.QW, G.KVW, G.HD
N_CORES, ACORES, NHL = G.N_CORES, G.ACORES, G.NHL
ELN, E_A = L.ELN, G.E_A
OS = ["-Os"]
PROBES = sorted(FR.probe_env().items())
ATTN_FLAGS = FR.attn_flags(G, [(k, v) for k, v in PROBES if k.startswith("ATTN_")])
NOGATE = bool(os.environ.get("LG_NOGATE"))             # bring-up bisect: no gate stream, attn_fin_ng
if NOGATE:
    ATTN_FLAGS = [f if f != "-DATTN_GATE=1" else "-DATTN_GATE=0" for f in ATTN_FLAGS]
LN_FLAGS = FR.ln_flags(G)
NULL = ["-DGEMV_NULL"] if os.environ.get("LG_NULL") else []
OGH = min(NHL, G.HPO)                  # heads per og element (attn.h kOGH)
N_OG = NHL // OGH
YB = BAND_ROWS * 4
RS = 2                                 # std_perm's 64-row band: two 32-row chunks a k-tile
SCW, TAPS = G.SCW, G.TAPS
SCB, SCYB = SCW * 4, SCW * 2
FF, NSLOT, UP_PC = F.FF, F.SLOTS, F.UP_PC
BB_F = FR.band_bytes(FF)


def pb(K):
    """q4_1 chunks per 64-row band"""
    return K // 128


def ng(K):
    """w elements per band"""
    return pb(K) // G.PER_CALL


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def rows_tap(total, off, rows, n, stride):
    """`rows` runs of n bytes, stride apart, from off: one DMA descriptor"""
    return TensorAccessPattern((1, total), off, [1, 1, rows, n], [0, 0, stride, 1])


BB_H = FR.band_bytes(HID)
NG_H, PB_H = ng(HID), pb(HID)
NG_F, PB_F = ng(FF), pb(FF)


def acq(fifo, n):
    e = fifo.acquire(n)
    return [e] if n == 1 else e


def gemv_bands(win, yout, tab, f_gy, nbands, ngroups, per_band):
    for _ in range_(nbands):
        ye = yout.acquire(1)
        for g in range_(ngroups):
            we = win.acquire(1)
            f_gy(we, tab, ye, g, per_band, RS)
            win.release(1)
        yout.release(1)


def _kernels(names, elem, x_ty, y_ty, tab_ty, ms_ty, acc_ty):
    i32 = np.int32
    inc = include_dirs() + [str(GEMV), str(ATTN), str(LN), str(LINL)]

    def ef(sym, args, flags=OS):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=flags)
    mk = {"gy": lambda: ef("lg_gy", [elem, tab_ty, y_ty, i32, i32, i32], OS + NULL),
          "gms": lambda: ef("lg_gms", [elem, tab_ty, ms_ty, i32, i32, i32], OS + NULL),
          "gacc": lambda: ef("lg_gacc", [elem, tab_ty, ms_ty, acc_ty, i32, i32, i32, i32, i32], OS + NULL),
          "act": lambda: ef("lg_act1", [ms_ty, y_ty]),
          "copy": lambda: ef("lg_copy", [acc_ty, y_ty, i32]),
          "prep": lambda: ef("lg_prep", [x_ty, tab_ty, i32, i32]),
          "prepf": lambda: ef("lg_prepf", [x_ty, tab_ty, i32, i32])}
    return [mk[n]() for n in names]


def _ln_fns(u8_ln, names=("nr", "nr32", "y", "xn")):
    i32 = np.int32
    inc = include_dirs() + [str(LN), str(LINL)]

    def af(sym, src, args):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=LN_FLAGS)
    mk = {"nr": lambda: af("ln_nr", LINL / "ln_nr.cc", [u8_ln] * 4),
          "nr32": lambda: af("ln_nr32", LN / "ln_nr32.cc", [u8_ln] * 4 + [i32]),
          "y": lambda: af("ln_y", LN / "ln_y.cc", [u8_ln] * 5 + [i32]),
          "xn": lambda: af("ln_xn", LN / "ln_xn.cc", [u8_ln] * 6)}
    return [mk[n]() for n in names]


def _post_norm(ain, aout, f_nr32):
    # a sandwich norm on a block output: [o0 o1 w] -> [t0] [t1] (fp32 halves)
    e = ain.acquire(3)
    for i in range(2):
        o = aout.acquire(1)
        f_nr32(e[0], e[1], e[2], o, i)
        aout.release(1)
    ain.release(3)


def _add_norm(ain, aout, f_lny, f_lnx, with_xn):
    # [x0 x1 w a0 a1] -> [y0] [y1] [xn]
    e = ain.acquire(5)
    for i in range(2):
        o = aout.acquire(1)
        f_lny(e[0], e[1], e[3], e[4], o, i)
        aout.release(1)
    if with_xn:
        o = aout.acquire(1)
        f_lnx(e[0], e[1], e[3], e[4], e[2], o)
        aout.release(1)
    ain.release(5)


def _ln_body(ain, aout, f_nr, f_nr32, f_lny, f_lnx):
    # 1. the layer-entry norm: [x0 x1 lnw] -> [xn]
    e = ain.acquire(3)
    o = aout.acquire(1)
    f_nr(e[0], e[1], e[2], o)
    aout.release(1)
    ain.release(3)
    _post_norm(ain, aout, f_nr32)             # 2. t = post_operator_norm(out)
    _add_norm(ain, aout, f_lny, f_lnx, True)  # 3. res = x + t; xm = ffn_norm(res)


def _entry_norm(lni, lno, c_xres, a_consts, a_act):
    tg = TaskGroup()
    lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg)
    lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_LNW, ELN), wait=True, group=tg)
    lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XN, ELN), wait=True, group=tg)
    return tg


def _res_norm(lni, lno, c_xres, a_consts, a_act, py):
    """t = post_operator_norm(out); res = x + t; xm = ffn_norm(res), once `out` is in DDR (py's drains)."""
    py.finish()                                               # out is in DDR
    tg_t = TaskGroup()
    lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT, HID * 4), wait=True, group=tg_t)
    lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTOP, ELN), wait=True, group=tg_t)
    lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_T, HID * 4), wait=True, group=tg_t)
    tg_t.finish()                                             # t is in DDR
    tg = TaskGroup()
    lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg)
    lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_FFNN, ELN), wait=True, group=tg)
    lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_T, HID * 4), wait=True, group=tg)
    lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_RES, HID * 4), wait=True, group=tg)
    lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XM, ELN), wait=True, group=tg)
    tg.finish()


def _types():
    return dict(
        elem=np.ndarray[(G.CALL_BYTES,), np.dtype[np.uint8]],
        x_ty=np.ndarray[(ELEM // 2,), np.dtype[bfloat16]],
        y_ty=np.ndarray[(BAND_ROWS,), np.dtype[np.float32]],
        tab_ty=np.ndarray[(G.TAB_BYTES,), np.dtype[np.uint8]],
        u8_ln=np.ndarray[(ELN,), np.dtype[np.uint8]],
        pool_ty=np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]],
        xres_ty=np.ndarray[(HID,), np.dtype[np.float32]],
        consts_ty=np.ndarray[(L.CD_BYTES,), np.dtype[np.uint8]],
        act_ty=np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]],
    )


def conv_main(win, xin, yout, tab, f_gy, f_prep):
    # B | C | u against xn (K = HID): three [hidden, hidden] slices of in_proj
    xe = acq(xin, G.XN_ELEMS)
    for i in range(G.XN_ELEMS):
        f_prep(xe[i], tab, HID, i)
    gemv_bands(win, yout, tab, f_gy, 3 * G.SC_PC, ng(HID), pb(HID))
    xin.release(G.XN_ELEMS)
    # out against the conv's y (K = HID)
    oe = acq(xin, G.XN_ELEMS)
    for i in range(G.XN_ELEMS):
        f_prep(oe[i], tab, HID, i)
    gemv_bands(win, yout, tab, f_gy, G.SC_PC, ng(HID), pb(HID))
    xin.release(G.XN_ELEMS)


def conv_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
             scin_p, scy_c, scn_c, st_bytes=None):
    # 1. layer-entry norm: xn -> act
    STB = st_bytes or L.STATE_BYTES
    tg_ln = _entry_norm(lni, lno, c_xres, a_consts, a_act)
    # 2. B | C | u: weights now, xn after the norm
    pw, py = Pipeline(3), Pipeline(3)
    for c in range(N_CORES):
        for base in (L.POOL_B, L.POOL_C, L.POOL_U):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, base + c * G.SC_PC * BB_H, G.SC_PC * BB_H))
        for dst in (L.AD_B, L.AD_C, L.AD_U):
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, dst + c * G.SC_PC * YB, G.SC_PC * YB))
    tg_ln.finish()                                            # xn is in DDR
    tg_x = TaskGroup()
    x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_XN, G.XN_ELEMS * ELEM), wait=True, group=tg_x)
    # 3. the conv core, element by element after the GEMVs: its taps and state rows (one strided
    #    descriptor each), B / C / u in; y and the new state out.
    ps_in, ps_out = Pipeline(3), Pipeline(3)
    py.finish(*y_conss)                                       # B, C, u are in DDR
    for i in range(G.SC_ELEMS):
        ps_in.fill(scin_p, a_consts, rows_tap(L.CD_BYTES, L.CD_CONV + i * SCB, TAPS, SCB, HID * 4))
        ps_in.fill(scin_p, a_state, rows_tap(STB, i * SCB, TAPS - 1, SCB, HID * 4))
        ps_in.fill(scin_p, a_act, bt(L.AD_BYTES, L.AD_B + i * SCB, SCB))
        ps_in.fill(scin_p, a_act, bt(L.AD_BYTES, L.AD_C + i * SCB, SCB))
        ps_in.fill(scin_p, a_act, bt(L.AD_BYTES, L.AD_U + i * SCB, SCB))
        ps_out.drain(scy_c, a_act, bt(L.AD_BYTES, L.AD_Y + i * SCYB, SCYB))
        ps_out.drain(scn_c, a_state, rows_tap(STB, i * SCB, TAPS - 1, SCB, HID * 4))
    ps_out.finish()                                           # y and the new state are in DDR
    ps_in.finish()
    # 4. out projection against y (bf16 hidden: one x element, like the norm's output)
    x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_Y, G.XN_ELEMS * ELEM), wait=True, group=tg_x)
    for c in range(N_CORES):
        pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, L.POOL_OUT + c * G.SC_PC * BB_H, G.SC_PC * BB_H))
        py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_OUT + c * G.SC_PC * YB, G.SC_PC * YB))
    # 5. t = post_operator_norm(out); res = x + t; xm = ffn_norm(res)
    _res_norm(lni, lno, c_xres, a_consts, a_act, py)
    pw.finish()
    tg_x.finish()


def att_main(win, xin, yout, tab, f_gy, f_prep):
    # q | gate | k | v against xn (K = HID)
    xe = acq(xin, G.XN_ELEMS)
    for i in range(G.XN_ELEMS):
        f_prep(xe[i], tab, HID, i)
    gemv_bands(win, yout, tab, f_gy, 2 * G.Q_PC + 2 * G.KV_PC, ng(HID), pb(HID))
    xin.release(G.XN_ELEMS)
    # o against og (K = QW)
    oe = acq(xin, G.OG_ELEMS)
    for i in range(G.OG_ELEMS):
        f_prep(oe[i], tab, QW, i)
    gemv_bands(win, yout, tab, f_gy, G.O_PC, ng(QW), pb(QW))
    xin.release(G.OG_ELEMS)


def att_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c,
             og_cs, pad=0, kv_bytes=None):
    # 1. layer-entry norm: xn -> act
    KVB = kv_bytes or L.KV_BYTES
    tg_ln = _entry_norm(lni, lno, c_xres, a_consts, a_act)
    # 2. q | gate | k | v: weights now, xn after the norm. FOUR transfers a core and channel are issued
    #    before xn exists, so the throttle must allow four (a fourth awaiting the first would never return).
    pw, py = Pipeline(4), Pipeline(4)
    for c in range(N_CORES):
        pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, L.POOL_Q + c * G.Q_PC * BB_H, G.Q_PC * BB_H))
        pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, L.POOL_GATE + c * G.Q_PC * BB_H, G.Q_PC * BB_H))
        pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, L.POOL_K + c * G.KV_PC * BB_H, G.KV_PC * BB_H))
        # pad: the shared image's main cores run the conv's band count, so v's fill carries `pad` more bands
        # (whatever follows in the pool) and their results go to scratch (t, rewritten by the close)
        pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, L.POOL_V + c * G.KV_PC * BB_H, (G.KV_PC + pad) * BB_H))
        if pad:     # q and gate in one 2-row drain, so the four drains a core fit the throttle
            assert L.AD_GATE - L.AD_Q == QW * 4
            py.drain(y_conss[c], a_act, TensorAccessPattern((1, L.AD_BYTES), L.AD_Q + c * G.Q_PC * YB,
                                                            [1, 1, 2, G.Q_PC * YB], [0, 0, QW * 4, 1]))
        else:
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_Q + c * G.Q_PC * YB, G.Q_PC * YB))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_GATE + c * G.Q_PC * YB, G.Q_PC * YB))
        py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_KVN + c * G.KV_PC * YB, G.KV_PC * YB))
        py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_KVN + KVW * 4 + c * G.KV_PC * YB, G.KV_PC * YB))
        if pad:
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_T + c * pad * YB, pad * YB))
    tg_ln.finish()                                            # xn is in DDR
    tg_x = TaskGroup()
    x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_XN, G.XN_ELEMS * ELEM), wait=True, group=tg_x)
    # 3. attention: meta + record now, q / k / v after the GEMVs, the window, the gate, the new row out
    pa_out, pa_in = Pipeline(3), Pipeline(3)
    pa_out.drain(aout_c, a_kv, bt(KVB, L.KV_ROW, L.KV_ROW))           # [k' | v'] -> row pos (attnpos)
    for c in range(ACORES):
        pa_out.drain(og_cs[c], a_act, bt(L.AD_BYTES, L.AD_OG + c * NHL * HD * 2, NHL * HD * 2))
    pa_in.fill(ain_p, a_consts, bt(L.CD_BYTES, L.CD_META, E_A))              # [qn | kn]
    pa_in.fill(ain_p, a_ptab, bt(L.PTAB_BYTES, L.PTAB_ROW, L.PTAB_ROW))      # the position record (attnpos)
    py.finish(*y_conss)                                       # q, gate, k, v are in DDR
    pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_Q, QW * 4))
    pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN, KVW * 4))
    pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN + KVW * 4, KVW * 4))
    pa_in.fill(ain_p, a_kv, bt(KVB, 0, L.KV_ROW))                     # the window: rows [0, nf) (attnpos)
    if not NOGATE:
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_GATE, QW * 4))          # the gate, after the window
    # 4. o projection against og
    for c in range(N_CORES):
        pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, L.POOL_O + c * G.O_PC * FR.band_bytes(QW),
                                       G.O_PC * FR.band_bytes(QW)))
        py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_OUT + c * G.O_PC * YB, G.O_PC * YB))
    pa_out.finish()                                           # og (and the new cache row) are in DDR
    x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OG, G.OG_ELEMS * ELEM), wait=True, group=tg_x)
    # 5. t = post_operator_norm(out); res = x + t; xm = ffn_norm(res)
    _res_norm(lni, lno, c_xres, a_consts, a_act, py)
    pw.finish()
    pa_in.finish()
    tg_x.finish()


def mlp_main(win, xin, yout, tab, ms, acc, gms, gacc, act, copy, prep, prepf):
    # xm
    xe = acq(xin, G.XM_ELEMS)
    for i in range(G.XN_ELEMS):
        prep(xe[i], tab, HID, i)
    xin.release(G.XM_ELEMS)
    # per slot: [gate | up] per band -> h_e band
    for e in range_(NSLOT):
        for _ in range_(UP_PC):
            for g in range_(NG_H):
                we = win.acquire(1)
                gms(we, tab, ms, g, PB_H, G.MS_G)
                win.release(1)
            for g in range_(NG_H):
                we = win.acquire(1)
                gms(we, tab, ms, g, PB_H, G.MS_U)
                win.release(1)
            ye = yout.acquire(1)
            act(ms, ye)
            yout.release(1)
    # down: h_e's table, then slot e's bands added onto the accumulator
    for e in range_(NSLOT):
        for i in range_(F.H_ELEMS):
            he = xin.acquire(1)
            prepf(he, tab, FF, i)
            xin.release(1)
        for b in range_(G.DOWN_PC):
            for g in range_(NG_F):
                we = win.acquire(1)
                gacc(we, tab, ms, acc, g, PB_F, b, e, NG_F)
                win.release(1)
    for b in range_(G.DOWN_PC):
        ye = yout.acquire(1)
        copy(acc, ye, b)
        yout.release(1)


def mlp_ln_body(ain, aout, f_nr32, f_lny):
    _post_norm(ain, aout, f_nr32)             # t = post_ffn_norm(down)
    e = ain.acquire(5)                        # [res0 res1 w t0 t1] -> xres halves
    for i in range(2):
        o = aout.acquire(1)
        f_lny(e[0], e[1], e[3], e[4], o, i)
        aout.release(1)
    ain.release(5)


def mlp_sequence(a_pool, c_xres, a_consts, a_act, lni, lno, w_prods, x_prod, y_conss):
    pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
    # 1. xm to every main core
    px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_XM, G.XM_ELEMS * ELEM))
    # 2. gate | up: each core's h bands of every slot in ONE drain (slot e's at AD_H + e * H_STRIDE),
    #    then slot e's fills
    for c in range(N_CORES):
        py.drain(y_conss[c], a_act, TensorAccessPattern((1, L.AD_BYTES), L.AD_H + c * UP_PC * YB,
                                                        [1, 1, NSLOT, UP_PC * YB], [0, 0, F.H_STRIDE, 1]))
    for e in range(NSLOT):
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, e * F.EXPERT_UG + c * UP_PC * 2 * BB_H, UP_PC * 2 * BB_H))
    py.finish()                                               # every h_e is in DDR
    # 3. down: h_e, then slot e's down bands; the cores hold the sum
    for c in range(N_CORES):
        py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_OUT2 + c * G.DOWN_PC * YB, G.DOWN_PC * YB))
    for e in range(NSLOT):
        px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_H + e * F.H_STRIDE, F.H_ELEMS * ELEM))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, F.POOL_DOWN + e * F.UP_BYTES + c * G.DOWN_PC * BB_F,
                                           G.DOWN_PC * BB_F))
    py.finish()                                               # out2 is in DDR
    # 4. t = post_ffn_norm(out2); xres = res + t
    tg_t = TaskGroup()
    lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT2, HID * 4), wait=True, group=tg_t)
    lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTFFN, ELN), wait=True, group=tg_t)
    lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_T2, HID * 4), wait=True, group=tg_t)
    tg_t.finish()                                             # t2 is in DDR
    tg = TaskGroup()
    lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_RES, HID * 4), wait=True, group=tg)
    lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTFFN, ELN), wait=True, group=tg)   # the unused w
    lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_T2, HID * 4), wait=True, group=tg)
    lno.drain(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg)
    tg.finish()
    pw.finish()
    px.finish()


def sc_body(scin, scy, scn, wl, acc, f_cp, f_mac0, f_mac, f_fin):
    """One token, element by element (a row at a time: a fifo has 16 buffer descriptors). Per element: the
    taps (copied out), the state rows (acc = sum s_k w_k, the window sliding out as it goes), then B, C, u
    in; the gated output (bf16) and the new state rows out."""
    for _ in range_(G.SC_ELEMS):
        for k in range(TAPS):
            t = scin.acquire(1)
            f_cp(t, wl, k)
            scin.release(1)
        s = scin.acquire(1)
        f_mac0(s, wl, acc)
        scin.release(1)
        for k in range(1, TAPS - 1):
            s = scin.acquire(1)
            n = scn.acquire(1)
            f_mac(s, wl, acc, n, k)
            scn.release(1)
            scin.release(1)
        e = scin.acquire(3)
        oy = scy.acquire(1)
        n = scn.acquire(1)
        f_fin(e[0], e[1], e[2], wl, acc, oy, n)
        scn.release(1)
        scy.release(1)
        scin.release(3)


def attn_core(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, h0, c,
         f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
    e = ain.acquire(2)                                     # [qn | kn], the position record
    f_meta(e[0], e[1], qn, kn, cs, pbuf)
    ain.release(2)
    for h in range_(G.Q_AIN_ELEMS):
        e = ain.acquire(1)
        f_q(e, qn, cs, qs, h)
        ain.release(1)
    for h in range_(G.K_AIN_ELEMS):
        e = ain.acquire(1)
        f_k(e, kn, cs, tmp, kout, h)
        ain.release(1)
    for h in range_(G.K_AIN_ELEMS):
        e = ain.acquire(1)
        f_v(e, vout, h)
        ain.release(1)
    if aout is not None:                                   # core 0 owns the cache row
        o = aout.acquire(1)
        for j in range_(KVW):
            o[j] = kout[j]
        aout.release(1)
        o = aout.acquire(1)
        for j in range_(KVW):
            o[j] = vout[j]
        aout.release(1)
    f_init(oacc, ml)
    for _ in range_(pbuf[4]):                              # whole blocks of RB cached rows
        e = ain.acquire(2 * G.RB)
        f_stepb(*[e[i] for i in range(2 * G.RB)], qs, oacc, ml, pbuf, h0)
        ain.release(2 * G.RB)
    for _ in range_(pbuf[5]):                              # what did not fill a block
        e = ain.acquire(2)
        f_step(e[0], e[1], qs, oacc, ml, pbuf, h0)
        ain.release(2)
    f_stepn(kout, vout, qs, oacc, ml, h0)
    # The gate arrives for EVERY og element on the broadcast stream (two elements an og element, in head
    # order): each core consumes all of them and finishes only its own N_OG (layer_x/xlayer.py's law).
    if NOGATE:
        for hp in range_(N_OG):
            o = ogout.acquire(1)
            f_fin(oacc, ml, o, hp)
            ogout.release(1)
        return
    for _ in range(c * N_OG):
        ain.acquire(2)
        ain.release(2)
    for hp in range_(N_OG):
        g = ain.acquire(2)
        o = ogout.acquire(1)
        f_fin(oacc, ml, g[0], g[1], o, hp)
        ogout.release(1)
        ain.release(2)
    for _ in range((ACORES - 1 - c) * N_OG):
        ain.acquire(2)
        ain.release(2)


def attn0(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf,
          f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
    attn_core(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, 0, 0,
         f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb)


def make_attn(c):
    def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf,
             f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
        attn_core(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, c * NHL, c,
             f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb)
    return body


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lg_conv(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, *, srchash: CompileTime[int] = 0):
    T = _types()
    elem, x_ty, y_ty, tab_ty, u8_ln = T["elem"], T["x_ty"], T["y_ty"], T["tab_ty"], T["u8_ln"]
    state_ty = np.ndarray[(L.STATE_BYTES,), np.dtype[np.uint8]]
    fsc = np.ndarray[(SCW,), np.dtype[np.float32]]              # one row of SCW channels: a tap, a state row, B, C, u
    bsc = np.ndarray[(SCW,), np.dtype[bfloat16]]                # y, bf16 (the out GEMV's activation)
    f_gy, f_prep = _kernels(("gy", "prep"), elem, x_ty, y_ty, tab_ty, None, None)
    f_nr, f_nr32, f_lny, f_lnx = _ln_fns(u8_ln)
    wloc = np.ndarray[(TAPS * SCW,), np.dtype[np.float32]]       # the core-local taps, tap-major
    f_cp = ExternalFunction("lg_cp", source_file=str(HERE / "lg_cp.cc"), arg_types=[fsc, wloc, np.int32],
                            include_dirs=include_dirs(), compile_flags=OS)
    f_mac0 = ExternalFunction("lg_scmac0", source_file=str(HERE / "lg_scmac0.cc"), arg_types=[fsc, wloc, fsc],
                              include_dirs=include_dirs(), compile_flags=OS)
    f_mac = ExternalFunction("lg_scmac", source_file=str(HERE / "lg_scmac.cc"),
                             arg_types=[fsc, wloc, fsc, fsc, np.int32], include_dirs=include_dirs(), compile_flags=OS)
    f_fin = ExternalFunction("lg_scfin", source_file=str(HERE / "lg_scfin.cc"),
                             arg_types=[fsc, fsc, fsc, wloc, fsc, bsc, fsc], include_dirs=include_dirs(),
                             compile_flags=OS)

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_scin = ObjectFifo(fsc, name="scin", depth=4)             # a row at a time; B, C, u acquired together
    of_scy = ObjectFifo(bsc, name="scy", depth=2)
    of_scn = ObjectFifo(fsc, name="scn", depth=2)               # the new state rows, a row at a time



    workers = [Worker(_ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_nr32, f_lny, f_lnx], tile=Tile(0, 3),
                      stack_size=FR.STACK)]
    for c in range(N_CORES):
        workers.append(Worker(conv_main, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                   Buffer(tab_ty, name=f"tab{c}"), f_gy, f_prep],
                              tile=Tile(c, 2), stack_size=FR.STACK))
    workers.append(Worker(sc_body, fn_args=[of_scin.cons(), of_scy.prod(), of_scn.prod(),
                                            Buffer(wloc, name="wloc"), Buffer(fsc, name="scacc"),
                                            f_cp, f_mac0, f_mac, f_fin],
                          tile=Tile(2, 3), stack_size=FR.STACK))


    # Shim channels: columns 0..2 already pair a producer with lni, x and the conv input (y on column 1's
    # second S2MM, the new state on column 2's).
    rt = Runtime(conv_sequence, [T["pool_ty"], T["xres_ty"], T["consts_ty"], state_ty, T["act_ty"],
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_scin.prod(tile=Tile(2, 0)),
                            of_scy.cons(tile=Tile(1, 0)), of_scn.cons(tile=Tile(2, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lg_att(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, *, srchash: CompileTime[int] = 0):
    T = _types()
    elem, x_ty, y_ty, tab_ty, u8_ln = T["elem"], T["x_ty"], T["y_ty"], T["tab_ty"], T["u8_ln"]
    u8_a = np.ndarray[(E_A,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(L.KV_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(L.PTAB_BYTES,), np.dtype[np.uint8]]
    pb_ty = np.ndarray[(8,), np.dtype[np.int32]]              # [pos, nf, seen, -, blocks, remainder]
    bhd = np.ndarray[(HD,), np.dtype[bfloat16]]
    brow = np.ndarray[(KVW,), np.dtype[bfloat16]]
    og_ty = np.ndarray[(OGH * HD,), np.dtype[bfloat16]]
    fcs = np.ndarray[(G.ROT,), np.dtype[np.float32]]
    fhd = np.ndarray[(HD,), np.dtype[np.float32]]
    fq = np.ndarray[(2 * QW,), np.dtype[bfloat16]]          # ATTN_VEXP: q pre-split, [hi | lo]
    fml = np.ndarray[(2 * G.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * HD,), np.dtype[np.float32]]
    i32 = np.int32
    inc = include_dirs() + [str(GEMV), str(ATTN), str(LN), str(LINL)]

    def af(sym, src, args, flags):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    f_gy, f_prep = _kernels(("gy", "prep"), elem, x_ty, y_ty, tab_ty, None, None)
    f_nr, f_nr32, f_lny, f_lnx = _ln_fns(u8_ln)
    f_meta = af("attn_meta", ATTN / "attn_meta.cc", [u8_a, u8_a, bhd, bhd, fcs, pb_ty], ATTN_FLAGS)
    f_q = af("attn_q", ATTN / "attn_q.cc", [u8_a, bhd, fcs, fq, i32], ATTN_FLAGS)
    f_k = af("attn_k", ATTN / "attn_k.cc", [u8_a, bhd, fcs, fhd, brow, i32], ATTN_FLAGS)
    f_v = af("attn_v", ATTN / "attn_v.cc", [u8_a, brow, i32], ATTN_FLAGS)
    f_init = af("attn_init", ATTN / "attn_init.cc", [foacc, fml], ATTN_FLAGS)
    f_step = af("attn_step", ATTN / "attn_step.cc", [u8_a, u8_a, fq, foacc, fml, pb_ty, i32], ATTN_FLAGS)
    f_stepb = af("attn_stepb", ATTN / "attn_stepb.cc", [u8_a] * (2 * G.RB) + [fq, foacc, fml, pb_ty, i32],
                 ATTN_FLAGS)
    f_stepn = af("attn_step_new", ATTN / "attn_step_new.cc", [brow, brow, fq, foacc, fml, i32], ATTN_FLAGS)
    f_fin = (af("attn_fin_ng", ATTN / "attn_fin_ng.cc", [foacc, fml, og_ty, i32], ATTN_FLAGS) if NOGATE else
             af("attn_fin", ATTN / "attn_fin.cc", [foacc, fml, u8_a, u8_a, og_ty, i32], ATTN_FLAGS))

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=max(G.XN_ELEMS, G.OG_ELEMS))
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_ain = ObjectFifo(u8_a, name="ain", depth=max(4, 2 * G.RB + 2))   # a block is acquired at once
    of_aout = ObjectFifo(brow, name="aout", depth=2)
    of_og = [ObjectFifo(og_ty, name=f"og{c}", depth=2) for c in range(ACORES)]





    workers = [Worker(_ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_nr32, f_lny, f_lnx], tile=Tile(0, 3),
                      stack_size=FR.STACK)]
    for c in range(N_CORES):
        workers.append(Worker(att_main, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                   Buffer(tab_ty, name=f"tab{c}"), f_gy, f_prep],
                              tile=Tile(c, 2), stack_size=FR.STACK))

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(bhd, name=f"qn{s}"), Buffer(bhd, name=f"kn{s}"), Buffer(fcs, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(fhd, name=f"tmp{s}"), Buffer(brow, name=f"kout{s}"),
                Buffer(brow, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}")]
    afns = [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb]
    workers.append(Worker(attn0, fn_args=[of_ain.cons(), of_aout.prod(), of_og[0].prod()] + abufs(0) + afns,
                          tile=Tile(2, 3), stack_size=FR.STACK))
    for c in range(1, ACORES):
        workers.append(Worker(make_attn(c), fn_args=[of_ain.cons(), of_og[c].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=FR.STACK))


    rt = Runtime(att_sequence, [T["pool_ty"], T["xres_ty"], T["consts_ty"], kv_ty, T["act_ty"], ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(1, 0)),
                            [of_og[c].cons(tile=Tile(2 + c, 0)) for c in range(ACORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# ---- one dispatch a layer: the mixer's program, then the MLP's, over the same fifos (LG_PART=convmlp | attmlp)
MLP_KN = ("gms", "gacc", "act", "copy", "prepf")     # the MLP's kernels beside the mixer's gy / prep


def fused_ln_body(ain, aout, f_nr, f_nr32, f_lny, f_lnx):
    _ln_body(ain, aout, f_nr, f_nr32, f_lny, f_lnx)       # entry norm, post_operator_norm, res / xm
    mlp_ln_body(ain, aout, f_nr32, f_lny)                 # post_ffn_norm, xres


def conv_mlp_main(win, xin, yout, tab, ms, acc, f_gy, f_prep, gms, gacc, act, copy, prepf):
    conv_main(win, xin, yout, tab, f_gy, f_prep)
    mlp_main(win, xin, yout, tab, ms, acc, gms, gacc, act, copy, f_prep, prepf)


def att_mlp_main(win, xin, yout, tab, ms, acc, f_gy, f_prep, gms, gacc, act, copy, prepf):
    att_main(win, xin, yout, tab, f_gy, f_prep)
    mlp_main(win, xin, yout, tab, ms, acc, gms, gacc, act, copy, f_prep, prepf)


def conv_mlp_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                      scin_p, scy_c, scn_c):
    conv_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss, scin_p, scy_c, scn_c)
    mlp_sequence(a_pool, c_xres, a_consts, a_act, lni, lno, w_prods, x_prod, y_conss)      # phase 1 is finished


def att_mlp_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c,
                     og_cs):
    att_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c, og_cs)
    mlp_sequence(a_pool, c_xres, a_consts, a_act, lni, lno, w_prods, x_prod, y_conss)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lg_convmlp(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, *, srchash: CompileTime[int] = 0):
    T = _types()
    elem, x_ty, y_ty, tab_ty, u8_ln = T["elem"], T["x_ty"], T["y_ty"], T["tab_ty"], T["u8_ln"]
    state_ty = np.ndarray[(L.STATE_BYTES,), np.dtype[np.uint8]]
    fsc = np.ndarray[(SCW,), np.dtype[np.float32]]              # one row of SCW channels: a tap, a state row, B, C, u
    bsc = np.ndarray[(SCW,), np.dtype[bfloat16]]                # y, bf16 (the out GEMV's activation)
    ms_ty = np.ndarray[(G.MS_FLOATS,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(G.DOWN_PC * BAND_ROWS,), np.dtype[np.float32]]
    f_gy, f_prep = _kernels(("gy", "prep"), elem, x_ty, y_ty, tab_ty, None, None)
    KF = _kernels(MLP_KN, elem, x_ty, y_ty, tab_ty, ms_ty, acc_ty)
    f_nr, f_nr32, f_lny, f_lnx = _ln_fns(u8_ln)
    wloc = np.ndarray[(TAPS * SCW,), np.dtype[np.float32]]       # the core-local taps, tap-major
    f_cp = ExternalFunction("lg_cp", source_file=str(HERE / "lg_cp.cc"), arg_types=[fsc, wloc, np.int32],
                            include_dirs=include_dirs(), compile_flags=OS)
    f_mac0 = ExternalFunction("lg_scmac0", source_file=str(HERE / "lg_scmac0.cc"), arg_types=[fsc, wloc, fsc],
                              include_dirs=include_dirs(), compile_flags=OS)
    f_mac = ExternalFunction("lg_scmac", source_file=str(HERE / "lg_scmac.cc"),
                             arg_types=[fsc, wloc, fsc, fsc, np.int32], include_dirs=include_dirs(), compile_flags=OS)
    f_fin = ExternalFunction("lg_scfin", source_file=str(HERE / "lg_scfin.cc"),
                             arg_types=[fsc, fsc, fsc, wloc, fsc, bsc, fsc], include_dirs=include_dirs(),
                             compile_flags=OS)

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_scin = ObjectFifo(fsc, name="scin", depth=4)             # a row at a time; B, C, u acquired together
    of_scy = ObjectFifo(bsc, name="scy", depth=2)
    of_scn = ObjectFifo(fsc, name="scn", depth=2)               # the new state rows, a row at a time


    def sc_body(scin, scy, scn, wl, acc, f_cp, f_mac0, f_mac, f_fin):
        """One token, element by element (a row at a time: a fifo has 16 buffer descriptors). Per element: the
        taps (copied out), the state rows (acc = sum s_k w_k, the window sliding out as it goes), then B, C, u
        in; the gated output (bf16) and the new state rows out."""
        for _ in range_(G.SC_ELEMS):
            for k in range(TAPS):
                t = scin.acquire(1)
                f_cp(t, wl, k)
                scin.release(1)
            s = scin.acquire(1)
            f_mac0(s, wl, acc)
            scin.release(1)
            for k in range(1, TAPS - 1):
                s = scin.acquire(1)
                n = scn.acquire(1)
                f_mac(s, wl, acc, n, k)
                scn.release(1)
                scin.release(1)
            e = scin.acquire(3)
            oy = scy.acquire(1)
            n = scn.acquire(1)
            f_fin(e[0], e[1], e[2], wl, acc, oy, n)
            scn.release(1)
            scy.release(1)
            scin.release(3)

    workers = [Worker(fused_ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_nr32, f_lny, f_lnx], tile=Tile(0, 3),
                      stack_size=FR.STACK)]
    for c in range(N_CORES):
        workers.append(Worker(conv_mlp_main, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                   Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                                                   Buffer(acc_ty, name=f"acc{c}"), f_gy, f_prep] + KF,
                              tile=Tile(c, 2), stack_size=FR.STACK))
    workers.append(Worker(sc_body, fn_args=[of_scin.cons(), of_scy.prod(), of_scn.prod(),
                                            Buffer(wloc, name="wloc"), Buffer(fsc, name="scacc"),
                                            f_cp, f_mac0, f_mac, f_fin],
                          tile=Tile(2, 3), stack_size=FR.STACK))


    # Shim channels: columns 0..2 already pair a producer with lni, x and the conv input (y on column 1's
    # second S2MM, the new state on column 2's).
    rt = Runtime(conv_mlp_sequence, [T["pool_ty"], T["xres_ty"], T["consts_ty"], state_ty, T["act_ty"],
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_scin.prod(tile=Tile(2, 0)),
                            of_scy.cons(tile=Tile(1, 0)), of_scn.cons(tile=Tile(2, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lg_attmlp(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, *, srchash: CompileTime[int] = 0):
    T = _types()
    elem, x_ty, y_ty, tab_ty, u8_ln = T["elem"], T["x_ty"], T["y_ty"], T["tab_ty"], T["u8_ln"]
    u8_a = np.ndarray[(E_A,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(L.KV_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(L.PTAB_BYTES,), np.dtype[np.uint8]]
    pb_ty = np.ndarray[(8,), np.dtype[np.int32]]              # [pos, nf, seen, -, blocks, remainder]
    bhd = np.ndarray[(HD,), np.dtype[bfloat16]]
    brow = np.ndarray[(KVW,), np.dtype[bfloat16]]
    og_ty = np.ndarray[(OGH * HD,), np.dtype[bfloat16]]
    fcs = np.ndarray[(G.ROT,), np.dtype[np.float32]]
    fhd = np.ndarray[(HD,), np.dtype[np.float32]]
    fq = np.ndarray[(2 * QW,), np.dtype[bfloat16]]          # ATTN_VEXP: q pre-split, [hi | lo]
    fml = np.ndarray[(2 * G.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * HD,), np.dtype[np.float32]]
    i32 = np.int32
    inc = include_dirs() + [str(GEMV), str(ATTN), str(LN), str(LINL)]

    def af(sym, src, args, flags):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    ms_ty = np.ndarray[(G.MS_FLOATS,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(G.DOWN_PC * BAND_ROWS,), np.dtype[np.float32]]
    f_gy, f_prep = _kernels(("gy", "prep"), elem, x_ty, y_ty, tab_ty, None, None)
    KF = _kernels(MLP_KN, elem, x_ty, y_ty, tab_ty, ms_ty, acc_ty)
    f_nr, f_nr32, f_lny, f_lnx = _ln_fns(u8_ln)
    f_meta = af("attn_meta", ATTN / "attn_meta.cc", [u8_a, u8_a, bhd, bhd, fcs, pb_ty], ATTN_FLAGS)
    f_q = af("attn_q", ATTN / "attn_q.cc", [u8_a, bhd, fcs, fq, i32], ATTN_FLAGS)
    f_k = af("attn_k", ATTN / "attn_k.cc", [u8_a, bhd, fcs, fhd, brow, i32], ATTN_FLAGS)
    f_v = af("attn_v", ATTN / "attn_v.cc", [u8_a, brow, i32], ATTN_FLAGS)
    f_init = af("attn_init", ATTN / "attn_init.cc", [foacc, fml], ATTN_FLAGS)
    f_step = af("attn_step", ATTN / "attn_step.cc", [u8_a, u8_a, fq, foacc, fml, pb_ty, i32], ATTN_FLAGS)
    f_stepb = af("attn_stepb", ATTN / "attn_stepb.cc", [u8_a] * (2 * G.RB) + [fq, foacc, fml, pb_ty, i32],
                 ATTN_FLAGS)
    f_stepn = af("attn_step_new", ATTN / "attn_step_new.cc", [brow, brow, fq, foacc, fml, i32], ATTN_FLAGS)
    f_fin = (af("attn_fin_ng", ATTN / "attn_fin_ng.cc", [foacc, fml, og_ty, i32], ATTN_FLAGS) if NOGATE else
             af("attn_fin", ATTN / "attn_fin.cc", [foacc, fml, u8_a, u8_a, og_ty, i32], ATTN_FLAGS))

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=max(2, G.XN_ELEMS, G.OG_ELEMS))
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_ain = ObjectFifo(u8_a, name="ain", depth=max(4, 2 * G.RB + 2))   # a block is acquired at once
    of_aout = ObjectFifo(brow, name="aout", depth=2)
    of_og = [ObjectFifo(og_ty, name=f"og{c}", depth=2) for c in range(ACORES)]


    def attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, h0, c,
             f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
        e = ain.acquire(2)                                     # [qn | kn], the position record
        f_meta(e[0], e[1], qn, kn, cs, pbuf)
        ain.release(2)
        for h in range_(G.Q_AIN_ELEMS):
            e = ain.acquire(1)
            f_q(e, qn, cs, qs, h)
            ain.release(1)
        for h in range_(G.K_AIN_ELEMS):
            e = ain.acquire(1)
            f_k(e, kn, cs, tmp, kout, h)
            ain.release(1)
        for h in range_(G.K_AIN_ELEMS):
            e = ain.acquire(1)
            f_v(e, vout, h)
            ain.release(1)
        if aout is not None:                                   # core 0 owns the cache row
            o = aout.acquire(1)
            for j in range_(KVW):
                o[j] = kout[j]
            aout.release(1)
            o = aout.acquire(1)
            for j in range_(KVW):
                o[j] = vout[j]
            aout.release(1)
        f_init(oacc, ml)
        for _ in range_(pbuf[4]):                              # whole blocks of RB cached rows
            e = ain.acquire(2 * G.RB)
            f_stepb(*[e[i] for i in range(2 * G.RB)], qs, oacc, ml, pbuf, h0)
            ain.release(2 * G.RB)
        for _ in range_(pbuf[5]):                              # what did not fill a block
            e = ain.acquire(2)
            f_step(e[0], e[1], qs, oacc, ml, pbuf, h0)
            ain.release(2)
        f_stepn(kout, vout, qs, oacc, ml, h0)
        # The gate arrives for EVERY og element on the broadcast stream (two elements an og element, in head
        # order): each core consumes all of them and finishes only its own N_OG (layer_x/xlayer.py's law).
        if NOGATE:
            for hp in range_(N_OG):
                o = ogout.acquire(1)
                f_fin(oacc, ml, o, hp)
                ogout.release(1)
            return
        for _ in range(c * N_OG):
            ain.acquire(2)
            ain.release(2)
        for hp in range_(N_OG):
            g = ain.acquire(2)
            o = ogout.acquire(1)
            f_fin(oacc, ml, g[0], g[1], o, hp)
            ogout.release(1)
            ain.release(2)
        for _ in range((ACORES - 1 - c) * N_OG):
            ain.acquire(2)
            ain.release(2)

    def attn0(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf,
              f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
        attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, 0, 0,
             f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb)

    def make_attn(c):
        def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf,
                 f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
            attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, c * NHL, c,
                 f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb)
        return body

    workers = [Worker(fused_ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_nr32, f_lny, f_lnx], tile=Tile(0, 3),
                      stack_size=FR.STACK)]
    for c in range(N_CORES):
        workers.append(Worker(att_mlp_main, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                   Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                                                   Buffer(acc_ty, name=f"acc{c}"), f_gy, f_prep] + KF,
                              tile=Tile(c, 2), stack_size=FR.STACK))

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(bhd, name=f"qn{s}"), Buffer(bhd, name=f"kn{s}"), Buffer(fcs, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(fhd, name=f"tmp{s}"), Buffer(brow, name=f"kout{s}"),
                Buffer(brow, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}")]
    afns = [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb]
    workers.append(Worker(attn0, fn_args=[of_ain.cons(), of_aout.prod(), of_og[0].prod()] + abufs(0) + afns,
                          tile=Tile(2, 3), stack_size=FR.STACK))
    for c in range(1, ACORES):
        workers.append(Worker(make_attn(c), fn_args=[of_ain.cons(), of_og[c].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=FR.STACK))


    rt = Runtime(att_mlp_sequence, [T["pool_ty"], T["xres_ty"], T["consts_ty"], kv_ty, T["act_ty"], ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(1, 0)),
                            [of_og[c].cons(tile=Tile(2 + c, 0)) for c in range(ACORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# ---- one image for both mixers (LG_PART=shared, LG_STREAM=conv | att): every core's program is the same in the two
# builds, only the instruction stream differs, so conv and attention layers never change hardware context. The main
# cores run the conv's band counts (12 then 4 a core); the attention stream pads its first GEMV phase with SHARED_PAD
# scratch bands (q 4 + gate 4 + k 1 + v 1 + 2). The conv core sits at Tile(6, 3) (Tile(2, 3) is attention core 0) on
# shim columns 6 / 7, which only the main cores' w / y use. Idle cores wait on their first fifo until a stream feeds
# them (IRON workers loop until reconfiguration).
STREAM = os.environ.get("LG_STREAM", "conv")
SHARED_PAD = 3 * G.SC_PC - (2 * G.Q_PC + 2 * G.KV_PC)
SHARED_ST_BYTES = max(L.KV_BYTES, L.STATE_BYTES)     # one signature: the state / kv argument typed at the larger


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lg_shared(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, ptab: In, *, srchash: CompileTime[int] = 0):
    assert PART == "shared" and STREAM in ("conv", "att")
    assert SHARED_PAD >= 0 and G.O_PC == G.SC_PC and QW == HID and G.OG_ELEMS == G.XN_ELEMS, \
        "lg_shared: the attention layer's GEMV phases must pad to the conv's"
    T = _types()
    elem, x_ty, y_ty, tab_ty, u8_ln = T["elem"], T["x_ty"], T["y_ty"], T["tab_ty"], T["u8_ln"]
    st_ty = np.ndarray[(SHARED_ST_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(L.PTAB_BYTES,), np.dtype[np.uint8]]
    ms_ty = np.ndarray[(G.MS_FLOATS,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(G.DOWN_PC * BAND_ROWS,), np.dtype[np.float32]]
    f_gy, f_prep = _kernels(("gy", "prep"), elem, x_ty, y_ty, tab_ty, None, None)
    KF = _kernels(MLP_KN, elem, x_ty, y_ty, tab_ty, ms_ty, acc_ty)
    f_nr, f_nr32, f_lny, f_lnx = _ln_fns(u8_ln)
    inc = include_dirs()
    # the conv core
    fsc = np.ndarray[(SCW,), np.dtype[np.float32]]
    bsc = np.ndarray[(SCW,), np.dtype[bfloat16]]
    wloc = np.ndarray[(TAPS * SCW,), np.dtype[np.float32]]
    f_cp = ExternalFunction("lg_cp", source_file=str(HERE / "lg_cp.cc"), arg_types=[fsc, wloc, np.int32],
                            include_dirs=inc, compile_flags=OS)
    f_mac0 = ExternalFunction("lg_scmac0", source_file=str(HERE / "lg_scmac0.cc"), arg_types=[fsc, wloc, fsc],
                              include_dirs=inc, compile_flags=OS)
    f_mac = ExternalFunction("lg_scmac", source_file=str(HERE / "lg_scmac.cc"),
                             arg_types=[fsc, wloc, fsc, fsc, np.int32], include_dirs=inc, compile_flags=OS)
    f_scf = ExternalFunction("lg_scfin", source_file=str(HERE / "lg_scfin.cc"),
                             arg_types=[fsc, fsc, fsc, wloc, fsc, bsc, fsc], include_dirs=inc, compile_flags=OS)
    # the attention cores
    u8_a = np.ndarray[(E_A,), np.dtype[np.uint8]]
    pb_ty = np.ndarray[(8,), np.dtype[np.int32]]
    bhd = np.ndarray[(HD,), np.dtype[bfloat16]]
    brow = np.ndarray[(KVW,), np.dtype[bfloat16]]
    og_ty = np.ndarray[(OGH * HD,), np.dtype[bfloat16]]
    fcs = np.ndarray[(G.ROT,), np.dtype[np.float32]]
    fhd = np.ndarray[(HD,), np.dtype[np.float32]]
    fq = np.ndarray[(2 * QW,), np.dtype[bfloat16]]
    fml = np.ndarray[(2 * G.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * HD,), np.dtype[np.float32]]
    i32 = np.int32
    ainc = include_dirs() + [str(GEMV), str(ATTN), str(LN), str(LINL)]

    def af(sym, args):
        return ExternalFunction(sym, source_file=str(ATTN / f"{sym}.cc"), arg_types=args, include_dirs=ainc,
                                compile_flags=ATTN_FLAGS)
    afns = [af("attn_meta", [u8_a, u8_a, bhd, bhd, fcs, pb_ty]), af("attn_q", [u8_a, bhd, fcs, fq, i32]),
            af("attn_k", [u8_a, bhd, fcs, fhd, brow, i32]), af("attn_v", [u8_a, brow, i32]),
            af("attn_init", [foacc, fml]), af("attn_step", [u8_a, u8_a, fq, foacc, fml, pb_ty, i32]),
            af("attn_step_new", [brow, brow, fq, foacc, fml, i32]),
            af("attn_fin", [foacc, fml, u8_a, u8_a, og_ty, i32]),
            af("attn_stepb", [u8_a] * (2 * G.RB) + [fq, foacc, fml, pb_ty, i32])]

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_scin = ObjectFifo(fsc, name="scin", depth=4)
    of_scy = ObjectFifo(bsc, name="scy", depth=2)
    of_scn = ObjectFifo(fsc, name="scn", depth=2)
    of_ain = ObjectFifo(u8_a, name="ain", depth=max(4, 2 * G.RB + 2))
    of_aout = ObjectFifo(brow, name="aout", depth=2)
    of_og = [ObjectFifo(og_ty, name=f"og{c}", depth=2) for c in range(ACORES)]

    workers = [Worker(fused_ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_nr32, f_lny, f_lnx],
                      tile=Tile(0, 3), stack_size=FR.STACK)]
    for c in range(N_CORES):
        workers.append(Worker(conv_mlp_main, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                       Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                                                       Buffer(acc_ty, name=f"acc{c}"), f_gy, f_prep] + KF,
                              tile=Tile(c, 2), stack_size=FR.STACK))
    workers.append(Worker(sc_body, fn_args=[of_scin.cons(), of_scy.prod(), of_scn.prod(),
                                            Buffer(wloc, name="wloc"), Buffer(fsc, name="scacc"),
                                            f_cp, f_mac0, f_mac, f_scf],
                          tile=Tile(6, 3), stack_size=FR.STACK))

    def abufs(c):
        s_ = "" if c == 0 else str(c)
        return [Buffer(bhd, name=f"qn{s_}"), Buffer(bhd, name=f"kn{s_}"), Buffer(fcs, name=f"cs{s_}"),
                Buffer(fq, name=f"qs{s_}"), Buffer(fhd, name=f"tmp{s_}"), Buffer(brow, name=f"kout{s_}"),
                Buffer(brow, name=f"vout{s_}"), Buffer(foacc, name=f"oacc{s_}"), Buffer(fml, name=f"ml{s_}"),
                Buffer(pb_ty, name=f"pb{s_}")]
    workers.append(Worker(attn0, fn_args=[of_ain.cons(), of_aout.prod(), of_og[0].prod()] + abufs(0) + afns,
                          tile=Tile(2, 3), stack_size=FR.STACK))
    for c in range(1, ACORES):
        workers.append(Worker(make_attn(c), fn_args=[of_ain.cons(), of_og[c].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=FR.STACK))

    def sequence(a_pool, c_xres, a_consts, a_st, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                 scin_p, scy_c, scn_c, ain_p, aout_c, og_cs):
        if STREAM == "conv":
            conv_sequence(a_pool, c_xres, a_consts, a_st, a_act, lni, lno, w_prods, x_prod, y_conss,
                          scin_p, scy_c, scn_c, st_bytes=SHARED_ST_BYTES)
        else:
            att_sequence(a_pool, c_xres, a_consts, a_st, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                         ain_p, aout_c, og_cs, pad=SHARED_PAD, kv_bytes=SHARED_ST_BYTES)
        mlp_sequence(a_pool, c_xres, a_consts, a_act, lni, lno, w_prods, x_prod, y_conss)

    rt = Runtime(sequence, [T["pool_ty"], T["xres_ty"], T["consts_ty"], st_ty, T["act_ty"], ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_scin.prod(tile=Tile(6, 0)), of_scy.cons(tile=Tile(6, 0)), of_scn.cons(tile=Tile(7, 0)),
                            of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(1, 0)),
                            [of_og[c].cons(tile=Tile(2 + c, 0)) for c in range(ACORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lg_mlp(pool: In, xres: InOut, consts: In, act: InOut, *, srchash: CompileTime[int] = 0):
    T = _types()
    elem, x_ty, y_ty, tab_ty, u8_ln = T["elem"], T["x_ty"], T["y_ty"], T["tab_ty"], T["u8_ln"]
    ms_ty = np.ndarray[(G.MS_FLOATS,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(G.DOWN_PC * BAND_ROWS,), np.dtype[np.float32]]
    KN = ("gms", "gacc", "act", "copy", "prep", "prepf")
    KF = _kernels(KN, elem, x_ty, y_ty, tab_ty, ms_ty, acc_ty)
    f_nr32, f_lny = _ln_fns(u8_ln, ("nr32", "y"))

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=max(2, G.XM_ELEMS))
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)



    workers = [Worker(mlp_ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr32, f_lny], tile=Tile(0, 3),
                      stack_size=FR.STACK)]
    for c in range(N_CORES):
        workers.append(Worker(mlp_main, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                   Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                                                   Buffer(acc_ty, name=f"acc{c}")] + KF,
                              tile=Tile(c, 2), stack_size=FR.STACK))


    rt = Runtime(mlp_sequence, [T["pool_ty"], T["xres_ty"], T["consts_ty"], T["act_ty"],
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = {"conv": lg_conv, "att": lg_att, "mlp": lg_mlp, "convmlp": lg_convmlp, "attmlp": lg_attmlp, "shared": lg_shared}[PART]
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + [bytes([NOGATE]), (HERE.parent.parent / "recipes" / "lfm3.py").read_bytes(), (LN / "ln.h").read_bytes(),
                   (LN / "ln_y.cc").read_bytes(), (LN / "ln_xn.cc").read_bytes(), (LN / "ln_nr32.cc").read_bytes(),
                   (LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode(), repr(PROBES).encode(), PART.encode()]
                + ([STREAM.encode()] if PART == "shared" else []))
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
