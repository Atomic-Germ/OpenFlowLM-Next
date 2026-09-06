r"""dx: a whole Qwen3 dense layer in ONE xclbin context and ONE instruction stream:

    ln -> gemv q | k | v -> attention (q/k RMSNorm, full RoPE, no gate) -> gemv o
       -> ln (+residual) -> gemv up | gate (per band) -> silu(gate) * up -> gemv down -> +residual

The same fabric as the MoE designs (layer_x): 8 main cores (Tile(c, 2)) fed by
w (10 KB weight elements), x (4 KB broadcast) and y (256 B results) streams;
the ln helper (Tile(0, 3)); the attention core (Tile(2, 3)). No routing read,
so the whole layer is one dispatch. Geometry from recipes/qwen3.py for the
spec named by OPEN_KERNELS_SPEC.

Activations that are not a whole number of 4 KB elements (xn: 2560 bf16 =
1.25 elements; h: 9728 f32 = 9.5 elements) are streamed as whole elements
(junk past the end) and prepared into the GEMV table by element index
(dense_prep / dense_prep_f32 derive the block range).

Args: pool (q k v o up gate down at their offsets), xres f32[HID] (in: the
layer input; out: the layer output), consts [lnw | postln | qn kn], kv (the
layer's KV cache, rows of [K_t | V_t]), act (scratch), ptab (position
records). The KV window / new-row / record words are patched per token by the
driver's attnpos (the stream is built for the placeholder position 1).
Build (WSL): OPEN_KERNELS_SPEC=<qwen3 spec> python build_design.py designs/dense/dx.py designs/dense/build_h2560
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
LX = HERE.parent / "layer_x"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes import dense as QR  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM, band_bytes  # noqa: E402
from aie.helpers.taplib import TensorAccessPattern  # noqa: E402

SPEC = current_spec()
R = QR.recipe(SPEC)
L, G = R.layout, R.geo
HID, FF, N_CORES = G.HID, G.FF, G.N_CORES
QW, KVW = G.QW, G.KVW
ELN, E_A = L.ELN, L.E_A
CALL_BYTES = G.CALL_BYTES
OS = ["-Os"]
STOP = int(os.environ.get("DX_STOP", 99))     # debug: 1 = after q/k/v, 2 = after attention, 3 = after the o proj + norm
assert not G.GATE, "dx.py: the Qwen3 dense recipe has no attention gate"


def per_band(K):
    return band_bytes(K) // 5120


def n_groups(K):
    return band_bytes(K) // CALL_BYTES


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


ATTN_FLAGS = [f"-DATTN_NH={G.NH}", f"-DATTN_KVH={G.KVH}", f"-DATTN_HD={G.HD}", f"-DATTN_ROT={G.ROT}", "-DATTN_GATE=0",
              f"-DATTN_QKNORM={1 if G.QKNORM else 0}", f"-DATTN_QKNORM_POST={1 if G.QKNORM_POST else 0}",
              f"-DATTN_EPS={G.EPS:g}f", f"-DATTN_VEXP={G.VEXP}"]
LN_FLAGS = [f"-DLN_N={HID}", f"-DLN_EPS={G.EPS:g}f"]


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dx(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, *, stop: CompileTime[int] = 99,
       srchash: CompileTime[int] = 0):
    elem = np.ndarray[(CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(G.TAB_BYTES,), np.dtype[np.uint8]]
    ms_ty = np.ndarray[(G.MS_FLOATS,), np.dtype[np.float32]]
    u8_ln = np.ndarray[(ELN,), np.dtype[np.uint8]]
    u8_a = np.ndarray[(E_A,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(L.CD_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(L.KV_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(L.PTAB_BYTES,), np.dtype[np.uint8]]
    i32_4 = np.ndarray[(4,), np.dtype[np.int32]]
    bhd = np.ndarray[(G.HD,), np.dtype[bfloat16]]
    brow = np.ndarray[(KVW,), np.dtype[bfloat16]]
    fcs = np.ndarray[(G.ROT,), np.dtype[np.float32]]
    fhd = np.ndarray[(G.HD,), np.dtype[np.float32]]
    fq = np.ndarray[(QW,), np.dtype[np.float32]]
    fml = np.ndarray[(2 * G.MLS,), np.dtype[np.float32]]   # [m | l], stride MLS (padded when ATTN_VEXP)
    i32 = np.int32

    inc = include_dirs() + [str(GEMV), str(ATTN), str(LN), str(LINL)]

    def ef(sym, src, args, flags=OS):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    f_gy = ef("gemv_q4_gy", HERE / "gemv_q4_gy.cc", [elem, tab_ty, y_ty, i32, i32, i32])
    f_gms = ef("gemv_q4_gms", HERE / "gemv_q4_gms.cc", [elem, tab_ty, ms_ty, i32, i32, i32])
    f_silu = ef("dense_act", HERE / "dense_act.cc", [ms_ty, y_ty])
    f_prep = ef("dense_prep", HERE / "dense_prep.cc", [x_ty, tab_ty, i32, i32])
    f_prepf = ef("dense_prep_f32", HERE / "dense_prep_f32.cc", [x_ty, tab_ty, i32, i32])
    f_nr = ef("ln_nr", LINL / "ln_nr.cc", [u8_ln] * 4, LN_FLAGS)
    f_lny = ef("ln_y", LN / "ln_y.cc", [u8_ln] * 5 + [i32], LN_FLAGS)
    f_lnx = ef("ln_xn", LN / "ln_xn.cc", [u8_ln] * 6, LN_FLAGS)
    f_nr32 = ef("ln_nr32", LN / "ln_nr32.cc", [u8_ln] * 4 + [i32], LN_FLAGS) if G.SANDWICH else None
    f_meta = ef("attn_meta", ATTN / "attn_meta.cc", [u8_a, u8_a, bhd, bhd, fcs, i32_4], ATTN_FLAGS)
    f_q = ef("attn_q", ATTN / "attn_q.cc", [u8_a, bhd, fcs, fq, i32], ATTN_FLAGS)
    f_k = ef("attn_k", ATTN / "attn_k.cc", [u8_a, bhd, fcs, fhd, brow, i32], ATTN_FLAGS)
    f_v = ef("attn_v", ATTN / "attn_v.cc", [u8_a, brow, i32], ATTN_FLAGS)
    f_init = ef("attn_init", ATTN / "attn_init.cc", [fq, fml], ATTN_FLAGS)
    f_step = ef("attn_step", ATTN / "attn_step.cc", [u8_a, u8_a, fq, fq, fml, i32_4], ATTN_FLAGS)
    f_stepn = ef("attn_step_new", ATTN / "attn_step_new.cc", [brow, brow, fq, fq, fml], ATTN_FLAGS)
    f_fin = ef("attn_fin_ng", ATTN / "attn_fin_ng.cc", [fq, fml, brow, i32], ATTN_FLAGS)

    # ---- fifos
    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)      # one output element at a time (8 KB elements at 4096 wide)
    of_ain = ObjectFifo(u8_a, name="ain", depth=4)
    of_aout = ObjectFifo(brow, name="aout", depth=2)

    PB_H, NG_H = per_band(HID), n_groups(HID)
    PB_Q, NG_Q = per_band(QW), n_groups(QW)
    PB_F, NG_F = per_band(FF), n_groups(FF)

    def gemv_bands(win, yout, tab, f_gy, nbands, ngroups, pb):
        for _ in range_(nbands):
            ye = yout.acquire(1)
            for g in range_(ngroups):
                we = win.acquire(1)
                f_gy(we, tab, ye, g, pb, 2)
                win.release(1)
            yout.release(1)

    def acq(fifo, n):
        e = fifo.acquire(n)
        return [e] if n == 1 else e            # acquire(1) yields the element itself

    def main_body(win, xin, yout, tab, ms, f_gy, f_gms, f_silu, f_prep, f_prepf):
        # q | k | v against xn (K = HID; XN_ELEMS elements)
        xe = acq(xin, G.XN_ELEMS)
        for i in range(G.XN_ELEMS):
            f_prep(xe[i], tab, HID, i)
        gemv_bands(win, yout, tab, f_gy, G.Q_PC + 2 * G.KV_PC, NG_H, PB_H)
        xin.release(G.XN_ELEMS)
        if stop == 1:
            return
        # o against og (K = QW)
        oe = acq(xin, G.OG_ELEMS)
        for i in range(G.OG_ELEMS):
            f_prep(oe[i], tab, QW, i)
        gemv_bands(win, yout, tab, f_gy, G.O_PC, NG_Q, PB_Q)
        xin.release(G.OG_ELEMS)
        if stop == 2:
            return
        # up | gate per band against xm (K = HID), silu -> h band
        me = acq(xin, G.XM_ELEMS)
        for i in range(G.XM_ELEMS):
            f_prep(me[i], tab, HID, i)
        for _ in range_(G.UP_PC):
            for g in range_(NG_H):
                we = win.acquire(1)
                f_gms(we, tab, ms, g, PB_H, G.MS_U)
                win.release(1)
            for g in range_(NG_H):
                we = win.acquire(1)
                f_gms(we, tab, ms, g, PB_H, G.MS_G)
                win.release(1)
            ye = yout.acquire(1)
            f_silu(ms, ye)
            yout.release(1)
        xin.release(G.XM_ELEMS)
        if stop == 3:
            return
        # down against h (K = FF; H_ELEMS f32 elements)
        for i in range_(G.H_ELEMS):
            he = xin.acquire(1)
            f_prepf(he, tab, FF, i)
            xin.release(1)
        gemv_bands(win, yout, tab, f_gy, G.DOWN_PC, NG_F, PB_F)

    def ln_body(ain, aout, f_nr, f_lny, f_lnx, *rest):
        # 1. the layer-entry norm: [x0 x1 lnw] -> [xn]
        e = ain.acquire(3)
        o = aout.acquire(1)
        f_nr(e[0], e[1], e[2], o)
        aout.release(1)
        ain.release(3)

        def post_norm(f_nr32):
            # a sandwich norm on a block output: [o0 o1 w] -> [t0] [t1] (fp32 halves)
            e = ain.acquire(3)
            for i in range(2):
                o = aout.acquire(1)
                f_nr32(e[0], e[1], e[2], o, i)
                aout.release(1)
            ain.release(3)

        def add_norm(with_xn):
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

        if G.SANDWICH:
            f_nr32 = rest[0]
            post_norm(f_nr32)                  # 2a. t = post_attn_norm(out)
            add_norm(True)                     # 2b. res = x + t; xm = pre_ffn_norm(res)
            if stop >= 3:
                post_norm(f_nr32)              # 3a. t2 = post_ffn_norm(out2)
                add_norm(False)                # 3b. xres = res + t2
        else:
            add_norm(True)                     # 2. res = x + out; xm = post_attn_norm(res)
            if stop >= 3:
                add_norm(True)                 # 3. xres = res + out2 (the xn is junk)

    def attn_body(ain, aout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin):
        e = ain.acquire(2)                                      # [qn | kn], the position record
        f_meta(e[0], e[1], qn, kn, cs, pb)
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
        o = aout.acquire(1)
        for j in range_(KVW):
            o[j] = kout[j]
        aout.release(1)
        o = aout.acquire(1)
        for j in range_(KVW):
            o[j] = vout[j]
        aout.release(1)
        f_init(oacc, ml)
        for _ in range_(pb[1]):                                 # nf cached rows (K_t, V_t)
            e = ain.acquire(2)
            f_step(e[0], e[1], qs, oacc, ml, pb)
            ain.release(2)
        f_stepn(kout, vout, qs, oacc, ml)
        for hp in range_(G.OG_AOUT_ELEMS):
            o = aout.acquire(1)
            f_fin(oacc, ml, o, hp)
            aout.release(1)

    workers = [Worker(ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_lny, f_lnx] + ([f_nr32] if G.SANDWICH else []),
                      tile=Tile(0, 3), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(main_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                  Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                                                  f_gy, f_gms, f_silu, f_prep, f_prepf],
                              tile=Tile(c, 2), stack_size=0x1800))
    workers.append(Worker(attn_body,
                          fn_args=[of_ain.cons(), of_aout.prod(),
                                   Buffer(bhd, name="qn"), Buffer(bhd, name="kn"), Buffer(fcs, name="cs"),
                                   Buffer(fq, name="qs"), Buffer(fhd, name="tmp"), Buffer(brow, name="kout"),
                                   Buffer(brow, name="vout"), Buffer(fq, name="oacc"), Buffer(fml, name="ml"),
                                   Buffer(i32_4, name="pb"), f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin],
                          tile=Tile(2, 3), stack_size=0x1800))

    BB_H, BB_Q, BB_F = band_bytes(HID), band_bytes(QW), band_bytes(FF)
    YB = BAND_ROWS * 4

    def sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c):
        # 1. layer-entry norm: xn -> act
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_LNW, ELN), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XN, ELN), wait=True, group=tg_ln)
        # 2. q | k | v: weights now, xn after the norm
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_Q + c * G.Q_PC * BB_H, G.Q_PC * BB_H))
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_K + c * G.KV_PC * BB_H, G.KV_PC * BB_H))
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_V + c * G.KV_PC * BB_H, G.KV_PC * BB_H))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_Q + c * G.Q_PC * YB, G.Q_PC * YB))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_KVN + c * G.KV_PC * YB, G.KV_PC * YB))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_KVN + KVW * 4 + c * G.KV_PC * YB, G.KV_PC * YB))
        tg_ln.finish()                                            # xn is in DDR
        tg_x = TaskGroup()
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_XN, G.XN_ELEMS * ELEM), wait=True, group=tg_x)
        if stop == 1:
            py.finish()
            pw.finish()
            tg_x.finish()
            return
        # 3. attention: meta + record now, q / k / v after the GEMVs, the window, the new row out
        pa_out, pa_in = Pipeline(3), Pipeline(3)
        pa_out.drain(aout_c, a_kv, bt(L.KV_BYTES, L.KV_ROW, L.KV_ROW))          # [k' | v'] -> row pos (attnpos)
        pa_out.drain(aout_c, a_act, bt(L.AD_BYTES, L.AD_OG, QW * 2))
        pa_in.fill(ain_p, a_consts, bt(L.CD_BYTES, L.CD_META, E_A))            # [qn | kn]
        pa_in.fill(ain_p, a_ptab, bt(L.PTAB_BYTES, L.PTAB_ROW, L.PTAB_ROW))    # the position record (attnpos)
        py.finish(*y_conss)                                       # q, k, v are in DDR
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_Q, QW * 4))
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN, KVW * 4))
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN + KVW * 4, KVW * 4))
        pa_in.fill(ain_p, a_kv, bt(L.KV_BYTES, 0, L.KV_ROW))                   # the window: rows [0, nf) (attnpos)
        # 4. o projection against og
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_O + c * G.O_PC * BB_Q, G.O_PC * BB_Q))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_OUT + c * G.O_PC * YB, G.O_PC * YB))
        pa_out.finish()                                           # og (and the new cache row) are in DDR
        if stop == 2:
            pa_in.finish()
            py.finish()
            pw.finish()
            tg_x.finish()
            return
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OG, G.OG_ELEMS * ELEM), wait=True, group=tg_x)
        # 5. residual + norms -> res, xm (sandwich: t = post_attn_norm(out) first, then res = x + t,
        #    xm = pre_ffn_norm(res); plain: res = x + out, xm = post_attn_norm(res))
        tg_ln2 = TaskGroup()
        if G.SANDWICH:
            py.finish()                                           # out is in DDR
            tg_t = TaskGroup()
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT, HID * 4), wait=True, group=tg_t)
            lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTLN, ELN), wait=True, group=tg_t)
            lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_T, HID * 4), wait=True, group=tg_t)
            tg_t.finish()                                         # t is in DDR
            lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
            lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_PREFFN, ELN), wait=True, group=tg_ln2)
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_T, HID * 4), wait=True, group=tg_ln2)
            lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_RES, HID * 4), wait=True, group=tg_ln2)
            lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XM, ELN), wait=True, group=tg_ln2)
        else:
            lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
            lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTLN, ELN), wait=True, group=tg_ln2)
            lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_RES, HID * 4), wait=True, group=tg_ln2)
            lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XM, ELN), wait=True, group=tg_ln2)
            py.finish()                                           # out is in DDR
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT, HID * 4), wait=True, group=tg_ln2)
        tg_ln2.finish()                                           # res, xm are in DDR
        # 6. up | gate per band, silu -> h
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_XM, G.XM_ELEMS * ELEM), wait=True, group=tg_x)
        # The drains first (a core blocks on a full y fifo after two bands), then the per-band
        # fills interleaved across the cores so every core streams while the host paces the
        # 38 fills per core three at a time (Pipeline).
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_H + c * G.UP_PC * YB, G.UP_PC * YB))
        for j in range(G.UP_PC):
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_UP + (c * G.UP_PC + j) * BB_H, BB_H))
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_GATE + (c * G.UP_PC + j) * BB_H, BB_H))
        py.finish()                                               # h is in DDR
        if stop == 3:
            pw.finish()
            pa_in.finish()
            tg_x.finish()
            return
        # 7. down against h, then the output residual -> xres
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_H, G.H_ELEMS * ELEM), wait=True, group=tg_x)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_DOWN + c * G.DOWN_PC * BB_F, G.DOWN_PC * BB_F))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_OUT2 + c * G.DOWN_PC * YB, G.DOWN_PC * YB))
        tg_ln3 = TaskGroup()
        if G.SANDWICH:
            py.finish()                                           # out2 is in DDR
            tg_t2 = TaskGroup()
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT2, HID * 4), wait=True, group=tg_t2)
            lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTFFN, ELN), wait=True, group=tg_t2)
            lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_T2, HID * 4), wait=True, group=tg_t2)
            tg_t2.finish()                                        # t2 is in DDR
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_RES, HID * 4), wait=True, group=tg_ln3)
            lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTFFN, ELN), wait=True, group=tg_ln3)   # unused w
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_T2, HID * 4), wait=True, group=tg_ln3)
            lno.drain(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln3)
        else:
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_RES, HID * 4), wait=True, group=tg_ln3)
            lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTLN, ELN), wait=True, group=tg_ln3)
            lno.drain(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln3)
            lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_JUNK, ELN), wait=True, group=tg_ln3)
            py.finish()                                           # out2 is in DDR
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT2, HID * 4), wait=True, group=tg_ln3)
        tg_ln3.finish()
        pw.finish()
        pa_in.finish()
        tg_x.finish()

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, kv_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(1, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = dx
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + sorted(f.read_bytes() for f in (HERE.parent.parent / "recipes").glob("*.py"))
                + [(LN / "ln.h").read_bytes(), (LN / "ln_y.cc").read_bytes(), (LN / "ln_xn.cc").read_bytes(), (LN / "ln_nr32.cc").read_bytes(), (LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"stop": STOP, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
