r"""cx: a whole LFM2 short-conv layer in ONE xclbin context and ONE instruction stream.

    ln -> gemv B | C | u -> short conv (helper core) -> gemv out
       -> ln (+residual) -> gemv up | gate -> silu(gate) * up -> gemv down -> +residual

The same fabric as designs/dense: 8 main cores (Tile(c, 2)) on the w / x / y streams and
the ln helper (Tile(0, 3)). What differs is the one helper core at Tile(2, 3), which runs
sc.h instead of attn.h, and what that core needs: no position table, no KV cache, no
rotation. A conv layer takes five buffers where an attention layer takes six.

The everything-after-the-block half -- residual, second norm, up | gate, silu, down, output
residual -- is the dense layer's, stage for stage, because an LFM2 FFN is the dense one.

The fused input projection arrives as three separate pool regions, B, C and u, because
recipes/lfm2.py splits it there with three std_perm ops at different source chunk offsets.
The core sees three ordinary [hidden, hidden] projections and never learns it was one
tensor.

State is (taps - 1) x hidden f32 in the `state` buffer: the two earlier tokens the third
tap still reaches. The core reads both rows and writes both back, so nothing outside it has
to know the window's parity.

Build (WSL):
    OPEN_KERNELS_SPEC=<lfm2 spec> OPEN_KERNELS_UNVALIDATED=1 \
        python build_design.py designs/short_conv/cx.py designs/short_conv/build_h2048_t3
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
LN = HERE.parent / "ln"
LINL = HERE.parent / "lin_layer"
DENSE = HERE.parent / "dense"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes import lfm2 as QR  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM, band_bytes  # noqa: E402
from aie.helpers.taplib import TensorAccessPattern  # noqa: E402

SPEC = current_spec()
R = QR.recipe(SPEC)
L, G = R.layout, R.geo
DL = L.dense
HID, FF, N_CORES = G.HID, G.FF, G.N_CORES
ELN = DL.ELN
TAPS = QR.TAPS
SCW = 512                       # channels in one conv-core element (2 KB of f32).
                                # The core holds B, C, u, two state rows, three taps and
                                # the output at this width; 512 keeps that near 24 KB of
                                # L1 against the ~60 KB a core has. The conv is not
                                # compute, so more iterations cost nothing that matters.
SC_ELEMS = HID // SCW           # iterations the core runs per token
SC_PC = HID // (BAND_ROWS * N_CORES)     # bands per core for a [hidden, hidden] projection

OS = ["-O2"]
GEMV_OS = ["-O2"]
LN_FLAGS = [f"-DLN_N={HID}", f"-DLN_EPS={G.EPS:g}f"]
SC_FLAGS = [f"-DSC_TAPS={TAPS}", f"-DSC_W={SCW}"]
STOP = int(os.environ.get("CX_STOP", 99))   # debug: 1 = after B|C|u, 2 = after the out proj, 3 = after up|gate


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def per_band(K):
    return band_bytes(K) // 5120 // G.PER_CALL


def n_groups(K):
    return band_bytes(K) // (G.PER_CALL * 5120)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def cx(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, *,
       stop: CompileTime[int] = 99, srchash: CompileTime[int] = 0):
    elem = np.ndarray[(G.CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(G.TAB_BYTES,), np.dtype[np.uint8]]
    ms_ty = np.ndarray[(G.MS_FLOATS,), np.dtype[np.float32]]
    u8_ln = np.ndarray[(ELN,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(L.CD_BYTES,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(L.STATE_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]]
    # the conv core's elements: f32 channel slices, and one tap's channels as bf16
    fsc = np.ndarray[(SCW,), np.dtype[np.float32]]
    bsc = np.ndarray[(SCW,), np.dtype[bfloat16]]
    i32 = np.int32

    inc = include_dirs() + [str(GEMV), str(LN), str(LINL), str(HERE)]

    def ef(sym, src, args, flags=OS):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    f_gy = ef("gemv_q4_gy", DENSE / "gemv_q4_gy.cc", [elem, tab_ty, y_ty, i32, i32, i32])
    f_gms = ef("gemv_q4_gms", DENSE / "gemv_q4_gms.cc", [elem, tab_ty, ms_ty, i32, i32, i32])
    f_act = ef("dense_act", DENSE / "dense_act.cc", [ms_ty, y_ty])
    f_prep = ef("dense_prep", DENSE / "dense_prep.cc", [x_ty, tab_ty, i32, i32])
    f_prepf = ef("dense_prep_f32", DENSE / "dense_prep_f32.cc", [x_ty, tab_ty, i32, i32])
    f_nr = ef("ln_nr", LINL / "ln_nr.cc", [u8_ln] * 4, LN_FLAGS)
    f_lny = ef("ln_y", LN / "ln_y.cc", [u8_ln] * 5 + [i32], LN_FLAGS)
    f_lnx = ef("ln_xn", LN / "ln_xn.cc", [u8_ln] * 6, LN_FLAGS)
    f_sc = ef("short_conv_step", HERE / "sc_elem.cc",
              [fsc, fsc, fsc, bsc, bsc, bsc, fsc, fsc, fsc, fsc, fsc], SC_FLAGS)

    # ---- fifos
    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    # The conv core's streams. `scin` carries B, C, u and the two state rows for one
    # element; `scw` the three taps; `scout` the gated output and the two new state rows.
    of_scin = ObjectFifo(fsc, name="scin", depth=6)   # five acquired at once, so depth > 5
    of_scw = ObjectFifo(bsc, name="scw", depth=4)
    of_scout = ObjectFifo(fsc, name="scout", depth=4)

    PB_H, NG_H = per_band(HID), n_groups(HID)
    PB_F, NG_F = per_band(FF), n_groups(FF)

    def gemv_bands(win, yout, tab, nbands, ngroups, pb):
        for _ in range_(nbands):
            ye = yout.acquire(1)
            for g in range_(ngroups):
                we = win.acquire(1)
                f_gy(we, tab, ye, g, pb, 2)
                win.release(1)
            yout.release(1)

    def acq(fifo, n):
        e = fifo.acquire(n)
        return [e] if n == 1 else e

    def main_body(win, xin, yout, tab, ms, f_gy, f_gms, f_act, f_prep, f_prepf):
        # 1. B | C | u against xn (K = HID): three [hidden, hidden] projections
        xe = acq(xin, G.XN_ELEMS)
        for i in range(G.XN_ELEMS):
            f_prep(xe[i], tab, HID, i)
        gemv_bands(win, yout, tab, 3 * SC_PC, NG_H, PB_H)
        xin.release(G.XN_ELEMS)
        if stop == 1:
            return
        # 2. out against the conv's y (K = HID)
        oe = acq(xin, G.XN_ELEMS)
        for i in range(G.XN_ELEMS):
            f_prep(oe[i], tab, HID, i)
        gemv_bands(win, yout, tab, SC_PC, NG_H, PB_H)
        xin.release(G.XN_ELEMS)
        if stop == 2:
            return
        # 3. up | gate per band against xm, silu -> h band
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
            f_act(ms, ye)
            yout.release(1)
        xin.release(G.XM_ELEMS)
        if stop == 3:
            return
        # 4. down against h (K = FF)
        for i in range_(G.H_ELEMS):
            he = xin.acquire(1)
            f_prepf(he, tab, FF, i)
            xin.release(1)
        gemv_bands(win, yout, tab, G.DOWN_PC, NG_F, PB_F)

    def ln_body(ain, aout, f_nr, f_lny, f_lnx):
        # 1. the layer-entry norm: [x0 x1 lnw] -> [xn]
        e = ain.acquire(3)
        o = aout.acquire(1)
        f_nr(e[0], e[1], e[2], o)
        aout.release(1)
        ain.release(3)

        def add_norm():
            # [x0 x1 w a0 a1] -> [y0] [y1] [xn]
            e = ain.acquire(5)
            for i in range(2):
                o = aout.acquire(1)
                f_lny(e[0], e[1], e[3], e[4], o, i)
                aout.release(1)
            o = aout.acquire(1)
            f_lnx(e[0], e[1], e[3], e[4], e[2], o)
            aout.release(1)
            ain.release(5)

        add_norm()                          # 2. res = x + out; xm = post_attn_norm(res)
        if stop >= 3:
            add_norm()                      # 3. xres = res + out2 (the xn is junk)

    def sc_body(scin, scw, scout, f_sc):
        """One token. Per element: the three taps, then B, C, u and the two state rows in,
        then the gated output and the two new state rows out."""
        for _ in range_(SC_ELEMS):
            w = scw.acquire(TAPS)
            e = scin.acquire(5)             # B, C, u, state0, state1
            o = scout.acquire(3)            # y, new state0, new state1
            f_sc(e[0], e[1], e[2], w[0], w[1], w[2], e[3], e[4], o[0], o[1], o[2])
            scout.release(3)
            scin.release(5)
            scw.release(TAPS)

    workers = [Worker(ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_lny, f_lnx],
                      tile=Tile(0, 3), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(main_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                  Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                                                  f_gy, f_gms, f_act, f_prep, f_prepf],
                              tile=Tile(c, 2), stack_size=0x1800))
    workers.append(Worker(sc_body, fn_args=[of_scin.cons(), of_scw.cons(), of_scout.prod(), f_sc],
                          tile=Tile(2, 3), stack_size=0x1800))

    BB_H, BB_F = band_bytes(HID), band_bytes(FF)
    BB_UG = band_bytes(HID)
    YB = BAND_ROWS * 4
    SCB = SCW * 4                            # one conv element in bytes (f32)
    TAPB = SCW * 2                           # one tap's channels for one element (bf16)

    def _sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                  scin_p, scw_p, scout_c):
        # 1. layer-entry norm: xn -> act
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(L.CD_BYTES, DL.CD_LNW, ELN), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(L.AD_BYTES, DL.AD_XN, ELN), wait=True, group=tg_ln)
        # 2. B | C | u: weights now, xn after the norm
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            for base in (L.POOL_SC_B, L.POOL_SC_C, L.POOL_SC_U):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, base + c * SC_PC * BB_H, SC_PC * BB_H))
            for dst in (L.AD_B, L.AD_C, L.AD_U):
                py.drain(y_conss[c], a_act, bt(L.AD_BYTES, dst + c * SC_PC * YB, SC_PC * YB))
        tg_ln.finish()
        tg_x = TaskGroup()
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, DL.AD_XN, G.XN_ELEMS * ELEM), wait=True, group=tg_x)
        if stop == 1:
            py.finish()
            pw.finish()
            tg_x.finish()
            return
        # 3. the conv core: taps, then B / C / u and the state; y and the new state out
        ps_in, ps_out = Pipeline(3), Pipeline(3)
        for i in range(SC_ELEMS):
            for k in range(TAPS):
                ps_in.fill(scw_p, a_consts, bt(L.CD_BYTES, L.CD_CONV + (k * HID + i * SCW) * 2, TAPB))
        py.finish(*y_conss)                                   # B, C, u are in DDR
        for i in range(SC_ELEMS):
            ps_in.fill(scin_p, a_act, bt(L.AD_BYTES, L.AD_B + i * SCB, SCB))
            ps_in.fill(scin_p, a_act, bt(L.AD_BYTES, L.AD_C + i * SCB, SCB))
            ps_in.fill(scin_p, a_act, bt(L.AD_BYTES, L.AD_U + i * SCB, SCB))
            ps_in.fill(scin_p, a_state, bt(L.STATE_BYTES, i * SCB, SCB))
            ps_in.fill(scin_p, a_state, bt(L.STATE_BYTES, HID * 4 + i * SCB, SCB))
            ps_out.drain(scout_c, a_act, bt(L.AD_BYTES, L.AD_Y + i * SCB, SCB))
            ps_out.drain(scout_c, a_state, bt(L.STATE_BYTES, i * SCB, SCB))
            ps_out.drain(scout_c, a_state, bt(L.STATE_BYTES, HID * 4 + i * SCB, SCB))
        ps_out.finish()                                       # y and the new state are in DDR
        ps_in.finish()
        # 4. out projection against y
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_Y, G.XN_ELEMS * ELEM), wait=True, group=tg_x)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_SC_OUT + c * SC_PC * BB_H, SC_PC * BB_H))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, DL.AD_OUT + c * SC_PC * YB, SC_PC * YB))
        if stop == 2:
            py.finish()
            pw.finish()
            tg_x.finish()
            return
        # 5. residual + second norm -> res, xm
        tg_ln2 = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
        lni.fill(a_consts, tap=bt(L.CD_BYTES, DL.CD_POSTLN, ELN), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(L.AD_BYTES, DL.AD_RES, HID * 4), wait=True, group=tg_ln2)
        lno.drain(a_act, tap=bt(L.AD_BYTES, DL.AD_XM, ELN), wait=True, group=tg_ln2)
        py.finish()                                           # out is in DDR
        lni.fill(a_act, tap=bt(L.AD_BYTES, DL.AD_OUT, HID * 4), wait=True, group=tg_ln2)
        tg_ln2.finish()
        # 6. up | gate per band, silu -> h
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, DL.AD_XM, G.XM_ELEMS * ELEM), wait=True, group=tg_x)
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, DL.AD_H + c * G.UP_PC * YB, G.UP_PC * YB))
        for j in range(G.UP_PC):
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_SC_UP + (c * G.UP_PC + j) * BB_UG, BB_UG))
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_SC_GATE + (c * G.UP_PC + j) * BB_UG, BB_UG))
        py.finish()                                           # h is in DDR
        if stop == 3:
            pw.finish()
            tg_x.finish()
            return
        # 7. down against h, then the output residual -> xres
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, DL.AD_H, G.H_ELEMS * ELEM), wait=True, group=tg_x)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_SC_DOWN + c * G.DOWN_PC * BB_F, G.DOWN_PC * BB_F))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, DL.AD_OUT2 + c * G.DOWN_PC * YB, G.DOWN_PC * YB))
        tg_ln3 = TaskGroup()
        lni.fill(a_act, tap=bt(L.AD_BYTES, DL.AD_RES, HID * 4), wait=True, group=tg_ln3)
        lni.fill(a_consts, tap=bt(L.CD_BYTES, DL.CD_POSTLN, ELN), wait=True, group=tg_ln3)
        lno.drain(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln3)
        lno.drain(a_act, tap=bt(L.AD_BYTES, DL.AD_JUNK, ELN), wait=True, group=tg_ln3)
        py.finish()                                           # out2 is in DDR
        lni.fill(a_act, tap=bt(L.AD_BYTES, DL.AD_OUT2, HID * 4), wait=True, group=tg_ln3)
        tg_ln3.finish()
        pw.finish()
        tg_x.finish()

    def sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                 scin_p, scw_p, scout_c):
        _sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                  scin_p, scw_p, scout_c)

    # Shim channels, following dx.py: columns 0..2 already pair a producer with lni, x and
    # the helper's input, so the taps take column 3 the way the q/k/v bias does there.
    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_scin.prod(tile=Tile(2, 0)),
                            of_scw.prod(tile=Tile(3, 0)),
                            of_scout.cons(tile=Tile(1, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = cx
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + sorted(f.read_bytes() for f in DENSE.glob("*.cc"))
                + sorted(f.read_bytes() for f in (HERE.parent.parent / "recipes").glob("*.py"))
                + [(LN / "ln.h").read_bytes(), (LN / "ln_y.cc").read_bytes(), (LN / "ln_xn.cc").read_bytes(),
                   (LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"stop": STOP, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
