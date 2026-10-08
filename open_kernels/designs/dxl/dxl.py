r"""dxl: L consecutive positions through one dense layer in ONE dispatch, the weights
streamed once for all L (the verify / draft pass of speculative decoding).

    for each row: ln -> L-row gemv q | k | v -> attention, one row at a time (row j's KV
    reaches DDR before row j+1 reads its window) -> L-row gemv o -> for each row: ln
    (+residual) -> L-row gemv up | gate, silu(gate) * up -> L-row gemv down -> for each
    row: +residual

dx's fabric and buffers: 8 main cores (Tile(c, 2)), the ln core (Tile(0, 3)), the
attention cores (Tile(2 + c, 3)), and the same per-layer pool, consts, kv and ptab, so
a kernel set gains dxl without a second copy of any weight. Its own buffers are the
L-row residual `xres` f32[L][HID] and the L-row scratch `act` (recipes/dxl.py).

Per row the arithmetic is dx's -- the same ln and attention kernels, a GEMV that is
bit-identical per token to gemv_q4_tile (dxl_gemv.h), the same silu -- so the L rows'
logits are the ones L decode steps would give.

The attention patch sites are built for the placeholder positions 1 .. L (row j at
1 + j); the driver's `attnrows` patch moves them to pos0 + j.

Build: OPEN_KERNELS_SPEC=<spec> DXL_L=4 python build_design.py designs/dxl/dxl.py designs/dxl/build_<tag>
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
DESIGNS = HERE.parent
GEMV, ATTN, LN, LINL = DESIGNS / "gemv_q4", DESIGNS / "attn", DESIGNS / "ln", DESIGNS / "lin_layer"
sys.path.insert(0, str(DESIGNS.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes.families import for_spec  # noqa: E402
from recipes import dxl as DXR  # noqa: E402

SPEC = current_spec()
QR = for_spec(SPEC)
R = QR.recipe(SPEC)
L0, G = R.layout, R.geo                     # dx's layout: pool / consts / kv / ptab offsets
NL_ROWS = int(os.environ.get("DXL_L", 4))
X = DXR.layout(SPEC, NL_ROWS)
LR = X.L
HID, FF, N_CORES = G.HID, G.FF, G.N_CORES
QW, KVW = G.QW, G.KVW
ELN, E_A = L0.ELN, L0.E_A
CHUNK = DXR.CHUNK
BB_H, BB_Q, BB_F = 2 * (HID // 256) * CHUNK, 2 * (QW // 256) * CHUNK, 2 * (FF // 256) * CHUNK
KSB, KSF = DXR.KS_BF16, DXR.KS_F32
YE = LR * 64                                # one band's [L][64] floats
BT, BP = X.BT, X.BP
YB = max(BT, BP, G.KV_PC)                   # accumulator bands; the gate bands follow at YB
OS = ["-Os"]

ATTN_FLAGS = [f"-DATTN_NH={G.NH}", f"-DATTN_KVH={G.KVH}", f"-DATTN_HD={G.HD}", f"-DATTN_ROT={G.ROT}", "-DATTN_GATE=0",
              f"-DATTN_QKNORM={1 if G.QKNORM else 0}", f"-DATTN_QKNORM_POST={1 if G.QKNORM_POST else 0}",
              f"-DATTN_EPS={G.EPS:g}f", f"-DATTN_VEXP={G.VEXP}", f"-DATTN_NHL={G.NHL}"]
if G.RB > 1:
    ATTN_FLAGS.append(f"-DATTN_RB={G.RB}")
NPTAB, CSE = G.PTAB_ELEMS, G.PTAB_CS_ELEM
if NPTAB > 1:
    ATTN_FLAGS.append("-DATTN_PTAB_SPLIT=1")
for _k, _v in QR.probe_env().items():
    if _k not in ("ATTN_RB", "ATTN_FAST"):
        ATTN_FLAGS.append(f"-D{_k}={_v}")
ACORES, NHL, RB = G.ACORES, G.NHL, G.RB
OGH = min(NHL, G.HPO)
N_OG = NHL // OGH
LN_FLAGS = [f"-DLN_N={HID}", f"-DLN_EPS={G.EPS:g}f"]
if SPEC.norm_groups != 1:
    LN_FLAGS.append(f"-DLN_GROUPS={SPEC.norm_groups}")
GL_FLAGS = OS + [f"-DDXL_L={LR}"]


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def tap(total, off, sizes, strides):
    return TensorAccessPattern((1, total), off, sizes, strides)


# The main cores' segments, in the order a layer runs them: (pool offset, bands per core,
# bands per tile, K, slice, f32 input?). k and v are one tile of their KV_PC bands each.
SEG_Q = (L0.POOL_Q, G.Q_PC, BT, HID, KSB, False, BB_H)
SEG_K = (L0.POOL_K, G.KV_PC, G.KV_PC, HID, KSB, False, BB_H)
SEG_V = (L0.POOL_V, G.KV_PC, G.KV_PC, HID, KSB, False, BB_H)
SEG_O = (L0.POOL_O, G.O_PC, BT, QW, KSB, False, BB_Q)
SEG_D = (L0.POOL_DOWN, G.DOWN_PC, BT, FF, KSF, True, BB_F)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dxl(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, *, srchash: CompileTime[int] = 0):
    elem = np.ndarray[(CHUNK,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(DXR.XE // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(YE,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(LR * DXR.tab_bytes(KSB),), np.dtype[np.uint8]]
    acc_ty = np.ndarray[((YB + BP) * YE,), np.dtype[np.float32]]
    u8_ln = np.ndarray[(ELN,), np.dtype[np.uint8]]
    u8_a = np.ndarray[(E_A,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(L0.POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(X.XRES_FLOATS,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(L0.CD_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(L0.KV_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(X.AD_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(L0.PTAB_BYTES,), np.dtype[np.uint8]]
    pb_ty = np.ndarray[(8 if RB > 1 else 4,), np.dtype[np.int32]]
    bhd = np.ndarray[(G.HD,), np.dtype[bfloat16]]
    brow = np.ndarray[(KVW,), np.dtype[bfloat16]]
    og_ty = np.ndarray[(OGH * G.HD,), np.dtype[bfloat16]]
    fcs = np.ndarray[(G.ROT,), np.dtype[np.float32]]
    fhd = np.ndarray[(G.HD,), np.dtype[np.float32]]
    fq = (np.ndarray[(2 * QW,), np.dtype[bfloat16]] if G.VEXP else np.ndarray[(QW,), np.dtype[np.float32]])
    fml = np.ndarray[(2 * G.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * G.HD,), np.dtype[np.float32]]
    i32 = np.int32

    inc = include_dirs() + [str(HERE), str(GEMV), str(ATTN), str(LN), str(LINL)]

    def ef(sym, src, args, flags=OS):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    f_gemv = ef("dxl_gemv_at", HERE / "dxl_gemv_at.cc", [elem, tab_ty, acc_ty, i32, i32, i32, i32, i32], GL_FLAGS)
    f_prep = ef("dxl_prep", HERE / "dxl_prep.cc", [x_ty, tab_ty, i32, i32], GL_FLAGS)
    f_prepf = ef("dxl_prep_f32", HERE / "dxl_prep_f32.cc", [x_ty, tab_ty, i32, i32], GL_FLAGS)
    f_act = ef("dxl_act", HERE / "dxl_act.cc", [acc_ty, y_ty, i32, i32], GL_FLAGS)
    f_out = ef("dxl_out", HERE / "dxl_out.cc", [acc_ty, y_ty, i32], GL_FLAGS)
    f_nr = ef("ln_nr", LINL / "ln_nr.cc", [u8_ln] * 4, LN_FLAGS)
    f_lny = ef("ln_y", LN / "ln_y.cc", [u8_ln] * 5 + [i32], LN_FLAGS)
    f_lnx = ef("ln_xn", LN / "ln_xn.cc", [u8_ln] * 6, LN_FLAGS)
    pz = [u8_a] if NPTAB > 1 else []
    f_meta = ef("attn_meta", ATTN / "attn_meta.cc", [u8_a, u8_a] + pz + [bhd, bhd, fcs, pb_ty], ATTN_FLAGS)
    f_q = ef("attn_q", ATTN / "attn_q.cc", [u8_a, bhd, fcs, fq, i32], ATTN_FLAGS)
    f_k = ef("attn_k", ATTN / "attn_k.cc", [u8_a, bhd, fcs, fhd, brow, i32], ATTN_FLAGS)
    f_v = ef("attn_v", ATTN / "attn_v.cc", [u8_a, brow, i32], ATTN_FLAGS)
    f_init = ef("attn_init", ATTN / "attn_init.cc", [foacc, fml], ATTN_FLAGS)
    h0_arg = [i32] if ACORES > 1 else []
    f_step = ef("attn_step", ATTN / "attn_step.cc", [u8_a, u8_a, fq, foacc, fml, pb_ty] + h0_arg, ATTN_FLAGS)
    f_stepn = ef("attn_step_new", ATTN / "attn_step_new.cc", [brow, brow, fq, foacc, fml] + h0_arg, ATTN_FLAGS)
    f_stepb = (ef("attn_stepb", ATTN / "attn_stepb.cc", [u8_a] * (2 * RB) + [fq, foacc, fml, pb_ty] + h0_arg,
                  ATTN_FLAGS) if RB > 1 else None)
    f_fin = ef("attn_fin_ng", ATTN / "attn_fin_ng.cc", [foacc, fml, og_ty, i32], ATTN_FLAGS)

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_ain = ObjectFifo(u8_a, name="ain", depth=max(4, 2 * RB + 2, 1 + NPTAB + 1))
    of_aout = ObjectFifo(brow, name="aout", depth=2)
    of_og = [ObjectFifo(og_ty, name=f"og{c}", depth=2) for c in range(ACORES)]

    def slices(win, xin, tab, acc, fg, fp, b0, nb, K, KS):
        """nb accumulator bands from b0 against K in KS-wide slices (the L tables rebuilt per slice)"""
        cps, kt = 2 * KS // 256, K // 256
        for s in range_(K // KS):
            for j in range_(LR):
                xe = xin.acquire(1)
                fp(xe, tab, j, KS)
                xin.release(1)
            for b in range_(nb):
                for c in range_(cps):
                    we = win.acquire(1)
                    fg(we, tab, acc, b0 + b, c, s, KS, kt)
                    win.release(1)

    def drain(yout, acc, fo, nb):
        for b in range_(nb):
            ye = yout.acquire(1)
            fo(acc, ye, b)
            yout.release(1)

    def main_body(win, xin, yout, tab, acc, fg, fp, fpf, fa, fo):
        for (nb, bt_, K, KS, prep) in ((G.Q_PC, BT, HID, KSB, fp), (G.KV_PC, G.KV_PC, HID, KSB, fp),
                                       (G.KV_PC, G.KV_PC, HID, KSB, fp), (G.O_PC, BT, QW, KSB, fp)):
            for _ in range_(nb // bt_):
                slices(win, xin, tab, acc, fg, prep, 0, bt_, K, KS)
                drain(yout, acc, fo, bt_)
        for _ in range_(G.UP_PC // BP):                          # up -> bands 0.., gate -> bands YB..
            slices(win, xin, tab, acc, fg, fp, 0, BP, HID, KSB)
            slices(win, xin, tab, acc, fg, fp, YB, BP, HID, KSB)
            for b in range_(BP):
                ye = yout.acquire(1)
                fa(acc, ye, b, YB)
                yout.release(1)
        for _ in range_(G.DOWN_PC // BT):
            slices(win, xin, tab, acc, fg, fpf, 0, BT, FF, KSF)
            drain(yout, acc, fo, BT)

    def ln_body(ain, aout, f_nr, f_lny, f_lnx):
        for _ in range_(LR):                                     # entry norm, per row: [x0 x1 lnw] -> xn
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

        for _ in range_(LR):                                     # res = x + out; xm = norm(res)
            add_norm()
        for _ in range_(LR):                                     # xres = res + out2 (xn is junk)
            add_norm()

    def _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
              f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, h0):
        for _ in range_(LR):                                     # one query row at a time
            e = ain.acquire(1 + NPTAB)
            if NPTAB > 1:
                f_meta(e[0], e[1], e[1 + CSE], qn, kn, cs, pb)
            else:
                f_meta(e[0], e[1], qn, kn, cs, pb)
            ain.release(1 + NPTAB)
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
            if aout is not None:
                o = aout.acquire(1)
                for j in range_(KVW):
                    o[j] = kout[j]
                aout.release(1)
                o = aout.acquire(1)
                for j in range_(KVW):
                    o[j] = vout[j]
                aout.release(1)
            f_init(oacc, ml)
            if RB > 1:
                for _ in range_(pb[4]):
                    e = ain.acquire(2 * RB)
                    args = [e[i] for i in range(2 * RB)] + [qs, oacc, ml, pb] + ([h0] if ACORES > 1 else [])
                    f_stepb(*args)
                    ain.release(2 * RB)
                for _ in range_(pb[5]):
                    e = ain.acquire(2)
                    f_step(e[0], e[1], qs, oacc, ml, pb, h0) if ACORES > 1 else f_step(e[0], e[1], qs, oacc, ml, pb)
                    ain.release(2)
            else:
                for _ in range_(pb[1]):
                    e = ain.acquire(2)
                    f_step(e[0], e[1], qs, oacc, ml, pb, h0) if ACORES > 1 else f_step(e[0], e[1], qs, oacc, ml, pb)
                    ain.release(2)
            f_stepn(kout, vout, qs, oacc, ml, h0) if ACORES > 1 else f_stepn(kout, vout, qs, oacc, ml)
            for hp in range_(N_OG):
                o = ogout.acquire(1)
                f_fin(oacc, ml, o, hp)
                ogout.release(1)

    afns = [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin] + ([f_stepb] if RB > 1 else [])

    def attn_body(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, *fns):
        fb = fns[8] if RB > 1 else None
        _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, *fns[:8], fb, 0)

    def make_attn_body(c):
        h0 = c * NHL

        def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, *fns):
            fb = fns[8] if RB > 1 else None
            _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb, *fns[:8], fb, h0)
        return body

    workers = [Worker(ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_lny, f_lnx],
                      tile=Tile(0, 3), stack_size=0x1800)]
    for c in range(N_CORES):
        workers.append(Worker(main_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                  Buffer(tab_ty, name=f"tab{c}"), Buffer(acc_ty, name=f"acc{c}"),
                                                  f_gemv, f_prep, f_prepf, f_act, f_out],
                              tile=Tile(c, 2), stack_size=0x1800))

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(bhd, name=f"qn{s}"), Buffer(bhd, name=f"kn{s}"), Buffer(fcs, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(fhd, name=f"tmp{s}"), Buffer(brow, name=f"kout{s}"),
                Buffer(brow, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}")]

    workers.append(Worker(attn_body, fn_args=[of_ain.cons(), of_aout.prod(), of_og[0].prod()] + abufs(0) + afns,
                          tile=Tile(2, 3), stack_size=0x1800))
    for c in range(1, ACORES):
        workers.append(Worker(make_attn_body(c), fn_args=[of_ain.cons(), of_og[c].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=0x1800))

    def sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p,
                 aout_c, og_cs):
        AB, XF = X.AD_BYTES, X.XRES_FLOATS
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)

        def x_rows(off, row, K, KS, f32, tiles):
            """the L rows' K-wide activation at act + off, slice-major, once per tile"""
            xb = KS * (4 if f32 else 2)
            for _ in range(tiles):
                px.fill(x_prod, a_act, tap(AB, off, [1, K // KS, LR, xb], [0, xb, row, 1]))

        def w_seg(pool_off, nb, bt_, K, KS, bb, tiles=None):
            """core c's nb bands at pool_off, tile by tile: [slices, bands, rows of 2 KB]"""
            sb = 2 * KS // 256 * CHUNK
            for t in range(nb // bt_):
                for c in range(N_CORES):
                    b0 = pool_off + (c * nb + t * bt_) * bb
                    pw.fill(w_prods[c], a_pool, tap(L0.POOL_BYTES, b0, [K // KS, bt_, sb // 2048, 2048],
                                                    [sb, bb, 2048, 1]))

        def y_seg(off, row, nb, bt_):
            """each core's [tile][band][row][64] -> act[row][band*64 ..]"""
            for c in range(N_CORES):
                py.drain(y_conss[c], a_act, tap(AB, off + c * nb * 256, [nb // bt_, bt_, LR, 256],
                                                 [bt_ * 256, 256, row, 1]))

        # 1. entry norm, per row: xn_j -> act
        pl = Pipeline(3)                                         # the ln channels: a shim channel queues 4 BDs
        for j in range(LR):
            pl.fill(lni, c_xres, bt(XF, j * HID, HID))
            pl.fill(lni, a_consts, bt(L0.CD_BYTES, L0.CD_LNW, ELN))
            pl.drain(lno, a_act, bt(AB, X.AD_XN + j * X.XN_ROW, ELN))
        # 2. q | k | v: weights and drains now, xn after the norm
        y_seg(X.AD_Q, X.Q_ROW, G.Q_PC, BT)
        y_seg(X.AD_K, X.KV_ROW_ACT, G.KV_PC, G.KV_PC)
        y_seg(X.AD_V, X.KV_ROW_ACT, G.KV_PC, G.KV_PC)
        pl.finish()
        for (pool_off, nb, bt_, K, KS, f32, bb) in (SEG_Q, SEG_K, SEG_V):
            x_rows(X.AD_XN, X.XN_ROW, K, KS, f32, nb // bt_)
            w_seg(pool_off, nb, bt_, K, KS, bb)
        py.finish()                                              # q, k, v are in DDR
        # 3. attention, one row at a time: row j's KV row is in DDR before row j + 1's window
        for j in range(LR):
            pa_out, pa_in = Pipeline(3), Pipeline(3)
            pa_out.drain(aout_c, a_kv, bt(L0.KV_BYTES, (1 + j) * L0.KV_ROW, L0.KV_ROW))    # attnrows
            for c in range(ACORES):
                pa_out.drain(og_cs[c], a_act, bt(AB, X.AD_OG + j * X.OG_ROW + c * NHL * G.HD * 2, NHL * G.HD * 2))
            pa_in.fill(ain_p, a_consts, bt(L0.CD_BYTES, L0.CD_META, E_A))
            pa_in.fill(ain_p, a_ptab, bt(L0.PTAB_BYTES, (1 + j) * L0.PTAB_ROW, L0.PTAB_ROW))  # attnrows
            pa_in.fill(ain_p, a_act, bt(AB, X.AD_Q + j * X.Q_ROW, QW * 4))
            pa_in.fill(ain_p, a_act, bt(AB, X.AD_K + j * X.KV_ROW_ACT, KVW * 4))
            pa_in.fill(ain_p, a_act, bt(AB, X.AD_V + j * X.KV_ROW_ACT, KVW * 4))
            pa_in.fill(ain_p, a_kv, bt(L0.KV_BYTES, 0, L0.KV_ROW))                          # attnrows window
            if j == 0:                                           # o's weights stream under the attention
                y_seg(X.AD_OUT, X.HID_ROW, G.O_PC, BT)
                w_seg(*SEG_O[:5], SEG_O[6])
            pa_out.finish()
            pa_in.finish()
        # 4. o against og
        x_rows(X.AD_OG, X.OG_ROW, QW, KSB, False, G.O_PC // BT)
        # 5. per row: res = x + out; xm = norm(res)
        py.finish()                                              # out is in DDR
        for j in range(LR):
            pl.fill(lni, c_xres, bt(XF, j * HID, HID))
            pl.fill(lni, a_consts, bt(L0.CD_BYTES, L0.CD_POSTLN, ELN))
            pl.fill(lni, a_act, bt(AB, X.AD_OUT + j * X.HID_ROW, HID * 4))
            pl.drain(lno, a_act, bt(AB, X.AD_RES + j * X.HID_ROW, HID * 4))
            pl.drain(lno, a_act, bt(AB, X.AD_XM + j * X.XM_ROW, ELN))
        # 6. up | gate per tile of BP pairs, silu -> h
        for c in range(N_CORES):
            py.drain(y_conss[c], a_act, tap(AB, X.AD_H + c * G.UP_PC * 256, [G.UP_PC // BP, BP, LR, 256],
                                             [BP * 256, 256, X.H_ROW, 1]))
        pl.finish()                                              # res, xm are in DDR
        sb = 2 * KSB // 256 * CHUNK
        for t in range(G.UP_PC // BP):
            for pool_off in (L0.POOL_UP, L0.POOL_GATE):
                x_rows(X.AD_XM, X.XM_ROW, HID, KSB, False, 1)
                for c in range(N_CORES):
                    b0 = pool_off + (c * G.UP_PC + t * BP) * BB_H
                    pw.fill(w_prods[c], a_pool, tap(L0.POOL_BYTES, b0, [HID // KSB, BP, sb // 2048, 2048],
                                                    [sb, BB_H, 2048, 1]))
        py.finish()                                              # h is in DDR
        # 7. down against h, then per row: xres = res + out2
        y_seg(X.AD_OUT2, X.HID_ROW, G.DOWN_PC, BT)
        x_rows(X.AD_H, X.H_ROW, FF, KSF, True, G.DOWN_PC // BT)
        w_seg(*SEG_D[:5], SEG_D[6])
        py.finish()                                              # out2 is in DDR
        for j in range(LR):
            pl.fill(lni, a_act, bt(AB, X.AD_RES + j * X.HID_ROW, HID * 4))
            pl.fill(lni, a_consts, bt(L0.CD_BYTES, L0.CD_POSTLN, ELN))
            pl.fill(lni, a_act, bt(AB, X.AD_OUT2 + j * X.HID_ROW, HID * 4))
            pl.drain(lno, c_xres, bt(XF, j * HID, HID))
            pl.drain(lno, a_act, bt(AB, X.AD_JUNK, ELN))
        pl.finish()
        pw.finish()
        px.finish()

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, kv_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(1, 0)),
                            [of_og[c].cons(tile=Tile(2 + c, 0)) for c in range(ACORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = dxl
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [Path(__file__).read_bytes()]
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + sorted(f.read_bytes() for f in (DESIGNS.parent / "recipes").glob("*.py"))
                + [(LN / "ln.h").read_bytes(), (LN / "ln_y.cc").read_bytes(), (LN / "ln_xn.cc").read_bytes(),
                   (LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (DESIGNS.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode(), str(LR).encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
