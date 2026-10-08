r"""lmhl2: lmhl with 10 KB weight elements; lmhl's whole-K tables left room for 5 KB only, DMA-bound at ~2x lm_head_q4."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
DESIGNS = HERE.parent
GEMV, LINL = DESIGNS / "gemv_q4", DESIGNS / "lin_layer"
sys.path.insert(0, str(DESIGNS.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes.families import for_spec  # noqa: E402
from recipes import dxl as DXR  # noqa: E402

SPEC = current_spec()
R = for_spec(SPEC).recipe(SPEC)
L0, G = R.layout, R.geo
LR = int(os.environ.get("DXL_L", 4))
H = DXR.head_layout(SPEC, LR)
HID, N_CORES, CHUNK, KS = G.HID, H.CORES, DXR.CHUNK, DXR.KS_BF16
BPC, BT, ROW = H.BPC, H.BT, H.ROW_FLOATS
S, KT = HID // KS, HID // 256
EPS = 2 * KS // 256 // 2                    # 10 KB elements per band per slice
BB = 2 * KT * CHUNK
SB = 2 * (KS // 256) * CHUNK                # a band's slice bytes
YE = LR * 64
ELN = L0.ELN
LN_FLAGS = [f"-DLN_N={HID}", f"-DLN_EPS={G.EPS:g}f"] + ([f"-DLN_GROUPS={SPEC.norm_groups}"] if SPEC.norm_groups != 1 else [])
GL_FLAGS = ["-Os", f"-DDXL_L={LR}"]


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def tap(total, off, sizes, strides):
    return TensorAccessPattern((1, total), off, sizes, strides)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lmhl2(pool: In, xres: In, normw: In, act: InOut, out: Out, *, srchash: CompileTime[int] = 0):
    elem = np.ndarray[(2 * CHUNK,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(DXR.XE // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(YE,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(LR * DXR.tab_bytes(KS),), np.dtype[np.uint8]]
    acc_ty = np.ndarray[(BT * YE,), np.dtype[np.float32]]
    i16_ty = np.ndarray[(16,), np.dtype[np.int32]]
    for ty in (elem, x_ty, y_ty, tab_ty, acc_ty, i16_ty):        # see dxl.py: every core buffer 64 B whole
        shape, dt = ty.__args__
        assert int(np.prod(shape)) * np.dtype(dt.__args__[0]).itemsize % 64 == 0, ty
    u8_ln = np.ndarray[(ELN,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(L0.LMHEAD_POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(LR * HID,), np.dtype[np.float32]]
    normw_ty = np.ndarray[(HID,), np.dtype[bfloat16]]
    act_ty = np.ndarray[(LR * HID * 2,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(H.OUT_FLOATS,), np.dtype[np.float32]]
    i32 = np.int32

    inc = include_dirs() + [str(HERE), str(GEMV), str(LINL)]

    def ef(sym, args, flags=GL_FLAGS, src=None):
        return ExternalFunction(sym, source_file=str(src or HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=flags)

    f_gemv = ef("dxl_gemv2", [elem, tab_ty, acc_ty, i32, i32, i32, i32, i32])
    f_prep = ef("dxl_prep", [x_ty, tab_ty, i32, i32])
    f_out = ef("dxl_out", [acc_ty, y_ty, i32])
    f_init = ef("dxl_lm_init2", [i16_ty, i16_ty])
    f_arg = ef("dxl_lm_arg2", [y_ty, i16_ty, i16_ty, i32, i32])
    f_bout = ef("dxl_lm_out", [i16_ty, y_ty])
    f_nr = ef("ln_nr", [u8_ln] * 4, LN_FLAGS, LINL / "ln_nr.cc")

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=3)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)

    def make_body(base: int):
        def body(win, xin, yout, tab, acc, best, cnt, fg, fp, fo, fi, fa, fb):
            fi(best, cnt)
            for _ in range_(BPC // BT):
                for s in range_(S):
                    for tok in range_(LR):
                        xe = xin.acquire(1)
                        fp(xe, tab, tok, KS)
                        xin.release(1)
                    for b in range_(BT):
                        for e in range_(EPS):
                            we = win.acquire(1)
                            fg(we, tab, acc, b, e, s, KS, KT)
                            win.release(1)
                for b in range_(BT):
                    ye = yout.acquire(1)
                    fo(acc, ye, b)
                    fa(ye, best, cnt, base, SPEC.real_vocab)
                    yout.release(1)
            ye = yout.acquire(1)
            fb(best, ye)
            yout.release(1)
        return body

    def ln_body(ain, aout, f_nr):
        for _ in range_(LR):
            e = ain.acquire(3)
            o = aout.acquire(1)
            f_nr(e[0], e[1], e[2], o)
            aout.release(1)
            ain.release(3)

    workers = [Worker(ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr], tile=Tile(0, 3), stack_size=0x1000)]
    for c in range(N_CORES):
        workers.append(Worker(make_body(c * BPC),
                              fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), Buffer(tab_ty, name=f"tab{c}"),
                                       Buffer(acc_ty, name=f"acc{c}"), Buffer(i16_ty, name=f"best{c}"),
                                       Buffer(i16_ty, name=f"cnt{c}"), f_gemv, f_prep, f_out, f_init, f_arg, f_bout],
                              tile=Tile(c, 2), stack_size=0x1000))

    def sequence(a_pool, c_xres, c_normw, a_act, a_out, lni, lno, w_prods, x_prod, y_conss):
        pl, pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            # [tile][band][row][64] -> logits[row][band*64 ..], then the core's argmax element
            py.drain(y_conss[c], a_out, tap(H.OUT_FLOATS, c * BPC * 64, [BPC // BT, BT, LR, 64], [BT * 64, 64, ROW, 1]))
            py.drain(y_conss[c], a_out, bt(H.OUT_FLOATS, LR * ROW + c * YE, YE))
        for j in range(LR):
            pl.fill(lni, c_xres, bt(LR * HID, j * HID, HID))
            pl.fill(lni, c_normw, bt(HID, 0, HID))
            pl.drain(lno, a_act, bt(LR * HID * 2, j * HID * 2, ELN))
        for c in range(N_CORES):                               # the first tile's weights under the norms
            pw.fill(w_prods[c], a_pool, tap(L0.LMHEAD_POOL_BYTES, c * BPC * BB, [S, BT, SB // 2048, 2048],
                                            [SB, BB, 2048, 1]))
        pl.finish()
        xb = KS * 2
        for t in range(BPC // BT):
            px.fill(x_prod, a_act, tap(LR * HID * 2, 0, [1, S, LR, xb], [0, xb, HID * 2, 1]))
            if t:
                for c in range(N_CORES):
                    b0 = (c * BPC + t * BT) * BB
                    pw.fill(w_prods[c], a_pool, tap(L0.LMHEAD_POOL_BYTES, b0, [S, BT, SB // 2048, 2048],
                                                    [SB, BB, 2048, 1]))
        px.finish()
        pw.finish()
        py.finish()

    rt = Runtime(sequence, [pool_ty, xres_ty, normw_ty, act_ty, out_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = lmhl2
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [Path(__file__).read_bytes(), (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (LINL / "ln_nr.cc").read_bytes(), SPEC.spec_hash().encode(), str(LR).encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
