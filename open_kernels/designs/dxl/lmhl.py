r"""lmhl: the L-row pass's tail in one dispatch -- the final norm of each row, the q4_1 head
for all L rows in one pass over its weights, and each row's argmax on the NPU.

    for each row: xn = norm(xres) (ln core) -> L-row head GEMV per band -> logits[row][..]
    and a running first-max per row (main cores) -> one [value | row] element per core

The head's bands split over the cores as lm_head_q4's do (lm_head_q8's uneven split) and
stream from the same head pool. K = hidden fits whole, so every token keeps a whole-K
table and a band is one pass of its chunks. Per token the logits are lm_head_q4's bit
for bit (the GEMV of dxl_gemv.h), so the argmax is the one the host would take from
them: the first maximal row, padding rows (>= real_vocab) excluded.

out = logits f32[L][VOCAB] | argmax int32[N_CORES][L*64] ([value L | row L] per core:
the host picks, per row, the largest value over the cores, the first core on a tie).

Build: OPEN_KERNELS_SPEC=<spec> DXL_L=4 python build_design.py designs/dxl/lmhl.py designs/dxl/build_lmhl_<tag>
"""

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
from recipes.dense import lm_rows  # noqa: E402

SPEC = current_spec()
R = for_spec(SPEC).recipe(SPEC)
L0, G = R.layout, R.geo
LR = int(os.environ.get("DXL_L", 4))
HID = G.HID
N_CORES = 8
CHUNK = 5120
VOCAB = lm_rows(SPEC)
BANDS = VOCAB // 64
BB = 2 * (HID // 256) * CHUNK
CPB = 2 * HID // 256                        # chunks per band
XE = 2048                                   # one x element: 1024 bf16
YE = LR * 64
ELN = L0.ELN
TABB = 2 * HID + HID // 4
assert LR * TABB + 2 * CHUNK + 2 * XE + 2 * YE * 4 + 0x1000 <= 60 * 1024, "whole-K tables do not fit at this L"
LN_FLAGS = [f"-DLN_N={HID}", f"-DLN_EPS={G.EPS:g}f"] + ([f"-DLN_GROUPS={SPEC.norm_groups}"] if SPEC.norm_groups != 1 else [])
GL_FLAGS = ["-Os", f"-DDXL_L={LR}"] + (["-DGEMV_NULL"] if os.environ.get("LMHL_NULL") == "1" else [])
NOARG = os.environ.get("LMHL_NOARG") == "1"             # probe: no argmax epilogue


def split_bands(bands: int, n: int) -> list[int]:
    q, r = divmod(bands, n)
    return [q + (1 if c < r else 0) for c in range(n)]


COUNTS = split_bands(BANDS, N_CORES)
BASES = [sum(COUNTS[:c]) for c in range(N_CORES)]
OUT_FLOATS = LR * VOCAB + N_CORES * YE


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


def tap(total, off, sizes, strides):
    return TensorAccessPattern((1, total), off, sizes, strides)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lmhl(pool: In, xres: In, normw: In, act: InOut, out: Out, *, srchash: CompileTime[int] = 0):
    elem = np.ndarray[(CHUNK,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(XE // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(YE,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(LR * TABB,), np.dtype[np.uint8]]
    best_ty = np.ndarray[(2 * LR,), np.dtype[np.int32]]
    u8_ln = np.ndarray[(ELN,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(L0.LMHEAD_POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(LR * HID,), np.dtype[np.float32]]
    normw_ty = np.ndarray[(HID,), np.dtype[bfloat16]]
    act_ty = np.ndarray[(LR * HID * 2,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(OUT_FLOATS,), np.dtype[np.float32]]
    i32 = np.int32

    inc = include_dirs() + [str(HERE), str(GEMV), str(LINL)]

    def ef(sym, src, args, flags=GL_FLAGS):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    f_gemv = ef("dxl_gemv_at", HERE / "dxl_gemv_at.cc", [elem, tab_ty, y_ty, i32, i32, i32, i32, i32])
    f_prep = ef("dxl_prep_whole", HERE / "dxl_prep_whole.cc", [x_ty, tab_ty, i32, i32, i32])
    f_init = ef("dxl_lm_init", HERE / "dxl_lm_init.cc", [best_ty])
    f_arg = ef("dxl_lm_arg", HERE / "dxl_lm_arg.cc", [y_ty, best_ty, i32, i32, i32])
    f_out = ef("dxl_lm_out", HERE / "dxl_lm_out.cc", [best_ty, y_ty])
    f_nr = ef("ln_nr", LINL / "ln_nr.cc", [u8_ln] * 4, LN_FLAGS)

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=3)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)

    def make_body(nb: int, base: int):
        def body(win, xin, yout, tab, best, fg, fp, fi, fa, fo):
            fi(best)
            for j in range_(LR):
                for i in range_(HID // 1024):
                    xe = xin.acquire(1)
                    fp(xe, tab, j, i, HID)
                    xin.release(1)
            for b in range_(nb):
                ye = yout.acquire(1)
                for c in range_(CPB):
                    we = win.acquire(1)
                    fg(we, tab, ye, 0, c, 0, HID, HID // 256)
                    win.release(1)
                if not NOARG:
                    fa(ye, best, base, b, SPEC.real_vocab)
                yout.release(1)
            ye = yout.acquire(1)
            fo(best, ye)
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
        workers.append(Worker(make_body(COUNTS[c], BASES[c]),
                              fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), Buffer(tab_ty, name=f"tab{c}"),
                                       Buffer(best_ty, name=f"best{c}"), f_gemv, f_prep, f_init, f_arg, f_out],
                              tile=Tile(c, 2), stack_size=0x1000))

    def sequence(a_pool, c_xres, c_normw, a_act, a_out, lni, lno, w_prods, x_prod, y_conss):
        pl, pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            nb = COUNTS[c]
            # [band][row][64] -> logits[row][band*64 ..], then the core's argmax element
            py.drain(y_conss[c], a_out, tap(OUT_FLOATS, BASES[c] * 64, [1, nb, LR, 64], [0, 64, VOCAB, 1]))
            py.drain(y_conss[c], a_out, bt(OUT_FLOATS, LR * VOCAB + c * YE, YE))
            pw.fill(w_prods[c], a_pool, bt(L0.LMHEAD_POOL_BYTES, BASES[c] * BB, nb * BB))
        for j in range(LR):
            pl.fill(lni, c_xres, bt(LR * HID, j * HID, HID))
            pl.fill(lni, c_normw, bt(HID, 0, HID))
            pl.drain(lno, a_act, bt(LR * HID * 2, j * HID * 2, ELN))
        pl.finish()
        px.fill(x_prod, a_act, bt(LR * HID * 2, 0, LR * HID * 2))
        px.finish()
        pw.finish()
        py.finish()

    rt = Runtime(sequence, [pool_ty, xres_ty, normw_ty, act_ty, out_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = lmhl
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [Path(__file__).read_bytes(), (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (LINL / "ln_nr.cc").read_bytes(), SPEC.spec_hash().encode(), str(LR).encode(),
                   repr(GL_FLAGS + [NOARG]).encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
