r"""gl: the L-row GEMV alone, Y[L, N] = X[L, K] @ W[N, K]^T over dx's pool-order q4_1 bands, sized by the GL_* env."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

CHUNK = 5120
N = int(os.environ.get("GL_N", 6144))
K = int(os.environ.get("GL_K", 4096))
L = int(os.environ.get("GL_L", 4))
N_CORES = int(os.environ.get("GL_CORES", 8))
BT = int(os.environ.get("GL_BT", 6))
F32 = int(os.environ.get("GL_F32", 0))
KS = int(os.environ.get("GL_KS", 512 if F32 else 1024))
NULL = int(os.environ.get("GL_NULL", 0))

BANDS = N // 64
BPC = BANDS // N_CORES
assert N % 64 == 0 and BANDS % N_CORES == 0, "whole bands per core"
assert BPC % BT == 0, f"{BPC} bands per core do not split into tiles of {BT}"
assert K % KS == 0 and KS % 256 == 0
T, S = BPC // BT, K // KS
CPS = 2 * KS // 256                       # chunks per band per slice (two 32-row halves per k-tile)
KT = K // 256
BB = 2 * KT * CHUNK                       # band bytes
SLICE_BYTES = CPS * CHUNK
XE = KS * (4 if F32 else 2)               # one token's slice in the x stream
TABB = 2 * KS + KS // 4                   # gemv_q4_tab_bytes(KS)
YE = L * 64                               # one band's [L][64] floats
assert SLICE_BYTES % 2048 == 0 and XE <= 4092 * 4


def tap(total, off, sizes, strides):
    return TensorAccessPattern((1, total), off, sizes, strides)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gl(w: In, x: In, y: Out, *, srchash: CompileTime[int] = 0):
    elem = np.ndarray[(CHUNK,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(XE // 2,), np.dtype[bfloat16]]
    tab_ty = np.ndarray[(L * TABB,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(YE,), np.dtype[np.float32]]
    w_ty = np.ndarray[(BANDS * BB,), np.dtype[np.uint8]]
    xx_ty = np.ndarray[(L * K * (4 if F32 else 2) // 2,), np.dtype[bfloat16]]
    yy_ty = np.ndarray[(L * N,), np.dtype[np.float32]]
    i32 = np.int32

    flags = [f"-DDXL_L={L}"] + (["-DGEMV_NULL"] if NULL else [])
    inc = include_dirs() + [str(HERE), str(HERE.parent / "gemv_q4")]
    f_gemv = ExternalFunction("dxl_gemv", source_file=str(HERE / "dxl_gemv.cc"),
                              arg_types=[elem, tab_ty, y_ty, i32, i32, i32, i32], include_dirs=inc,
                              compile_flags=flags)
    prep_name = "dxl_prep_f32" if F32 else "dxl_prep"
    f_prep = ExternalFunction(prep_name, source_file=str(HERE / f"{prep_name}.cc"),
                              arg_types=[x_ty, tab_ty, i32, i32], include_dirs=inc, compile_flags=flags)

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=BT) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)

    def core_body(win, xin, yout, tab, fg, fp):
        for _ in range_(T):
            ys = yout.acquire(BT)
            ys = [ys] if BT == 1 else ys
            for s in range_(S):
                for j in range(L):
                    xe = xin.acquire(1)
                    fp(xe, tab, j, KS)
                    xin.release(1)
                for b in range(BT):
                    for c in range_(CPS):
                        we = win.acquire(1)
                        fg(we, tab, ys[b], c, s, KS, KT)
                        win.release(1)
            yout.release(BT)

    workers = [Worker(core_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                          Buffer(tab_ty, name=f"tab{c}"), f_gemv, f_prep],
                      tile=Tile(c, 2), stack_size=0x1800) for c in range(N_CORES)]

    def sequence(a_w, a_x, a_y, w_prods, x_prod, y_conss):
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            # [tile][band][token][64] -> y[token][band*64 ..]
            py.drain(y_conss[c], a_y, tap(L * N, c * BPC * 64, [T, BT, L, 64], [BT * 64, 64, N, 1]))
        xs = XE // 2                                        # bf16 units of one token's slice
        xrow = K * (4 if F32 else 2) // 2
        for t in range(T):
            px.fill(x_prod, a_x, tap(L * xrow, 0, [1, S, L, xs], [0, xs, xrow, 1]))
            for c in range(N_CORES):
                # tile t of core c: for s: for b: band (c*BPC + t*BT + b) slice s, as rows of 2 KB
                b0 = (c * BPC + t * BT) * BB
                pw.fill(w_prods[c], a_w, tap(BANDS * BB, b0, [S, BT, SLICE_BYTES // 2048, 2048],
                                             [SLICE_BYTES, BB, 2048, 1]))
        px.finish()
        pw.finish()
        py.finish()

    rt = Runtime(sequence, [w_ty, xx_ty, yy_ty, [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(0, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = gl
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE.parent / "gemv_q4" / "gemv_q4.h").read_bytes(),
                   (HERE.parent / "gemv_q4" / "gemv_tab.h").read_bytes(),
                   repr((N, K, L, N_CORES, BT, KS, F32, NULL)).encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
