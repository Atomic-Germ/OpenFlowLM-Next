r"""glj: gl driven by a job table, a probe of dxl's job-driven main cores."""

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
STATIC = int(os.environ.get("GL_STATIC", 0))      # 1: constant loop bounds (jp still read by the kernels)
KOLD = os.environ.get("GL_KOLD", "")             # probe: this piece through the pre-job kernel

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
def glj(w: In, x: In, y: Out, *, srchash: CompileTime[int] = 0):
    elem = np.ndarray[(CHUNK,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(XE // 2,), np.dtype[bfloat16]]
    tab_ty = np.ndarray[(L * TABB,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(YE,), np.dtype[np.float32]]
    w_ty = np.ndarray[(BANDS * BB,), np.dtype[np.uint8]]
    xx_ty = np.ndarray[(L * K * (4 if F32 else 2) // 2,), np.dtype[bfloat16]]
    yy_ty = np.ndarray[(L * N,), np.dtype[np.float32]]
    i32 = np.int32

    flags = [f"-DDXL_L={L}", "-DDXL_JMAX=32"] + (["-DGEMV_NULL"] if NULL else [])
    inc = include_dirs() + [str(HERE), str(HERE.parent / "gemv_q4"), str(HERE.parent.parent / "include")]
    JMAX, NFLD = 32, 12
    PAD = os.environ.get("GL_PAD") == "1"            # probe: every new buffer a whole number of 64 B
    table_ty = np.ndarray[(784 if PAD else 2 * (1 + JMAX * NFLD),), np.dtype[np.int32]]
    rtp_ty = np.ndarray[(16 if PAD else 4,), np.dtype[np.int32]]
    q4_ty = np.ndarray[(16 if PAD else 4,), np.dtype[np.int32]]
    jp_ty = np.ndarray[(16,), np.dtype[np.int32]]
    acc_ty = np.ndarray[(BT * YE,), np.dtype[np.float32]]
    tbl = np.zeros((2, 1 + JMAX * NFLD), np.int32)
    tblf = np.zeros(784 if PAD else tbl.size, np.int32)
    tbl[0, 0] = T
    for t in range(T):
        tbl[0, 1 + t * NFLD:1 + (t + 1) * NFLD] = [S, BT, CPS, BT, KS, 1 if F32 else 0, 0, 0, 0, KT, 0, 0]

    def ef(sym, args):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=flags)
    f_nj = ef("dxl_njobs", [table_ty, rtp_ty, q4_ty])
    f_job = ef("dxl_job", [table_ty, rtp_ty, i32, jp_ty])
    f_prep = ef("dxl_prep_job", [x_ty, tab_ty, i32, jp_ty])
    f_gemv = ef("dxl_gemv_job", [elem, tab_ty, acc_ty, jp_ty, i32, i32, i32])
    f_out = ef("dxl_out_job", [acc_ty, y_ty, jp_ty, i32])
    o_gemv = ef("dxl_gemv_at", [elem, tab_ty, acc_ty, i32, i32, i32, i32, i32])
    o_prep = ef("dxl_prep_f32" if F32 else "dxl_prep", [x_ty, tab_ty, i32, i32])
    o_out = ef("dxl_out", [acc_ty, y_ty, i32])

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)

    NOBUF = os.environ.get("GL_NOBUF") == "1"
    NOINIT = os.environ.get("GL_NOBUF") == "noinit"

    def core_nobuf(win, xin, yout, tab, acc, og, op, oo):
        for j in range_(T):
            for s in range_(S):
                for tok in range_(L):
                    xe = xin.acquire(1)
                    op(xe, tab, tok, KS)
                    xin.release(1)
                for b in range_(BT):
                    for c in range_(CPS):
                        we = win.acquire(1)
                        og(we, tab, acc, b, c, s, KS, KT)
                        win.release(1)
            for b in range_(BT):
                ye = yout.acquire(1)
                oo(acc, ye, b)
                yout.release(1)

    def core_body(win, xin, yout, tab, acc, table, rtp, nj, jp, fnj, fjob, fprep, fgemv, fout, og, op, oo):
        if KOLD != "all":
            fnj(table, rtp, nj)
        for j in range_(T if KOLD == "all" else nj[0]):
            if KOLD != "all":
                fjob(table, rtp, j, jp)
            for s in range_(S if STATIC else jp[0]):
                for tok in range_(L):
                    xe = xin.acquire(1)
                    if KOLD in ("prep", "all"):
                        op(xe, tab, tok, KS)
                    else:
                        fprep(xe, tab, tok, jp)
                    xin.release(1)
                for b in range_(BT if STATIC else jp[1]):
                    for c in range_(CPS if STATIC else jp[2]):
                        we = win.acquire(1)
                        if KOLD in ("gemv", "all"):
                            og(we, tab, acc, b, c, s, KS, KT)
                        else:
                            fgemv(we, tab, acc, jp, b, c, s)
                        win.release(1)
            for b in range_(BT if STATIC else jp[3]):
                ye = yout.acquire(1)
                if KOLD in ("out", "all"):
                    oo(acc, ye, b)
                else:
                    fout(acc, ye, jp, b)
                yout.release(1)

    def core_bufonly(win, xin, yout, tab, acc, table, rtp, nj, jp, og, op, oo):
        core_nobuf(win, xin, yout, tab, acc, og, op, oo)

    def core_konly(win, xin, yout, tab, acc, og, op, oo, fnj, fjob, fprep, fgemv, fout):
        core_nobuf(win, xin, yout, tab, acc, og, op, oo)

    BUFONLY = os.environ.get("GL_NOBUF") == "bufonly"
    KONLY = os.environ.get("GL_NOBUF") == "konly"
    if NOBUF:
        workers = [Worker(core_nobuf, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                               Buffer(tab_ty, name=f"tab{c}"), Buffer(acc_ty, name=f"acc{c}"),
                                               o_gemv, o_prep, o_out],
                          tile=Tile(c, 2), stack_size=0x1800) for c in range(N_CORES)]
    elif BUFONLY:
        workers = [Worker(core_bufonly, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                 Buffer(tab_ty, name=f"tab{c}"), Buffer(acc_ty, name=f"acc{c}"),
                                                 Buffer(table_ty, name=f"jobs{c}"), Buffer(rtp_ty, name=f"rtp{c}"),
                                                 Buffer(q4_ty, name=f"nj{c}"), Buffer(jp_ty, name=f"jp{c}"),
                                                 o_gemv, o_prep, o_out],
                          tile=Tile(c, 2), stack_size=0x1800) for c in range(N_CORES)]
    elif KONLY:
        workers = [Worker(core_konly, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                               Buffer(tab_ty, name=f"tab{c}"), Buffer(acc_ty, name=f"acc{c}"),
                                               o_gemv, o_prep, o_out, f_nj, f_job, f_prep, f_gemv, f_out],
                          tile=Tile(c, 2), stack_size=0x1800) for c in range(N_CORES)]
    else:
        workers = [Worker(core_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                          Buffer(tab_ty, name=f"tab{c}"), Buffer(acc_ty, name=f"acc{c}"),
                                          (Buffer(table_ty, name=f"jobs{c}") if NOINIT else
                                           Buffer(table_ty, name=f"jobs{c}", initial_value=np.concatenate([tbl.reshape(-1), tblf[tbl.size:]]))),
                                          (Buffer(rtp_ty, name=f"rtp{c}") if NOINIT else
                                           Buffer(rtp_ty, name=f"rtp{c}", initial_value=np.zeros(16 if PAD else 4, np.int32))),
                                          Buffer(q4_ty, name=f"nj{c}"), Buffer(jp_ty, name=f"jp{c}"),
                                          f_nj, f_job, f_prep, f_gemv, f_out, o_gemv, o_prep, o_out],
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


DESIGN = glj
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE.parent / "gemv_q4" / "gemv_q4.h").read_bytes(),
                   (HERE.parent / "gemv_q4" / "gemv_tab.h").read_bytes(),
                   repr((N, K, L, N_CORES, BT, KS, F32, NULL, STATIC, KOLD, os.environ.get('GL_NOBUF', ''), os.environ.get('GL_PAD', ''), 'glj')).encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
