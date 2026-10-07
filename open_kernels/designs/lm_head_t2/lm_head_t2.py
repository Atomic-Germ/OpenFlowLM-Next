r"""lm_head on the NPU from 2-bit ternary chunks, for a rotated-basis model (Ternary Bonsai 2):
logits[N] = W'[N, K] @ (H(x) / 32 per 1024 block), N = vocab (N/64 bands of 64 rows), K = hidden.

W' is PrismML's own ternary head (OPEN-QUANT-T2): q4nx-build writes it as exact-ternary q4_1 in
the rotated basis, and the engine packs it into t2 chunks at load (pools.cpp `t2_perm`, std_perm
band order) at LMT2_CHUNK bytes each -- unpadded, since this design owns its pool. The input's
+-1 signs are folded into the final norm's gain, so the prep applies only H/32 (wht.h).

Dataflow: lm_head_q4's. n_cores workers, each with its own shim weight stream (elements of
LMHEAD_PER_CALL chunks, double-buffered), x broadcast once as ONE element of K bf16, one 64-float
result per band. Bands split as evenly as possible over the cores, so the taps are hand-built.
Kernels: lm_head_t2_prep.cc (the FWHT, then gemv_t2's per-128 table) and lm_head_t2.cc (one
entry per w element over gemv_t2.h's tile).

L1 per core at K = 5120, PER_CALL 5: x 10 KB + table 10.3 KB + w 2 x 10.6 KB + y 0.5 KB + the
FWHT scratch 4 KB + stack 4 KB = 50 KB of 64.

Build (WSL):  LMHEAD_N=248320 LMHEAD_K=5120 python build_design.py designs/lm_head_t2/lm_head_t2.py [out]
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import config

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"          # gemv_t2.h, wht.h, gemv_tab.h

TILE_BYTES = 2176        # the unpadded t2 chunk: 128 B of bf16 scales + 2048 B of codes
BAND_ROWS = 64
ROW_SPLIT = 2            # 32-row halves per 64-row band

N = int(os.environ.get("LMHEAD_N", 248320))
K = int(os.environ.get("LMHEAD_K", 5120))
N_CORES = int(os.environ.get("LMHEAD_CORES", 8))
PER_CALL = int(os.environ.get("LMHEAD_PER_CALL", 5))
NULL = int(os.environ.get("LMHEAD_NULL", 0))      # 1: DMA only (gemv_t2.h GEMV_NULL), for timing


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    inc.append(str(GEMV))
    return inc


def split_bands(bands: int, n_cores: int) -> list[int]:
    q, r = divmod(bands, n_cores)
    return [q + (1 if c < r else 0) for c in range(n_cores)]


def tab_bytes(k: int) -> int:
    return 2 * k + k // 32 + k // 32         # gemv_t2_tab_bytes: int16 xi | int32 s | bf16 hi | bf16 lo per 128


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lm_head_t2(w: In, x: In, y: Out, *, n: CompileTime[int], k: CompileTime[int], n_cores: CompileTime[int],
               per_call: CompileTime[int], null: CompileTime[int] = 0, srchash: CompileTime[int] = 0):
    assert n % BAND_ROWS == 0 and k % 1024 == 0
    per_band = ROW_SPLIT * k // 256              # chunks per 64-row band (40 at K = 5120)
    assert per_band % per_call == 0
    n_groups = per_band // per_call
    band_bytes = per_band * TILE_BYTES
    bands = n // BAND_ROWS
    counts = split_bands(bands, n_cores)

    elem_ty = np.ndarray[(per_call * TILE_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(k,), np.dtype[bfloat16]]
    tab_ty = np.ndarray[(tab_bytes(k),), np.dtype[np.uint8]]
    acc_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    w_ty = np.ndarray[(bands * band_bytes,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(n,), np.dtype[np.float32]]

    flags = [f"-DLMT2_K={k}", f"-DLMT2_PER_CALL={per_call}", f"-DGEMV_T2_CHUNK={TILE_BYTES}"]
    flags += ["-DGEMV_NULL"] if null else []
    kernel = ExternalFunction("lm_head_t2_group", source_file=str(HERE / "lm_head_t2.cc"),
                              arg_types=[elem_ty, tab_ty, acc_ty, np.int32],
                              include_dirs=_include_dirs(), compile_flags=flags)
    prep = ExternalFunction("lm_head_t2_prep", source_file=str(HERE / "lm_head_t2_prep.cc"),
                            arg_types=[x_ty, tab_ty], include_dirs=_include_dirs(), compile_flags=flags)

    of_w = [ObjectFifo(elem_ty, name=f"w{c}", depth=2) for c in range(n_cores)]
    of_y = [ObjectFifo(acc_ty, name=f"y{c}", depth=2) for c in range(n_cores)]
    of_x = ObjectFifo(x_ty, name="x", depth=1)

    def make_body(nb: int):
        def core_body(win, xin, yout, tab, fprep, fn):
            xe = xin.acquire(1)
            fprep(xe, tab)                       # H/32 in place, then the per-128 table
            for _ in range_(nb):
                ye = yout.acquire(1)
                for g in range_(n_groups):
                    we = win.acquire(1)
                    fn(we, tab, ye, g)
                    win.release(1)
                yout.release(1)
            xin.release(1)
        return core_body

    workers = [
        Worker(make_body(counts[c]),
               fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), Buffer(tab_ty, name=f"tab{c}"), prep, kernel],
               stack_size=0x1000)
        for c in range(n_cores)
    ]

    w_taps, y_taps = [], []
    off = 0
    for c in range(n_cores):
        nb = counts[c]
        w_taps.append(TensorAccessPattern((1, bands * band_bytes), off * band_bytes,
                                          [1, 1, 1, nb * band_bytes], [0, 0, 0, 1]))
        y_taps.append(TensorAccessPattern((1, n), off * BAND_ROWS,
                                          [1, 1, 1, nb * BAND_ROWS], [0, 0, 0, 1]))
        off += nb

    def sequence(a_w, a_x, c_y, w_prods, x_prod, y_conss):
        tg = TaskGroup()
        x_prod.fill(a_x, group=tg)
        for c in range(n_cores):
            w_prods[c].fill(a_w, tap=w_taps[c], group=tg)
            y_conss[c].drain(c_y, tap=y_taps[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [w_ty, x_ty, y_ty,
                            [f.prod() for f in of_w], of_x.prod(), [f.cons() for f in of_y]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = lm_head_t2
# The jit cache keys on this file + CompileTime args only: hash the kernel sources in.
_src = b"".join([(HERE / f).read_bytes() for f in ("lm_head_t2.cc", "lm_head_t2_prep.cc")]
                + [(GEMV / f).read_bytes() for f in ("gemv_t2.h", "wht.h", "gemv_tab.h")])
SPECIALIZE = {"n": N, "k": K, "n_cores": N_CORES, "per_call": PER_CALL, "null": NULL,
              "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
