r"""gemv_t2n: Y[8, N] = X[8, K] @ W[N, K]^T on 2-bit ternary chunks, the DFlash 2 drafter's GEMV
(OPEN-DRAFT-DFLASH2). lm_head_t2's dataflow -- n_cores workers, each with its own shim weight stream
-- carrying eight rows through gemv_t2n.cc's quad tile, with the activation table built on the host
for one K slice at a time (eight rows of a full 5120-wide table do not fit L1).

A core works through the ITEMS t2n_plan.plan() gives it: a run of 64-row bands over one K slice.
Each item is one table element (header: nb, nkt; then the slice's records), nb x nkt w elements of
one k-tile of one band each (both 32-row halves, 2 x 2176 B), and nb y elements of 512 floats. The
core reads nb / nkt from the table element, so the core program is the same for every shape: one
xclbin and one hardware context for the whole drafter, one instruction stream per (N, K).

W is the t2 pool in std_perm band order (pools.cpp t2_perm at chunk_bytes 2176) -- a drafter weight
buffer or the target's own lmpool. Items over different K slices of the same band are partial sums;
the host adds them (dflash2.cpp).

L1 per core at T2N_KT_MAX 8: table 40 KB (one element) + w 2 x 4352 B + y 2 x 2 KB + stack 4 KB.

Build (WSL):  T2N_N=6144 T2N_K=5120 python build_design.py designs/gemv_t2n/gemv_t2n.py [out]
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import config

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from t2n_plan import CHUNK, Y_BAND, plan  # noqa: E402

N = int(os.environ.get("T2N_N", 6144))
K = int(os.environ.get("T2N_K", 5120))
N_CORES = int(os.environ.get("T2N_CORES", 8))
KT_MAX = int(os.environ.get("T2N_KT_MAX", 8))
NULL = int(os.environ.get("T2N_NULL", 0))       # 1: DMA only, for timing


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    return inc


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemv_t2n(w: In, tab: In, y: Out, *, n: CompileTime[int], k: CompileTime[int], n_cores: CompileTime[int],
             kt_max: CompileTime[int], null: CompileTime[int] = 0, srchash: CompileTime[int] = 0):
    p = plan(n, k, n_cores, kt_max)
    tab_words = p["tab_bytes"] // 4
    band_bytes = 2 * (k // 256) * CHUNK
    pool_bytes = (n // 64) * band_bytes

    elem_ty = np.ndarray[(2 * CHUNK,), np.dtype[np.uint8]]
    tab_ty = np.ndarray[(tab_words,), np.dtype[np.int32]]
    yb_ty = np.ndarray[(Y_BAND,), np.dtype[np.float32]]
    w_ty = np.ndarray[(pool_bytes,), np.dtype[np.uint8]]
    t_ty = np.ndarray[(p["n_tabs"] * tab_words,), np.dtype[np.int32]]
    y_ty = np.ndarray[(p["y_floats"],), np.dtype[np.float32]]

    flags = ["-DGEMV_NULL"] if null else []
    kernel = ExternalFunction("gemv_t2n_kt", source_file=str(HERE / "gemv_t2n.cc"),
                              arg_types=[elem_ty, tab_ty, yb_ty, np.int32],
                              include_dirs=_include_dirs(), compile_flags=flags)

    of_w = [ObjectFifo(elem_ty, name=f"w{c}", depth=2) for c in range(n_cores)]
    of_t = [ObjectFifo(tab_ty, name=f"t{c}", depth=1) for c in range(n_cores)]
    of_y = [ObjectFifo(yb_ty, name=f"y{c}", depth=2) for c in range(n_cores)]

    def core_body(win, tin, yout, fn):
        te = tin.acquire(1)
        nb = te[0]                               # read after the acquire: the fill has landed
        nkt = te[1]
        for _ in range_(nb):
            ye = yout.acquire(1)
            for kt in range_(nkt):
                we = win.acquire(1)
                fn(we, te, ye, kt)
                win.release(1)
            yout.release(1)
        tin.release(1)

    workers = [Worker(core_body, fn_args=[of_w[c].cons(), of_t[c].cons(), of_y[c].prod(), kernel],
                      stack_size=0x1000) for c in range(n_cores)]

    def sequence(a_w, a_t, a_y, w_prods, t_prods, y_conss):
        rounds = max(len(i) for i in p["items"])
        for r in range(rounds):
            tg = TaskGroup()
            for c in range(n_cores):
                if r >= len(p["items"][c]):
                    continue
                it = p["items"][c][r]
                t_prods[c].fill(a_t, tap=TensorAccessPattern((p["n_tabs"] * tab_words,), it["tab"] * tab_words,
                                                             [tab_words // 512, 512], [512, 1]), group=tg)
                w_prods[c].fill(a_w, tap=TensorAccessPattern((pool_bytes,), it["b0"] * band_bytes + it["kt0"] * 2 * CHUNK,
                                                             [it["nb"], 2 * it["nkt"], CHUNK], [band_bytes, CHUNK, 1]),
                                group=tg)
                y_conss[c].drain(a_y, tap=TensorAccessPattern((p["y_floats"],), it["y"], [it["nb"], Y_BAND], [Y_BAND, 1]),
                                 wait=True, group=tg)
            tg.finish()

    rt = Runtime(sequence, [w_ty, t_ty, y_ty, [f.prod() for f in of_w], [f.prod() for f in of_t],
                            [f.cons() for f in of_y]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = gemv_t2n
_src = b"".join((HERE / f).read_bytes() for f in ("gemv_t2n.cc", "t2n_plan.py"))
SPECIALIZE = {"n": N, "k": K, "n_cores": N_CORES, "kt_max": KT_MAX, "null": NULL,
              "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
