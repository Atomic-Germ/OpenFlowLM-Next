"""Isolated Q4 projection using actual layer_x main-core buffers and prep/GEMV.

Compile through utilities/probe-qwen35-wide.py --scope projection. A synthetic
FFN permits resource probing before segmented FFN; this is not a layer build.
"""
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from ironutil import include_dirs
import xcommon as X

K = int(os.environ.get("PROBE_K", str(X.HID)))
BANDS_PER_CORE = 2
N = X.N_CORES * BANDS_PER_CORE * X.BAND_ROWS
XN_ELEMS = (K * 2 + X.ELEM - 1) // X.ELEM
W_BYTES = N * K // 8192 * X.TILE
if X.KIND != "dense" or X.Q8 or K % 256 or not 0 < K <= X.KWIDE:
    raise ValueError("projection probe requires dense Q4 and 256-aligned K within its table")


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def projection(pool: In, activation: In, result: Out, *, source_hash: CompileTime[int] = 0):
    t = X.types()
    inc = include_dirs() + [str(HERE), str(HERE.parent / "gemv_q4")]
    kernels = X.kernels(inc, t)
    wf = [ObjectFifo(t["elem"], name=f"w{c}", depth=2) for c in range(X.N_CORES)]
    xf = ObjectFifo(t["x"], name="x", depth=2)
    yf = [ObjectFifo(t["y"], name=f"y{c}", depth=2) for c in range(X.N_CORES)]

    def core_body(win, xin, yout, *args):
        buffers, functions = X.unpack_args(args)
        X.prep_bands(win, xin, yout, buffers, functions, K, XN_ELEMS, BANDS_PER_CORE, "linear")

    workers = [Worker(core_body,
                      fn_args=[wf[c].cons(), xf.cons(), yf[c].prod(),
                               *X.worker_args(X.core_buffers(t, c), kernels)],
                      tile=Tile(c, 2), stack_size=0x1800) for c in range(X.N_CORES)]
    wty = np.ndarray[(W_BYTES,), np.dtype[np.uint8]]
    xty = np.ndarray[(XN_ELEMS * X.ELEM // 2,), np.dtype[bfloat16]]
    yty = np.ndarray[(N,), np.dtype[np.float32]]

    def sequence(a_w, a_x, c_y, w_prods, x_prod, y_conss):
        def tap(total, off, size):
            return TensorAccessPattern((1, total), off, [1, 1, 1, size], [0, 0, 0, 1])
        group = TaskGroup()
        for c in range(X.N_CORES):
            y_conss[c].drain(c_y, tap=tap(N, c * BANDS_PER_CORE * X.BAND_ROWS,
                                         BANDS_PER_CORE * X.BAND_ROWS), wait=True, group=group)
        x_prod.fill(a_x, tap=tap(XN_ELEMS * X.ELEM // 2, 0, XN_ELEMS * X.ELEM // 2), wait=True, group=group)
        for c in range(X.N_CORES):
            w_prods[c].fill(a_w, tap=tap(W_BYTES, c * W_BYTES // X.N_CORES,
                                        W_BYTES // X.N_CORES), wait=True, group=group)
        group.finish()

    rt = Runtime(sequence, [wty, xty, yty,
                            [f.prod(tile=Tile(c, 0)) for c, f in enumerate(wf)],
                            xf.prod(tile=Tile(1, 0)),
                            [f.cons(tile=Tile(c, 0)) for c, f in enumerate(yf)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = projection
_sources = [Path(__file__), HERE / "xcommon.py", HERE / "gen_kernels.py", ROOT / "ironutil.py",
            *sorted(HERE.glob("*.cc")), *sorted(HERE.glob("*.h")),
            *sorted((HERE.parent / "gemv_q4").glob("*.h"))]
SPECIALIZE = {"source_hash": int(hashlib.sha256(b"".join(p.read_bytes() for p in _sources)
                                             + repr((K, N, X.C, X.FFN)).encode()).hexdigest()[:8], 16)}
