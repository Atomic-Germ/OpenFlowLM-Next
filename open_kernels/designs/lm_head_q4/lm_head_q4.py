r"""lm_head on the NPU from pool-order q4_1 chunks (a dense family's head):
logits[N] = W[N, K] @ x[K], N = vocab (N/64 bands of 64 rows), K = hidden.

Dataflow: n_cores workers, each with its own shim weight stream (10 KB
elements = 2 chunks, double-buffered), x broadcast once as ONE element of K
bf16, one 64-float result per band. Bands split as evenly as possible over the
cores (lm_head_q8's split: 2374 bands of the Qwen3-4B head are 297 x 6 + 296 x 2),
so the taps are hand-built. Kernel: gemv_q4_gy (generated in this directory) with the
runtime band law (K/128 chunks per band, row split 2) and gemv_q4_prep_rt.

Build (WSL):  LMHEAD_N=151936 LMHEAD_K=2560 python build_design.py designs/lm_head_q4/lm_head_q4.py [out]
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
GEMV = HERE.parent / "gemv_q4"

TILE_BYTES = 5120
BAND_ROWS = 64
PER_CALL = 2
CALL_BYTES = PER_CALL * TILE_BYTES


def _ensure_gy() -> Path:
    """The gemv_q4_gy band entry, generated here rather than borrowed from a
    sibling design's gen_kernels.py output. lm_head_q4 always streams PER_CALL
    chunks per weight element, so the wrapper is fixed and belongs to this
    design; it is a generated TU (git-ignored), not source."""
    src = f'''#define GEMV_PER_CALL {PER_CALL}
#include "gemv_q4.h"
// A band into its y element: runtime band law (per_band chunks, row split rs).
extern "C" {{
void gemv_q4_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
                int32_t group, int32_t per_band, int32_t rs) {{
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}}
}}
'''
    p = HERE / "gemv_q4_gy.cc"
    if not p.is_file() or p.read_text(encoding="utf-8") != src:
        p.write_text(src, encoding="utf-8", newline="\n")
    return p


GY = _ensure_gy()

N = int(os.environ.get("LMHEAD_N", 151936))
K = int(os.environ.get("LMHEAD_K", 2560))
N_CORES = int(os.environ.get("LMHEAD_CORES", 8))


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    inc.append(str(GEMV))
    inc.append(str(HERE.parent.parent / "include"))
    return inc


def split_bands(bands: int, n_cores: int) -> list[int]:
    q, r = divmod(bands, n_cores)
    return [q + (1 if c < r else 0) for c in range(n_cores)]


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lm_head_q4(w: In, x: In, y: Out, *, n: CompileTime[int], k: CompileTime[int],
               n_cores: CompileTime[int], srchash: CompileTime[int] = 0):
    assert n % BAND_ROWS == 0 and k % 256 == 0
    per_band = k // 128                       # chunks per 64-row band
    n_groups = per_band // PER_CALL
    assert per_band % PER_CALL == 0
    band_bytes = per_band * TILE_BYTES
    bands = n // BAND_ROWS
    counts = split_bands(bands, n_cores)
    tab_bytes = 2 * k + k // 8 + k // 8

    elem_ty = np.ndarray[(CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(k,), np.dtype[bfloat16]]
    tab_ty = np.ndarray[(tab_bytes,), np.dtype[np.uint8]]
    acc_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    w_ty = np.ndarray[(bands * band_bytes,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(n,), np.dtype[np.float32]]
    i32 = np.int32

    inc = _include_dirs()
    kernel = ExternalFunction("gemv_q4_gy", source_file=str(HERE / "gemv_q4_gy.cc"),
                              arg_types=[elem_ty, tab_ty, acc_ty, i32, i32, i32], include_dirs=inc, compile_flags=["-Os"])
    prep = ExternalFunction("gemv_q4_prep_rt", source_file=str(GEMV / "gemv_q4_prep_rt.cc"),
                            arg_types=[x_ty, tab_ty, i32, i32, i32], include_dirs=inc, compile_flags=["-Os"])

    of_w = [ObjectFifo(elem_ty, name=f"w{c}", depth=2) for c in range(n_cores)]
    of_y = [ObjectFifo(acc_ty, name=f"y{c}", depth=2) for c in range(n_cores)]
    of_x = ObjectFifo(x_ty, name="x", depth=1)

    def make_body(nb: int):
        def core_body(win, xin, yout, tab, fprep, fn):
            xe = xin.acquire(1)
            fprep(xe, tab, k, 0, k // 32)
            for _ in range_(nb):
                ye = yout.acquire(1)
                for g in range_(n_groups):
                    we = win.acquire(1)
                    fn(we, tab, ye, g, per_band, 2)
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


DESIGN = lm_head_q4
_src = b"".join([(GEMV / f).read_bytes() for f in ("gemv_q4.h", "gemv_tab.h", "gemv_q4_prep_rt.cc")]
                + [(HERE / "gemv_q4_gy.cc").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"n": N, "k": K, "n_cores": N_CORES, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
