r"""lm_head on the NPU from phlegm's pool-order q8 chunks:
logits[N] = W[N, K] @ x[K], N = 248320 and K = 2048 for the 27B (1940 bands of
128 rows). K is LMHEAD_K (a compile-time knob of lm_head_q8.h too, since it sizes
the activation table the kernel indexes); Qwen3.5 dense builds it at K = 4096.

Dataflow: n_cores workers, each with its own shim weight stream (elements of
PER_CALL chunks, double-buffered), x broadcast once, one 128-float result per
band. Bands split as evenly as possible over the cores (1940 is not a multiple
of 8), so the taps are hand-built rather than simple_tiler's equal tiles.

Build (WSL):  [LMHEAD_N=<rows>] [LMHEAD_K=<hidden>] python build_design.py designs/lm_head_q8/lm_head_q8.py [out]
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
GEMV = HERE.parent / "gemv_q4"          # gemv_tab.h + the activation prep entry point

# GGUF-direct pool chunks (f32 scales, 9216 B) vs the q4nx ones (bf16, 8704 B);
# see lm_head_q8.h. The entry point is lm_head_q8s32_group in the f32 build.
SCALES_F32 = int(os.environ.get("LMHEAD_SCALES_F32", 0))
TILE_BYTES = 9216 if SCALES_F32 else 8704
KERNEL_SYM = "lm_head_q8s32_group" if SCALES_F32 else "lm_head_q8_group"
EXTRA_FLAGS = ["-DLMHEAD_SCALES_F32=1", "-DLMHEAD_Q8_PREFIX=lm_head_q8s32"] if SCALES_F32 else []
BAND_ROWS = 128
ROW_SPLIT = 4           # 32-row quarters per 128-row band (lm_head_q8.h's kRowSplit)
K = int(os.environ.get("LMHEAD_K", 2048))      # the hidden width; 2048 for the 27B, 4096 for Qwen3.5 dense
PER_BAND = ROW_SPLIT * (K // 256)              # chunks per band: K/256 k-tiles x 4 row quarters (32 at K=2048)
BAND_BYTES = PER_BAND * TILE_BYTES             # 278528 at K = 2048

N = int(os.environ.get("LMHEAD_N", 248320))
N_CORES = int(os.environ.get("LMHEAD_CORES", 8))
PER_CALL = 2            # 2 x 8704 = 17408 B per element, x2 buffered -> 34 KB of L1


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


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lm_head_q8(w: In, x: In, y: Out, *, n: CompileTime[int],
               n_cores: CompileTime[int], per_call: CompileTime[int], srchash: CompileTime[int] = 0):
    assert n % BAND_ROWS == 0 and PER_BAND % per_call == 0
    bands = n // BAND_ROWS
    n_groups = PER_BAND // per_call
    call_bytes = per_call * TILE_BYTES
    counts = split_bands(bands, n_cores)

    elem_ty = np.ndarray[(call_bytes,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(K,), np.dtype[bfloat16]]
    tab_ty = np.ndarray[(2 * K + K // 4,), np.dtype[np.uint8]]      # gemv_q4_tab_bytes(2048)
    acc_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    w_ty = np.ndarray[(bands * BAND_BYTES,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(n,), np.dtype[np.float32]]

    kernel = ExternalFunction(
        KERNEL_SYM,
        source_file=str(HERE / "lm_head_q8.cc"),
        arg_types=[elem_ty, tab_ty, acc_ty, np.int32],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DLMHEAD_PER_CALL={per_call}", f"-DLMHEAD_K={K}"] + EXTRA_FLAGS,
    )
    # the prep TU is generated on demand (gemv_q4.ensure_prep_entry rewrites it only when the
    # text differs, so the K values already checked in stay byte-identical)
    import importlib.util
    _spec = importlib.util.spec_from_file_location("_gemv_q4_gen", GEMV / "gemv_q4.py")
    _gemv = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gemv)
    prep = ExternalFunction(f"gemv_q4_prep_k{K}", source_file=str(_gemv.ensure_prep_entry(K)),
                            arg_types=[x_ty, tab_ty], include_dirs=_include_dirs())

    of_w = [ObjectFifo(elem_ty, name=f"w{c}", depth=2) for c in range(n_cores)]
    of_y = [ObjectFifo(acc_ty, name=f"y{c}", depth=2) for c in range(n_cores)]
    of_x = ObjectFifo(x_ty, name="x", depth=1)

    def make_body(nb: int):
        def core_body(win, xin, yout, tab, fprep, fn):
            xe = xin.acquire(1)
            fprep(xe, tab)                       # block-quantise x once (gemv_tab.h)
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

    # Hand-built flat taps (what simple_tiler emits for a (1, L) tensor is a
    # 4-D pattern sizes [1,1,1,len] strides [0,0,0,1]); uneven lengths per core.
    w_taps, y_taps = [], []
    off = 0
    for c in range(n_cores):
        nb = counts[c]
        w_taps.append(TensorAccessPattern((1, bands * BAND_BYTES), off * BAND_BYTES,
                                          [1, 1, 1, nb * BAND_BYTES], [0, 0, 0, 1]))
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


DESIGN = lm_head_q8
_src = (b"".join((HERE / f).read_bytes() for f in ("lm_head_q8.h", "lm_head_q8.cc"))
        + (GEMV / "gemv_tab.h").read_bytes() + f"K={K}".encode())
SPECIALIZE = {"n": N, "n_cores": N_CORES, "per_call": PER_CALL,
              "srchash": int(hashlib.sha1(_src + str(SCALES_F32).encode()).hexdigest()[:8], 16)}
