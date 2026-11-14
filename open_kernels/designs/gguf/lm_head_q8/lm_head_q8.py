"""LM head on the NPU from pool-order q8_f32 chunks (GGUF-direct f32-scale build).

This design builds the f32-scale (GGUF-direct) twin of the lm_head_q8 design.
The quant is set to "q4_1_f32" in the ModelSpec, which causes the recipe
to emit the f32-scale pool layout (9216 B chunks) and the kernel to be
compiled with -DLMHEAD_SCALES_F32=1.

Dataflow: n_cores workers, each with its own shim weight stream (elements of
PER_CALL chunks, double-buffered), x broadcast once, one 128-float result per
band. Bands split as evenly as possible over the cores.

Build (WSL):  [LMHEAD_N=<rows>] [LMHEAD_K=<hidden>] python build_design.py
  designs/gguf/lm_head_q8/lm_head_q8.py [out]
  # To build with GGUF f32 scales: LMHEAD_SCALES_F32=1 python build_design.py
  designs/gguf/lm_head_q8/lm_head_q8.py [out]
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