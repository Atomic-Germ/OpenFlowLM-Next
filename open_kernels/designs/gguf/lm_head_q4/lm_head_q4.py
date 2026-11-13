"""LM head on the NPU from pool-order q4_1_f32 chunks (GGUF-direct f32-scale build).

This design builds the f32-scale (GGUF-direct) twin of the lm_head_q4 design.
The quant is set to "q4_1_f32" in the ModelSpec, which causes the recipe
to emit the f32-scale pool layout (6144 B chunks) and the kernel to be
compiled with -DGEMV_Q4_SCALES_F32=1.

Dataflow: n_cores workers, each with its own shim weight stream (10 KB
elements = 2 chunks, double-buffered), x broadcast once as ONE element of K
bf16, one 64-float result per band. Bands split as evenly as possible over
the cores.

Build (WSL):  LMHEAD_N=<rows> LMHEAD_K=<hidden> python build_design.py
  designs/gguf/lm_head_q4/lm_head_q4.py [out]
  # To build with GGUF f32 scales: LMHEAD_SCALES_F32=1 python build_design.py
  designs/gguf/lm_head_q4/lm_head_q4.py [out]
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

# GGUF-direct pool chunks (f32 scales, 6144 B) vs the q4nx ones (bf16, 5120 B).
# The kernel is this design's own generated gemv_q4_gy, compiled with the f32-scale
# define and the gemv_q4s32 symbol prefix in the f32 build.
SCALES_F32 = int(os.environ.get("LMHEAD_SCALES_F32", 0))
TILE_BYTES = 6144 if SCALES_F32 else 5120
KERNEL_SYM = "gemv_q4s32_gy" if SCALES_F32 else "gemv_q4_gy"
KERNEL_FLAGS = ["-Os"] + (["-DGEMV_Q4_SCALES_F32=1", "-DGEMV_Q4_PREFIX=gemv_q4s32"] if SCALES_F32 else [])
BAND_ROWS = 64
PER_CALL = 2
CALL_BYTES = PER_CALL * TILE_BYTES


def _ensure_gy() -> Path:
    """The gemv_q4_gy band entry, generated here rather than borrowed from a
    sibling design's gen_kernels.py output. lm_head_q4 always streams PER_CALL
    chunks per weight element, so the wrapper is fixed and belongs to this
    design; it is a generated TU (git-ignored), not source.

    Borrowing was tried and does not work: designs/dense and designs/layer_x write
    theirs per spec and delete them again for a spec that does not need one, so the
    file is simply absent half the time - on 2026-09-08 Gemma's lm_head failed to
    build right after a Qwen3.5 export, for exactly that reason."""
    src = f'''#define GEMV_PER_CALL {PER_CALL}
#include "gemv_q4.h"
// A band into its y element: runtime band law (per_band chunks, row split rs).
// The entry name carries GEMV_Q4_PREFIX, so the f32-scale build
// (-DGEMV_Q4_PREFIX=gemv_q4s32) references gemv_q4s32_gy from its MLIR while both
// builds compile the one source.
#define GEMV_Q4_WRAP__(PFX, NAME) PFX##_g##NAME
#define GEMV_Q4_WRAP(PFX, NAME) GEMV_Q4_WRAP__(PFX, NAME)
extern "C" {{
void GEMV_Q4_WRAP(GEMV_Q4_PREFIX, y)(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y,
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