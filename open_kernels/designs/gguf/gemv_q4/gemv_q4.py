"""GEMV with q4_1_f32 chunks (GGUF-direct f32-scale build).

This design provides the GEMV kernel for GGUF-direct f32-scale pool chunks.
The quant is set to "q4_1_f32" in the ModelSpec, which causes the recipe
to use the f32-scale pool layout (6144 B chunks) and the f32-scale kernels.

The design provides both the bf16 (q4nx) and f32 (GGUF-direct) kernel versions.
When LMHEAD_SCALES_F32=1 is set, the f32-scale kernels are used.

Build (WSL):  python build_design.py designs/gguf/gemv_q4/gemv_q4.py [out]
  # To build with GGUF f32 scales: LMHEAD_SCALES_F32=1 python build_design.py
  designs/gguf/gemv_q4/gemv_q4.py [out]
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

# GGUF-direct pool chunks (f32 scales, 6144 B) vs the q4nx ones (bf16, 5120 B).
# The kernel is this design's own gemv_q4s32_* entry points, compiled with the
# f32-scale define and the gemv_q4s32 symbol prefix.
SCALES_F32 = int(os.environ.get("LMHEAD_SCALES_F32", 0))
TILE_BYTES = 6144 if SCALES_F32 else 5120
KERNEL_SYM = "gemv_q4s32" if SCALES_F32 else "gemv_q4"
KERNEL_FLAGS = (["-DGEMV_Q4_SCALES_F32=1", "-DGEMV_Q4_PREFIX=gemv_q4s32"]
                if SCALES_F32 else [])
BAND_ROWS = 64
PER_CALL = 2
CALL_BYTES = PER_CALL * TILE_BYTES


def kernel_rne(f32: np.ndarray) -> np.ndarray:
    """The kernel's f32 -> bf16 convert, as bf16 bit patterns.

    ``load_scale`` uses the AIE native RNE convert; the integer form below is
    valid for both the bf16 and f32-scale builds since fp16 -> f32 is exact
    and the RNE round-trip is identical.
    """
    from ml_dtypes import bfloat16 as _bf
    return np.array([_bf(f32i).astype(np.uint16) for f32i in f32])


def _codes_to_nibbles(q: np.ndarray, rows0: int, cols0: int) -> np.ndarray:
    """[32 r, 256 k] codes -> the chunk's 4096 interleaved nibble bytes."""
    qq = q[rows0:rows0 + 32, cols0:cols0 + 256].reshape(2, 16, 256) \
        .transpose(0, 2, 1).reshape(-1)                      # [rb, k, r16], r16 fastest
    return qq[0::2] | (qq[1::2] << 4)


def _codes_to_int8(v: np.ndarray, rows0: int, cols0: int) -> np.ndarray:
    """[32 r, 256 k] int8 codes -> the chunk's 8192 interleaved bytes."""
    return v[rows0:rows0 + 32, cols0:cols0 + 256].reshape(2, 16, 256) \
        .transpose(0, 2, 1).reshape(-1)


def q4_chunk_bytes(raw: np.ndarray, r0: int, c0: int, gguf_type: str = "Q4_1") -> np.ndarray:
    """One 6144 B f32-scale q4 chunk covering rows r0..r0+31, cols c0..c0+255."""
    from .gguf_pool import unpack_q4, CH_Q4, Q4_1_BYTES
    q = unpack_q4(raw, gguf_type)[2]
    out = np.empty(CH_Q4, np.uint8)
    out[0:1024] = (_widened_f32(raw, r0, c0, slice(0, 2))
                   if hasattr(_, '__wrapped__') else _widened_f32(raw, r0, c0, slice(0, 2)))
    if gguf_type == "Q4_1":
        out[1024:2048] = _chunk_m32(raw, r0, c0)
    else:
        out[1024:2048] = np.ascontiguousarray(-8.0 * out[0:1024].view(np.float32)).view(np.uint8)
    out[2048:] = _codes_to_nibbles(q, r0, c0)
    return out


def _widened_f32(raw, rows0, cols0, sl):
    """GGUF fp16 scale bytes for one chunk, widened EXACTLY to f32 and
    permuted to the kernel's j = kb*32 + r. sl = scale byte slice in a block."""
    nb = raw.shape[1]
    kb0 = cols0 // 32
    f16 = raw[rows0:rows0 + 32, kb0:kb0 + 8, sl] \
        .reshape(32, 8, 2).view(np.float16)[..., 0]
    f32 = f16.astype(np.float32)                     # exact widening
    return np.ascontiguousarray(f32.T).reshape(-1).view(np.uint8)  # [8 kb, 32 r]


def _chunk_d32(raw, rows0, cols0):
    return _widened_f32(raw, rows0, cols0, slice(0, 2))


def _chunk_m32(raw, rows0, cols0):
    return _widened_f32(raw, rows0, cols0, slice(2, 4))


def pack_q4_pool(raw: np.ndarray, rs: int, gguf_type: str = "Q4_1") -> np.ndarray:
    """GGUF Q4 raw blocks uint8[n, nb, B] -> pool chunk bytes uint8[nch*6144].

    Scales widen EXACTLY fp16 -> f32; Q4_0 gets the exact ``-8*d`` min.
    Codes are permuted only. No requantization.
    """
    from .gguf_pool import chunk_geometry, pack_q4_pool as _base_pack_q4_pool
    assert raw.shape[-1] == (Q4_1_BYTES if gguf_type == "Q4_1" else Q4_0_BYTES)
    n, nb, _ = raw.shape
    k = nb * 32
    _, _, rows0, cols0 = chunk_geometry(n, k, rs)
    nch = len(rows0)
    q = unpack_q4(raw, gguf_type)[2]
    out = np.empty((nch, CH_Q4), np.uint8)
    for c in range(nch):
        r0, c0 = int(rows0[c]), int(cols0[c])
        out[c, 0:1024] = _chunk_d32(raw, r0, c0)
        if gguf_type == "Q4_1":
            out[c, 1024:2048] = _chunk_m32(raw, r0, c0)
        else:
            d = out[c, 0:1024].view(np.float32)
            out[c, 1024:2048] = np.ascontiguousarray(-8.0 * d).view(np.uint8)
        out[c, 2048:] = _codes_to_nibbles(q, r0, c0)
    return out.reshape(-1)


def pack_q8_pool(raw: np.ndarray, rs: int) -> np.ndarray:
    """GGUF Q8_0 raw blocks uint8[n, nb, 34] -> pool chunk bytes uint8[nch*9216]
    in the q4 band law (expert / dense-GEMV shapes use this when a q8 tensor
    rides the standard band order)."""
    from .gguf_pool import chunk_geometry
    n, nb, _ = raw.shape
    k = nb * 32
    _, _, rows0, cols0 = chunk_geometry(n, k, rs)
    nch = len(rows0)
    out = np.empty((nch, 9216), np.uint8)
    for c in range(nch):
        out[c] = q8_chunk_bytes(raw, int(rows0[c]), int(cols0[c]))
    return out.reshape(-1)


def q8_chunk_bytes(raw: np.ndarray, r0: int, c0: int) -> np.ndarray:
    """One 9216 B f32-scale q8 chunk covering rows r0..r0+31, cols c0..c0+255."""
    from .gguf_pool import unpack_q8_0
    v = unpack_q8_0(raw)[1]
    out = np.empty(9216, np.uint8)
    out[0:1024] = _chunk_d32(raw, r0, c0)
    out[1024:] = _codes_to_int8(v, r0, c0)
    return out


def _ensure_gy() -> Path:
    """The gemv_q4_gy band entry, generated here rather than borrowed from a
    sibling design's gen_kernels.py output. gemv_q4 always streams PER_CALL
    chunks per weight element, so the wrapper is fixed and belongs to this
    design; it is a generated TU (git-ignored), not source.

    Borrowing was tried and does not work: designs/dense and designs/layer_x write
    theirs per spec and delete them again for a spec that does not need one, so the
    file is simply absent half the time."""
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