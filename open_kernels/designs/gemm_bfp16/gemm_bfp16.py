r"""gemm_bfp16: whole-array bfp16 x bfp16 -> fp32 GEMM, C[M, N] = A[M, K] @ B[N, K]^T, one xclbin for every shape.

Derived from ggml-xdna's kernels/bfp16_gemm/designs/whole_array_bfp_rtp.py (Cyrus Attoun,
https://github.com/Cyronius/ggml-xdna), itself derived from mlir-aie v1.4.2's block_datatypes
whole_array example (Apache-2.0 WITH LLVM-exception, Copyright (C) Advanced Micro Devices, Inc.).
Dropped from the ggml-xdna design: the bf16 output modes (--out-mode) and the b_groups / b_kfull
attention layouts. The shipped dataflow is otherwise theirs: A and B pre-tiled on the host so every
shim transfer is one contiguous run, K walked in chunks of k with the fp32 sums kept in the output
tile, C written as each column produces it (--c-tiled).

A is the activations (one token a row), B the weights (one output channel a row, K along the row),
the layout of a GGUF tensor. Both are bfp16ebs8: blocks of eight K values sharing an exponent byte,
nine bytes a block (recipes/pack.py pack_bfp16). The K chunk count reaches the cores as a runtime
parameter, so the core program does not depend on K, nor on M or N beyond the tile counts.

Shape constraints: M % (m * 4) == 0, N % (n * 8) == 0, K % k == 0.

    GB_M=256 GB_K=2560 GB_N=4096 python open_kernels/build_design.py open_kernels/designs/gemm_bfp16/gemm_bfp16.py <out>
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import (Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, StreamDims,
                      TaskGroup, Worker)
from aie.iron.controlflow import range_
from aie.utils import config as _aie_config

HERE = Path(__file__).resolve().parent
_KERNEL_SRC = HERE / "mm_bfp_acc.cc"
_AIE_KERNELS_INC = Path(_aie_config.cxx_header_path()) / "aie_kernels"

N_ROWS = 4
N_COLS = 8


def check_shape(M: int, K: int, N: int, m: int, k: int, n: int) -> str | None:
    if M % (m * N_ROWS) or N % (n * N_COLS) or K % k or K // k < 2:
        return f"gemm_bfp16 needs M % {m * N_ROWS}, N % {n * N_COLS}, K % {k} with 2+ chunks (got {M}x{K}x{N})"
    return None


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def gemm_bfp16(
    A: In, B: In, C: Out, *,
    gm: CompileTime[int], gk: CompileTime[int], gn: CompileTime[int],
    tm: CompileTime[int] = 64, tk: CompileTime[int] = 64, tn: CompileTime[int] = 128,
):
    why = check_shape(gm, gk, gn, tm, tk, tn)
    assert why is None, why
    n_aie_rows, n_aie_cols = N_ROWS, N_COLS
    fifo_depth = 2
    r = t = 8  # the 8x8x8 bfp16 mac

    A_l2_ty = np.ndarray[(tm * tk // 8,), np.dtype[v8bfp16ebs8]]
    B_l2_ty = np.ndarray[(tk * tn // 8,), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(tm * tn * n_aie_rows,), np.dtype[np.float32]]
    A_l1_ty = np.ndarray[(tm, tk // 8), np.dtype[v8bfp16ebs8]]
    B_l1_ty = np.ndarray[(tk, tn // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(tm, tn), np.dtype[np.float32]]

    flags = [f"-DDIM_M={tm}", f"-DDIM_K={tk}", f"-DDIM_N={tn}", f"-I{_AIE_KERNELS_INC}"]
    k_first = ExternalFunction("mm_f32_first", source_file=str(_KERNEL_SRC), arg_types=[A_l1_ty, B_l1_ty, C_l1_ty],
                               compile_flags=flags + ["-DF32_FIRST"])
    k_mid = ExternalFunction("mm_f32_mid", source_file=str(_KERNEL_SRC), arg_types=[A_l1_ty, B_l1_ty, C_l1_ty],
                             compile_flags=flags + ["-DF32_MID"])

    A_l3l2, A_l2l1, B_l3l2, B_l2l1, C_l2l3 = [], [], [], [], []
    C_l1l2 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    for i in range(n_aie_rows):
        a = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        A_l3l2.append(a)
        A_l2l1.append(a.cons().split([0], obj_types=[A_l1_ty], names=[f"A_L2L1_{i}"])[0])
    for col in range(n_aie_cols):
        b = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2.append(b)
        B_l2l1.append(b.cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}"))
        # 8x8 blocks -> row-major on the way to DDR
        c = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth,
                       dims_to_stream=[(tm // r, r * tn), (r, t), (tn // t, r * t), (t, 1)])
        C_l2l3.append(c)
        tmp = c.prod().join([tm * tn * i for i in range(n_aie_rows)], obj_types=[C_l1_ty] * n_aie_rows,
                            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)], depths=[1] * n_aie_rows)
        for row in range(n_aie_rows):
            C_l1l2[row][col] = tmp[row]

    # gk/tk - 1 arrives per dispatch from the instruction stream. The initial value is zeros so no shape's image differs
    # from another's; the core reads it after acquiring the tile's first chunk, which cannot arrive before the write.
    rtps = [[Buffer(np.ndarray[(2,), np.dtype[np.int32]], name=f"nmid_{row}_{col}",
                    initial_value=np.zeros(2, dtype=np.int32), use_write_rtp=True)
             for col in range(n_aie_cols)] for row in range(n_aie_rows)]

    def core_fn(in_a, in_b, out_c, rtp, first, mid):
        elem_out = out_c.acquire(1)
        elem_a = in_a.acquire(1)
        elem_b = in_b.acquire(1)
        n_mid = rtp[0]
        first(elem_a, elem_b, elem_out)
        in_a.release(1)
        in_b.release(1)
        for _ in range_(n_mid):
            ea = in_a.acquire(1)
            eb = in_b.acquire(1)
            mid(ea, eb, elem_out)
            in_a.release(1)
            in_b.release(1)
        out_c.release(1)

    workers = Worker.grid(
        n_aie_rows, n_aie_cols,
        lambda row, col: Worker(core_fn, [A_l2l1[row].cons(), B_l2l1[col].cons(), C_l1l2[row][col].prod(),
                                          rtps[row][col], k_first, k_mid], stack_size=0x1000))

    A_ty = np.ndarray[(gm * gk // 8,), np.dtype[v8bfp16ebs8]]
    B_ty = np.ndarray[(gn * gk // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(gm * gn,), np.dtype[np.float32]]

    # A: row block rb (tm rows), then gk chunk; B: array column c, then its tn-row groups, then gk chunk. One contiguous
    # run each, repeated: A once per tn-group a column walks, B not at all.
    def lin_tap(total, offset, length, repeat=1):
        d0 = 256
        rows = length // d0
        assert length % d0 == 0
        d1 = max(d for d in range(1, 1024) if rows % d == 0)
        d2 = rows // d1
        assert d2 < 1024 and repeat <= 64
        return TensorAccessPattern((total,), offset, [repeat, d2, d1, d0], [0, d1 * d0, d0, 1])

    A_tiles = [lin_tap(gm * gk // 8, rb * tm * (gk // 8), tm * (gk // 8), gn // tn // n_aie_cols) for rb in range(gm // tm)]
    B_tiles = [lin_tap(gn * gk // 8, c * (gn // n_aie_cols) * (gk // 8), (gn // n_aie_cols) * (gk // 8))
               for c in range(n_aie_cols)]

    # Each column drains its results as one contiguous run: block (tm * rows) x tn, row block by row block, then the
    # column's gn tiles c, c + cols, ... in order, each block row-major.
    c_blk = tm * n_aie_rows * tn
    c_per_rb = (gn // tn // n_aie_cols) * c_blk
    # a drain of two row blocks strides tm * rows * gn words, which a shim BD caps at 2^20
    tb_max_n_rows = 2 if tm * n_aie_rows * gn > (1 << 20) else 4
    n_rb = gm // tm // n_aie_rows
    n_taps = (n_rb // min(tb_max_n_rows // 2, n_rb)) * n_aie_cols

    def sequence(a, b, c, A_prods, B_prods, C_conses):
        for row in rtps:
            for rtp in row:
                rtp[0] = gk // tk - 1
                rtp[1] = 0
        tg = TaskGroup()
        c_index = 0
        for tb in range(iron.ceildiv(n_rb, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= n_taps:
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current = min([tb_max_n_rows // 2, n_rb - row_base])
                for col in range(n_aie_cols):
                    C_conses[col].drain(c, tap=lin_tap(gm * gn, col * (gm * gn // n_aie_cols) + row_base * c_per_rb,
                                                       current * c_per_rb), wait=True, group=tg)
                    c_index += 1
                    for tile_row in range(current):
                        tile_offset = ((row_base + tile_row) * n_aie_rows + col) % len(A_tiles)
                        if col < n_aie_rows:
                            A_prods[col].fill(a, tap=A_tiles[tile_offset], group=tg)
                        B_prods[col].fill(b, tap=B_tiles[col], group=tg)
                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, [f.prod() for f in A_l3l2], [f.prod() for f in B_l3l2],
                            [f.cons() for f in C_l2l3]])
    return Program(iron.get_current_device(), rt, workers=[w for row in workers for w in row]).resolve_program()


M = int(os.environ.get("GB_M", 256))
K = int(os.environ.get("GB_K", 2560))
N = int(os.environ.get("GB_N", 4096))
TILE_M = int(os.environ.get("GB_TM", 64))
TILE_K = int(os.environ.get("GB_TK", 64))
TILE_N = int(os.environ.get("GB_TN", 128))

DESIGN = gemm_bfp16
SPECIALIZE = {"gm": M, "gk": K, "gn": N, "tm": TILE_M, "tk": TILE_K, "tn": TILE_N}
