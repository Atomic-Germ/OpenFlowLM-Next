r"""dit_gemm_bfp16a: a TIMING PROBE of dit_gemm with A already in bfp16ebs8.

A copy of open_kernels/designs/dit_gemm/dit_gemm.py with one change. A arrives as
bfp16ebs8 blocks (1.125 bytes a value instead of 2), pre-arranged in the order the
cores consume it. It moves L3 -> L2 -> L1 as a linear copy, like B, and the microkernel
(mm_dit_bfp16a.cc) pops it as ready bfp16 instead of converting bf16 in the core for
every pair of output columns.

The arrangement of A's blocks is not the real one, and the test data is random bytes, so
the output is meaningless. Only the time counts: it bounds what pre-converted activations
(specs/open-diffusion/archive/phase7-speed.md, step 1) could buy. No layout options
(gather, lda, epilogue).

    DG_M=4608 DG_K=3072 DG_N=27648 python open_kernelsuild_design.py utilities\dit-gemm-benchfp16a\dit_gemm_bfp16a.py <out>
    python utilities\dit-gemm-benchfp16a\probe.py ...   (builds, makes the cfgs, benches)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorTiler2D
from aie.iron import (
    Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime,
    StreamDims, TaskGroup, Worker, WorkerRuntimeBarrier,
)
from aie.iron.controlflow import range_
from aie.utils import config as _aie_config

HERE = Path(__file__).resolve().parent
_KERNEL_SRC = HERE / "mm_dit_bfp16a.cc"
_AIE_KERNELS_INC = Path(_aie_config.cxx_header_path()) / "aie_kernels"

M_T, K_T, N_T = 128, 64, 128     # per-core C tile and K step; mm_dit.cc is built for these
MTK = 512                         # width of the L2 A strip
N_COLS, N_ROWS = 8, 4
MAX_INFLIGHT = 4                  # tile groups in flight (upstream's rotation depth)


RTP_TILES, RTP_KB, RTP_COL_GROUPS, RTP_EPI_FIRST = range(4)
RTP_LEN = 4


def check_shape(M: int, K: int, N: int) -> str | None:
    if M % (M_T * N_ROWS) or K % MTK or N % (N_T * N_COLS):
        return f"dit_gemm needs M % {M_T * N_ROWS}, K % {MTK}, N % {N_T * N_COLS} (got {M}x{K}x{N})"
    return None


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def dit_gemm(
    A: In, B: In, C: Out, *,
    M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
    layout: CompileTime[str] = "{}",   # must be "{}"
    stack_size: CompileTime[int] = 0xF00,
):
    """layout (JSON): {"lda": row stride of A (default K, or 2K gathered), "a_col": A's
    first column, "a_gather": A's K columns are the first 64 of every 128, "ldc": C's
    row stride (default N), "c_off": C's element offset, "a_size"/"c_size": the argument
    sizes in elements, "epi": {"first_cb": the first column group (1024 columns) that
    gets the SwiGLU epilogue, "gap": extra columns before those groups in C}}."""
    lay = json.loads(layout)
    why = check_shape(M, K, N)
    assert why is None, why
    m, k, n = M_T, K_T, N_T
    r, s, t = 8, 8, 8

    A_l2_ty = np.ndarray[(m * MTK // 8,), np.dtype[v8bfp16ebs8]]
    B_l2_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(N_ROWS * m, n), np.dtype[bfloat16]]
    A_l1_ty = np.ndarray[(m // 4 * k // 8,), np.dtype[v8bfp16ebs8]]
    B_l1_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[bfloat16]]

    flags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}", f"-I{_AIE_KERNELS_INC}"]
    zero_kernel = ExternalFunction("dit_zero_c", source_file=str(_KERNEL_SRC),
                                   arg_types=[C_l1_ty], compile_flags=flags + ["-DZERO_ONLY"])
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]
    epi_kernel = ExternalFunction("dit_swiglu_epi", source_file=str(_KERNEL_SRC),
                                  arg_types=[C_l1_ty, rtp_ty, np.int32],
                                  compile_flags=flags + ["-DEPI_ONLY"])
    matmul_kernel = ExternalFunction("dit_matmul_quarter", source_file=str(_KERNEL_SRC),
                                     arg_types=[A_l1_ty, B_l1_ty, C_l1_ty],
                                     compile_flags=flags + ["-DMATMUL_ONLY"])

    # PROBE: A pre-arranged on the host in consumption order, so a linear copy like B.
    A_l3l2, A_l2l1 = [], []
    for row in range(N_ROWS):
        f = ObjectFifo(A_l2_ty, name=f"A_L3L2_{row}", depth=2)
        A_l3l2.append(f)
        A_l2l1.append(f.cons().forward(obj_type=A_l1_ty, name=f"A_L2L1_{row}", depth=2))

    # B: one shim per column -> memtile -> the 4 cores of that column. Pre-packed on the
    # host, so this is a linear copy.
    B_l3l2, B_l2l1 = [], []
    for col in range(N_COLS):
        f = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=2)
        B_l3l2.append(f)
        B_l2l1.append(f.cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}", depth=2))

    # C: the 4 cores of a column join at the memtile, which writes row-major out.
    c_l2l3_dims: StreamDims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    C_l1l2 = [[] for _ in range(N_ROWS)]
    C_l2l3 = []
    for col in range(N_COLS):
        f = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=2, dims_to_stream=c_l2l3_dims)
        C_l2l3.append(f)
        parts = f.prod().join([m * n * i for i in range(N_ROWS)],
                              obj_types=[C_l1_ty] * N_ROWS,
                              names=[f"C_L1L2_{col}_{row}" for row in range(N_ROWS)],
                              depths=[1] * N_ROWS)
        for row in range(N_ROWS):
            C_l1l2[row].append(parts[row])

    # OFLM: runtime parameters -- output tiles per core, K blocks, column groups, the
    # first column group of the SwiGLU epilogue (RTP_*).
    rtp = [[Buffer(rtp_ty, name=f"rtp_{row}_{col}",
                   initial_value=np.zeros(RTP_LEN, dtype=np.int32), use_write_rtp=True)
            for col in range(N_COLS)] for row in range(N_ROWS)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(N_COLS)] for _ in range(N_ROWS)]

    def core_fn(in_a, in_b, out_c, zero, matmul, epi, my_rtp, barrier):
        barrier.wait_for_value(1)
        n_tiles = my_rtp[RTP_TILES]
        n_kb = my_rtp[RTP_KB]
        for t in range_(n_tiles):
            c = out_c.acquire(1)
            zero(c)
            for _ in range_(n_kb):
                b = in_b.acquire(1)
                for _ in range(4):
                    a = in_a.acquire(1)
                    matmul(a, b, c)
                    in_a.release(1)
                in_b.release(1)
            epi(c, my_rtp, t)
            out_c.release(1)
        barrier.release_with_value(1)

    workers = Worker.grid(N_ROWS, N_COLS, lambda row, col: Worker(
        core_fn,
        [A_l2l1[row].cons(), B_l2l1[col].cons(), C_l1l2[row][col].prod(),
         zero_kernel, matmul_kernel, epi_kernel, rtp[row][col], barriers[row][col]],
        stack_size=stack_size))

    assert lay == {}, "the probe has no layout options"
    ldc, c_off, first_cb, gap = N, 0, 1 << 30, 0
    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    A_taps = TensorTiler2D.group_tiler((1, M * K // 8), (1, m * K // 8), (1, 1))
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    c_size = lay.get("c_size", c_off + M * ldc)
    assert c_off + (M - 1) * ldc + N + gap <= c_size, "C layout overruns the argument"
    C_ty = np.ndarray[(c_size,), np.dtype[bfloat16]]
    B_taps = TensorTiler2D.group_tiler((1, N * K // 8), (1, n * K // 8), (1, 1))

    n_row_groups = M // m // N_ROWS
    n_col_groups = N // n // N_COLS
    n_groups = n_row_groups * n_col_groups     # one 512x1024 block of C each
    tiles_per_core = n_groups                  # each group gives every core one C tile

    def sequence(a, b, c, A_prods, B_prods, C_conses):
        for row in range(N_ROWS):
            for col in range(N_COLS):
                rtp[row][col][RTP_TILES] = tiles_per_core
                rtp[row][col][RTP_KB] = K // k
                rtp[row][col][RTP_COL_GROUPS] = n_col_groups
                rtp[row][col][RTP_EPI_FIRST] = min(first_cb, n_col_groups)
        for row in range(N_ROWS):
            for col in range(N_COLS):
                barriers[row][col].set(1)

        # OFLM: at most MAX_INFLIGHT groups outstanding; retire the oldest two when full.
        inflight: list[TaskGroup] = []
        for g in range(n_groups):
            tg = TaskGroup()
            rb, cb = divmod(g, n_col_groups)
            for row in range(N_ROWS):
                A_prods[row].fill(a, tap=A_taps[rb * N_ROWS + row], group=tg, wait=False)
            for col in range(N_COLS):
                B_prods[col].fill(b, tap=B_taps[cb * N_COLS + col], group=tg, wait=False)
            for col in range(N_COLS):
                col0 = (cb * N_COLS + col) * n + (gap if cb >= first_cb else 0)
                C_conses[col].drain(c, offset=c_off + rb * N_ROWS * m * ldc + col0,
                                    sizes=[N_ROWS * m, n], strides=[ldc, 1],
                                    group=tg, wait=True)
            inflight.append(tg)
            if len(inflight) == MAX_INFLIGHT:
                inflight.pop(0).finish()
                inflight.pop(0).finish()
        for tg in inflight:
            tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty,
                            [f.prod() for f in A_l3l2], [f.prod() for f in B_l3l2],
                            [f.cons() for f in C_l2l3]])
    return Program(iron.get_current_device(), rt,
                   workers=[w for row in workers for w in row]).resolve_program()


# build_design.py convention
DESIGN = dit_gemm
SPECIALIZE = dict(M=int(os.environ.get("DG_M", 4096)),
                  K=int(os.environ.get("DG_K", 3072)),
                  N=int(os.environ.get("DG_N", 9216)),
                  layout=os.environ.get("DG_LAYOUT", "{}"))
