r"""dit_gemm: the diffusion transformer's GEMM -- bf16 activations x bfp16ebs8 weights -> bf16.

    C[M, N] = A[M, K] @ B[K, N]      A bf16 row-major, B bfp16ebs8 pre-packed, C bf16 row-major

The dataflow is mlir-aie's asymmetric-tile-buffering GEMM
(programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering/config1,
mlir-aie 760932a4, Apache-2.0 WITH LLVM-exception; Wang et al., "Can Asymmetric Tile
Buffering Be Beneficial?", arXiv:2511.16041). Each of the 32 cores keeps a 128x128 bf16 C
tile in L1 for the whole K walk and takes A a quarter (32 rows) at a time, so the C tile --
the reuse -- is 4x bigger than the A buffer that feeds it. The microkernel is mlir-aie's
stock bf16 x bfp16 one on each quarter (mm_dit.cc, which says why not config1's).

Measured (utilities/dit-gemm-bench/README.md, 2026-09-25, HX 370, turbo, the same dataflow
and kernel as `atbs` there): 13.6-17.1 TFLOPS on FLUX.2 [klein] 4B's shapes, rel_fro
1.25e-2 (K = 3072) to 1.9e-2 (K = 12288) against the unquantized product; the error is
the bfp16 operands and the accumulator's re-rounding to bf16 every 64 of K. At image level
that arithmetic is indistinguishable from bf16 (utilities/dit-ref/README.md).

Changes from upstream config1, marked OFLM:

  - Runtime loop bounds. The only shape-dependent values in the core program are the
    output-tile count and the K-block count; they reach the cores as runtime parameters
    (the rtp pattern of npu_offload/gemm_rtp/gemm_pretiled.py), so every GEMM of a model
    is an instruction stream over ONE xclbin and ONE hardware context. The RTP buffers
    start at zero for the same reason gemm_pretiled's do: a shape-dependent initialiser
    is the few bytes that keep two shapes' xclbins from being identical.
  - Any number of tile groups. Upstream's 4-slot rotation needs (M/128)*(N/128) to be a
    multiple of 128; klein's joint single-stream sequence (M = 4608 against N = 27648)
    is 243 groups. The runtime sequence keeps at most four groups in flight and
    finishes the oldest two when a fifth would be issued -- the same descriptor
    pressure, and exactly upstream's order when the count does divide.
  - The stock microkernel (mm_dit.cc), not config1's chess-scheduled one.
  - A SwiGLU epilogue (runtime-selected per column group): with the weight's columns
    interleaved 64 gate + 64 up per 128-column tile (pack.interleave_swiglu), a finished
    tile's first 64 columns become silu(gate) * up in place and the tile leaves whole.
    The next GEMM reads its A "gathered" -- 64 of every 128 columns (layout a_gather),
    one 4-D descriptor per strip -- so the SwiGLU output is never compacted. This
    replaces a separate SwiGLU pass (read 2x, write 1x the MLP activations).
    (Routing the unused half to scratch in the drain instead does not work: a BD stride
    is at most 2^20 words, so the scratch cannot sit a buffer's length away.)
  - A runtime layout (spec DG_LAYOUT): A's row stride and column offset, C's row stride,
    base offset, and an extra column offset for the epilogue column groups. klein's
    single blocks use the gap to leave room, before the MLP tiles, for the attention
    output (dit_fa writes it 64-of-128 too), so sgl_out gathers [attention | SwiGLU]
    in one pattern.

Shape constraints: M % 512, K % 512 (the L2 A strip), N % 1024.

B is packed by pack.py (pack_b): per 64x128 tile, [n-block 16][k-block 8][8 n][8 k] with
a bfp block being 8 values along K; tiles column-major (all K tiles of the first 128
output columns, then the next). The same function feeds tests and the model packer.

    DG_M=4096 DG_K=3072 DG_N=9216 python build_design.py designs/dit_gemm/dit_gemm.py <out>
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
_KERNEL_SRC = HERE / "mm_dit.cc"
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
    layout: CompileTime[str] = "{}",
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

    A_l2_ty = np.ndarray[(m, MTK), np.dtype[bfloat16]]
    B_l2_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(N_ROWS * m, n), np.dtype[bfloat16]]
    A_l1_ty = np.ndarray[(m // 4, k), np.dtype[bfloat16]]
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

    # A: one shim per row -> memtile (an m x MTK strip, re-laid out as MTK/k blocks of
    # m x k) -> the 8 cores of that row, a quarter (m/4 x k) at a time, in the 8x8-block
    # order the stock kernel reads.
    a_l3l2_dims: StreamDims = [(m, k), (MTK // k, m * k), (k, 1)]
    a_l2l1_in_dims: StreamDims = [(MTK // k * 4, m * k // 4), (k // s, s), (m // 4, k), (s, 1)]
    a_l2l1_out_dims: StreamDims = [(k // s, r * s), (m // 4 // r, r * k), (r * s, 1)]
    A_l3l2, A_l2l1 = [], []
    for row in range(N_ROWS):
        f = ObjectFifo(A_l2_ty, name=f"A_L3L2_{row}", depth=2,
                       dims_from_stream_per_cons=a_l3l2_dims)
        A_l3l2.append(f)
        A_l2l1.append(f.cons().forward(obj_type=A_l1_ty, name=f"A_L2L1_{row}", depth=2,
                                       dims_to_stream=a_l2l1_in_dims,
                                       dims_from_stream=a_l2l1_out_dims))

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

    gather = bool(lay.get("a_gather"))
    lda = lay.get("lda", 2 * K if gather else K)
    a_col, ldc, c_off = lay.get("a_col", 0), lay.get("ldc", N), lay.get("c_off", 0)
    epi = lay.get("epi")
    first_cb = epi["first_cb"] if epi else 1 << 30
    gap = epi.get("gap", 0) if epi else 0
    A_ty = np.ndarray[(lay.get("a_size", M * lda),), np.dtype[bfloat16]]
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
                # the row strip [m rows, K] as K/MTK consecutive m x MTK tiles; gathered,
                # a tile's MTK columns are the first 64 of each of MTK/64 128-col tiles
                a_base = a_col + (rb * N_ROWS + row) * m * lda
                if gather:
                    A_prods[row].fill(a, offset=a_base, sizes=[K // MTK, m, MTK // 64, 64],
                                      strides=[2 * MTK, lda, 128, 1], group=tg, wait=False)
                else:
                    A_prods[row].fill(a, offset=a_base, sizes=[K // MTK, m, MTK],
                                      strides=[MTK, lda, 1], group=tg, wait=False)
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
