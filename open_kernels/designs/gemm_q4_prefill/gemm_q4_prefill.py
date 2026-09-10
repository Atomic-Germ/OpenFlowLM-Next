r"""gemm_q4_prefill: whole-array Y[N_WEIGHT,T] = W(q4_1)[N_WEIGHT,K] @ X^T[K,T]
prefill GEMM on the NPU array (task 0167 stage 14).

Architectural mapping (see the task brief / NpuEmbeddings
experiments/m5-pretiled-gemm/gemm_pretiled.py for the validated whole-array
bf16 GEMM this is structurally derived from):

  * WEIGHT takes the "A" (row) role: not reused across T-tiles (refetched,
    like the reference's own A operand), broadcast across all n_aie_cols=8
    columns of its row via one ObjectFifo with multiple .cons() ports (the
    same broadcast idiom the reference uses for its A operand). Fetched at
    the q4 pool's native granularity -- one 10240 B "band-k-group" (64 output
    rows x 256 K) per DMA element -- and DEQUANTIZED ON-CORE into a bf16
    [64,64] scratch tile immediately before each of the 4 k=64 matmul
    sub-steps that one band covers. See gemm_q4_dequant.h for the dequant
    kernel and why it is split into a scalar integer gather pass and a
    vectorized bf16-arithmetic pass.

  * ACTIVATION takes the "B" (col) role: bf16, pre-tiled offline exactly like
    the reference's B operand (npu_offload/gemm_rtp/npue.py's tile_b(), same
    "k,n" tile order), and reused across ALL row-blocks via `b_reuse="asym"`
    (one mem-tile-resident ObjectFifo, `repeat_count=NRB`) -- the reference's
    own validated mechanism for "this operand does not depend on which
    row-block is currently computing", applied to activation instead of
    weight because here it is the activation, not the weight, that is
    row-block-invariant.

Mem-tile budget check (the brief's own explicit ask -- "verify this is really
free before trusting it"): the asym-reuse mem-tile footprint per column is
T*K*2/n_aie_cols bytes (n and k_tile cancel out of b_slice_tiles*k*n*2), i.e.
T*K/4 at n_aie_cols=8. That exceeds the ~512 KB mem tile at (T=1024,K=2560)
and even at (T=256,K=8192) -- see the printed budget check at import time and
the task report for the actual numbers measured against this build.

Weight is NOT reused across T-tiles (T_TILES = T//tile_n//n_aie_cols > 1): a
row's whole weight column is refetched from DRAM once per T-tile iteration
(a real inefficiency above T=256, reported honestly rather than hidden).

Build (from this directory, mlir-aie env dot-sourced):
    GQP_N=2560 GQP_K=2560 GQP_T=256 python ../../build_design.py gemm_q4_prefill.py build_qkv_t256
Test:
    python make_test.py --shape qkv --tokens 256
    ..\..\harness\out\run_kernel.exe run_qkv_t256.cfg && python compare.py qkv_t256
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker, kernels, str_to_dtype
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D

HERE = Path(__file__).parent

# npue.py (tile_b/untile_b) already lives in THIS repo, synced from
# NpuEmbeddings/experiments/m5-pretiled-gemm (tasks/0156, T63) -- reuse it
# rather than re-deriving the pre-tiling logic.
sys.path.insert(0, str(HERE.parents[2] / "npu_offload" / "gemm_rtp"))
from npue import tile_b, untile_b  # noqa: E402

N_AIE_ROWS = 4
N_AIE_COLS = 8
M_TILE = 64        # weight rows per band / per matmul m-tile (= mac_dims-compatible, matches q4 band's 64 rows)
K_TILE = 64        # matmul k-tile (mac_dims s=8 divides it)
BAND_K = 256       # q4 pool band k-width (one q4_1_pack.py "chunk pair")
BAND_BYTES = 10240 # 64 rows x 256 K x 5/8 B/elem

# ---- shape this build is specialised to (env-driven, like gemv_q4.py) ----
N_WEIGHT = int(os.environ.get("GQP_N", 2560))
K = int(os.environ.get("GQP_K", 2560))
T = int(os.environ.get("GQP_T", 256))
TILE_N = int(os.environ.get("GQP_TILE_N", 32))
# "auto" (default): pick b_reuse="asym" only if the per-column mem-tile
# footprint (T*K*2/n_aie_cols bytes) leaves a safety margin under the ~512 KB
# mem tile once C's own mem-tile footprint is added. "1"/"0" forces it.
_BREUSE_ENV = os.environ.get("GQP_BREUSE", "auto")


def mem_tile_budget_bytes(k: int, t: int, tile_n: int = TILE_N, n_aie_cols: int = N_AIE_COLS,
                          n_aie_rows: int = N_AIE_ROWS, m: int = M_TILE) -> dict:
    """b_reuse=asym mem-tile bytes per column, plus C's own mem-tile fifo
    bytes, for the safety-margin check. b_slice_tiles*(k_tile*n)*2 collapses
    to t*k*2/n_aie_cols (k_tile, n cancel) -- see the module docstring."""
    b_bytes = t * k * 2 // n_aie_cols
    c_bytes = m * tile_n * n_aie_rows * 4 * 2  # C_l2_ty, itemsize 4 (fp32), depth 2
    return dict(b_bytes=b_bytes, c_bytes=c_bytes, total=b_bytes + c_bytes)


def recommend_b_reuse(k: int, t: int, tile_n: int = TILE_N) -> bool:
    if _BREUSE_ENV in ("1", "0"):
        return _BREUSE_ENV == "1"
    # HARD BLOCKER, confirmed on real hardware (0167 stage 14): at
    # n_aie_cols=8, b_reuse="asym" (a single ObjectFifo whose PRODUCER is a
    # shim tile, consumer_obj_type=B_l1_ty, repeat_count=NRB) fails MLIR
    # verification outright --
    #   error: unknown: `repeat_count` unavailable for shim tiles
    # -- regardless of mem-tile byte budget (this is a HARDWARE CAPABILITY
    # gap, not a capacity one: repeat_count is only legal on an L2->L1
    # forward whose producer is a MEM TILE, and asym's whole point was
    # collapsing the L3->L2 and L2->L1 hops into one fifo, which places the
    # repeat on the shim side instead). This corroborates -- with a more
    # precise root cause -- NpuEmbeddings experiments/m5-pretiled-gemm's own
    # documented verdict (tasks/0046, T48): "'asym' DOES NOT BUILD... at 8
    # columns", there attributed primarily to mem-tile channel/BD-pool
    # exhaustion. The plain b_reuse=True path (L3->L2 fifo depth=b_slice_
    # tiles, repeat_count on nothing -- the WHOLE slice just sits resident)
    # independently fails the mem-tile BD-pool cap (~4-6 at 8 cols) per that
    # same verdict, so no b_reuse form is available at cols=8 today. Always
    # refuse here; GQP_BREUSE=1 overrides for an isolated repro if a future
    # mlir-aie fixes the shim-tile restriction.
    return False


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base
    from aie.utils import config
    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    return inc


def ensure_dequant_entries() -> list[Path]:
    """One .cc per ky (0..3): gqd_dequant_ky's `ky` baked in as a C++
    literal, matching gemv_q4.py's own per-entry-point .cc discipline (an
    ExternalFunction's compiled object comes straight from its own source
    file's signature -- CLAUDE.md trap 8 -- so ky cannot just be an extra
    Python-side argument; it has to be a distinct compiled symbol, or
    a distinct runtime-plumbed RTP buffer, and 4 tiny symbols is far simpler
    than RTP here). gqd_dequant_ky and its two helpers stay `inline` +
    noinline (see gemm_q4_dequant.h): four TUs each emit a copy, and the
    linker COMDAT-folds them to one (trap 9).
    """
    out = []
    for ky in range(4):
        p = HERE / f"gemm_q4_dequant_entry_k{ky}.cc"
        src = (
            f"// GENERATED by gemm_q4_prefill.py: dequant entry, ky={ky}\n"
            f'#include "gemm_q4_dequant.h"\n\n'
            f'extern "C" {{\n'
            f"void gqd_entry_k{ky}(const uint8_t *__restrict band, uint8_t *__restrict nib_scr,\n"
            f"                     bfloat16 *__restrict scratch) {{\n"
            f"  gqd_dequant_ky(band, {ky}u, nib_scr, scratch);\n"
            f"}}\n"
            f"}}\n"
        )
        if not p.is_file() or p.read_text(encoding="utf-8") != src:
            p.write_text(src, encoding="utf-8", newline="\n")
        out.append(p)
    return out


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemm_q4_prefill(
    W: In, X: In, Y: Out, *,
    n_weight: CompileTime[int], k: CompileTime[int], t: CompileTime[int],
    tile_n: CompileTime[int] = 32, b_reuse: CompileTime[bool] = True,
    srchash: CompileTime[int] = 0,
):
    m, k_tile = M_TILE, K_TILE
    n = tile_n
    n_aie_rows, n_aie_cols = N_AIE_ROWS, N_AIE_COLS

    assert n_weight % (m * n_aie_rows) == 0, "N_WEIGHT must tile into 256-row blocks"
    assert k % BAND_K == 0, "K must be a multiple of 256 (one q4 band-k-group)"
    assert t % (n * n_aie_cols) == 0, f"T must be a multiple of {n * n_aie_cols}"

    NRB = n_weight // (m * n_aie_rows)   # weight row-block-groups
    NBG = k // BAND_K                    # band-k-groups per row-block (weight DMA granularity)
    NKT = k // k_tile                    # matmul k-tile steps == NBG*4
    T_TILES = t // (n * n_aie_cols)      # T-tile iterations per core
    assert NKT == NBG * 4

    dtype_in = str_to_dtype("bf16")
    dtype_out = str_to_dtype("f32")

    matmul_kernel = kernels.mm(
        dim_m=m, dim_k=k_tile, dim_n=n, input_dtype=dtype_in, output_dtype=dtype_out,
        b_col_maj=False, c_col_maj=False, use_chess=False,
        emulate_bf16_mmul_with_bfp16=False, vectorized=True)
    zero_kernel = matmul_kernel.zero
    r, s, mt = matmul_kernel.mac_dims
    assert (r, s, mt) == (4, 8, 8), f"unexpected aie2p bf16 mac_dims {(r, s, mt)}"
    assert m % r == 0 and k_tile % s == 0 and n % mt == 0

    dequant_srcs = ensure_dequant_entries()
    band_ty = np.ndarray[(BAND_BYTES,), np.dtype[np.uint8]]
    nib_ty = np.ndarray[(m * k_tile,), np.dtype[np.uint8]]
    scratch_ty = np.ndarray[(m * k_tile,), np.dtype[dtype_in]]
    # 0167 stage 16: GQP_SCALAR_GATHER=1 rebuilds Pass 1 with its original
    # scalar integer gather (gemm_q4_dequant.h's GQD_SCALAR_GATHER branch),
    # for a same-toolchain before/after A-B against the vectorized default.
    _dequant_flags = ["-DGQD_SCALAR_GATHER"] if os.environ.get("GQP_SCALAR_GATHER") == "1" else []
    # Timing-only ablation (0167 stage 16): skip Pass 1 and/or Pass 2's
    # compute to attribute cost. Output is garbage with either set -- never
    # combine with a correctness (compare.py) run.
    if os.environ.get("GQP_NULL_GATHER") == "1":
        _dequant_flags.append("-DGQD_NULL_GATHER")
    if os.environ.get("GQP_NULL_DEQUANT") == "1":
        _dequant_flags.append("-DGQD_NULL_DEQUANT")
    dequant_kernels = [
        ExternalFunction(f"gqd_entry_k{ky}", source_file=str(dequant_srcs[ky]),
                         arg_types=[band_ty, nib_ty, scratch_ty], include_dirs=_include_dirs(),
                         compile_flags=_dequant_flags)
        for ky in range(4)
    ]

    pool_bytes = n_weight * k * 5 // 8
    W_ty = np.ndarray[(pool_bytes,), np.dtype[np.uint8]]
    X_ty = np.ndarray[(k * t,), np.dtype[dtype_in]]
    Y_ty = np.ndarray[(n_weight * t,), np.dtype[dtype_out]]

    B_l1_ty = np.ndarray[(k_tile, n), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k_tile * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    # ---- WEIGHT (A role): one band-fifo per row, broadcast to all columns ----
    A_l3l2_fifos = [ObjectFifo(band_ty, name=f"A_L3L2_{row}", depth=2) for row in range(n_aie_rows)]
    A_l2l1_fifos = [
        A_l3l2_fifos[row].cons().forward(obj_type=band_ty, name=f"A_L2L1_{row}", depth=2)
        for row in range(n_aie_rows)
    ]

    # ---- ACTIVATION (B role): pre-tiled bf16, b_reuse across row-blocks ----
    b_slice_tiles = T_TILES * NKT
    B_l2_mega_ty = np.ndarray[(b_slice_tiles * k_tile * n,), np.dtype[dtype_in]]
    B_l2l1_fifos = [None] * n_aie_cols
    B_l3l2_fifos = [None] * n_aie_cols
    for col in range(n_aie_cols):
        if b_reuse:
            B_l2l1_fifos[col] = ObjectFifo(
                B_l2_mega_ty, name=f"B_L3L2_{col}", depth=1,
                consumer_obj_type=B_l1_ty, repeat_count=NRB)
            B_l3l2_fifos[col] = B_l2l1_fifos[col]
        else:
            B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=2)
            B_l2l1_fifos[col] = B_l3l2_fifos[col].cons().forward(
                obj_type=B_l1_ty, name=f"B_L2L1_{col}", depth=2)

    # ---- OUTPUT (C): four rows join per column, exactly as the reference ----
    C_l2l3_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    for col in range(n_aie_cols):
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=2,
            dims_to_stream=[(m // r, r * n), (r, mt), (n // mt, r * mt), (mt, 1)])
        tmp = C_l2l3_fifos[col].prod().join(
            [m * n * i for i in range(n_aie_rows)],
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[2] * n_aie_rows,
        )
        for row in range(n_aie_rows):
            C_l1l2_fifos[row][col] = tmp[row]

    nib_bufs = [[Buffer(nib_ty, name=f"nib_{row}_{col}") for col in range(n_aie_cols)] for row in range(n_aie_rows)]
    scratch_bufs = [[Buffer(scratch_ty, name=f"ascr_{row}_{col}") for col in range(n_aie_cols)] for row in range(n_aie_rows)]

    def core_fn(in_a, in_b, out_c, zero, matmul, nib_scr, a_scr, *dequants):
        for _ in range_(NRB):
            for _ in range_(T_TILES):
                elem_out = out_c.acquire(1)
                zero(elem_out)
                for _ in range_(NBG):
                    band = in_a.acquire(1)
                    for ky in range(4):  # compile-time (Python) unroll: 4 distinct entry symbols
                        dequants[ky](band, nib_scr, a_scr)
                        elem_in_b = in_b.acquire(1)
                        matmul(a_scr, elem_in_b, elem_out)
                        in_b.release(1)
                    in_a.release(1)
                out_c.release(1)

    def _mk(row, col):
        return Worker(
            core_fn,
            [A_l2l1_fifos[row].cons(), B_l2l1_fifos[col].cons(), C_l1l2_fifos[row][col].prod(),
             zero_kernel, matmul_kernel, nib_bufs[row][col], scratch_bufs[row][col], *dequant_kernels],
            stack_size=0x1000,
        )

    workers = Worker.grid(n_aie_rows, n_aie_cols, _mk)

    # ---- weight taps: one per (row, row-block-group) -- NOT one compound tap
    # across all NRB, and NOT (see below) all issued into one TaskGroup. The
    # shim DMA has 16 buffer descriptors PER TILE (CLAUDE.md trap 17); a
    # naive "issue everything, finish() once at the end" schedule was tried
    # first here and failed to compile with exactly that error ("Too many
    # simultaneously active buffer descriptors on tile (3,0), which supports
    # up to 16") once activation had to be refetched NRB times per column
    # (b_reuse does not build at n_aie_cols=8 -- see recommend_b_reuse). The
    # fix is the same one CLAUDE.md trap 18 already names for the reference
    # design: chunk the sequence and call tg.finish() between chunks so BDs
    # are freed (aiex.dma_await_task) before the next batch is issued. Here
    # chunked at the finest granularity, one (row-block-group, T-tile) pair
    # per TaskGroup, since correctness -- not dispatch-count -- is this
    # build's priority; this is strictly more barriers than necessary and a
    # real, reported inefficiency (trap 18's "+4.9%" cost applies again here,
    # likely worse given the finer granularity).
    row_k_bytes = NBG * BAND_BYTES         # bytes for one band_idx across the WHOLE K
    band_idx_stride = n_aie_rows * row_k_bytes

    def a_tap(row: int, rbg: int) -> TensorAccessPattern:
        off = row * row_k_bytes + rbg * band_idx_stride
        return TensorAccessPattern((pool_bytes,), off, [1, row_k_bytes], [row_k_bytes, 1])

    # ---- activation taps: same "k,n" pre-tiled access pattern as the
    # reference (npu_offload/gemm_rtp/gemm_pretiled.py's B_taps, "pretiled"
    # branch), K/N relabelled to K/T here. Rbg-independent (activation does
    # not depend on which weight row-block is being computed), so ONE tap
    # per column is reused for every (rbg, ntile) refill.
    TE = k_tile * n
    KB, NB = k // k_tile, t // n
    kb_stride, nb_stride = NB * TE, TE  # "k,n" order

    def b_tap(col: int, ntile: int) -> TensorAccessPattern:
        # ONE n-block's worth (NKT k-tiles) -- NOT the column's whole
        # T_TILES-wide compound slice. Column `col`'s ntile-th n-block (its
        # own NBC=T_TILES sequence) is nb = ntile*n_aie_cols + col, matching
        # the "n,k,kt,nt" pre-tiled layout tile_b() wrote. A first attempt
        # filled the WHOLE compound (all T_TILES ntiles) on every (rbg,ntile)
        # refill -- silently overfeeding the B fifo by a factor of T_TILES
        # per refill with no compile-time or run-time error, caught only by
        # the T_TILES>1 correctness gate going from PASS (T=256) to FAIL with
        # a negative per-token cosine (T=512/1024) -- see the task report.
        off = col * nb_stride + ntile * (n_aie_cols * nb_stride)
        return TensorAccessPattern((k * t,), off, [KB, k_tile, n], [kb_stride, n, 1])

    # ---- C drain taps: one per (row-block-group, T-tile, column) output
    # block -- plain row-major [n_weight, t], block shape (m*n_aie_rows, n).
    def c_tap(rbg: int, ntile: int, col: int) -> TensorAccessPattern:
        off = (rbg * m * n_aie_rows) * t + ntile * (n * n_aie_cols) + col * n
        return TensorAccessPattern((n_weight * t,), off, [m * n_aie_rows, n], [t, 1])

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conss = [f.cons() for f in C_l2l3_fifos]

    def sequence(a_W, a_X, c_Y, A_prod_hs, B_prod_hs, C_cons_hs):
        for rbg in range(NRB):
            for ntile in range(T_TILES):
                tg = TaskGroup()
                for row in range(n_aie_rows):
                    A_prod_hs[row].fill(a_W, tap=a_tap(row, rbg), group=tg)
                for col in range(n_aie_cols):
                    B_prod_hs[col].fill(a_X, tap=b_tap(col, ntile), group=tg)
                for col in range(n_aie_cols):
                    C_cons_hs[col].drain(c_Y, tap=c_tap(rbg, ntile, col), wait=True, group=tg)
                tg.finish()

    rt = Runtime(sequence, [W_ty, X_ty, Y_ty, A_prods, B_prods, C_conss])
    program = Program(iron.get_current_device(), rt, workers=[w for row in workers for w in row])
    return program.resolve_program()


DESIGN = gemm_q4_prefill
SPECIALIZE = {
    "n_weight": N_WEIGHT, "k": K, "t": T, "tile_n": TILE_N,
    "b_reuse": recommend_b_reuse(K, T, TILE_N),
}

if __name__ == "__main__":
    b = mem_tile_budget_bytes(K, T, TILE_N)
    print(f"N_WEIGHT={N_WEIGHT} K={K} T={T} tile_n={TILE_N} "
          f"b_reuse={SPECIALIZE['b_reuse']} mem_tile_budget: B={b['b_bytes']} B "
          f"C={b['c_bytes']} B total={b['total']} B (of ~512 KB/col)")
