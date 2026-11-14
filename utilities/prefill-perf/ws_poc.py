r"""ws_poc: weight-stationary proof of concept for the q4_1 prefill GEMM.

STOPPED experiment: see utilities/prefill-perf/BASELINE.md for the measured
incomplete-output failure and L1 limit. Retained only for reproducibility.

NOT part of the model runtime. A standalone experiment that answers one question with
hardware numbers: does reusing one dequantized weight tile across R activation
microtiles beat the frozen paired baseline (utilities/prefill-perf/BASELINE.md) by
>= 1.4x?

Same format (q4_1), same arithmetic (the identical bf16 matmul over the same pool
bytes), same tile sizes -- GQP_TILE_N is NOT increased and `m x n` stays 64 x 32. Only
the ORDER of the loops changes.

Baseline (gemm_q4_prefill.py), per core, whole dispatch:

    for t in T_TILES:                       # T / (tile_n * n_aie_cols) = T/256
        for band_group in N_band_groups:    # 14 at N=14336, 4 rows x 8 cols
            acquire(band); for ky in 0..3: dequant -> scratch; acquire(b); matmul

The weight band is acquired once per band group and the four ky slices come from it,
so it is already tight *within* one T-tile. The cost is that the whole walk repeats for
the next T-tile: the same band is re-fetched from DRAM and the same nibbles are
re-dequantized R = T_TILES times. The baseline's own module docstring names this
("Weight is NOT reused across T-tiles ... a real inefficiency above T=256").

This POC hoists the dequant out of the T-tile loop:

    for band_group:                         # 14
        acquire(band)                       # ONCE
        for batch in 0,1:                   # two scratch tiles, ky 2*batch..+1
            for ky in batch: dequant -> scratch[ky]
            for t in T_TILES:               # the R activation microtiles
                ...matmul with that scratch...
        release(band)

so the dequant and the weight DMA drop by R while the matmul count is unchanged.

Why two scratch tiles and not four: four cost 32768 B and, together with the fifos,
overrun the 64 KB L1 (`aie.tile op allocated buffers exceeded available memory`, the
same wall that stops GQP_TILE_N=64). Two tiles are 16384 B, and the hoist is unaffected
-- the band is still fetched once and dequantized once, just in two batches.

Why this is the right experiment at all: at n14336_k5120_t256 a call spends 85.7 % of
its time in "the matmul and the streams" and 14.3 % in dequant
(utilities/prefill-perf/BASELINE.md), yet 2048 MAC-cycles per band group is ~2 us
against ~302 us measured per core. The core is paying ~1.35 us per `(dma or matmul)`
op, so the lever is ISSUE DENSITY, not MAC throughput or tile shape.

R is an independent variable, so it must be reachable: R <= T_TILES = T/(32*8) = T/256.
At T=256 that is R=1, i.e. no experiment, which is why the POC is driven at T >= 512 and
R={2,4} at T=512, R={2,4,8} at T=2048.

Build from the repository root (needs the full toolchain):
    GQP_N=14336 GQP_K=5120 GQP_T=2048 GQP_R=4 \
      python open_kernels/build_design.py utilities/prefill-perf/ws_poc.py \
        Models/qwen38-27b/prefill-perf/ws-t2048-r4

Drive it with the baseline's own vectors (make_test.py --shape n14336_k5120 --tokens T)
and score separate per-variant dumps with utilities/prefill-perf/ws_measure.py.
Do not use compare.py for multiple variants: it reads one shared output dump.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker,
                      kernels, str_to_dtype)
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

KERNEL_SOURCES = Path(__file__).resolve().parents[2] / "open_kernels/designs/gemm_q4_prefill"

N_AIE_ROWS = 4
N_AIE_COLS = 8
M_TILE = 64
K_TILE = 64
BAND_K = 256
BAND_BYTES = 10240

FMT = os.environ.get("GQP_FMT", "q4_1")
N_WEIGHT = int(os.environ.get("GQP_N", 14336))
K = int(os.environ.get("GQP_K", 5120))
T = int(os.environ.get("GQP_T", 2048))
TILE_N = int(os.environ.get("GQP_TILE_N", 32))
# R = activation microtiles fed by ONE dequantized weight tile. The independent
# variable of the experiment, so it is a parameter, not a consequence of T.
R_TILES = int(os.environ.get("GQP_R", 1))
# Resident A tiles, one per ky. MUST equal kys: with fewer, a band's four partial
# products cannot share one output, so the accumulator would be reopened per batch and
# `zero()` would discard the earlier partial sum. Kept as a named constant rather than a
# free parameter so nobody can build the numerically wrong variant by accident.
NSCR = 4
# Fifo depths. The four resident A tiles cost 32768 B, which is half of the 64 KB L1, so
# the fifos have to give up depth to fit -- that is itself part of the measurement, not
# an implementation detail: keeping the weight stationary in L1 spends pipelining
# headroom. Byte counts are unaffected by depth; only in-flight overlap is.
ADEPTH = int(os.environ.get("GQP_ADEPTH", 2))
BDEPTH = int(os.environ.get("GQP_BDEPTH", 2))
CDEPTH = int(os.environ.get("GQP_CDEPTH", 2))

_FORMATS = {"q4_1": (BAND_BYTES, BAND_K, 4, 5, 8)}


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base
    from aie.utils import config
    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    return inc


def ensure_dequant_entries() -> list[Path]:
    """The per-ky entries gemm_q4_prefill.py generates; this design reuses them."""
    out = []
    for ky in range(4):
        p = KERNEL_SOURCES / f"gemm_q4_dequant_entry_k{ky}.cc"
        if not p.is_file():
            raise RuntimeError(f"missing {p}; build gemm_q4_prefill.py once so the entries exist")
        out.append(p)
    return out


def traffic_model(n_weight: int, k: int, t: int, tile_n: int, r_tiles: int) -> dict:
    """Analytic byte and count accounting for one dispatch, all 32 cores.

    The BASELINE re-fetches and re-dequantizes each band once per T-tile, i.e. R times
    over R consecutive T-tiles. This design does it once per T-tile GROUP. Both run the
    SAME number of matmuls and move the same activation and accumulator bytes, so any
    difference the hardware shows is attributable to the weight/dequant side.
    """
    m, n, cols, rows = M_TILE, tile_n, N_AIE_COLS, N_AIE_ROWS
    NRB = n_weight // (m * rows)
    NBG = k // 256
    T_TILES = t // (n * cols)
    R = r_tiles
    NTG = T_TILES // R
    band_groups_per_core = NRB // rows
    w_bytes_ws = band_groups_per_core * NTG * 10240 * rows * cols
    w_bytes_base = w_bytes_ws * R
    dequant_ws = band_groups_per_core * NTG * 4 * rows * cols
    dequant_base = dequant_ws * R
    b_bytes = T_TILES * (k // K_TILE) * K_TILE * n * 2 * rows * cols
    c_bytes = T_TILES * m * n * 4 * rows * cols
    matmuls = T_TILES * (k // K_TILE) * rows * cols
    return dict(R=R, NTG=NTG, T_TILES=T_TILES, NRB=NRB, NBG=NBG,
                band_groups_per_core=band_groups_per_core, matmuls=matmuls,
                weight_bytes_ws=w_bytes_ws, weight_bytes_base=w_bytes_base,
                dequant_ws=dequant_ws, dequant_base=dequant_base,
                activation_bytes=b_bytes, accumulator_bytes=c_bytes,
                accumulator_spill_bytes=0,
                weight_ratio=(w_bytes_base / w_bytes_ws if w_bytes_ws else 0.0),
                dequant_ratio=(dequant_base / dequant_ws if dequant_ws else 0.0))


def l1_budget() -> dict:
    """Bytes a core must hold, so the build's refusal is predicted rather than discovered."""
    scratch = NSCR * M_TILE * K_TILE * 2        # bf16 A operand, one per resident ky
    nibble = M_TILE * K_TILE                   # nibble staging, shared by all ky
    accum = M_TILE * TILE_N * 4                # [m, n] f32
    a_fifo = ADEPTH * BAND_BYTES
    b_fifo = BDEPTH * (K_TILE * TILE_N * 2)
    c_fifo = CDEPTH * (M_TILE * TILE_N * 4)
    data = scratch + nibble + accum
    fifo = a_fifo + b_fifo + c_fifo
    return dict(scratch=scratch, nibble=nibble, accum=accum,
                data=data, fifo=fifo, total=data + fifo)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def ws_poc(
    W: In, X: In, Y: Out, *,
    n_weight: CompileTime[int], k: CompileTime[int], t: CompileTime[int],
    tile_n: CompileTime[int] = 32, srchash: CompileTime[int] = 0, fmt: CompileTime[str] = "q4_1",
    r_tiles: CompileTime[int] = 1,
):
    assert fmt == "q4_1", "the weight-stationary POC exists to test q4_1 only"
    m, k_tile = M_TILE, K_TILE
    n = tile_n
    n_aie_rows, n_aie_cols = N_AIE_ROWS, N_AIE_COLS
    elem_bytes, elem_k, kys, pool_num, pool_den = _FORMATS[fmt]

    assert n_weight % (m * n_aie_rows) == 0
    assert k % elem_k == 0
    assert t % (n * n_aie_cols) == 0

    NRB = n_weight // (m * n_aie_rows)
    NBG = k // elem_k
    NKT = k // k_tile
    T_TILES = t // (n * n_aie_cols)
    assert NKT == NBG * kys
    assert kys == 4 and NSCR == kys, "every ky slice must be resident at once (see one_band)"
    R = r_tiles
    assert 1 <= R <= T_TILES, f"R={R} needs 1 <= R <= T_TILES={T_TILES}"
    assert T_TILES % R == 0, f"T_TILES={T_TILES} is not a multiple of R={R}"
    NTG = T_TILES // R

    dtype_in = str_to_dtype("bf16")
    dtype_out = str_to_dtype("f32")

    matmul_kernel = kernels.mm(
        dim_m=m, dim_k=k_tile, dim_n=n, input_dtype=dtype_in, output_dtype=dtype_out,
        b_col_maj=False, c_col_maj=False, use_chess=False,
        emulate_bf16_mmul_with_bfp16=False, vectorized=True)
    zero_kernel = matmul_kernel.zero
    r, s, mt = matmul_kernel.mac_dims
    assert (r, s, mt) == (4, 8, 8)
    assert m % r == 0 and k_tile % s == 0 and n % mt == 0

    nib_ty = np.ndarray[(m * k_tile,), np.dtype[np.uint8]]
    scratch_ty = np.ndarray[(m * k_tile,), np.dtype[dtype_in]]
    band_ty = np.ndarray[(elem_bytes,), np.dtype[np.uint8]]

    pool_bytes = n_weight * k * pool_num // pool_den
    W_ty = np.ndarray[(pool_bytes,), np.dtype[np.uint8]]
    X_ty = np.ndarray[(k * t,), np.dtype[dtype_in]]
    Y_ty = np.ndarray[(n_weight * t,), np.dtype[dtype_out]]

    B_l1_ty = np.ndarray[(k_tile, n), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k_tile * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    A_l3l2_fifos = [ObjectFifo(band_ty, name=f"A_L3L2_{row}", depth=ADEPTH)
                    for row in range(n_aie_rows)]
    A_l2l1_fifos = [
        A_l3l2_fifos[row].cons().forward(obj_type=band_ty, name=f"A_L2L1_{row}", depth=ADEPTH)
        for row in range(n_aie_rows)
    ]

    # B: plain depth-2 ping-pong. b_reuse="asym" is a documented dead end at 8 columns
    # (repeat_count on a shim tile / mem-tile BD pool), and not changing the B side keeps
    # the comparison to the baseline honest: only the A side's order differs.
    B_l3l2_fifos = [ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=BDEPTH)
                    for col in range(n_aie_cols)]
    B_l2l1_fifos = [
        B_l3l2_fifos[col].cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}", depth=BDEPTH)
        for col in range(n_aie_cols)
    ]

    C_l2l3_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    for col in range(n_aie_cols):
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=CDEPTH,
            dims_to_stream=[(m // r, r * n), (r, mt), (n // mt, r * mt), (mt, 1)])
        tmp = C_l2l3_fifos[col].prod().join(
            [m * n * i for i in range(n_aie_rows)],
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[CDEPTH] * n_aie_rows,
        )
        for row in range(n_aie_rows):
            C_l1l2_fifos[row][col] = tmp[row]

    # NSCR resident bf16 A tiles (the hoist's whole point) and ONE nibble staging buffer:
    # the ky dequant batches are sequential, so the staging is dead between them.
    nib_grid = [[Buffer(nib_ty, name=f"nib_{rr}_{cc}") for cc in range(n_aie_cols)]
                for rr in range(n_aie_rows)]
    scr_grid = [[[Buffer(scratch_ty, name=f"ascr{b}_{rr}_{cc}") for cc in range(n_aie_cols)]
                 for rr in range(n_aie_rows)] for b in range(NSCR)]

    rtp_bufs = [[Buffer(np.ndarray[(2,), np.dtype[np.int32]], name=f"rtp_{row}_{col}",
                        initial_value=np.zeros(2, dtype=np.int32), use_write_rtp=True)
                 for col in range(n_aie_cols)] for row in range(n_aie_rows)]

    dequant_srcs = ensure_dequant_entries()
    dequant_kernels = [
        ExternalFunction(f"gqd_entry_k{ky}", source_file=str(dequant_srcs[ky]),
                         arg_types=[band_ty, nib_ty, scratch_ty], include_dirs=_include_dirs(),
                         compile_flags=[])
        for ky in range(4)
    ]

    def core_fn(in_a, in_b, out_c, zero, matmul, *extra):
        my_scr = extra[:NSCR]
        my_nib = extra[NSCR]
        my_rtp = extra[NSCR + 1]
        dequants = extra[NSCR + 2:]

        def one_band(band):
            # Dequantize ALL ky slices of this band once, keep them resident, then run the
            # R x NTG activation microtiles against them. Every microtile accumulates all
            # ky into ONE output before releasing it, so no partial sum is ever written
            # and re-read -- there is no accumulator spill, by construction.
            #
            # This is why the four tiles cannot be batched: with fewer than kys resident
            # tiles the output would have to be reopened per batch, and `zero()` would
            # wipe the partial sum from the batch before. That is a correctness
            # constraint, not a tuning choice -- see the report.
            for j in range(kys):
                dequants[j](band, my_nib, my_scr[j])
            for g in range_(NTG):
                for rr in range_(R):
                    elem_out = out_c.acquire(1)
                    zero(elem_out)
                    for j in range(kys):
                        elem_in_b = in_b.acquire(1)
                        matmul(my_scr[j], elem_in_b, elem_out)
                        in_b.release(1)
                    out_c.release(1)

        band = in_a.acquire(1)
        n_bandgroups = my_rtp[1]
        one_band(band)
        in_a.release(1)
        for _ in range_(n_bandgroups - 1):
            band = in_a.acquire(1)
            one_band(band)
            in_a.release(1)

    def _mk(row, col):
        return Worker(
            core_fn,
            [A_l2l1_fifos[row].cons(), B_l2l1_fifos[col].cons(), C_l1l2_fifos[row][col].prod(),
             zero_kernel, matmul_kernel,
             *[scr_grid[b][row][col] for b in range(NSCR)],
             nib_grid[row][col],
             rtp_bufs[row][col], *dequant_kernels],
            stack_size=0x1000,
        )

    workers = Worker.grid(n_aie_rows, n_aie_cols, _mk)

    row_k_bytes = NBG * elem_bytes
    band_idx_stride = n_aie_rows * row_k_bytes

    def a_tap(row: int, rbg: int) -> TensorAccessPattern:
        off = row * row_k_bytes + rbg * band_idx_stride
        return TensorAccessPattern((pool_bytes,), off, [1, row_k_bytes], [row_k_bytes, 1])

    TE = k_tile * n
    KB, NB = k // k_tile, t // n
    kb_stride, nb_stride = NB * TE, TE

    def b_tap(col: int, ntile: int) -> TensorAccessPattern:
        off = col * nb_stride + ntile * (n_aie_cols * nb_stride)
        return TensorAccessPattern((k * t,), off, [KB, k_tile, n], [kb_stride, n, 1])

    def c_tap(rbg: int, ntile: int, col: int) -> TensorAccessPattern:
        off = (rbg * m * n_aie_rows) * t + ntile * (n * n_aie_cols) + col * n
        return TensorAccessPattern((n_weight * t,), off, [m * n_aie_rows, n], [t, 1])

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conss = [f.cons() for f in C_l2l3_fifos]

    def sequence(a_W, a_X, c_Y, A_prod_hs, B_prod_hs, C_cons_hs):
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                rtp_bufs[row][col][0] = T_TILES
                rtp_bufs[row][col][1] = NBG
        # The A fill happens once per T-tile GROUP, not once per T-tile: that is the
        # weight-stationary schedule. Every C drain is still issued for every T-tile, so
        # the output layout is byte-identical to the baseline's.
        for rbg in range(NRB):
            for g in range(NTG):
                tg = TaskGroup()
                for row in range(n_aie_rows):
                    A_prod_hs[row].fill(a_W, tap=a_tap(row, rbg), group=tg)
                for j in range(R):
                    for col in range(n_aie_cols):
                        B_prod_hs[col].fill(a_X, tap=b_tap(col, g * R + j), group=tg)
                for j in range(R):
                    for col in range(n_aie_cols):
                        C_cons_hs[col].drain(c_Y, tap=c_tap(rbg, g * R + j, col),
                                             wait=True, group=tg)
                tg.finish()

    rt = Runtime(sequence, [W_ty, X_ty, Y_ty, A_prods, B_prods, C_conss])
    program = Program(iron.get_current_device(), rt, workers=[w for row in workers for w in row])
    return program.resolve_program()


DESIGN = ws_poc
SPECIALIZE = {"n_weight": N_WEIGHT, "k": K, "t": T, "tile_n": TILE_N, "fmt": FMT,
              "r_tiles": R_TILES}

if __name__ == "__main__":
    tr = traffic_model(N_WEIGHT, K, T, TILE_N, R_TILES)
    b = l1_budget()
    print(f"ws_poc N={N_WEIGHT} K={K} T={T} tile_n={TILE_N} R={tr['R']} NTG={tr['NTG']} "
          f"NSCR={NSCR}")
    print(f"  per core: {tr['band_groups_per_core']} band groups, {tr['matmuls']} matmuls, "
          f"{tr['dequant_ws'] // (N_AIE_ROWS * N_AIE_COLS)} dequants")
    print(f"  weight MB   ws {tr['weight_bytes_ws'] / 1e6:8.1f}  baseline "
          f"{tr['weight_bytes_base'] / 1e6:8.1f}  ({tr['weight_ratio']:.2f}x less)")
    print(f"  dequant      ws {tr['dequant_ws']:8d}  baseline {tr['dequant_base']:8d}  "
          f"({tr['dequant_ratio']:.2f}x less)")
    print(f"  activation MB {tr['activation_bytes'] / 1e6:.1f}   accumulator MB "
          f"{tr['accumulator_bytes'] / 1e6:.1f}   spill B {tr['accumulator_spill_bytes']}")
    print(f"  L1/core data {b['data']} B (scratch {b['scratch']} + nibble {b['nibble']} + "
          f"accum {b['accum']}) + fifos {b['fifo']} B = {b['total']} B")
