r"""moe_batch: the routed experts of a MoE layer over a block of tokens, NT
token slots per expert visit (OPEN-MOE-BATCH, stage 2 of the block prefill route).

Per token the sequential stream (layer_x's moe_body) re-reads its eight experts'
2 MB each; over a 256-token block that is 181 GB per layer set. Here an expert
is streamed once for up to NT of its tokens: per expert slot the host gathers
the tokens into x[slot] = [HID, NT] bf16 (tokens as the fast axis), the kernel
computes h = silu(gate x) * (up x) and y = down h, and the host scatters
y[slot] = [HID, NT] f32 back with the router weights. An expert with more than
NT tokens takes several slots (a slot is patched to any expert, the way
moeroute2 patches mx); what does not fit one dispatch goes into a shorter one.

NT is a whole number of 8-token MAC tiles (NG = NT / 8 of them) and comes from
MB_NT in the environment, because the driver reads the same number out of the
manifest. Every buffer and every core loop scales with it; at NG = 1 this is the
kernel exactly as it was before the parameter existed.

Mapping: one expert per column, eight in flight, the four rows of a column
splitting the expert's output rows 64 at a time (up and gate: row r owns
stripe r, its two bands; down: 2048 rows = eight groups of four bands). The
weight side is gemm_q4_prefill's stream: one 10240 B band-k-group per element,
four k-tiles of 64 each. A 128-row expert stripe is two bands interleaved at
k-tile granularity, so a band is a strided read of the pool as packed
(specs/open-engine/tests/test_moe_batch.py derives the offsets from the
packer): no repack, no new weight law.

The product is taken transposed (moe_batch.h): the eight tokens are the mmul's
A rows and the weight is its B operand, whose 8 k x 8 rows block is what a
q4_1 chunk's nibbles already are (k-major, 16 rows per k), so a weight tile
costs a mask and a convert, no gather. The even and odd rows come out as
separate C blocks; the silu kernel puts h's rows back in order for the down,
and the host un-interleaves y (layout.py).

The four rows' bands of one k-group travel as ONE 40 KB shim element and are
split at the column's mem tile (a shim tile has two input channels: one for
weights, one for activations). x is broadcast from the shim to the four rows;
h and y leave through the column's C join in row order, so h lands in DDR as
the [512 k, 8 tokens] A tiles the down projection reads back.

The core program does not depend on the slot count: one xclbin serves every
dispatch length, each its own instruction stream (MB_SLOTS).

Three timing-only ablations, all of which compute garbage: MB_NULL_MM=1 drops the
core arithmetic and leaves the streams, MB_NULL_DQ=1 keeps the product but drops
the q4_1 scales, and MB_CONTIG=1 makes the weight taps read the pool in order
(same bytes, same footprint, no band stride).

Build (WSL): MB_SLOTS=256 MB_NT=16 python build_design.py designs/moe_batch/moe_batch.py designs/moe_batch/build_s256
Test:        python make_test.py --slots 16 (builds against a 16-expert pool: MB_SLOTS=16 MB_EXPERTS=16)
             ..\..\harness\out\run_kernel.exe run_s16.cfg && python compare.py s16
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

N_ROWS, N_COLS = 4, 8
# Token slots per expert visit, a whole number of 8-row MAC tiles (NG of them). Widening it
# is what amortises an expert's 1.97 MB over more of its tokens (.claude/plans/prefill-parity.md,
# B(1)); the core program's arithmetic per token does not change, only how often the weights
# are streamed. The driver reads the same number out of the manifest (`moe_batch.nt`), so it
# comes from the environment the recipe's `builds` entry sets, not from a literal here.
NT = int(os.environ.get("MB_NT", 8))
NG = NT // 8
BAND = 10240                # one band-k-group: 64 rows x 256 K of q4_1
CHUNK = 5120
K_TILE = 64

# ---- the model's expert geometry (env, the 35B's by default) and the slot count
HID = int(os.environ.get("MB_HID", 2048))
FF = int(os.environ.get("MB_FF", 512))
SLOTS = int(os.environ.get("MB_SLOTS", 256))
EXPERTS = int(os.environ.get("MB_EXPERTS", 256))      # experts in the pool (the placeholder range)
STRIPE = 128 * HID * 5 // 8                            # one 128-row up (or gate) stripe
SPP = FF // 128                                        # stripes per projection
EXP_UG = 2 * SPP * STRIPE                              # one expert's up + gate stripes
DOWN_BAND = 128 * FF * 5 // 8                          # one 128-row band of the down law
DOWN_BYTES = HID * FF * 5 // 8                         # one expert's down
POOL_DOWN = int(os.environ.get("MB_POOL_DOWN", EXPERTS * EXP_UG))
POOL_BYTES = int(os.environ.get("MB_POOL_BYTES", 512 << 20))
# MB_CONTIG=1: the weight taps walk the pool in order instead of by band stride. Same bytes,
# same footprint, wrong arithmetic - a probe for whether the stride costs stream rate.
CONTIG = os.environ.get("MB_CONTIG") == "1"
X_BYTES = SLOTS * HID * NT * 2
H_BYTES = SLOTS * FF * NT * 2
Y_BYTES = SLOTS * HID * NT * 4
NBG_UP, NRB_UP = HID // 256, FF // (64 * N_ROWS)       # band-k-groups per band, band groups per projection
NBG_DN, NRB_DN = FF // 256, HID // (64 * N_ROWS)

# The core's 64 KB of L1, spent: the stack, the weight fifo (one band-k-group, double
# buffered), the f32 up and gate accumulators, the C join, and the B fifo -- which the down
# phase holds ENTIRELY, because every output band needs all of h, and which the toolchain
# gives one buffer more than the depth asked for. Widening NT grows everything but the
# weights, and at NT = 16 what it takes is the C join's second buffer: the drain is one
# 4 KB element against a whole band's worth of mmul, so the core waits on it for very
# little, where a shallower weight fifo would stall the stream that is the point of all
# this. NT = 32 needs the down to stream h in k-tiles instead of holding it.
C_DEPTH = 2 if NT <= 8 else 1
L1_BYTES = (0x1000 + 2 * BAND + 2 * (NRB_UP * 64 * NT * 4) + C_DEPTH * (64 * NT * 4)
            + (NBG_DN * 4 + 1) * (K_TILE * NT * 2))

assert NT % 8 == 0 and NT >= 8, "NT is a whole number of 8-token MAC tiles"
assert L1_BYTES <= 64 << 10, f"NT={NT} wants {L1_BYTES} B of a core's 65536"
assert HID % 256 == 0 and FF % 256 == 0, "HID and FF must be multiples of 256 (one band-k-group)"
assert SLOTS % N_COLS == 0 and SLOTS <= EXPERTS, "slots come in rounds of eight, one expert per column"
assert SPP == N_ROWS, "up / gate: one 128-row stripe per row (FF = 512)"
assert POOL_DOWN + EXPERTS * DOWN_BYTES <= POOL_BYTES


def _srchash() -> int:
    parts = [f.read_bytes() for f in sorted(HERE.glob("*.cc")) + sorted(HERE.glob("*.h")) + [HERE / "moe_batch.py"]]
    parts += [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()]
    # NT reaches the kernels as a -D, not as source, so it has to be hashed in by hand or two
    # widths would share a compiled core program.
    parts += [f"NT={NT}".encode()]
    return int(hashlib.sha1(b"".join(parts)).hexdigest()[:8], 16)


def lin(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((total,), off, [1, n], [n, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def moe_batch(pool: In, x: In, h: Out, y: Out, *, slots: CompileTime[int], srchash: CompileTime[int] = 0):
    band_ty = np.ndarray[(BAND,), np.dtype[np.uint8]]
    a4_ty = np.ndarray[(N_ROWS * BAND,), np.dtype[np.uint8]]
    b_ty = np.ndarray[(K_TILE * NT,), np.dtype[bfloat16]]       # one k-tile of x or h: 8 NG A blocks of [8, 8]
    c_ty = np.ndarray[(64 * NT,), np.dtype[np.float32]]         # one band of output: 8 NG C blocks of [8, 8]
    c4_ty = np.ndarray[(N_ROWS * 64 * NT,), np.dtype[np.float32]]
    ug_ty = np.ndarray[(NRB_UP * 64 * NT,), np.dtype[np.float32]]   # the core's up (or gate) rows
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(X_BYTES,), np.dtype[np.uint8]]
    h_ty = np.ndarray[(H_BYTES,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(Y_BYTES,), np.dtype[np.uint8]]

    inc = include_dirs()
    # EVERY kernel here sizes its buffers off MB_NT, so every one of them gets the -D. (When
    # only the two product kernels had it, the zero and silu kernels compiled at the default
    # 8 and left the second half of the up / gate accumulator untouched: h came back NaN from
    # exactly the bands mb_zero_ug had stopped clearing.)
    # Timing-only ablations (output garbage): MB_NULL_MM=1 skips the core work and leaves the
    # streams, MB_NULL_DQ=1 keeps the product but drops the q4_1 scales.
    flags = [f"-DMB_NT={NT}"] + [f"-D{v}" for v in ("MB_NULL_MM", "MB_NULL_DQ") if os.environ.get(v) == "1"]
    step_ug = ExternalFunction("mb_step_ug", source_file=str(HERE / "mb_step_ug.cc"),
                               arg_types=[band_ty, b_ty, ug_ty, np.int32, np.int32], include_dirs=inc, compile_flags=flags)
    step_dn = ExternalFunction("mb_step_dn", source_file=str(HERE / "mb_step_dn.cc"),
                               arg_types=[band_ty, b_ty, c_ty, np.int32], include_dirs=inc, compile_flags=flags)
    zero_ug = ExternalFunction("mb_zero_ug", source_file=str(HERE / "mb_zero_ug.cc"), arg_types=[ug_ty],
                               include_dirs=inc, compile_flags=flags)
    zero_y = ExternalFunction("mb_zero_y", source_file=str(HERE / "mb_zero_y.cc"), arg_types=[c_ty],
                              include_dirs=inc, compile_flags=flags)
    silu = ExternalFunction("mb_silu", source_file=str(HERE / "mb_silu.cc"), arg_types=[ug_ty, ug_ty, c_ty],
                            include_dirs=inc, compile_flags=flags)

    # ---- weights: one 4-row element per column into the mem tile, split by row
    A_l3l2 = [ObjectFifo(a4_ty, name=f"A{c}", depth=2) for c in range(N_COLS)]
    A_l2l1 = [A_l3l2[c].cons().split([r * BAND for r in range(N_ROWS)], obj_types=[band_ty] * N_ROWS,
                                     names=[f"A{c}_{r}" for r in range(N_ROWS)], depths=[2] * N_ROWS, tile=Tile(c, 1))
              for c in range(N_COLS)]
    # ---- activations: x (up / gate) and h (down) k-tiles, broadcast to the column's rows; depth 8
    # so the down holds the whole [512, 8] h at once
    B = [ObjectFifo(b_ty, name=f"B{c}", depth=NBG_DN * 4) for c in range(N_COLS)]
    # ---- output: the rows join in order (h: the core's 128 hidden rows; y: its 64-row band)
    C_l2l3 = [ObjectFifo(c4_ty, name=f"C{c}", depth=2) for c in range(N_COLS)]
    C_l1l2 = [C_l2l3[c].prod().join([r * 64 * NT for r in range(N_ROWS)], obj_types=[c_ty] * N_ROWS,
                                    names=[f"C{c}_{r}" for r in range(N_ROWS)], depths=[C_DEPTH] * N_ROWS,
                                    tile=Tile(c, 1))
              for c in range(N_COLS)]

    ubuf = [[Buffer(ug_ty, name=f"u_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]
    gbuf = [[Buffer(ug_ty, name=f"g_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]

    # One expert per pass; the slot count is a property of the instruction stream. The A
    # element is one band-k-group (4 k-tiles); the x / h k-tiles ride the B fifo one per k-tile.
    def core_fn(in_a, in_b, out_c, k_ug, k_dn, k_zero_ug, k_zero_y, k_silu, u_s, g_s):
        k_zero_ug(u_s)
        k_zero_ug(g_s)
        for proj in (u_s, g_s):
            for rbg in range(NRB_UP):
                for _ in range_(NBG_UP):
                    band = in_a.acquire(1)
                    for ky in range(4):
                        xe = in_b.acquire(1)
                        k_ug(band, xe, proj, ky, rbg)
                        in_b.release(1)
                    in_a.release(1)
        he = out_c.acquire(1)
        k_silu(u_s, g_s, he)
        out_c.release(1)
        hb = in_b.acquire(NBG_DN * 4)                     # the whole h, one k-tile per element
        for _ in range_(NRB_DN):
            ye = out_c.acquire(1)
            k_zero_y(ye)
            for g in range(NBG_DN):
                band = in_a.acquire(1)
                for ky in range(4):
                    k_dn(band, hb[g * 4 + ky], ye, ky)
                in_a.release(1)
            out_c.release(1)
        in_b.release(NBG_DN * 4)

    workers = [Worker(core_fn,
                      [A_l2l1[c][r].cons(), B[c].cons(), C_l1l2[c][r].prod(), step_ug, step_dn, zero_ug, zero_y, silu,
                       ubuf[r][c], gbuf[r][c]],
                      tile=Tile(c, 2 + r), stack_size=0x1000)
               for c in range(N_COLS) for r in range(N_ROWS)]

    # ---- taps. Up / gate: row r owns stripe r (its two bands, rbg = the band inside the stripe),
    # so the core's u / g rows and the h it drains are contiguous rows 128r.. of the expert. A 4-row
    # element is the four stripes' band rbg at one k-group. Down: row r owns band rbg*4 + r, so a
    # drain is y rows rbg*256.. in order; rows 0, 1 (one stripe) are contiguous 20 KB. The innermost
    # wrap stays under 4 KB.
    def up_tap(slot: int, proj: int, rbg: int) -> TensorAccessPattern:
        if CONTIG:                                        # traffic probe: same bytes, read in order
            off = slot * EXP_UG + (proj * NRB_UP + rbg) * (NBG_UP * N_ROWS * BAND)
            return TensorAccessPattern((POOL_BYTES,), off, [NBG_UP, N_ROWS, 4, 2560], [N_ROWS * BAND, BAND, 2560, 1])
        off = slot * EXP_UG + proj * STRIPE + rbg * BAND
        return TensorAccessPattern((POOL_BYTES,), off, [NBG_UP, N_ROWS, 4, 2560], [2 * BAND, 2 * STRIPE, 2560, 1])

    def down_tap(slot: int, rbg: int) -> TensorAccessPattern:
        if CONTIG:
            off = POOL_DOWN + slot * DOWN_BYTES + rbg * (NBG_DN * 4 * BAND)
            return TensorAccessPattern((POOL_BYTES,), off, [NBG_DN, 2, 8, 2560], [4 * BAND, 2 * BAND, 2560, 1])
        off = POOL_DOWN + slot * DOWN_BYTES + rbg * 2 * DOWN_BAND
        return TensorAccessPattern((POOL_BYTES,), off, [NBG_DN, 2, 8, 2560], [2 * BAND, DOWN_BAND, 2560, 1])

    A_prods = [A_l3l2[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    B_prods = [B[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    C_conss = [C_l2l3[c].cons(tile=Tile(c, 0)) for c in range(N_COLS)]

    # Every phase is issued across all eight columns before the next, so a throttle wait on one
    # column's oldest fill never leaves the others without work (issued column by column, the
    # down phase serialised the columns and the stream ran at half its rate).
    def sequence(a_pool, a_x, a_h, a_y, A_hs, B_hs, C_hs):
        pw, px, py = Pipeline(3), Pipeline(3), Pipeline(3)
        for rnd in range(slots // N_COLS):
            slot = [rnd * N_COLS + c for c in range(N_COLS)]
            for proj in range(2):
                for rbg in range(NRB_UP):
                    for c in range(N_COLS):
                        pw.fill(A_hs[c], a_pool, up_tap(slot[c], proj, rbg))
                        px.fill(B_hs[c], a_x, lin(X_BYTES, slot[c] * HID * NT * 2, HID * NT * 2))
            for c in range(N_COLS):
                py.drain(C_hs[c], a_h, lin(H_BYTES, slot[c] * FF * NT * 2, FF * NT * 2))
            for c in range(N_COLS):
                py.finish(C_hs[c])                        # h is in DDR
            for c in range(N_COLS):
                px.fill(B_hs[c], a_h, lin(H_BYTES, slot[c] * FF * NT * 2, FF * NT * 2))
            for rbg in range(NRB_DN):
                for c in range(N_COLS):
                    pw.fill(A_hs[c], a_pool, down_tap(slot[c], rbg))
            for c in range(N_COLS):
                py.drain(C_hs[c], a_y, lin(Y_BYTES, slot[c] * HID * NT * 4, HID * NT * 4))
        pw.finish()
        px.finish()
        py.finish()

    rt = Runtime(sequence, [pool_ty, x_ty, h_ty, y_ty, A_prods, B_prods, C_conss])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = moe_batch
SPECIALIZE = {"slots": SLOTS, "srchash": _srchash()}

if __name__ == "__main__":
    print(f"HID={HID} FF={FF} NT={NT} L1={L1_BYTES} SLOTS={SLOTS} EXPERTS={EXPERTS} POOL_DOWN={POOL_DOWN} "
          f"POOL_BYTES={POOL_BYTES} "
          f"x={X_BYTES} h={H_BYTES} y={Y_BYTES}")
