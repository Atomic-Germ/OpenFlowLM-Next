r"""mbl: LFM2.5-24B-A2B's routed experts over a block of tokens, NT token slots per expert visit (the block
prefill's expert kernel; gemma4_moe_batch/mb4.py's design on the Flint / LFM2-MoE pool law).

Per expert slot the host gathers its tokens into x[slot] = [HID, NT] bf16, the kernel computes
h = silu(gate x) * (up x) and y = down h, and the host scatters y[slot] = [HID, NT] f32 back with the
router weights. An expert is streamed once for up to NT of its tokens instead of once per token.

What differs from mb4.py:
  * q4_1 chunks (5120 B: bf16 d plane, bf16 m plane, codes), the fl_q4 pool law recipes/lfm2moe.py
    writes: 64-row bands, per expert [gate band b | up band b] for b < 24, then (POOL_DOWN) its 32 down
    bands. The m plane is read as stored (mb4 forms m = -8 d on the core).
  * HID 2048, FF 1536: a core owns SIX gate / up bands (band 4 rbg + r) and EIGHT down bands. Six bands
    of u and g in L1 do not fit beside h (24 KB), so the up / gate loop runs per rbg -- gate band, up band,
    silu -> one h element -- and u / g hold one band.
  * SwiGLU (silu) instead of GeGLU.
  * x / h / y layouts as mb4 (layout.py), at these dimensions.

Build: MBL_SLOTS=16 MBL_EXPERTS=16 python build_design.py designs/lfm2moe_batch/mbl.py <out>
Test:  python make_test.py --slots 16 ; run_kernel run_s16.cfg ; python compare.py s16
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
NT = int(os.environ.get("MBL_NT", 8))
NG = NT // 8
CHUNK = 5120
BAND = 2 * CHUNK            # one band-k-group: 64 rows x 256 K of Q4_0-native
K_TILE = 64
ELEM_B = 32 * NT * 4        # one output element: h band (64 rows bf16) or a y half (32 rows f32)

HID = int(os.environ.get("MBL_HID", 2048))
FF = int(os.environ.get("MBL_FF", 1536))
SLOTS = int(os.environ.get("MBL_SLOTS", 64))
EXPERTS = int(os.environ.get("MBL_EXPERTS", 64))
NBG_UP, NBG_DN = HID // 256, FF // 256                 # band-k-groups down a gate / up band, a down band
NB_UP, NB_DN = FF // 64, HID // 64                     # bands in a projection, in the down
NRB_UP, NRB_DN = NB_UP // N_ROWS, NB_DN // N_ROWS      # bands a core owns
BB_H = 2 * NBG_UP * CHUNK                              # one gate (or up) band: 101376 B
BB_F = 2 * NBG_DN * CHUNK                              # one down band: 27648 B
EXP_UG = 2 * NB_UP * BB_H                              # gate | up bands of one expert
UP_BYTES = NB_DN * BB_F                                # one expert's down (= one projection, as moeroute2 needs)
POOL_DOWN = int(os.environ.get("MBL_POOL_DOWN", EXPERTS * EXP_UG))
POOL_BYTES = int(os.environ.get("MBL_POOL_BYTES", POOL_DOWN + EXPERTS * UP_BYTES))
X_BYTES = SLOTS * HID * NT * 2
H_BYTES = SLOTS * FF * NT * 2
Y_BYTES = SLOTS * HID * NT * 4

# A core has 16 buffer descriptors for all its channels, so the B fifo cannot hold h (12 k-tiles):
# the down copies h out of a shallow fifo into an L1 buffer, once, and reads the tiles from there.
B_DEPTH = 4
UG_BANDS = 1                                           # u / g bands held in L1 (the loop is per rbg)
FLAGS = ("MBL_NULL_MM", "MBL_NULL_DQ")
# The core's L1: stack, the weight fifo (a band-k-group, double buffered), u and g (f32), the
# y accumulator, the output join, and the B fifo -- which the down holds entirely (all of h).
L1_BYTES = (0x1000 + 2 * BAND + 2 * (UG_BANDS * 64 * NT * 4) + 64 * NT * 4 + 2 * ELEM_B
            + (B_DEPTH + 1) * (K_TILE * NT * 2) + FF * NT * 2)

assert NT % 8 == 0 and NT >= 8
assert L1_BYTES <= 64 << 10, f"NT={NT} wants {L1_BYTES} B of a core's 65536"
assert HID % 256 == 0 and FF % 256 == 0
assert NB_UP % N_ROWS == 0 and NB_DN % N_ROWS == 0
assert UP_BYTES == NB_UP * BB_H, "an expert's gate and down must be the same size for moeroute2"
assert SLOTS % N_COLS == 0 and SLOTS <= EXPERTS
assert POOL_DOWN + EXPERTS * UP_BYTES <= POOL_BYTES


def _srchash() -> int:
    parts = [f.read_bytes() for f in sorted(HERE.glob("*.cc")) + sorted(HERE.glob("*.h")) + [HERE / "mbl.py"]]
    parts += [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()]
    parts += [f"NT={NT} UG_BANDS={UG_BANDS}".encode()]
    parts += [f"{v}=1".encode() for v in FLAGS if os.environ.get(v) == "1"]
    return int(hashlib.sha1(b"".join(parts)).hexdigest()[:8], 16)


def lin(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((total,), off, [1, n], [n, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def mbl(pool: In, x: In, h: Out, y: Out, *, slots: CompileTime[int], srchash: CompileTime[int] = 0):
    band_ty = np.ndarray[(BAND,), np.dtype[np.uint8]]
    a4_ty = np.ndarray[(N_ROWS * BAND,), np.dtype[np.uint8]]
    b_ty = np.ndarray[(K_TILE * NT,), np.dtype[bfloat16]]       # one k-tile of x or h
    c_ty = np.ndarray[(64 * NT,), np.dtype[np.float32]]         # a band of f32 output
    e_ty = np.ndarray[(ELEM_B // 4,), np.dtype[np.float32]]     # one output element
    e4_ty = np.ndarray[(N_ROWS * ELEM_B // 4,), np.dtype[np.float32]]
    ug_ty = np.ndarray[(UG_BANDS * 64 * NT,), np.dtype[np.float32]]
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(X_BYTES,), np.dtype[np.uint8]]
    h_ty = np.ndarray[(H_BYTES,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(Y_BYTES,), np.dtype[np.uint8]]

    inc = include_dirs()
    flags = [f"-DMBL_NT={NT}", f"-DMBL_NRB_UP={UG_BANDS}"] + [f"-D{v}" for v in FLAGS if os.environ.get(v) == "1"]

    def ext(name, src, args):
        return ExternalFunction(name, source_file=str(HERE / src), arg_types=args, include_dirs=inc, compile_flags=flags)

    step_ug = ext("mbl_step_ug", "mbl_step_ug.cc", [band_ty, b_ty, ug_ty, np.int32, np.int32])
    hb_ty = np.ndarray[(FF * NT,), np.dtype[bfloat16]]
    step_dn = ext("mbl_step_dn", "mbl_step_dn.cc", [band_ty, hb_ty, c_ty, np.int32, np.int32])
    copy_h = ext("mbl_copy_h", "mbl_copy_h.cc", [b_ty, hb_ty, np.int32])
    zero_ug = ext("mbl_zero_ug", "mbl_zero_ug.cc", [ug_ty])
    zero_y = ext("mbl_zero_y", "mbl_zero_y.cc", [c_ty])
    gelu = ext("mbl_silu", "mbl_silu.cc", [ug_ty, ug_ty, e_ty, np.int32])
    copy_y = ext("mbl_copy_y", "mbl_copy_y.cc", [c_ty, e_ty, np.int32])

    A_l3l2 = [ObjectFifo(a4_ty, name=f"A{c}", depth=2) for c in range(N_COLS)]
    A_l2l1 = [A_l3l2[c].cons().split([r * BAND for r in range(N_ROWS)], obj_types=[band_ty] * N_ROWS,
                                     names=[f"A{c}_{r}" for r in range(N_ROWS)], depths=[2] * N_ROWS, tile=Tile(c, 1))
              for c in range(N_COLS)]
    B = [ObjectFifo(b_ty, name=f"B{c}", depth=B_DEPTH) for c in range(N_COLS)]
    C_l2l3 = [ObjectFifo(e4_ty, name=f"C{c}", depth=2) for c in range(N_COLS)]
    C_l1l2 = [C_l2l3[c].prod().join([r * ELEM_B // 4 for r in range(N_ROWS)], obj_types=[e_ty] * N_ROWS,
                                    names=[f"C{c}_{r}" for r in range(N_ROWS)], depths=[2] * N_ROWS,
                                    tile=Tile(c, 1))
              for c in range(N_COLS)]

    ubuf = [[Buffer(ug_ty, name=f"u_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]
    gbuf = [[Buffer(ug_ty, name=f"g_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]
    hbuf = [[Buffer(hb_ty, name=f"h_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]
    ybuf = [[Buffer(c_ty, name=f"y_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]

    def core_fn(in_a, in_b, out_c, k_ug, k_dn, k_zero_ug, k_zero_y, k_gelu, k_copy, k_copy_h, u_s, g_s, y_s, h_s):
        for _ in range_(NRB_UP):
            k_zero_ug(g_s)
            k_zero_ug(u_s)
            for proj in (g_s, u_s):                       # the pool's order: [gate band | up band]
                for _ in range_(NBG_UP):
                    band = in_a.acquire(1)
                    for ky in range(4):
                        xe = in_b.acquire(1)
                        k_ug(band, xe, proj, ky, 0)
                        in_b.release(1)
                    in_a.release(1)
            he = out_c.acquire(1)
            k_gelu(u_s, g_s, he, 0)
            out_c.release(1)
        for t in range(NBG_DN * 4):                       # the whole h, one k-tile per element, into L1
            he = in_b.acquire(1)
            k_copy_h(he, h_s, t)
            in_b.release(1)
        for _ in range_(NRB_DN):
            k_zero_y(y_s)
            for g in range(NBG_DN):
                band = in_a.acquire(1)
                for ky in range(4):
                    k_dn(band, h_s, y_s, g * 4 + ky, ky)
                in_a.release(1)
            for half in range(2):
                ye = out_c.acquire(1)
                k_copy(y_s, ye, half)
                out_c.release(1)

    workers = [Worker(core_fn,
                      [A_l2l1[c][r].cons(), B[c].cons(), C_l1l2[c][r].prod(), step_ug, step_dn, zero_ug, zero_y, gelu,
                       copy_y, copy_h, ubuf[r][c], gbuf[r][c], ybuf[r][c], hbuf[r][c]],
                      tile=Tile(c, 2 + r), stack_size=0x1000)
               for c in range(N_COLS) for r in range(N_ROWS)]

    # ---- taps. Gate / up: row r owns band 4 rbg + r of the projection; the 4-row element at k-group
    # kg is the four bands' band-k-group kg. Bands of one projection are 2 BB_H apart (gate and up
    # interleave); a band-k-group is 9216 contiguous bytes, split [4, 2304] to keep the innermost
    # wrap under 4 KB. Down: row r owns band 4 rbg + r, the bands consecutive.
    def up_tap(slot: int, proj: int, rbg: int) -> TensorAccessPattern:
        off = slot * EXP_UG + (2 * N_ROWS * rbg + proj) * BB_H
        return TensorAccessPattern((POOL_BYTES,), off, [NBG_UP, N_ROWS, 4, BAND // 4], [BAND, 2 * BB_H, BAND // 4, 1])

    def down_tap(slot: int, rbg: int) -> TensorAccessPattern:
        off = POOL_DOWN + slot * UP_BYTES + rbg * N_ROWS * BB_F
        return TensorAccessPattern((POOL_BYTES,), off, [NBG_DN, N_ROWS, 4, BAND // 4], [BAND, BB_F, BAND // 4, 1])

    A_prods = [A_l3l2[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    B_prods = [B[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    C_conss = [C_l2l3[c].cons(tile=Tile(c, 0)) for c in range(N_COLS)]

    def sequence(a_pool, a_x, a_h, a_y, A_hs, B_hs, C_hs):
        pw, px, py = Pipeline(3), Pipeline(3), Pipeline(3)
        for rnd in range(slots // N_COLS):
            slot = [rnd * N_COLS + c for c in range(N_COLS)]
            for c in range(N_COLS):                   # h leaves one band per rbg: drain first, so a core never stalls on it
                py.drain(C_hs[c], a_h, lin(H_BYTES, slot[c] * FF * NT * 2, FF * NT * 2))
            for rbg in range(NRB_UP):
                for proj in range(2):
                    for c in range(N_COLS):
                        pw.fill(A_hs[c], a_pool, up_tap(slot[c], proj, rbg))
                        px.fill(B_hs[c], a_x, lin(X_BYTES, slot[c] * HID * NT * 2, HID * NT * 2))
            for c in range(N_COLS):
                py.finish(C_hs[c])                        # h is in DDR
            for c in range(N_COLS):
                px.fill(B_hs[c], a_h, lin(H_BYTES, slot[c] * FF * NT * 2, FF * NT * 2))
            # y leaves a band row (two 4096 B objects) at a time, interleaved with the weight fills: the
            # core's output holds a few objects, so draining only after the last fill would stall it
            # (the fills' throttle waits on the core consuming them) -- a deadlock at 11 bands.
            yrow = 2 * N_ROWS * ELEM_B
            for rbg in range(NRB_DN):
                for c in range(N_COLS):
                    pw.fill(A_hs[c], a_pool, down_tap(slot[c], rbg))
                for c in range(N_COLS):
                    py.drain(C_hs[c], a_y, lin(Y_BYTES, slot[c] * HID * NT * 4 + rbg * yrow, yrow))
        pw.finish()
        px.finish()
        py.finish()

    rt = Runtime(sequence, [pool_ty, x_ty, h_ty, y_ty, A_prods, B_prods, C_conss])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = mbl
SPECIALIZE = {"slots": SLOTS, "srchash": _srchash()}

if __name__ == "__main__":
    print(f"HID={HID} FF={FF} NT={NT} L1={L1_BYTES} SLOTS={SLOTS} EXPERTS={EXPERTS} POOL_DOWN={POOL_DOWN} "
          f"POOL_BYTES={POOL_BYTES} NRB_UP={NRB_UP} NRB_DN={NRB_DN} x={X_BYTES} h={H_BYTES} y={Y_BYTES}")
