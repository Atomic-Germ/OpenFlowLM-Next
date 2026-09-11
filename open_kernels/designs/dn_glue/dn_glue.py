r"""Linear-attention layer glue on the NPU (decode, one core): conv1d + SiLU +
q/k L2-norm, alpha/beta projections -> decay/beta, per-head records for the
DeltaNet step, new conv state. Math in dn_glue.h.

Args:
  side   uint8[SIDE_BYTES]   our packed side blob (4 KB elements):
                             [xn bf16 2048][Wa bf16 2048x32 = 32 elems][Wb 32 elems]
                             [small f32: A[32], dt_bias[32], pad][convw bf16 [8 tiles][4][1024]
                             = 2 elems per tile]
  qkv    f32[8192]           this token's qkv projection
  state  bf16[3*8192]        conv state rows (OFLM layout)
  nstate bf16[3*8192]        out: shifted state [s1, s2, bf16(qkv)]
  vec    f32[32*512]         out: per-head records for designs/deltanet

Streams: side = 4 KB elements, one fill; act = 2 KB elements, per tile 2 (qkv)
+ 3 (state rows) from two fills; out = 2 KB elements, per tile 3 state rows,
then 8 records for v tiles. Fills/drains are throttled by ironutil.Pipeline
(shim start queue = 4 BDs). L1 note: the objectfifo lowering allocates
depth+1 buffers when a side acquires `depth` at once, so element sizes were
chosen from the aie-opt memory map (stack 3.3 KB + qk 16 + vt 4 + side 4x4 +
act 6x2 + out 4x2 = ~60 KB).
"""

from __future__ import annotations

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

# DNGLUE_NHEAD=16 builds the Qwen3.5 2B / 0.8B point: 16 linear value heads over the same
# 16 key heads (so one value head per key head, not two), 6144 conv channels, and an alpha /
# beta projection zero-padded out to the accumulator's 32 lanes. make_test.py slices the
# captured 32-head fixture down to it; compare.py reads the same knob.
NHEAD = int(os.environ.get("DNGLUE_NHEAD", 32))
HID, HD = 2048, 128
KEY_WIDTH = 2048
NCH = 2 * KEY_WIDTH + NHEAD * HD             # 8192 at 32 heads, 6144 at 16
TILE = 1024
NT = NCH // TILE
KEY_TILES = 2 * KEY_WIDTH // TILE            # 4
VALUE_TILES = NT - KEY_TILES                 # 4 at 32 heads, 2 at 16
HEADS_PER_TILE = TILE // HD                  # 8
SIDE_ELEM = 4096
ACT_ELEM = 2048
AB_LANES = 32                                # dn_glue.h's kV: a W element is 64 rows x 32 bf16
AB_ELEMS = HID * AB_LANES * 2 // SIDE_ELEM   # 32, whatever the head count
SIDE_ELEMS = 1 + AB_ELEMS + AB_ELEMS + 1 + 2 * NT    # xn, Wa, Wb, small, convw
SIDE_BYTES = SIDE_ELEMS * SIDE_ELEM          # 335872 at 32 heads, 319488 at 16
# dn_glue.h's default IS 32, so the 32-head build's compile command is unchanged.
GLUE_FLAGS = {} if NHEAD == 32 else {"compile_flags": [f"-DDNGLUE_NHEAD={NHEAD}"]}


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dn_glue(side: In, qkv: In, state: In, nstate: Out, vec: Out, *, dummy: CompileTime[int] = 0):
    u8s = np.ndarray[(SIDE_ELEM,), np.dtype[np.uint8]]
    u8a = np.ndarray[(ACT_ELEM,), np.dtype[np.uint8]]
    f32 = np.ndarray[(AB_LANES,), np.dtype[np.float32]]
    fqk = np.ndarray[(2 * KEY_WIDTH,), np.dtype[np.float32]]
    fvt = np.ndarray[(TILE,), np.dtype[np.float32]]
    fxn = np.ndarray[(HID,), np.dtype[bfloat16]]
    side_ty = np.ndarray[(SIDE_BYTES,), np.dtype[np.uint8]]
    qkv_ty = np.ndarray[(NCH,), np.dtype[np.float32]]
    st_ty = np.ndarray[(3 * NCH,), np.dtype[bfloat16]]
    vec_ty = np.ndarray[(NHEAD * 512,), np.dtype[np.float32]]

    inc = include_dirs()
    f_ab = ExternalFunction("glue_ab", source_file=str(HERE / "glue_ab.cc"),
                            arg_types=[u8s, fxn, f32, np.int32], include_dirs=inc, **GLUE_FLAGS)
    f_small = ExternalFunction("glue_small_fn", source_file=str(HERE / "glue_small.cc"),
                               arg_types=[u8s, f32, f32, f32, f32], include_dirs=inc, **GLUE_FLAGS)
    f_conv = ExternalFunction("glue_conv", source_file=str(HERE / "glue_conv.cc"),
                              arg_types=[u8a, u8a, u8a, u8a, u8a, u8s, u8s, u8a, u8a, u8a, fqk, fvt, np.int32, np.int32],
                              include_dirs=inc, **GLUE_FLAGS)
    f_emit = ExternalFunction("glue_emit_fn", source_file=str(HERE / "glue_emit.cc"),
                              arg_types=[fqk, fvt, f32, f32, u8a, np.int32, np.int32], include_dirs=inc, **GLUE_FLAGS)
    f_copy = ExternalFunction("glue_copy_xn", source_file=str(HERE / "glue_copy.cc"),
                              arg_types=[u8s, fxn], include_dirs=inc, **GLUE_FLAGS)

    of_side = ObjectFifo(u8s, name="side", depth=2)
    of_act = ObjectFifo(u8a, name="act", depth=5)
    of_out = ObjectFifo(u8a, name="out", depth=3)

    acc_a = Buffer(f32, name="acc_a")
    acc_b = Buffer(f32, name="acc_b")
    decay = Buffer(f32, name="decay")
    beta = Buffer(f32, name="beta")
    qk = Buffer(fqk, name="qk")
    vt = Buffer(fvt, name="vt")
    xn = Buffer(fxn, name="xn")

    # Rolled runtime loops: unrolling these in Python overflowed the core's
    # 16 KB program memory (XAie_LoadElf: XAIE_INVALID_ELF). Tile indices go
    # to the kernels as runtime ints.
    def core_body(sin, ain, oout, acc_a, acc_b, decay, beta, qk, vt, xn, fab, fsmall, fconv, femit, fcopy):
        e0 = sin.acquire(1)
        fcopy(e0, xn)                             # xn -> scratch (release(n) frees the OLDEST n)
        sin.release(1)
        for acc in (acc_a, acc_b):
            for tile in range_(AB_ELEMS):
                w = sin.acquire(1)
                fab(w, xn, acc, tile)
                sin.release(1)
        sm = sin.acquire(1)
        fsmall(sm, acc_a, acc_b, decay, beta)
        sin.release(1)
        for base, ntiles in ((0, KEY_TILES), (KEY_TILES, VALUE_TILES)):
            for t in range_(ntiles):
                w = sin.acquire(2)
                e = ain.acquire(5)
                o = oout.acquire(3)
                fconv(e[0], e[1], e[2], e[3], e[4], w[0], w[1], o[0], o[1], o[2], qk, vt, t, base)
                oout.release(3)
                ain.release(5)
                sin.release(2)
                if base == KEY_TILES:
                    for i in range_(HEADS_PER_TILE):
                        r = oout.acquire(1)
                        femit(qk, vt, decay, beta, r, t, i)
                        oout.release(1)

    worker = Worker(core_body,
                    fn_args=[of_side.cons(), of_act.cons(), of_out.prod(),
                             acc_a, acc_b, decay, beta, qk, vt, xn, f_ab, f_small, f_conv, f_emit, f_copy],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(a_side, a_qkv, a_state, c_nstate, c_vec, side_p, act_p, out_c):
        pipe = Pipeline(3)
        pipe.fill(side_p, a_side, TensorAccessPattern((1, SIDE_BYTES), 0, [1, 1, 1, SIDE_BYTES], [0, 0, 0, 1]))
        for t in range(NT):
            pipe.drain(out_c, c_nstate, TensorAccessPattern((3, NCH), t * TILE, [1, 1, 3, TILE], [0, 0, NCH, 1]))
            if t >= KEY_TILES:
                pipe.drain(out_c, c_vec, TensorAccessPattern((1, NHEAD * 512), (t - KEY_TILES) * HEADS_PER_TILE * 512,
                                                             [1, 1, 1, HEADS_PER_TILE * 512], [0, 0, 0, 1]))
            pipe.fill(act_p, a_qkv, TensorAccessPattern((1, NCH), t * TILE, [1, 1, 1, TILE], [0, 0, 0, 1]))
            pipe.fill(act_p, a_state, TensorAccessPattern((3, NCH), t * TILE, [1, 1, 3, TILE], [0, 0, NCH, 1]))
        pipe.finish()

    rt = Runtime(sequence, [side_ty, qkv_ty, st_ty, st_ty, vec_ty,
                            of_side.prod(), of_act.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = dn_glue
# IRON's design cache is keyed on this function's source + CompileTime args and
# never looks at the kernel sources, so a header edit alone silently reuses the
# old xclbin. Pass a hash of every source file in this directory.
import hashlib as _h
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h")))
SPECIALIZE = {"dummy": int(_h.sha1(_src + str(NHEAD).encode()).hexdigest()[:8], 16)}
