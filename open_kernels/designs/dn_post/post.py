r"""DeltaNet post step on the NPU (one core): og = bf16(rms128(o)*nw * silu(z)).

Args: o f32[4096] (32 heads x 128), z f32[4096], nw bf16[2048] (first 128 used; a
4 KB element), og bf16[4096] (out).
Streams: in = 4 KB elements [nw][o g0][z g0][o g1][z g1]... (4 groups of 8 heads),
out = 2 KB elements, one per group.
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
from recipes.wide_deltanet_layer import post_groups

HEADS = int(os.environ.get("DN_POST_HEADS", "32"))
D = HEADS * 128
G = 1024            # floats per group (8 heads)
NG = post_groups(HEADS)
CARRY = os.environ.get("POST_CARRY") == "1"
TRACE = os.environ.get("POST_TRACE") == "1"


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dn_post(o: In, z: In, nw: In, og: Out, trace: Out, *, srchash: CompileTime[int] = 0):
    u8i = np.ndarray[(4096,), np.dtype[np.uint8]]
    u8o = np.ndarray[(2048,), np.dtype[np.uint8]]
    nw_ty = np.ndarray[(128,), np.dtype[bfloat16]]
    f_ty = np.ndarray[(D,), np.dtype[np.float32]]
    b_in = np.ndarray[(2048,), np.dtype[bfloat16]]
    b_out = np.ndarray[(D,), np.dtype[bfloat16]]
    trace_ty = np.ndarray[(6 * G,), np.dtype[np.float32]]
    traces_ty = np.ndarray[(6 * D,), np.dtype[np.float32]]
    inc = include_dirs() + [str(HERE.parent / "ln"), str(HERE.parent / "layer_x")]
    fn = ExternalFunction("post_fn", source_file=str(HERE / "post.cc"),
                          arg_types=[u8i, u8i, nw_ty, u8o] + ([trace_ty] if TRACE else []), include_dirs=inc,
                          compile_flags=["-DPOST_PRECISE=1", f"-DPOST_TRACE={int(TRACE)}", f"-DPOST_CARRY={int(CARRY)}"])
    fcopy = ExternalFunction("post_copy_nw", source_file=str(HERE / "post_copy.cc"),
                             arg_types=[u8i, nw_ty], include_dirs=inc)
    of_in = ObjectFifo(u8i, name="in", depth=2)
    of_out = ObjectFifo(u8o, name="out", depth=2)
    tf = ObjectFifo(trace_ty, name="trace", depth=1) if TRACE else None
    nwb = Buffer(nw_ty, name="nwb")

    def core_body(ain, aout, nwb, f, fc, *extra):
        e = ain.acquire(1)
        fc(e, nwb)
        ain.release(1)
        for _ in range_(NG):
            e = ain.acquire(2)
            r = aout.acquire(1)
            if TRACE:
                tr = extra[0].acquire(1)
                f(e[0], e[1], nwb, r, tr)
                extra[0].release(1)
            else:
                f(e[0], e[1], nwb, r)
            aout.release(1)
            ain.release(2)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), nwb, fn, fcopy] + ([tf.prod()] if TRACE else []),
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(a_o, a_z, a_nw, c_og, *rest):
        c_trace, inp, outc, tc = rest if TRACE else (None, *rest, None)
        pipe = Pipeline(3)
        pipe.fill(inp, a_nw, TensorAccessPattern((1, 2048), 0, [1, 1, 1, 2048], [0, 0, 0, 1]))
        for g in range(NG):
            if TRACE:
                pipe.drain(tc, c_trace, TensorAccessPattern((1, 6 * D), g * 6 * G, [1, 1, 1, 6 * G], [0, 0, 0, 1]))
            pipe.drain(outc, c_og, TensorAccessPattern((1, D), g * G, [1, 1, 1, G], [0, 0, 0, 1]))
            pipe.fill(inp, a_o, TensorAccessPattern((1, D), g * G, [1, 1, 1, G], [0, 0, 0, 1]))
            pipe.fill(inp, a_z, TensorAccessPattern((1, D), g * G, [1, 1, 1, G], [0, 0, 0, 1]))
        pipe.finish()

    rt = Runtime(sequence, [f_ty, f_ty, b_in, b_out] + ([traces_ty] if TRACE else []) +
                 [of_in.prod(), of_out.cons()] + ([tf.cons()] if TRACE else []))
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = dn_post
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath_precise.h").read_bytes()]) + f"heads={HEADS},trace={TRACE},carry={CARRY}".encode() + Path(__file__).read_bytes()
if CARRY:
    _src += b"".join(p.read_bytes() for p in [HERE / "post_carry.h", HERE.parent / "ln/ln_rne.h",
                   HERE.parent / "layer_x/sigmoid_series.h", *sorted((HERE.parent.parent / "include").glob("fp32_*.h"))])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
