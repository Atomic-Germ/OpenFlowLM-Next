r"""MoE router on the NPU (one core): p = softmax(xm @ W), top-8 + renormalised weights.

Args: xm bf16[2048], W bf16[2048*256] (pack @12288), out f32[1024]
      (out: p[256] @0, idx int32[8] @1024 B, w[8] @1056 B).
Streams: in = 4 KB elements [xm][W as 256 elements of 8 rows] (two fills), out = one 4 KB element.
"""

from __future__ import annotations

import hashlib
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

HID, E = 2048, 256
ELEM = 4096
W_ELEMS = HID * E * 2 // ELEM      # 256


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def router(xm: In, w: In, out: Out, *, srchash: CompileTime[int] = 0):
    u8 = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    xb = np.ndarray[(HID,), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(E,), np.dtype[np.float32]]
    w_ty = np.ndarray[(HID * E,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(1024,), np.dtype[np.float32]]
    inc = include_dirs()
    f_copy = ExternalFunction("router_copy_x", source_file=str(HERE / "router_copy.cc"), arg_types=[u8, xb], include_dirs=inc)
    f_acc = ExternalFunction("router_acc", source_file=str(HERE / "router.cc"), arg_types=[u8, xb, acc_ty, np.int32], include_dirs=inc)
    f_fin = ExternalFunction("router_fin", source_file=str(HERE / "router_fin.cc"), arg_types=[acc_ty, u8], include_dirs=inc)
    of_in = ObjectFifo(u8, name="in", depth=2)
    of_out = ObjectFifo(u8, name="out", depth=1)
    xs = Buffer(xb, name="xs")
    acc = Buffer(acc_ty, name="acc")

    def core_body(ain, aout, xs, acc, fc, fa, ff):
        e = ain.acquire(1)
        fc(e, xs)
        ain.release(1)
        for rb in range_(W_ELEMS):
            e = ain.acquire(1)
            fa(e, xs, acc, rb)
            ain.release(1)
        o = aout.acquire(1)
        ff(acc, o)
        aout.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), xs, acc, f_copy, f_acc, f_fin],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(a_xm, a_w, c_out, inp, outc):
        pipe = Pipeline(3)
        pipe.drain(outc, c_out, TensorAccessPattern((1, 1024), 0, [1, 1, 1, 1024], [0, 0, 0, 1]))
        pipe.fill(inp, a_xm, TensorAccessPattern((1, HID), 0, [1, 1, 1, HID], [0, 0, 0, 1]))
        pipe.fill(inp, a_w, TensorAccessPattern((1, HID * E), 0, [1, 1, 1, HID * E], [0, 0, 0, 1]))
        pipe.finish()

    rt = Runtime(sequence, [xb, w_ty, out_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = router
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h")) + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
