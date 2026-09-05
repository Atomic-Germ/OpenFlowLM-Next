r"""moe_axpy: acc_out = (e == 0 ? 0 : acc_in) + w[e] * y   (one run per routed expert)

Args: rout f32[1024] (router output, w[8] at 264..271), y f32[2048], acc_in f32[2048],
eb int32[1024] (expert slot at [0]), acc_out f32[2048].
Streams: in = 4 KB elements [rout][y (2)][acc_in (2)][eb]; out = 2 elements.
Build: python build_design.py designs/moe_combine/moe_axpy.py designs/moe_combine/build_axpy
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

N = 2048


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def moe_axpy(rout: In, y: In, acc_in: In, eb: In, acc_out: Out, *, srchash: CompileTime[int] = 0):
    u8 = np.ndarray[(4096,), np.dtype[np.uint8]]
    r_ty = np.ndarray[(1024,), np.dtype[np.float32]]
    f_ty = np.ndarray[(N,), np.dtype[np.float32]]
    i_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    fn = ExternalFunction("mc_axpy", source_file=str(HERE / "mc_axpy.cc"),
                          arg_types=[u8, u8, u8, u8, u8, u8, u8, u8], include_dirs=include_dirs())
    of_in = ObjectFifo(u8, name="in", depth=6)
    of_out = ObjectFifo(u8, name="out", depth=2)

    def core_body(ain, aout, f):
        e = ain.acquire(6)
        o = aout.acquire(2)
        f(e[0], e[1], e[2], e[3], e[4], e[5], o[0], o[1])
        aout.release(2)
        ain.release(6)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), fn], tile=Tile(0, 2), stack_size=0x1800)

    def sequence(a_r, a_y, a_acc, a_eb, c_out, inp, outc):
        pipe = Pipeline(3)
        pipe.drain(outc, c_out, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_r, TensorAccessPattern((1, 1024), 0, [1, 1, 1, 1024], [0, 0, 0, 1]))
        pipe.fill(inp, a_y, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_acc, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_eb, TensorAccessPattern((1, 1024), 0, [1, 1, 1, 1024], [0, 0, 0, 1]))
        pipe.finish()

    rt = Runtime(sequence, [r_ty, f_ty, f_ty, i_ty, f_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = moe_axpy
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h")) + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
