r"""moe_fin: out = xres + acc + sigmoid(xm . sgw) * shared

Args: acc f32[2048] (sum_e w_e y_e from moe_axpy), xres f32[2048], shared f32[2048],
xm bf16[2048], sgw bf16[2048], out f32[2048].
Streams: in = 4 KB elements [acc (2)][xres (2)][shared (2)][xm][sgw]; out = 2 elements.
Build: python build_design.py designs/moe_combine/moe_fin.py designs/moe_combine/build_fin
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

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
def moe_fin(acc: In, xres: In, shared: In, xm: In, sgw: In, out: Out, *, srchash: CompileTime[int] = 0):
    u8 = np.ndarray[(4096,), np.dtype[np.uint8]]
    f_ty = np.ndarray[(N,), np.dtype[np.float32]]
    b_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    fn = ExternalFunction("mc_fin", source_file=str(HERE / "mc_fin.cc"),
                          arg_types=[u8] * 10, include_dirs=include_dirs())
    of_in = ObjectFifo(u8, name="in", depth=8)
    of_out = ObjectFifo(u8, name="out", depth=2)

    def core_body(ain, aout, f):
        e = ain.acquire(8)
        o = aout.acquire(2)
        f(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], o[0], o[1])
        aout.release(2)
        ain.release(8)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), fn], tile=Tile(0, 2), stack_size=0x1800)

    def sequence(a_acc, a_x, a_s, a_xm, a_sgw, c_out, inp, outc):
        pipe = Pipeline(3)
        pipe.drain(outc, c_out, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        for t in (a_acc, a_x, a_s, a_xm, a_sgw):
            pipe.fill(inp, t, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.finish()

    rt = Runtime(sequence, [f_ty, f_ty, f_ty, b_ty, b_ty, f_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = moe_fin
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h")) + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
