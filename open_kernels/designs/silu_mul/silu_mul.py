r"""h = bf16(silu(g) * u), g/u fp32[512] -> h bf16[512] (one expert hidden), one core.
Streams: in = 2 KB elements [g][u]; out = one 1 KB element."""

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

N = 512


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def silu_mul(g: In, u: In, h: Out, *, srchash: CompileTime[int] = 0):
    u8i = np.ndarray[(2048,), np.dtype[np.uint8]]
    u8o = np.ndarray[(1024,), np.dtype[np.uint8]]
    f_ty = np.ndarray[(N,), np.dtype[np.float32]]
    b_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    fn = ExternalFunction("silu_mul", source_file=str(HERE / "silu_mul.cc"), arg_types=[u8i, u8i, u8o], include_dirs=include_dirs())
    of_in = ObjectFifo(u8i, name="in", depth=2)
    of_out = ObjectFifo(u8o, name="out", depth=1)

    def core_body(ain, aout, f):
        e = ain.acquire(2)
        o = aout.acquire(1)
        f(e[0], e[1], o)
        aout.release(1)
        ain.release(2)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), fn], tile=Tile(0, 2), stack_size=0x1800)

    def sequence(a_g, a_u, c_h, inp, outc):
        pipe = Pipeline(3)
        pipe.drain(outc, c_h, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_g, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_u, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.finish()

    rt = Runtime(sequence, [f_ty, f_ty, b_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = silu_mul
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
