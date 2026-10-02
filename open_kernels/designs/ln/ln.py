r"""Layer RMSNorm + residual on the NPU (one core, one call):
  y = x + add (fp32[2048]); xn = bf16(y * rsqrt(mean(y^2)+1e-6) * w)

Args: x f32[2048], add f32[2048], w bf16[2048], y f32[2048] (out), xn bf16[2048] (out)
All streams use 4 KB byte elements: in = [x (2), add (2), w (1)], out = [y (2), xn (1)].
"""

from __future__ import annotations

import hashlib
import os
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

N = int(os.environ.get("LN_N", 2048))     # the width; elements are N*2 bytes (ln.cc LN_N)
EPS = float(os.environ.get("LN_EPS", "1e-6"))
ELEM = N * 2
# Five inputs and one output held at once, over the 0x1800 stack, must fit the core's 64 KB:
# true through N = 4096 (55 296 B), false at the 27B's 5120 (67 584 B). Past it the residual
# streams half by half and the norm reads the sum back (recipes/qwen36moe.py norm_split).
SPLIT = 6 * ELEM + 0x1800 > 64 * 1024


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def ln(x: In, add: In, w: In, y: Out, xn: Out, *, n: CompileTime[int] = 2048, eps: CompileTime[int] = 0,
       srchash: CompileTime[int] = 0):
    u8 = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    f_ty = np.ndarray[(N,), np.dtype[np.float32]]
    b_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    flags = [f"-DLN_N={N}", f"-DLN_EPS={EPS:g}f"]
    if N <= 2048:
        # the fused kernel: five inputs and three outputs held at once (32 KB of 4 KB elements)
        fn = ExternalFunction("ln_fn", source_file=str(HERE / "ln.cc"),
                              arg_types=[u8, u8, u8, u8, u8, u8, u8, u8], include_dirs=include_dirs(), compile_flags=flags)
        of_in = ObjectFifo(u8, name="in", depth=5)
        of_out = ObjectFifo(u8, name="out", depth=3)

        def core_body(ain, aout, f):
            e = ain.acquire(5)
            o = aout.acquire(3)
            f(e[0], e[1], e[2], e[3], e[4], o[0], o[1], o[2])
            aout.release(3)
            ain.release(5)

        worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), fn], tile=Tile(0, 2), stack_size=0x1800)
    elif SPLIT:
        # widest: [x_i a_i] -> y_i for each half, y to DDR, then [y0 y1 w] -> xn through the
        # layer-entry norm's kernel. ln_add2 forms the sum exactly as ln_y / ln_xn do, so xn is
        # the same bits; three inputs at most, never more than four elements held.
        f_add = ExternalFunction("ln_add2", source_file=str(HERE / "ln_add2.cc"), arg_types=[u8] * 3,
                                 include_dirs=include_dirs(), compile_flags=flags)
        f_nr = ExternalFunction("ln_nr", source_file=str(HERE.parent / "lin_layer" / "ln_nr.cc"), arg_types=[u8] * 4,
                                include_dirs=include_dirs(), compile_flags=flags)
        of_in = ObjectFifo(u8, name="in", depth=3)
        of_out = ObjectFifo(u8, name="out", depth=1)

        def core_body(ain, aout, fa, fn):
            for _ in range(2):
                e = ain.acquire(2)                # [x_i a_i]
                o = aout.acquire(1)
                fa(e[0], e[1], o)
                aout.release(1)
                ain.release(2)
            e = ain.acquire(3)                    # [y0 y1 w]
            o = aout.acquire(1)
            fn(e[0], e[1], e[2], o)
            aout.release(1)
            ain.release(3)

        worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), f_add, f_nr], tile=Tile(0, 2), stack_size=0x1800)
    else:
        # wider: 8 KB elements would not fit three outputs beside the five inputs -- one output element per call
        f_y = ExternalFunction("ln_y", source_file=str(HERE / "ln_y.cc"), arg_types=[u8] * 5 + [np.int32],
                               include_dirs=include_dirs(), compile_flags=flags)
        f_xn = ExternalFunction("ln_xn", source_file=str(HERE / "ln_xn.cc"), arg_types=[u8] * 6,
                                include_dirs=include_dirs(), compile_flags=flags)
        of_in = ObjectFifo(u8, name="in", depth=5)
        of_out = ObjectFifo(u8, name="out", depth=1)

        def core_body(ain, aout, fy, fx):
            e = ain.acquire(5)                    # [x0 x1 a0 a1 w] (the fills' order below)
            for i in range(2):
                o = aout.acquire(1)
                fy(e[0], e[1], e[2], e[3], o, i)
                aout.release(1)
            o = aout.acquire(1)
            fx(e[0], e[1], e[2], e[3], e[4], o)
            aout.release(1)
            ain.release(5)

        worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), f_y, f_xn], tile=Tile(0, 2), stack_size=0x1800)

    def split_sequence(a_x, a_add, a_w, c_y, c_xn, inp, outc):
        h = N // 2

        def tap(off, n):
            return TensorAccessPattern((1, N), off, [1, 1, 1, n], [0, 0, 0, 1])

        pipe = Pipeline(3)
        for i in range(2):
            pipe.fill(inp, a_x, tap(i * h, h))
            pipe.fill(inp, a_add, tap(i * h, h))
        pipe.drain(outc, c_y, tap(0, N))
        pipe.finish()                            # y is in DDR
        pipe.fill(inp, c_y, tap(0, N))
        pipe.fill(inp, a_w, tap(0, N))
        pipe.drain(outc, c_xn, tap(0, N))
        pipe.finish()

    def sequence(a_x, a_add, a_w, c_y, c_xn, inp, outc):
        if SPLIT:
            split_sequence(a_x, a_add, a_w, c_y, c_xn, inp, outc)
            return
        pipe = Pipeline(3)
        pipe.drain(outc, c_y, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.drain(outc, c_xn, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_x, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_add, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_w, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.finish()

    rt = Runtime(sequence, [f_ty, f_ty, b_ty, f_ty, b_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = ln
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"n": N, "eps": int(round(-1e6 * __import__("math").log10(EPS))) if EPS > 0 else 0, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
