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
GROUPS = int(os.environ.get("LN_GROUPS", "1"))  # K2's GroupRMSNorm(2): one RMS per contiguous half
ELEM = N * 2
if GROUPS not in (1, 2):
    raise SystemExit(f"LN_GROUPS={GROUPS}: only 1 (plain RMSNorm) or 2 (K2 halves) supported")
if GROUPS == 2 and N <= 2048:
    # the fused ln_fn below has no grouped reduction; a grouped model must take the
    # split path (K2 hidden 2560 does) -- refuse loudly instead of silently wrong math
    raise SystemExit(f"LN_GROUPS=2 with N={N}: the fused (N<=2048) kernel is single-group")


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def ln(x: In, add: In, w: In, y: Out, xn: Out, *, n: CompileTime[int] = 2048, eps: CompileTime[int] = 0,
       srchash: CompileTime[int] = 0):
    u8 = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    f_ty = np.ndarray[(N,), np.dtype[np.float32]]
    b_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    flags = [f"-DLN_N={N}", f"-DLN_EPS={EPS:g}f"]
    if GROUPS != 1:
        flags.append(f"-DLN_GROUPS={GROUPS}")  # default 1: identical flags to the unpatched build
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

    def sequence(a_x, a_add, a_w, c_y, c_xn, inp, outc):
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
