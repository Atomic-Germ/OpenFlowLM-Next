"""One-shot probe: run the REAL `ln_nr32.cc` for half 0 at N=5120 and score it.

Not a design the recipe uses -- two byte streams in, one byte stream out, one call --
so the actual production norm source can be measured at fp32 output, where the
arithmetic is visible (a bf16 output hides it behind a 2e-3 quantisation step).
The reference is FP64; the pass bar is the block route's whole-layer 4.129e-06,
because this one stage feeds every element the layer later uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.device import Tile, from_name
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).resolve().parents[2]          # open_kernels/
LN = HERE / "designs" / "ln"
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402

N = 5120
HALF = N // 2
ELEM = N * 2                                        # bytes of one 4 KB element


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def nr32(x0: In, x1: In, w: In, y: Out, *, n: CompileTime[int] = 5120,
         srchash: CompileTime[int] = 0):
    u8 = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    flags = [f"-DLN_N={N}", "-DLN_EPS=1e-06f"]
    fn = ExternalFunction("ln_nr32", source_file=str(LN / "ln_nr32.cc"),
                          arg_types=[u8, u8, u8, u8, np.int32],
                          include_dirs=include_dirs(), compile_flags=flags)
    of_in = ObjectFifo(u8, name="in", depth=3)
    of_out = ObjectFifo(u8, name="out", depth=1)

    def core_body(ain, aout, f):
        e = ain.acquire(3)                          # [x0 x1 w]
        o = aout.acquire(1)
        f(e[0], e[1], e[2], o, 0)                   # half 0 -> fp32[HALF]
        aout.release(1)
        ain.release(3)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), fn],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(x0, x1, w, cy, inp, outc):
        """The two fp32 halves of x as separate elements, then w -- the shape the
        production ln_xn/ln_nr32 helpers feed."""
        half = TensorAccessPattern((1, HALF), 0, [1, 1, 1, HALF], [0, 0, 0, 1])
        whole = TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1])
        p = Pipeline(3)
        p.drain(outc, cy, half)                     # y is one fp32 HALF
        p.fill(inp, x0, half)
        p.fill(inp, x1, half)
        p.fill(inp, w, whole)
        p.finish()

    f_ty = np.ndarray[(N,), np.dtype[np.float32]]
    b_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    rt = Runtime(sequence, [f_ty, f_ty, b_ty, f_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def make_fixture(seed=0):
    """Writes the two halves of `x` separately, because the kernel takes them as two
    arguments: `x` is fp32[N] and each half is its own element."""
    rng = np.random.default_rng(seed)
    w = (1.0 + rng.standard_normal(N) * 0.1).astype(np.float32).astype(bfloat16)
    x = (rng.standard_normal(N) * 0.5).astype(np.float32)
    y = x.astype(np.float64)
    yn = y / np.sqrt((y ** 2).mean() + 1e-6)
    ref = (yn * w.astype(np.float64))[:HALF].astype(np.float32)
    x[:HALF].tofile(HERE / "probe_x0.bin")
    x[HALF:].tofile(HERE / "probe_x1.bin")
    w.tofile(HERE / "probe_w.bin")
    ref.tofile(HERE / "probe_ref.bin")
    return x, w, ref


import hashlib  # noqa: E402

DESIGN = nr32
_src = (LN / "ln_nr32.cc").read_bytes() + (HERE / "include" / "vecmath.h").read_bytes()
SPECIALIZE = {"n": N, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}


if __name__ == "__main__":
    make_fixture()
    print("fixture written: probe_x.bin probe_w.bin probe_ref.bin")
