r"""Short-conv element on the NPU (one core, one SC_W slice): y = C*(s0*w0+s1*w1+B*u*w2).

Buffers (4, the harness limit): BCuS f32[5*SC_W] (B,C,u,s0,s1), W bf16[3*SC_W]
(w0,w1,w2 tap-major), y bf16[SC_W], NS f32[2*SC_W] (n0,n1). Env SC_W (default 512),
SC_TAPS (default 3, the header needs 3). A full hidden W is W//SC_W slices;
the gate proves the element, the layer loop proves the count (cx/sx).
"""
from __future__ import annotations
import hashlib, os, sys
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

SCW = int(os.environ.get("SC_W", 512))
TAPS = int(os.environ.get("SC_TAPS", 3))
FLAGS = [f"-DSC_TAPS={TAPS}", f"-DSC_W={SCW}", "-DSC_PASSTHROUGH=0"]

@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def sc(bcus: In, w: In, y: Out, ns: Out, *, srchash: CompileTime[int] = 0):
    f_ty = np.ndarray[(SCW,), np.dtype[np.float32]]
    b_ty = np.ndarray[(SCW,), np.dtype[bfloat16]]
    bcus_ty = np.ndarray[(5 * SCW,), np.dtype[np.float32]]
    w_ty = np.ndarray[(3 * SCW,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(SCW,), np.dtype[bfloat16]]
    ns_ty = np.ndarray[(2 * SCW,), np.dtype[np.float32]]
    inc = include_dirs() + [str(HERE)]
    f = ExternalFunction("short_conv_step", source_file=str(HERE / "sc_elem.cc"),
                         arg_types=[f_ty, f_ty, f_ty, b_ty, b_ty, b_ty, f_ty, f_ty, b_ty, f_ty, f_ty],
                         include_dirs=inc, compile_flags=FLAGS)
    of_in = ObjectFifo(f_ty, name="in", depth=6)
    of_w = ObjectFifo(b_ty, name="win", depth=4)
    of_y = ObjectFifo(b_ty, name="out", depth=2)
    of_st = ObjectFifo(f_ty, name="st", depth=2)

    def core_body(ain, awin, aout, ast, fn):
        w = awin.acquire(TAPS)
        e = ain.acquire(5)
        oy = aout.acquire(1)
        os = ast.acquire(2)
        fn(e[0], e[1], e[2], w[0], w[1], w[2], e[3], e[4], oy, os[0], os[1])
        ast.release(2)
        aout.release(1)
        ain.release(5)
        awin.release(TAPS)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_w.cons(), of_y.prod(), of_st.prod(), f],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(a_bcus, a_w, c_y, c_ns, inp, winp, outp, stp):
        pipe = Pipeline(3)
        pipe.drain(outp, c_y, TensorAccessPattern((1, 1), 0, [1, 1, 1, SCW], [0, 0, 0, 1]))
        pipe.drain(stp, c_ns, TensorAccessPattern((1, 1), 0, [1, 1, 1, 2 * SCW], [0, 0, 0, 1]))
        pipe.fill(inp, a_bcus, TensorAccessPattern((1, 5 * SCW), 0, [1, 1, 1, 5 * SCW], [0, 0, 0, 1]))
        pipe.fill(winp, a_w, TensorAccessPattern((1, 3 * SCW), 0, [1, 1, 1, 3 * SCW], [0, 0, 0, 1]))
        pipe.finish()

    rt = Runtime(sequence, [bcus_ty, w_ty, y_ty, ns_ty,
                            of_in.prod(), of_w.prod(), of_y.cons(), of_st.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = sc
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h")) + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
