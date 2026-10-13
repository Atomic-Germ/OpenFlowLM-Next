"""Separate banked AB dispatch, two input DMA channels and 32-lane accumulators.

side: per bank Wa[H,32], Wb[H,32] bf16, small[A,dt_bias] padded to 4096 B.
xn: bf16[H] padded to whole 4096 B elements, replayed per projection/bank.
result: f32[4, heads] = alpha, beta logits, decay, sigmoid(beta).
No model-specific layout or whole-layer support is implied by this primitive.
"""
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
GLUE = HERE.parent / "dn_glue"
sys.path.insert(0, str(ROOT))
from ironutil import Pipeline, include_dirs
from recipes.wide_deltanet import WideDeltaNet

G = WideDeltaNet(hidden=int(os.environ.get("WIDE_DN_HIDDEN", "5120")),
                key_heads=16, value_heads=48, key_dim=128, value_dim=128)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def wide_ab(side: In, xn: In, result: Out, *, dummy: CompileTime[int] = 0):
    elem = np.ndarray[(4096,), np.dtype[np.uint8]]
    xchunk = np.ndarray[(2048,), np.dtype[bfloat16]]
    acc = np.ndarray[(32,), np.dtype[np.float32]]
    res = np.ndarray[(4 * G.value_heads,), np.dtype[np.float32]]
    side_ty = np.ndarray[(G.side_bytes,), np.dtype[np.uint8]]
    xn_ty = np.ndarray[(G.xn_chunks * 4096,), np.dtype[np.uint8]]
    inc = include_dirs() + [str(GLUE)]
    flags = [f"-DDNGLUE_NHEAD={G.value_heads}"]
    fab = ExternalFunction("glue_ab_e", source_file=str(GLUE / "glue_ab_e.cc"),
                           arg_types=[elem, xchunk, acc, np.int32, np.int32], include_dirs=inc,
                           compile_flags=flags)
    fcopy = ExternalFunction("glue_copy_xn", source_file=str(GLUE / "glue_copy.cc"),
                             arg_types=[elem, xchunk], include_dirs=inc, compile_flags=flags)
    fstore = ExternalFunction("wide_ab_store", source_file=str(HERE / "ab_store.cc"),
                              arg_types=[elem, acc, acc, res, np.int32, np.int32],
                              include_dirs=inc, compile_flags=flags)
    weights = ObjectFifo(elem, name="ab_weights", depth=2)
    xs = ObjectFifo(elem, name="ab_xn", depth=1)
    out = ObjectFifo(res, name="ab_result", depth=1)
    a = Buffer(acc, name="acc_a")
    b = Buffer(acc, name="acc_b")
    x = Buffer(xchunk, name="xn_chunk")

    def core_body(sin, xin, oout, acc_a, acc_b, xn, fab, fstore, fcopy):
        result = oout.acquire(1)
        for base, active in G.banks:
            for acc in (acc_a, acc_b):
                for h, ntiles in enumerate(G.ab_tiles):
                    e = xin.acquire(1)
                    fcopy(e, xn)
                    xin.release(1)
                    for tile in range_(ntiles):
                        w = sin.acquire(1)
                        fab(w, xn, acc, tile, int(h == 0))
                        sin.release(1)
            sm = sin.acquire(1)
            fstore(sm, acc_a, acc_b, result, base, active)
            sin.release(1)
        oout.release(1)

    worker = Worker(core_body, fn_args=[weights.cons(), xs.cons(), out.prod(),
                                       a, b, x, fab, fstore, fcopy],
                    tile=Tile(0, 2), stack_size=0x1000)

    def sequence(a_side, a_xn, c_result, side_p, xn_p, out_c):
        def tap(total, offset, size):
            return TensorAccessPattern((1, total), offset, [1, 1, 1, size], [0, 0, 0, 1])
        pipe = Pipeline(3)
        pipe.drain(out_c, c_result, tap(4 * G.value_heads, 0, 4 * G.value_heads))
        pipe.fill(side_p, a_side, tap(G.side_bytes, 0, G.side_bytes))
        for _ in range(2 * G.ab_banks):
            pipe.fill(xn_p, a_xn, tap(G.xn_chunks * 4096, 0, G.xn_chunks * 4096))
        pipe.finish()

    rt = Runtime(sequence, [side_ty, xn_ty, res, weights.prod(), xs.prod(), out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = wide_ab
_sources = [Path(__file__), ROOT / "recipes/wide_deltanet.py", ROOT / "ironutil.py",
            ROOT / "include/vecmath.h", HERE / "ab_store.cc",
            *sorted(GLUE.glob("*.h")), GLUE / "glue_ab_e.cc", GLUE / "glue_copy.cc"]
SPECIALIZE = {"dummy": int(hashlib.sha256(b"".join(p.read_bytes() for p in _sources)
                                        + repr(G).encode()).hexdigest()[:8], 16)}
