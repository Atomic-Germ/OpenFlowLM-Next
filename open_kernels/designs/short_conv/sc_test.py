r"""Fixture design for sc.h: the LFM2 3-tap depthwise conv at any hidden width.

`cx.py` runs this core as part of a whole layer, so there was no standalone
fixture and a new width could not be gated (NOT_AUTOMATED). This design is the
same call -- `short_conv_step` over SCW=512 channel slices -- with the layer's
projections replaced by DDR vectors:

    [B|C|u], [s0|s1], [w0|w1|w2] -> [y], [n0|n1]

One core has only 2 input / 2 output DMA channels, so the five inputs and two
state outputs share FIFOs the way cx.py's scin/scw/scy/scst do: `in1` carries
B,C,u,s0,s1 (five fills per element), `in2` the three taps; `out1` is y (bf16),
`out2` the two new state rows (f32). W/512 elements -- the same loop cx.py runs.

Build:
    SC_TEST_WIDTH=1536 python build_design.py designs/short_conv/sc_test.py \
        designs/short_conv/build_w1536_t3
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import include_dirs  # noqa: E402

W = int(os.environ.get("SC_TEST_WIDTH", 2048))   # hidden width, in channels
SCW = 512                                        # channels per element (cx.py's choice)
TAPS = 3
if W % SCW or W <= 0:
    raise SystemExit(f"SC_TEST_WIDTH={W}: must be a positive multiple of {SCW}")
ELEMS = W // SCW


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def sc_test(a_x: In, a_s: In, a_w: In, c_y: Out, c_sn: Out, *,
            width: CompileTime[int] = 2048, srchash: CompileTime[int] = 0):
    fsc = np.ndarray[(SCW,), np.dtype[np.float32]]
    bsc = np.ndarray[(SCW,), np.dtype[bfloat16]]
    flags = [f"-DSC_TAPS={TAPS}", f"-DSC_W={SCW}"]
    f_sc = ExternalFunction("short_conv_step", source_file=str(HERE / "sc_elem.cc"),
                            arg_types=[fsc, fsc, fsc, bsc, bsc, bsc, fsc, fsc, bsc, fsc, fsc],
                            include_dirs=include_dirs() + [str(HERE)], compile_flags=flags)

    # Two input / two output FIFOs (the core's channel limit). Depths cover the
    # 5-at-once / 3-at-once / 2-at-once acquires; wait=True on every DMA keeps
    # each well under the core's 64 KB at W=1536/2048.
    of_in1 = ObjectFifo(fsc, name="in1", depth=6)   # B, C, u, state0, state1
    of_in2 = ObjectFifo(bsc, name="in2", depth=4)   # w0, w1, w2
    of_out1 = ObjectFifo(bsc, name="out1", depth=2)  # y
    of_out2 = ObjectFifo(fsc, name="out2", depth=4)  # new state0, new state1

    def core_body(i1, i2, o1, o2, f):
        for _ in range_(ELEMS):
            b, c, u, s0, s1 = i1.acquire(5)
            w0, w1, w2 = i2.acquire(3)
            y = o1.acquire(1)
            n0, n1 = o2.acquire(2)
            f(b, c, u, w0, w1, w2, s0, s1, y, n0, n1)
            o2.release(2)
            o1.release(1)
            i2.release(3)
            i1.release(5)

    worker = Worker(core_body, fn_args=[of_in1.cons(), of_in2.cons(), of_out1.prod(),
                                        of_out2.prod(), f_sc],
                    tile=Tile(0, 2), stack_size=0x1000)

    def tap(total, off, n):
        return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])

    def sequence(a_x, a_s, a_w, c_y, c_sn, p1, p2, c1, c2):
        # A fresh TaskGroup per element: 11 DMAs/element at ELEMS=3 would exceed
        # the shim's 16 BDs if queued together. Finishing each element reuses BDs.
        for i in range(ELEMS):
            tg = TaskGroup()
            off = i * SCW
            p1.fill(a_x, tap=tap(3 * W, 0 * W + off, SCW), wait=True, group=tg)   # B
            p1.fill(a_x, tap=tap(3 * W, 1 * W + off, SCW), wait=True, group=tg)   # C
            p1.fill(a_x, tap=tap(3 * W, 2 * W + off, SCW), wait=True, group=tg)   # u
            p1.fill(a_s, tap=tap(2 * W, 0 * W + off, SCW), wait=True, group=tg)   # state0
            p1.fill(a_s, tap=tap(2 * W, 1 * W + off, SCW), wait=True, group=tg)   # state1
            for j in range(TAPS):
                p2.fill(a_w, tap=tap(TAPS * W, j * W + off, SCW), wait=True, group=tg)
            c1.drain(c_y, tap=tap(W, off, SCW), wait=True, group=tg)
            for j in range(2):
                c2.drain(c_sn, tap=tap(2 * W, j * W + off, SCW), wait=True, group=tg)
            tg.finish()

    rt = Runtime(sequence, [np.ndarray[(3 * W,), np.dtype[np.float32]],    # B|C|u
                            np.ndarray[(2 * W,), np.dtype[np.float32]],    # s0|s1
                            np.ndarray[(TAPS * W,), np.dtype[bfloat16]],   # w0|w1|w2
                            np.ndarray[(W,), np.dtype[bfloat16]],          # y
                            np.ndarray[(2 * W,), np.dtype[np.float32]],    # n0|n1
                            of_in1.prod(), of_in2.prod(), of_out1.cons(), of_out2.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = sc_test
# jit cache is keyed on this file + CompileTime args, so a sc.h edit must move srchash.
_SRCHASH = int(hashlib.sha1((HERE / "sc.h").read_bytes()).hexdigest()[:8], 16)
SPECIALIZE = {"width": W, "srchash": _SRCHASH}
