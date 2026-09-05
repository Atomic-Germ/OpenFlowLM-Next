r"""Gated DeltaNet decode step on the NPU: 32 v-heads, S[32,128,128] fp32 in/out,
per-head (k, q, v, decay, beta) in, o[32,128] fp32 out.

Per core (8 cores x 4 heads): S streams in as 16-row slices (8 KB elements),
TWICE per head (pass 1 forms S^T k; pass 2 updates S and forms S'^T q) -- the
host access pattern repeats each head's S via a zero-stride dimension, so it is
one BD per core. See dn_step.h for the math and the bf16-split arithmetic.

Args: S_in fp32[32*128*128], vec fp32[32*512], S_out fp32[32*128*128], o fp32[32*128]
Build (WSL):  python build_design.py designs/deltanet/dn_step.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import config

HERE = Path(__file__).parent

D = 128
HEADS = 32
SLICE_ROWS = 16
NBLK = D // SLICE_ROWS
VEC = 512                       # fp32 per head: k, q, v, decay, beta, pad
N_CORES = int(os.environ.get("DN_CORES", 8))
# The whole-layer designs (layer_x) keep vec and o inside their `act` scratch BO: build with
# DN_ACT_F32 = its length in floats and the two offsets (floats) to read/write them there.
ACT_F32 = int(os.environ.get("DN_ACT_F32", 0))
VEC_OFF = int(os.environ.get("DN_VEC_OFF", 0))
O_OFF = int(os.environ.get("DN_O_OFF", 0))


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    return inc


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dn_step(s_in: In, vec: In, s_out: Out, o: Out, *, n_cores: CompileTime[int],
            act_f32: CompileTime[int] = 0, vec_off: CompileTime[int] = 0, o_off: CompileTime[int] = 0):
    heads_per_core = HEADS // n_cores
    s_elems = D * D                       # fp32 per head
    slice_ty = np.ndarray[(SLICE_ROWS * D,), np.dtype[np.float32]]
    vec_ty = np.ndarray[(VEC,), np.dtype[np.float32]]
    o_ty = np.ndarray[(D,), np.dtype[np.float32]]
    f128 = np.ndarray[(D,), np.dtype[np.float32]]
    b256 = np.ndarray[(2 * D,), np.dtype[bfloat16]]
    S_all = np.ndarray[(HEADS * s_elems,), np.dtype[np.float32]]
    vec_all = np.ndarray[(act_f32 or HEADS * VEC,), np.dtype[np.float32]]
    o_all = np.ndarray[(act_f32 or HEADS * D,), np.dtype[np.float32]]

    pass1 = ExternalFunction("dn_pass1", source_file=str(HERE / "dn_pass1.cc"),
                             arg_types=[slice_ty, vec_ty, f128, b256, b256, np.int32],
                             include_dirs=_include_dirs())
    pass2 = ExternalFunction("dn_pass2", source_file=str(HERE / "dn_pass2.cc"),
                             arg_types=[slice_ty, slice_ty, vec_ty, f128, f128, b256, b256, b256, np.int32],
                             include_dirs=_include_dirs())

    of_s = [ObjectFifo(slice_ty, name=f"s{c}", depth=2) for c in range(n_cores)]
    of_so = [ObjectFifo(slice_ty, name=f"so{c}", depth=2) for c in range(n_cores)]
    of_v = [ObjectFifo(vec_ty, name=f"v{c}", depth=2) for c in range(n_cores)]
    of_o = [ObjectFifo(o_ty, name=f"o{c}", depth=2) for c in range(n_cores)]

    def core_body(sin, vin, sout, oout, t, ob, k_hl, q_hl, d_hl, p1, p2):
        for _ in range_(heads_per_core):
            ve = vin.acquire(1)
            for blk in range_(NBLK):
                se = sin.acquire(1)
                p1(se, ve, t, k_hl, q_hl, blk)
                sin.release(1)
            for blk in range_(NBLK):
                se = sin.acquire(1)
                so = sout.acquire(1)
                p2(se, so, ve, t, ob, k_hl, q_hl, d_hl, blk)
                sin.release(1)
                sout.release(1)
            oe = oout.acquire(1)
            # copy the finished o (scratch) into the output element: cheap, 128 floats
            for j in range_(D):
                oe[j] = ob[j]
            oout.release(1)
            vin.release(1)

    workers = []
    for c in range(n_cores):
        t = Buffer(f128, name=f"t{c}")
        ob = Buffer(f128, name=f"ob{c}")
        k_hl = Buffer(b256, name=f"khl{c}")
        q_hl = Buffer(b256, name=f"qhl{c}")
        d_hl = Buffer(b256, name=f"dhl{c}")
        # One core per column: each head costs the shim 2 S fills, and a column
        # shim has 16 BDs -- four cores per column (IRON's default placement)
        # overflows it ("Too many simultaneously active buffer descriptors").
        workers.append(Worker(core_body,
                              fn_args=[of_s[c].cons(), of_v[c].cons(), of_so[c].prod(),
                                       of_o[c].prod(), t, ob, k_hl, q_hl, d_hl, pass1, pass2],
                              tile=Tile(c, 2),
                              stack_size=0xD00))

    # S in: per core, heads_per_core heads, each sent twice. (A single 4-D tap
    # with a zero-stride repeat dimension is rejected by the dma_bd verifier,
    # so it is two plain fills per head, queued on the core's shim channel.)
    s_taps = [[TensorAccessPattern((1, HEADS * s_elems), (c * heads_per_core + hh) * s_elems,
                                   [1, 1, 1, s_elems], [0, 0, 0, 1])
               for hh in range(heads_per_core)]
              for c in range(n_cores)]
    so_taps = [TensorAccessPattern((1, HEADS * s_elems), c * heads_per_core * s_elems,
                                   [1, 1, 1, heads_per_core * s_elems], [0, 0, 0, 1])
               for c in range(n_cores)]
    v_taps = [TensorAccessPattern((1, act_f32 or HEADS * VEC), vec_off + c * heads_per_core * VEC,
                                  [1, 1, 1, heads_per_core * VEC], [0, 0, 0, 1])
              for c in range(n_cores)]
    o_taps = [TensorAccessPattern((1, act_f32 or HEADS * D), o_off + c * heads_per_core * D,
                                  [1, 1, 1, heads_per_core * D], [0, 0, 0, 1])
              for c in range(n_cores)]

    # A shim DMA channel's start queue holds 4 BDs; pushing more silently drops
    # them and the core waits forever (measured: 8 queued S fills -> timeout).
    # So S fills are issued at most INFLIGHT heads (2 fills each) ahead per
    # core, awaiting the oldest head's fills before issuing the next. Drains go
    # first so the cores always have somewhere to put their output.
    INFLIGHT = 2

    def sequence(a_s, a_v, c_so, c_o, s_prods, v_prods, so_conss, o_conss):
        tg_end = TaskGroup()
        for c in range(n_cores):
            so_conss[c].drain(c_so, tap=so_taps[c], wait=True, group=tg_end)
            o_conss[c].drain(c_o, tap=o_taps[c], wait=True, group=tg_end)
            v_prods[c].fill(a_v, tap=v_taps[c], group=tg_end)
        tgs = [[None] * heads_per_core for _ in range(n_cores)]
        for hh in range(heads_per_core):
            for c in range(n_cores):
                if hh >= INFLIGHT:
                    tgs[c][hh - INFLIGHT].finish()
                tg = TaskGroup()
                tgs[c][hh] = tg
                for _rep in range(2):
                    s_prods[c].fill(a_s, tap=s_taps[c][hh], wait=True, group=tg)
        for c in range(n_cores):
            for hh in range(max(0, heads_per_core - INFLIGHT), heads_per_core):
                tgs[c][hh].finish()
        tg_end.finish()

    rt = Runtime(sequence, [S_all, vec_all, S_all, o_all,
                            [f.prod() for f in of_s], [f.prod() for f in of_v],
                            [f.cons() for f in of_so], [f.cons() for f in of_o]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = dn_step
SPECIALIZE = {"n_cores": N_CORES, "act_f32": ACT_F32, "vec_off": VEC_OFF, "o_off": O_OFF}
