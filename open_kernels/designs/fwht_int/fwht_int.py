r"""Decode-prep Hadamard on one core, wht_int.h vs wht.h; FWHT_KIND 0/1 int bf16/f32, 2/3 fp bf16/f32.

Build (WSL):  python build_design.py designs/fwht_int/fwht_int.py designs/fwht_int/build_k<kind>_r<reps>
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils import config

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"

KIND = int(os.environ.get("FWHT_KIND", 0))
K = int(os.environ.get("FWHT_K", 8192))
REPS = int(os.environ.get("FWHT_REPS", 1))
CFLAGS = os.environ.get("FWHT_CFLAGS", "-Oz")     # the decode image compiles its preps -Oz (xcommon.py)
ELEM = 4096


def tab_bytes(k: int) -> int:
    return 2 * k + k // 16


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc += [str(root), str(root / _detect_arch()), str(HERE), str(GEMV)]
    return inc


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def fwht_int(x: In, t: Out, *, kind: CompileTime[int], k: CompileTime[int], reps: CompileTime[int], cflags: CompileTime[str] = "",
             srchash: CompileTime[int] = 0):
    vbytes = 2 if kind in (0, 2) else 4
    n_elem = k * vbytes // ELEM
    elem_ty = np.ndarray[(ELEM,), np.dtype[np.uint8]]
    tab_ty = np.ndarray[(tab_bytes(k),), np.dtype[np.uint8]]
    x_ty = np.ndarray[(k * vbytes,), np.dtype[np.uint8]]

    fn = ExternalFunction("fwht_probe", source_file=str(HERE / "fwht_probe.cc"),
                          arg_types=[elem_ty, tab_ty, np.int32, np.int32, np.int32],
                          include_dirs=_include_dirs(), compile_flags=[f"-DFWHT_KIND={kind}"] + cflags.split())
    of_x = ObjectFifo(elem_ty, name="x", depth=2)
    of_t = ObjectFifo(tab_ty, name="t", depth=1)

    def core_body(xin, tout, f):
        te = tout.acquire(1)
        for i in range_(n_elem):
            xe = xin.acquire(1)
            f(xe, te, k, i, reps)
            xin.release(1)
        tout.release(1)

    w = Worker(core_body, fn_args=[of_x.cons(), of_t.prod(), fn], stack_size=0x1000)

    def sequence(a_x, c_t, x_prod, t_cons):
        tg = TaskGroup()
        x_prod.fill(a_x, group=tg)
        t_cons.drain(c_t, wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [x_ty, tab_ty, of_x.prod(), of_t.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


DESIGN = fwht_int
_SRCHASH = int(hashlib.sha1(b"".join(p.read_bytes() for p in
                                     [HERE / "fwht_probe.cc", GEMV / "wht_int.h", GEMV / "wht.h",
                                      GEMV / "gemv_t2.h"])).hexdigest()[:8], 16)
SPECIALIZE = {"kind": KIND, "k": K, "reps": REPS, "cflags": CFLAGS, "srchash": _SRCHASH}
