r"""bfp_cvt: read the NPU's bf16 -> bfp16ebs8 conversion bytes (workstream B2, Bonsai 2 round 2).

One worker converts TILES elements of 64 x 64 bf16 values each (bfp_cvt.cc: accum<accfloat, 64>
then to_v64bfp16ebs8 under conv_even, the bfp16 GEMM's operand conversion) and writes the raw
72-byte block vectors. check.py compares them with a host model, so the GEMM can be fed bfp16
activations made on the host without changing a bit of its arithmetic.

Build (WSL, ironenv):  python ../../build_design.py bfp_cvt.py build
Run (Windows):         run_kernel.exe run_bfp_cvt.cfg   (written by check.py gen)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent
VECS = 64                    # 64-value vectors per element (BFP_VECS in bfp_cvt.cc)
TILES = int(os.environ.get("BFP_TILES", 16))


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base
    from aie.utils import config

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    return inc


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def bfp_cvt(x_in: In, y_out: Out, *, tiles: CompileTime[int] = TILES):
    x_ty = np.ndarray[(VECS * 64,), np.dtype[np.int16]]          # bf16 bits
    y_ty = np.ndarray[(VECS * 72,), np.dtype[np.uint8]]          # bfp16ebs8 blocks
    X_ty = np.ndarray[(tiles * VECS * 64,), np.dtype[np.int16]]
    Y_ty = np.ndarray[(tiles * VECS * 72,), np.dtype[np.uint8]]
    kernel = ExternalFunction("bfp_cvt", source_file=str(HERE / "bfp_cvt.cc"), arg_types=[x_ty, y_ty],
                              include_dirs=_include_dirs(), compile_flags=[f"-DBFP_VECS={VECS}"])
    of_in = ObjectFifo(x_ty, name="x_in", depth=2)
    of_out = ObjectFifo(y_ty, name="y_out", depth=2)

    def core_body(rx, tx, fn):
        for _ in range_(tiles):
            a = rx.acquire(1)
            b = tx.acquire(1)
            fn(a, b)
            rx.release(1)
            tx.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), kernel], stack_size=0xD00)

    def sequence(src, dst, rx_prod, tx_cons):
        rx_prod.fill(src)
        tx_cons.drain(dst, wait=True)

    rt = Runtime(sequence, [X_ty, Y_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = bfp_cvt
SPECIALIZE = {"tiles": TILES}
