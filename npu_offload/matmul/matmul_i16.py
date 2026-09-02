# my-first-npu-matmul-int16: single-tile i16 -> i32 matmul (64x64x64) on npu2.
# Uses matmul_i16_i32 from the shipped AIE2P microkernel (mm.cc). Integer exact.

import argparse
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli

DIM_M = 64
DIM_K = 64
DIM_N = 64

tile_ty = np.ndarray[(DIM_M, DIM_K), np.dtype[np.int16]]
tile_ty_b = np.ndarray[(DIM_K, DIM_N), np.dtype[np.int16]]
tile_ty_c = np.ndarray[(DIM_M, DIM_N), np.dtype[np.int32]]

_MM_SRC = Path("/home/atomic-germ/Projects/FastFlowLM_v1.0.1-add/ironvenv/"
               "lib/python3.13/site-packages/mlir_aie/include/aie_kernels/aie2p/mm.cc")


@iron.jit
def matmul_i16(a_in: In, b_in: In, c_out: Out):
    mm_fn = ExternalFunction(
        "matmul_i16_i32",
        source_file=str(_MM_SRC),
        arg_types=[tile_ty, tile_ty_b, tile_ty_c],
        compile_flags=["-Di16_i32_ONLY"],
    )

    of_a = ObjectFifo(tile_ty, name="a")
    of_b = ObjectFifo(tile_ty_b, name="b")
    of_c = ObjectFifo(tile_ty_c, name="c")

    def core_fn(of_a, of_b, of_c, mm):
        a = of_a.acquire(1)
        b = of_b.acquire(1)
        c = of_c.acquire(1)
        mm(a, b, c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    my_worker = Worker(core_fn, [of_a.cons(), of_b.cons(), of_c.prod(), mm_fn])

    rt = Runtime()
    with rt.sequence(tile_ty, tile_ty_b, tile_ty_c) as (ia, ib, ic):
        rt.start(my_worker)
        rt.fill(of_a.prod(), ia)
        rt.fill(of_b.prod(), ib)
        rt.drain(of_c.cons(), ic, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="single-tile i16 matmul")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        matmul_i16,
        opts,
        compile_kwargs={},
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()