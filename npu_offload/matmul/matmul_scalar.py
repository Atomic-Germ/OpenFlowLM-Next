# my-first-npu-matmul: single-tile bf16 matmul (64x64x64) on npu2.
# Uses the shipped AIE2P microkernel from mlir_aie/include/aie_kernels/aie2p/mm.cc
# ("matmul_scalar_bf16_bf16"), validated end-to-end against a CPU reference.

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

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
from aie.utils.verify import assert_pass

DIM_M = 64
DIM_K = 64
DIM_N = 64

tile_ty = np.ndarray[(DIM_M, DIM_K), np.dtype[bfloat16]]
tile_ty_b = np.ndarray[(DIM_K, DIM_N), np.dtype[bfloat16]]
tile_ty_c = np.ndarray[(DIM_M, DIM_N), np.dtype[bfloat16]]

_MM_SRC = Path("/home/atomic-germ/Projects/FastFlowLM_v1.0.1-add/ironvenv/"
               "lib/python3.13/site-packages/mlir_aie/include/aie_kernels/aie2p/mm.cc")


@iron.jit
def matmul_bf16(a_in: In, b_in: In, c_out: Out):
    mm_fn = ExternalFunction(
        "matmul_scalar_bf16_bf16",
        source_file=str(_MM_SRC),
        arg_types=[tile_ty, tile_ty_b, tile_ty_c],
        
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


def _make_inputs():
    rng = np.random.default_rng(42)
    a = (rng.normal(size=(DIM_M, DIM_K)) * 0.1).astype(bfloat16)
    b = (rng.normal(size=(DIM_K, DIM_N)) * 0.1).astype(bfloat16)
    return a.astype(np.float32).astype(bfloat16), b.astype(np.float32).astype(bfloat16)


def _run_and_verify(opts):
    a, b = _make_inputs()
    c_ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)
    c = iron.zeros((DIM_M, DIM_N), dtype=bfloat16, device="npu")
    matmul_bf16(a, b, c)
    assert_pass(
        c.numpy().astype(np.float32),
        c_ref.astype(np.float32),
        atol=0.1,
        fail_msg="matmul_bf16 output mismatch",
    )


def main():
    p = argparse.ArgumentParser(prog="single-tile bf16 matmul")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        matmul_bf16,
        opts,
        compile_kwargs={},
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()