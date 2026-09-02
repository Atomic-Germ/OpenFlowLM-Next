# single-core matmul (ported from Xilinx/mlir-aie
# programming_examples/getting_started/03_matrix_multiplication_single_core)
# adapted to the installed iron Runtime API (rt.sequence context manager).
# Uses kernels.mm() (mm.cc) with DMA layout taps (r,s,t sub-tile streaming).

import argparse

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

_TILE_M = _TILE_K = _TILE_N = 64


@iron.jit
def matrix_multiplication_single_core(
    input0: In,
    input1: In,
    output: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    element_type: CompileTime[type],
):
    m, k, n = _TILE_M, _TILE_K, _TILE_N

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=element_type,
        output_dtype=element_type,
        vectorized=True,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    A_ty = np.ndarray[(M, K), np.dtype[element_type]]
    B_ty = np.ndarray[(K, N), np.dtype[element_type]]
    C_ty = np.ndarray[(M, N), np.dtype[element_type]]
    a_ty = np.ndarray[(m * k,), np.dtype[element_type]]
    b_ty = np.ndarray[(k * n,), np.dtype[element_type]]
    c_ty = np.ndarray[(m * n,), np.dtype[element_type]]

    fifo_A_L3L2 = ObjectFifo(a_ty, name="A_L3L2")
    tap_A_L2L1 = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
        dims_to_stream=tap_A_L2L1.transformation_dims, name="A_L2L1"
    )

    fifo_B_L3L2 = ObjectFifo(b_ty, name="B_L3L2")
    tap_B_L2L1 = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
    fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
        dims_to_stream=tap_B_L2L1.transformation_dims, name="B_L2L1"
    )

    fifo_C_L1L2 = ObjectFifo(c_ty, name="C_L1L2")
    tap_C_L1L2 = TensorAccessPattern(
        tensor_dims=(m, n),
        offset=0,
        sizes=[m // r, r, n // t, t],
        strides=[r * n, t, r * t, 1],
    )
    fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
        dims_to_stream=list(tap_C_L1L2.transformation_dims), name="C_L2L3"
    )

    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(M // m * N // n):
            elem_out = of_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [
            fifo_A_L2L1.cons(),
            fifo_B_L2L1.cons(),
            fifo_C_L1L2.prod(),
            zero_kernel,
            matmul_kernel,
        ],
    )

    a_taps = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K // k), pattern_repeat=(N // n)
    )
    b_tap = TensorTiler2D.group_tiler(
        (K, N), (k, n), (K // k, N // n), tile_group_col_major=True
    )[0]
    c_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, N // n))

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        for tile_row in range(M // m):
            tg = rt.task_group()
            rt.fill(fifo_A_L3L2.prod(), A, tap=a_taps[tile_row], task_group=tg)
            rt.fill(fifo_B_L3L2.prod(), B, tap=b_tap, task_group=tg)
            rt.drain(fifo_C_L2L3.cons(), C, tap=c_taps[tile_row], task_group=tg, wait=True)
            rt.finish_task_group(tg)

    return Program(iron.get_current_device(), rt).resolve_program()


def _compile_kwargs(opts):
    import os

    try:
        m, k, n = (int(x) for x in os.environ.get("MM_SHAPE", "64,64,64").split(","))
    except ValueError:
        m, k, n = 64, 64, 64
    element_type = bfloat16 if os.environ.get("MM_DTYPE") == "bf16" else np.int16
    return dict(M=m, K=k, N=n, element_type=element_type)


def _run_and_verify(opts):
    kw = _compile_kwargs(opts)
    M, K, N = kw["M"], kw["K"], kw["N"]
    et = kw["element_type"]
    input0 = iron.randint(0, 256, (M, K), dtype=et, device="npu")
    input1 = iron.randint(0, 256, (K, N), dtype=et, device="npu")
    output = iron.zeros(M * N, dtype=et, device="npu")
    ref = np.matmul(input0.numpy(), input1.numpy())
    matrix_multiplication_single_core(input0, input1, output, **kw)
    assert_pass(output.numpy(), ref.flatten(), fail_msg="matmul mismatch")


def main():
    p = argparse.ArgumentParser(prog="single-core matmul (canonical port)")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        matrix_multiplication_single_core,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
