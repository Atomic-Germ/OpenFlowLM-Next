# Whole-array bf16 matmul for NPU2, adapted from mlir-aie v1.3.4
# programming_examples/basic/matrix_multiplication/whole_array/whole_array.py.

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.helpers.taplib import TensorTiler2D
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def whole_array_bf16(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    tile_n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    dtype_in_str: CompileTime[str] = "bf16",
    dtype_out_str: CompileTime[str] = "bf16",
):
    m = k = 64
    n = tile_n
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    assert M % (m * n_aie_rows) == 0
    assert K % k == 0
    assert N % (n * n_aie_cols) == 0

    dtype_in = iron.str_to_dtype(dtype_in_str)
    dtype_out = iron.str_to_dtype(dtype_out_str)

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        vectorized=True,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    fifo_depth = 2
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
    n_shim_mem_a = min(n_aie_cols, n_aie_rows)
    n_a_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_a_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    A_l3l2 = [None] * n_shim_mem_a
    A_l2l1 = [None] * n_aie_rows
    B_l3l2 = [None] * n_aie_cols
    B_l2l1 = [None] * n_aie_cols
    C_l1l2 = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3 = [None] * n_aie_cols

    for shim in range(n_shim_mem_a):
        A_l3l2[shim] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{shim}", depth=fifo_depth)
        start_row = shim * n_a_tiles_per_shim
        stop_row = start_row + n_a_tiles_per_shim
        offsets = [m * k * row for row in range(stop_row - start_row)]
        dims = [[(m // r, r * k), (k // s, s), (r, k), (s, 1)]] * len(offsets)
        split = A_l3l2[shim].cons().split(
            offsets,
            obj_types=[A_l1_ty] * len(offsets),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=dims,
        )
        for index, fifo in enumerate(split):
            A_l2l1[start_row + index] = fifo

    for col in range(n_aie_cols):
        B_l3l2[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l2l1[col] = B_l3l2[col].cons().forward(
            obj_type=B_l1_ty,
            name=f"B_L2L1_{col}",
            dims_to_stream=[
                (k // s, s * n),
                (n // t, t),
                (s, n),
                (t, 1),
            ],
        )

        C_l2l3[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=[
                (m // r, r * n),
                (r, t),
                (n // t, r * t),
                (t, 1),
            ],
        )
        joined = C_l2l3[col].prod().join(
            [m * n * row for row in range(n_aie_rows)],
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[fifo_depth] * n_aie_rows,
        )
        for row, fifo in enumerate(joined):
            C_l1l2[row][col] = fifo

    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range(1) if n_tiles_per_core == 1 else range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1[row].cons(),
                B_l2l1[col].cons(),
                C_l1l2[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
        ),
    )

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m * n_a_tiles_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, N),
        (k, n),
        tile_group_repeats=(K // k, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        tile_group_col_major=True,
        prune_step=False,
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N),
        (m * n_aie_rows, n),
        tile_group_repeats=(2, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        prune_step=False,
    )

    c_index = 0
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A_arg, B_arg, C_arg):
        rt.start(*[worker for row in workers for worker in row])
        tg = rt.task_group()
        for transfer_block in range(iron.ceildiv(M // m // n_aie_rows, 4)):
            for pingpong in (0, 1):
                if c_index >= len(C_tiles):
                    break
                row_base = transfer_block * 4 + pingpong * 2
                current_rows = min(2, M // m // n_aie_rows - row_base)
                for col in range(n_aie_cols):
                    rt.drain(
                        C_l2l3[col].cons(),
                        C_arg,
                        tap=C_tiles[c_index],
                        wait=True,
                        task_group=tg,
                    )
                    c_index += 1
                    for tile_row in range(current_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_a + col
                        ) % len(A_tiles)
                        if col < n_aie_rows:
                            rt.fill(
                                A_l3l2[col].prod(),
                                A_arg,
                                tap=A_tiles[tile_offset],
                                task_group=tg,
                            )
                        rt.fill(
                            B_l3l2[col].prod(),
                            B_arg,
                            tap=B_tiles[col],
                            task_group=tg,
                        )
                if transfer_block > 0 or pingpong > 0:
                    rt.finish_task_group(tg)
                    tg = rt.task_group()
        rt.finish_task_group(tg)

    return Program(iron.get_current_device(), rt).resolve_program()


def _compile_kwargs(opts):
    return dict(
        M=opts.M,
        K=opts.K,
        N=opts.N,
        tile_n=opts.tile_n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
    )


def _validate(opts):
    if opts.M % 256:
        sys.exit("M must be a multiple of 256")
    if opts.K % 64:
        sys.exit("K must be a multiple of 64")
    if opts.tile_n % 16:
        sys.exit("--tile-n must be a multiple of 16")
    if opts.N % (opts.tile_n * opts.n_aie_cols):
        sys.exit("N must be divisible by --tile-n * --n-aie-cols")


def main():
    parser = argparse.ArgumentParser(prog="whole-array bf16 matmul")
    add_compile_args(parser, with_emit_mlir=True)
    parser.add_argument("-M", type=int, default=512)
    parser.add_argument("-K", type=int, default=768)
    parser.add_argument("-N", type=int, default=768)
    parser.add_argument("--tile-n", type=int, default=32)
    parser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--dtype-in", type=str, choices=["bf16", "i16", "i8"], default="bf16", dest="dtype_in")
    parser.add_argument("--dtype-out", type=str, choices=["bf16", "f32", "i32", "i16", "i8"], default="bf16", dest="dtype_out")
    opts = parser.parse_args()
    run_design_cli(
        whole_array_bf16,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda o: from_name(o.dev, n_cols=None if o.dev == "npu2" else o.n_aie_cols),
        validate=_validate,
    )


if __name__ == "__main__":
    main()
