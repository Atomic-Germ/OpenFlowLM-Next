r"""vae_ew: the elementwise ops of FLUX.2 [klein]'s VAE decoder, one xclbin.

    gn_stats   per-core partial GroupNorm sums of A                  -> S (the layer's block)
    gn_apply   Y = GroupNorm(A) (* SiLU): reads the layer's block from S, then A
    add        Y = A + B (the resnet residual); "stats": also the partial GroupNorm sums
               of Y into the next layer's block (every residual feeds a GroupNorm except
               the three before an upsample conv)
    rgba       Y = RGBA8 of the decoder's output (channels 0-2 of A; x/2 + 1/2, clamped)

Activations are NHWC bf16 views: {"off", "pitch" (pixels per row), "border": 1 if the
image starts at row 1 / column 1 of a zero-bordered buffer (dit_conv's layout), 0 for a
plain [H*W, C] tensor, "px_stride": elements from one pixel to the next when a pixel's C
channels are the start of a wider row (default C)}; H, W, C are the op's. Kernels: vew.cc.

Topology: dit_ew's -- 16 cores, 2 per column (rows 2-3); per column two input streams (A,
B) split between the pair and two output streams (Y, Z) joined; the ops are DDR-bound.
The unit is an element of EL = 4096 bf16, whole pixels. Column c takes image rows
[c H/8, (c+1) H/8); a column's elements alternate between its two cores.

GroupNorm over 32 groups needs statistics of the whole tensor, so it is two dispatches:
gn_stats (or add with "stats") leaves each of the 16 cores' partial sums in the layer's
parameter block, and gn_apply reads the block ahead of the data. A block is 19 elements
at "stats_off" in S:
    [pad] [gamma[C] at 0, beta[C] at 512] [partials of cores 0..15] [pad]
(the host writes gamma/beta once; the pads only need to be readable). gn_apply streams it
p[-1], p[0], p[0], p[1], ..., p[17] -- a shim BD cannot repeat with stride 0 -- so each
core of a pair sees the whole block, one element apart (vew_param's shift).

rgba reads 4 of each pixel's channels, 1024 pixels an element, and writes 1024 RGBA8
pixels into the first half of an output element: Y is [H*W/1024][8192 bytes], the host
takes the first 4096 bytes of each.

    VE_SPEC='{"op": "gn_apply", ...}' python build_design.py designs/vae_ew/vae_ew.py <out>
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    Buffer, CompileTime, ExternalFunction, In, Kernel, ObjectFifo, Out, Program, Runtime,
    TaskGroup, Worker, WorkerRuntimeBarrier,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile

HERE = Path(__file__).resolve().parent
KERNEL_SRC = str(HERE / "vew.cc")

EL = 4096
CHUNK = 1024                     # innermost DMA dimension (a shim BD's d0 is <= 1023 words)
N_COLS, PER_COL = 8, 2
N_CORES = N_COLS * PER_COL
BLOCK = 19                       # elements of a GroupNorm parameter block
STACK = 0x800
P_LEN = 5 * 512 + 64             # floats of per-core state (vew.cc P_LEN)

(R_NPAR, R_CNT_STATS, R_STATS_OUT, R_CNT_APPLY, R_CNT_ADD, R_ADD_STATS_OUT, R_CNT_RGBA,
 R_C, R_SILU, R_ADD_STATS, R_NPIX) = range(11)
RTP_LEN = 12


def view_base(v: dict, C: int, row: int) -> int:
    b = v.get("border", 1)
    return v["off"] + ((b + row) * v["pitch"] + b) * v.get("px_stride", C)


def col_elements(s: dict) -> int:
    rows = s["H"] // N_COLS
    per_px = 4 if s["op"] == "rgba" else s["C"]
    n = rows * s["W"] * per_px
    assert n % (PER_COL * EL) == 0, f"{n} values per column is not a whole number of element pairs"
    return n // EL


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def vae_ew(X: In, B: In, S: In, Y: Out, *, spec: CompileTime[str]):
    s = json.loads(spec)
    op, C, H, W = s["op"], s["C"], s["H"], s["W"]
    assert H % N_COLS == 0 and C % 32 == 0 and C <= 512 and EL % (4 if op == "rgba" else C) == 0

    el_ty = np.ndarray[(EL,), np.dtype[bfloat16]]
    l2_ty = np.ndarray[(PER_COL * EL,), np.dtype[bfloat16]]
    par_ty = np.ndarray[(P_LEN,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]

    _ef = ExternalFunction("vew_begin", source_file=KERNEL_SRC, arg_types=[par_ty, rtp_ty],
                           compile_flags=[f"-DVEW_EL={EL}", "-Wno-deprecated-declarations"])
    obj = _ef.object_file_name
    kernels = [_ef,
               Kernel("vew_param", obj, [el_ty, par_ty, np.int32, np.int32, rtp_ty]),
               Kernel("vew_stats", obj, [el_ty, par_ty, rtp_ty]),
               Kernel("vew_stats_out", obj, [el_ty, par_ty, rtp_ty]),
               Kernel("vew_apply", obj, [el_ty, el_ty, par_ty, rtp_ty]),
               Kernel("vew_add", obj, [el_ty, el_ty, el_ty, par_ty, rtp_ty]),
               Kernel("vew_rgba", obj, [el_ty, el_ty, rtp_ty])]

    offs = [i * EL for i in range(PER_COL)]
    A3, B3, Y3, Z3 = [], [], [], []
    A1, B1, Y1, Z1 = {}, {}, {}, {}
    for c in range(N_COLS):
        for name, lst, l1, depth in (("A", A3, A1, 2), ("B", B3, B1, 1)):
            f = ObjectFifo(l2_ty, name=f"{name}_L3L2_{c}", depth=2)
            lst.append(f)
            parts = f.cons().split(offs, tile=Tile(c, 1), obj_types=[el_ty] * PER_COL,
                                   names=[f"{name}_L2L1_{c}_{r}" for r in range(PER_COL)],
                                   depths=[depth] * PER_COL)
            for r in range(PER_COL):
                l1[(c, r)] = parts[r].cons()
        for name, lst, l1, depth in (("Y", Y3, Y1, 2), ("Z", Z3, Z1, 1)):
            f = ObjectFifo(l2_ty, name=f"{name}_L2L3_{c}", depth=2)
            lst.append(f)
            parts = f.prod().join(offs, tile=Tile(c, 1), obj_types=[el_ty] * PER_COL,
                                  names=[f"{name}_L1L2_{c}_{r}" for r in range(PER_COL)],
                                  depths=[depth] * PER_COL)
            for r in range(PER_COL):
                l1[(c, r)] = parts[r].prod()

    def make_core_fn(r):
        return lambda *args: core_fn(r, *args)

    def core_fn(r, a_in, b_in, y_out, z_out, kern, par, my_rtp, barrier):
        k_begin, k_param, k_stats, k_sout, k_apply, k_add, k_rgba = kern
        barrier.wait_for_value(1)
        k_begin(par, my_rtp)
        for i in range_(my_rtp[R_NPAR]):
            e = a_in.acquire(1)
            k_param(e, par, i, r - 1, my_rtp)
            a_in.release(1)
        for _ in range_(my_rtp[R_CNT_STATS]):
            x = a_in.acquire(1)
            k_stats(x, par, my_rtp)
            a_in.release(1)
        for _ in range_(my_rtp[R_STATS_OUT]):
            y = y_out.acquire(1)
            k_sout(y, par, my_rtp)
            y_out.release(1)
        for _ in range_(my_rtp[R_CNT_APPLY]):
            x = a_in.acquire(1)
            y = y_out.acquire(1)
            k_apply(x, y, par, my_rtp)
            a_in.release(1)
            y_out.release(1)
        for _ in range_(my_rtp[R_CNT_ADD]):
            a = a_in.acquire(1)
            b = b_in.acquire(1)
            y = y_out.acquire(1)
            k_add(a, b, y, par, my_rtp)
            a_in.release(1)
            b_in.release(1)
            y_out.release(1)
        for _ in range_(my_rtp[R_ADD_STATS_OUT]):
            z = z_out.acquire(1)
            k_sout(z, par, my_rtp)
            z_out.release(1)
        for _ in range_(my_rtp[R_CNT_RGBA]):
            x = a_in.acquire(1)
            y = y_out.acquire(1)
            k_rgba(x, y, my_rtp)
            a_in.release(1)
            y_out.release(1)
        barrier.release_with_value(1)

    rtp, barriers, workers = {}, {}, []
    for c in range(N_COLS):
        for r in range(PER_COL):
            rtp[(c, r)] = Buffer(rtp_ty, name=f"rtp_{c}_{r}",
                                 initial_value=np.zeros(RTP_LEN, dtype=np.int32),
                                 use_write_rtp=True)
            barriers[(c, r)] = WorkerRuntimeBarrier()
            workers.append(Worker(make_core_fn(r), [
                A1[(c, r)], B1[(c, r)], Y1[(c, r)], Z1[(c, r)], kernels,
                Buffer(par_ty, name=f"par_{c}_{r}"), rtp[(c, r)], barriers[(c, r)]],
                tile=Tile(c, 2 + r), stack_size=STACK))

    n_col = col_elements(s)
    per_core = n_col // PER_COL
    rows = H // N_COLS
    sizes = s["sizes"]
    tys = [np.ndarray[(sizes[k],), np.dtype[bfloat16]] for k in ("X", "B", "S", "Y")]
    stats_off = s.get("stats_off", 0)

    def data_fill(v, c):
        """(offset, sizes, strides) of column c's rows of view v (C channels per pixel)."""
        ps = v.get("px_stride", C)
        if ps != C:
            return view_base(v, C, c * rows), [rows, W, C], [v["pitch"] * ps, ps, 1]
        return (view_base(v, C, c * rows), [rows, W * C // CHUNK, CHUNK],
                [v["pitch"] * C, CHUNK, 1])

    def sequence(x, b, st, y, a_prods, b_prods, y_conses, z_conses):
        cnt = {"gn_stats": R_CNT_STATS, "gn_apply": R_CNT_APPLY, "add": R_CNT_ADD,
               "rgba": R_CNT_RGBA}[op]
        add_stats = op == "add" and bool(s.get("stats"))
        for (c, r), buf in rtp.items():
            vals = {cnt: per_core, R_C: C, R_SILU: int(bool(s.get("silu"))),
                    R_NPIX: H * W, R_ADD_STATS: int(add_stats),
                    R_NPAR: BLOCK - 1 if op == "gn_apply" else 0,
                    R_STATS_OUT: int(op == "gn_stats"), R_ADD_STATS_OUT: int(add_stats)}
            for k in range(RTP_LEN):
                buf[k] = vals.get(k, 0)
        for bar in barriers.values():
            bar.set(1)
        tg = TaskGroup()
        for c in range(N_COLS):
            if op == "gn_apply":        # the block, p[-1], p[0], p[0], ... (module docstring)
                a_prods[c].fill(st, offset=stats_off, sizes=[BLOCK - 1, PER_COL, EL // CHUNK, CHUNK],
                                strides=[EL, EL, CHUNK, 1], group=tg)
            if op == "rgba":            # 4 channels of each pixel, <= 64 rows per BD
                v = s["a"]
                for r0 in range(0, rows, 64):
                    n = min(64, rows - r0)
                    a_prods[c].fill(x, offset=view_base(v, C, c * rows + r0),
                                    sizes=[n, W // 512, 512, 4] if W > 512 else [n, W, 4],
                                                    strides=([v["pitch"] * C, 512 * C, C, 1] if W > 512
                                             else [v["pitch"] * C, C, 1]), group=tg)
            else:
                off, sz, stv = data_fill(s["a"], c)
                a_prods[c].fill(x, offset=off, sizes=sz, strides=stv, group=tg)
            if op == "add":
                off, sz, stv = data_fill(s["b"], c)
                b_prods[c].fill(b, offset=off, sizes=sz, strides=stv, group=tg)
            if op == "gn_stats":
                y_conses[c].drain(st, offset=stats_off + (2 + PER_COL * c) * EL,
                                  sizes=[1, 1, 1, PER_COL * EL], strides=[0, 0, 0, 1],
                                  wait=True, group=tg)
            elif op == "rgba":
                y_conses[c].drain(y, offset=s["y"]["off"] + c * n_col * EL,
                                  sizes=[1, 1, 1, n_col * EL], strides=[0, 0, 0, 1],
                                  wait=True, group=tg)
            else:
                off, sz, stv = data_fill(s["y"], c)
                y_conses[c].drain(y, offset=off, sizes=sz, strides=stv, wait=True, group=tg)
            if add_stats:
                z_conses[c].drain(st, offset=stats_off + (2 + PER_COL * c) * EL,
                                  sizes=[1, 1, 1, PER_COL * EL], strides=[0, 0, 0, 1],
                                  wait=True, group=tg)
        tg.finish()

    a_prods = [A3[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    b_prods = [B3[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    y_conses = [Y3[c].cons(tile=Tile(c, 0)) for c in range(N_COLS)]
    z_conses = [Z3[c].cons(tile=Tile(c, 0)) for c in range(N_COLS)]
    rt = Runtime(sequence, tys + [a_prods, b_prods, y_conses, z_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# build_design.py convention: VE_SPEC is the stream's JSON spec.
DESIGN = vae_ew
SPECIALIZE = dict(spec=os.environ.get("VE_SPEC", json.dumps({
    "op": "add", "C": 128, "H": 64, "W": 128,
    "a": {"off": 0, "pitch": 130}, "b": {"off": 0, "pitch": 130}, "y": {"off": 0, "pitch": 130},
    "sizes": {"X": 66 * 130 * 128, "B": 66 * 130 * 128, "S": BLOCK * EL, "Y": 66 * 130 * 128}})))
