r"""dit_ew: the row-wise and elementwise ops of a diffusion transformer step, one xclbin.

    ln_mod       Y = LN(A) * (1 + scale) + shift                 A: rows      params: shift, scale
                 "norm": "rms":  Y = RMSNorm(A) * w                                 params: w
    res_ln_mod   Z = A + gate * B;  Y = LN(Z) * (1 + scale) + shift          params: gate, shift, scale
                 "norm": "rms", "unit_gate": 1:  Z = A + B;  Y = RMSNorm(Z) * w      params: w
    qk           Y = rope(rmsnorm(A; wq)), Z = rope(rmsnorm(B; wk)) (A = q rows, B = k rows)
                 "rope": "qwen", "b_q_heads": 8, "b_k_heads": 8: Qwen3's fused q|k|v row as two
                 elements (A = q heads 0-23; B = q 24-31, k 0-7, v 0-7 copied through),
                 rotate-half RoPE
    swiglu       Y = silu(A) * B
    euler        Y = A + dt * B                                  params: dt
    silu         Y = silu(A)

Every op is an instruction stream over ONE static configuration: the core program runs
each op's loop a runtime number of times (zero for the ops not asked for), so a whole
model's elementwise work shares one hardware context. Kernels: ew.cc.

Topology: 16 cores, 2 per column (rows 2-3). Per column, two input streams (A, B) come
in through the memtile and split between the pair; two output streams (Y, Z) join there
on the way out -- 6 memtile S2MM and 6 MM2S, the channel budget, which is why it is 2
cores per column and not 4. The ops are DDR-bound: 16 cores are plenty.

The unit of work is a row element of EL = 3072 bf16 (FLUX.2 [klein]'s hidden size; a
q/k row is 24 heads). A stream reads a "view" of a buffer: T tokens, each E consecutive
elements of 3072 starting at column `off`, row stride `ld` (the fused QKV/MLP buffers
are read in place). Column c takes tokens [c*T/8, (c+1)*T/8); within a column the
elements alternate between its two cores. Euler reads latents in "tile24" views instead
(24 tokens x 128 channels per element).

Parameter vectors (P) ride stream A ahead of the rows. The pair's split hands
alternate elements to its two cores, and a shim BD cannot repeat with stride 0, so one
fill streams p[-1], p[0], p[0], p[1], p[1], ..., p[n]: core 0 sees p[-1], p[0], .. and
core 1 sees p[0], p[1], .., and load_param stores each at its slot (dropping p[-1] and
p[n]). The run [p_off, p_off + n_par*3072) must therefore have one vector of readable
memory on each side. The modulation layout is the engine's: the spec names the run and
which vector in it is gate/shift/scale.

The spec (JSON, env DE_SPEC for build_design.py) -- see make_test.py for examples:
    {"op": "ln_mod", "a": view, "y": view, "p_off": 0, "n_par": 2,
     "idx": {"shift": 0, "scale": 1}, "W": 3072, "sizes": {"X": .., "B": .., "P": .., "Y": .., "Z": ..}}
    view = {"off": int, "ld": int, "T": int, "E": int}  or  {"kind": "tile24", "off", "ld", "n": elements}
    qk: also "tok0" (the first token's index in the joint sequence), "n_txt", "grid_w", "heads"
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
KERNEL_SRC = str(HERE / "ew.cc")

EL = 3072
CHUNK = 1024                      # innermost DMA dimension (a shim BD's d0 is <= 1023 words)
N_COLS, PER_COL = 8, 2
N_PAR_MAX = 3
STACK = 0x800

(R_NPAR, R_CNT_LN, R_CNT_RES, R_CNT_QK, R_CNT_BIN, R_CNT_UN, R_W, R_IDX_GATE, R_IDX_SHIFT,
 R_IDX_SCALE, R_BINOP, R_ROW0, R_ROW_STEP, R_N_TXT, R_GRID_W, R_HEADS, R_NORM, R_UNIT_GATE,
 R_ROPE, R_B_QH, R_B_KH) = range(21)
RTP_LEN = 24
BINOPS = {"swiglu": 1, "euler": 2}
OPS_2IN = {"res_ln_mod", "qk", "swiglu", "euler"}
OPS_2OUT = {"res_ln_mod", "qk"}


def view_elems(v: dict) -> int:
    return v["n"] if v.get("kind") == "tile24" else v["T"] * v["E"]


def check_spec(s: dict) -> str | None:
    n = view_elems(s["a"])
    if n % (N_COLS * PER_COL):
        return f"dit_ew needs a multiple of {N_COLS * PER_COL} row elements per stream (got {n})"
    for k in ("b", "y", "z"):
        if k in s and view_elems(s[k]) != n:
            return f"view {k} has {view_elems(s[k])} elements, a has {n}"
    for k in ("a", "b", "y", "z"):
        v = s.get(k)
        if v and v.get("kind") != "tile24" and (v["T"] % N_COLS or (v["T"] // N_COLS * v["E"]) % PER_COL):
            return f"view {k}: T must split evenly over {N_COLS} columns x {PER_COL} cores"
    return None


def _col_pattern(v: dict, c: int):
    """(offset, sizes, strides) of column c's share of a view."""
    if v.get("kind") == "tile24":
        per = v["n"] // N_COLS
        return (v["off"] + c * per * 24 * v["ld"], [per, 24, 128], [24 * v["ld"], v["ld"], 1])
    per = v["T"] // N_COLS
    return (v["off"] + c * per * v["ld"], [per, v["E"] * EL // CHUNK, CHUNK],
            [v["ld"], CHUNK, 1])


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def dit_ew(X: In, B: In, P: In, Y: Out, Z: Out, *, spec: CompileTime[str]):
    s = json.loads(spec)
    why = check_spec(s)
    assert why is None, why
    op = s["op"]

    el_ty = np.ndarray[(EL,), np.dtype[bfloat16]]
    l2_ty = np.ndarray[(PER_COL * EL,), np.dtype[bfloat16]]
    par_ty = np.ndarray[(N_PAR_MAX * EL,), np.dtype[bfloat16]]
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]

    _ef = ExternalFunction("load_param", source_file=KERNEL_SRC,
                           arg_types=[el_ty, par_ty, np.int32, np.int32, rtp_ty],
                           compile_flags=[f"-DEW_EL={EL}", "-Wno-deprecated-declarations"]
                           + (["-DEW_NOP"] if os.environ.get("DE_NOP") else []))
    obj = _ef.object_file_name
    k_param = Kernel("load_param", obj, [el_ty, par_ty, np.int32, np.int32, rtp_ty])
    k_ln = Kernel("ln_mod_row", obj, [el_ty, el_ty, par_ty, rtp_ty])
    k_res = Kernel("res_ln_mod_row", obj, [el_ty, el_ty, el_ty, el_ty, par_ty, rtp_ty])
    k_qk = Kernel("qk_row", obj, [el_ty, el_ty, el_ty, el_ty, par_ty, rtp_ty, np.int32])
    k_bin = Kernel("bin_row", obj, [el_ty, el_ty, el_ty, par_ty, rtp_ty])
    k_un = Kernel("un_row", obj, [el_ty, el_ty, rtp_ty])
    kernels = [k_param, k_ln, k_res, k_qk, k_bin, k_un]

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

    rtp, barriers = {}, {}

    def make_core_fn(r):
        return lambda *args: core_fn(r, *args)

    def core_fn(r, a_in, b_in, y_out, z_out, kern, par, my_rtp, barrier):
        k_param, k_ln, k_res, k_qk, k_bin, k_un = kern
        barrier.wait_for_value(1)
        for i in range_(my_rtp[R_NPAR]):
            e = a_in.acquire(1)
            k_param(e, par, i, r - 1, my_rtp)
            a_in.release(1)
        for _ in range_(my_rtp[R_CNT_LN]):
            x = a_in.acquire(1)
            y = y_out.acquire(1)
            k_ln(x, y, par, my_rtp)
            a_in.release(1)
            y_out.release(1)
        for _ in range_(my_rtp[R_CNT_RES]):
            x = a_in.acquire(1)
            yb = b_in.acquire(1)
            z = y_out.acquire(1)
            xo = z_out.acquire(1)
            k_res(x, yb, z, xo, par, my_rtp)
            a_in.release(1)
            b_in.release(1)
            y_out.release(1)
            z_out.release(1)
        for t in range_(my_rtp[R_CNT_QK]):
            q = a_in.acquire(1)
            k = b_in.acquire(1)
            qo = y_out.acquire(1)
            ko = z_out.acquire(1)
            k_qk(q, k, qo, ko, par, my_rtp, t)
            a_in.release(1)
            b_in.release(1)
            y_out.release(1)
            z_out.release(1)
        for _ in range_(my_rtp[R_CNT_BIN]):
            a = a_in.acquire(1)
            b = b_in.acquire(1)
            y = y_out.acquire(1)
            k_bin(a, b, y, par, my_rtp)
            a_in.release(1)
            b_in.release(1)
            y_out.release(1)
        for _ in range_(my_rtp[R_CNT_UN]):
            a = a_in.acquire(1)
            y = y_out.acquire(1)
            k_un(a, y, my_rtp)
            a_in.release(1)
            y_out.release(1)
        barrier.release_with_value(1)

    workers = []
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

    n_el = view_elems(s["a"])
    per_core = n_el // (N_COLS * PER_COL)
    n_par = s.get("n_par", 0)
    idx = s.get("idx", {})
    cnt_slot = {"ln_mod": R_CNT_LN, "res_ln_mod": R_CNT_RES, "qk": R_CNT_QK,
                "swiglu": R_CNT_BIN, "euler": R_CNT_BIN, "silu": R_CNT_UN}[op]
    sizes = s["sizes"]
    tys = [np.ndarray[(sizes[k],), np.dtype[bfloat16]] for k in ("X", "B", "P", "Y", "Z")]

    def sequence(x, b, p, y, z, a_prods, b_prods, y_conses, z_conses):
        for (c, r), buf in rtp.items():
            vals = {R_NPAR: n_par + 1 if n_par else 0, cnt_slot: per_core, R_W: s.get("W", EL),
                    R_IDX_GATE: idx.get("gate", 0), R_IDX_SHIFT: idx.get("shift", 0),
                    R_IDX_SCALE: idx.get("scale", 0), R_BINOP: BINOPS.get(op, 0),
                    R_HEADS: s.get("heads", EL // 128),
                    R_NORM: int(s.get("norm") == "rms"), R_UNIT_GATE: int(s.get("unit_gate", 0)),
                    R_ROPE: int(s.get("rope") == "qwen"), R_B_QH: s.get("b_q_heads", 0),
                    R_B_KH: s.get("b_k_heads", EL // 128)}
            if op == "qk":
                vals |= {R_ROW0: s.get("tok0", 0) + c * (s["a"]["T"] // N_COLS) + r,
                         R_ROW_STEP: PER_COL, R_N_TXT: s.get("n_txt", 0),
                         R_GRID_W: s.get("grid_w", 1)}
            for k in range(RTP_LEN):
                buf[k] = vals.get(k, 0)
        for bar in barriers.values():
            bar.set(1)
        tg = TaskGroup()
        for c in range(N_COLS):
            if n_par:
                # p[-1], p[0], p[0], p[1], ..., p[n]: see the module docstring
                a_prods[c].fill(p, offset=s["p_off"] - EL,
                                sizes=[n_par + 1, PER_COL, EL // CHUNK, CHUNK],
                                strides=[EL, EL, CHUNK, 1], group=tg)
            off, sz, st = _col_pattern(s["a"], c)
            a_prods[c].fill(x, offset=off, sizes=sz, strides=st, group=tg)
            if op in OPS_2IN:
                off, sz, st = _col_pattern(s["b"], c)
                b_prods[c].fill(b, offset=off, sizes=sz, strides=st, group=tg)
            off, sz, st = _col_pattern(s["y"], c)
            y_conses[c].drain(y, offset=off, sizes=sz, strides=st, wait=True, group=tg)
            if op in OPS_2OUT:
                off, sz, st = _col_pattern(s["z"], c)
                z_conses[c].drain(z, offset=off, sizes=sz, strides=st, wait=True, group=tg)
        tg.finish()

    a_prods = [A3[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    b_prods = [B3[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    y_conses = [Y3[c].cons(tile=Tile(c, 0)) for c in range(N_COLS)]
    z_conses = [Z3[c].cons(tile=Tile(c, 0)) for c in range(N_COLS)]
    rt = Runtime(sequence, tys + [a_prods, b_prods, y_conses, z_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# build_design.py convention: DE_SPEC is the stream's JSON spec.
DESIGN = dit_ew
SPECIALIZE = dict(spec=os.environ.get("DE_SPEC", json.dumps({
    "op": "ln_mod", "a": {"off": 0, "ld": EL, "T": 512, "E": 1},
    "y": {"off": 0, "ld": EL, "T": 512, "E": 1}, "p_off": EL, "n_par": 2,
    "idx": {"shift": 0, "scale": 1},
    "sizes": {"X": 512 * EL, "B": EL, "P": 4 * EL, "Y": 512 * EL, "Z": EL}})))
