r"""attn_fa: causal flash attention for the dense block prefill at head dim 64 (OPEN-PREFILL-ATTN-FA).

    O[t, h*64:(h+1)*64] = softmax_causal(Q_h K_kv(h)^T) V_kv(h)     (Q pre-scaled by log2(e)/sqrt(64))

One dispatch is one 256-token block of one layer against a window of L cached rows; one xclbin
serves every L (one instruction stream per 256 rows of window), so a layer's attention is one
hardware context whatever the prompt length. Scores, mask, softmax and both products stay on the
array; nothing between them comes back to the host.

Topology: 4 groups of 8 cores (two columns each). A pass is one query head's 256 tokens; each core
owns 32 of them (core i: tokens 32i .. 32i+31) and walks every key chunk of its kv head's window
itself, so there is no merge. K reaches a group through its first column's memtile, V through its
second's, broadcast to the 8 cores; the pass's record and Q (hi, then lo) ride the same two
broadcasts ahead of the keys, and each core keeps its own rows (dit_fa's capture). A group takes
heads 10g .. 10g+9 one pass each. Nothing is re-laid on the way: the kernels read K and V as the
cache's rows and Q as the host wrote it, and write O as token rows.

Buffers (the engine's argument order):
  Q   bf16, per head [record 64x64 | Q hi: 8 cores x Q^T [64 dims][32 tokens] | Q lo: the same];
      the record's first int32 is p0, the block's first position (token t's causal bound is p0 + t)
  KV  the layer's KV cache, bf16 rows of kv_ld elements: K at kvh*64, V at kv_heads*64 + kvh*64
  O   fp32 [256 tokens, heads*64] row-major (the out-projection's input)

    AF_L=2048 python open_kernels/build_design.py open_kernels/designs/attn_fa/attn_fa.py <out>
"""

from __future__ import annotations

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
KERNEL_SRC = str(HERE / "attn_fa.cc")

T, TQ, LKP, HD = 256, 32, 64, 64
N_GROUPS, GROUP_COLS, N_ROWS = 4, 2, 4
CORES = GROUP_COLS * N_ROWS            # per group: T // TQ
Q_ELEMS = T // (2 * TQ)                # Q elements a pass (two cores' rows each)
ELEM = LKP * HD                        # bf16 per fifo element
MAX_INFLIGHT = int(os.environ.get("AF_INFLIGHT", 2))   # passes whose BDs are queued ahead
STACK = 0x2000      # af_sm's four interleaved exponentials take a 5 KB frame; 4 KB hung the array
QSPLIT = int(os.environ.get("AF_QSPLIT", 1))
TAU = int(os.environ.get("AF_TAU", 16))
KDEPTH = int(os.environ.get("AF_KDEPTH", 2))
VDEPTH = int(os.environ.get("AF_VDEPTH", 1))
ABL = int(os.environ.get("AF_ABL", 0))            # attn_fa.cc AF_ABL, timing ablations only
RTP_PASSES, RTP_CHUNKS, RTP_TBL, RTP_PQ, RTP_PKV = 0, 1, 2, 3, 4
RTP_LEN = 8
# AF_MODE=prep: the same cores and dataflow, another instruction stream (OPEN-PREFILL-GLUE-PREP)
MODE = os.environ.get("AF_MODE", "attn")
PREP_KV_PER_GROUP = 2     # kv heads a group rotates and caches: 8 over 4 groups
# bisecting a hang: passes a prep dispatch runs (-1: all), and whether it loads the table
PREP_NQ = int(os.environ.get("AF_PREP_NQ", -1))
PREP_NKV = int(os.environ.get("AF_PREP_NKV", -1))
PREP_NTBL = int(os.environ.get("AF_PREP_NTBL", 1))
# two prep passes in flight hang the array (bisected); one at a time
PREP_INFLIGHT = int(os.environ.get("AF_PREP_INFLIGHT", 1))
# fills of pass k+1 are queued before pass k's drains are awaited
PREP_PIPE = int(os.environ.get("AF_PREP_PIPE", 1))

assert CORES * TQ == T
PREP_TBL_ELEMS = T * 2 * HD * 2     # the block's table, fp32 [256 tokens][cos 32 | sin 32], as bf16 pairs
QKV_NPAD = int(os.environ.get("AF_QKV_NPAD", 4096))   # the qkv GEMM's padded output width, c-tiled by 128 columns


def q_head_elems(qsplit: int = QSPLIT) -> int:
    """bf16 per head in the Q buffer: the record, then hi (and lo) as Q_ELEMS elements each."""
    return ELEM * (1 + Q_ELEMS * (2 if qsplit else 1))


def check_shape(L: int, heads: int, kv_heads: int) -> str | None:
    if L % 256 or L // LKP > 64 or heads % N_GROUPS or heads % kv_heads:
        return (f"attn_fa needs L % 256 with at most 64 key chunks, heads % {N_GROUPS}, heads % kv_heads "
                f"(got L={L} heads={heads} kv_heads={kv_heads})")
    return None


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def attn_fa(Q: In, KV: In, O: Out, TB: In, *, L: CompileTime[int], heads: CompileTime[int],
            kv_heads: CompileTime[int], kv_ld: CompileTime[int], max_rows: CompileTime[int]):
    why = check_shape(L, heads, kv_heads)
    assert why is None, why
    hpg = heads // N_GROUPS
    gqa = heads // kv_heads
    qw = heads * HD
    qh_elems = q_head_elems()
    n_chunks = L // LKP

    elem_ty = np.ndarray[(LKP, HD), np.dtype[bfloat16]]
    q_ty = np.ndarray[(TQ * HD,), np.dtype[bfloat16]]
    s_ty = np.ndarray[(TQ * LKP,), np.dtype[np.float32]]
    st_ty = np.ndarray[(2 * TQ * 8 + 16,), np.dtype[np.float32]]
    o_l1_ty = np.ndarray[(TQ * HD,), np.dtype[np.float32]]
    o_l2_ty = np.ndarray[(N_ROWS * TQ * HD,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]

    flags = [f"-DAF_QSPLIT={QSPLIT}", f"-DAF_TAU={TAU}", f"-DAF_ABL={ABL}", "-Wno-deprecated-declarations"]
    _ef = ExternalFunction("af_init", source_file=KERNEL_SRC, arg_types=[o_l1_ty, st_ty], compile_flags=flags)
    obj = _ef.object_file_name
    k_meta = Kernel("af_meta", obj, [elem_ty, st_ty])
    k_cap = Kernel("af_capture", obj, [elem_ty, q_ty, np.int32])
    k_init = Kernel("af_init", obj, [o_l1_ty, st_ty])
    k_qk = Kernel("af_qk", obj, [q_ty, q_ty, elem_ty, s_ty, st_ty, np.int32, np.int32])
    k_sm = Kernel("af_sm", obj, [s_ty, st_ty, o_l1_ty, np.int32, np.int32])
    k_pv = Kernel("af_pv", obj, [s_ty, elem_ty, o_l1_ty, st_ty])
    k_fin = Kernel("af_fin", obj, [o_l1_ty, st_ty])
    k_ptab = Kernel("af_ptab", obj, [elem_ty, s_ty])
    k_prep_q = Kernel("af_prep_q", obj, [elem_ty, s_ty, q_ty, q_ty, o_l1_ty])
    k_prep_kv = Kernel("af_prep_kv", obj, [elem_ty, elem_ty, s_ty, o_l1_ty])
    kernels = [k_meta, k_cap, k_init, k_qk, k_sm, k_pv, k_fin, k_ptab, k_prep_q, k_prep_kv]

    def core_tile(g, i):
        return Tile(g * GROUP_COLS + i // N_ROWS, 2 + i % N_ROWS)

    A_l3l2, B_l3l2, A_l1, B_l1 = [], [], [], []
    for g in range(N_GROUPS):
        fa = ObjectFifo(elem_ty, name=f"A_L3L2_{g}", depth=2)
        fb = ObjectFifo(elem_ty, name=f"B_L3L2_{g}", depth=2)
        A_l3l2.append(fa)
        B_l3l2.append(fb)
        a2 = fa.cons().forward(tile=Tile(g * GROUP_COLS, 1), obj_type=elem_ty, depth=2, name=f"A_L2L1_{g}")
        b2 = fb.cons().forward(tile=Tile(g * GROUP_COLS + 1, 1), obj_type=elem_ty, depth=2, name=f"B_L2L1_{g}")
        A_l1.append([a2.cons(depth=KDEPTH) for _ in range(CORES)])
        B_l1.append([b2.cons(depth=VDEPTH) for _ in range(CORES)])

    # O: a column's 4 cores join at its memtile; each part is already token rows
    O_l2l3, O_l1 = [], {}
    for c in range(N_GROUPS * GROUP_COLS):
        f = ObjectFifo(o_l2_ty, name=f"O_L2L3_{c}", depth=2)
        O_l2l3.append(f)
        parts = f.prod().join([r * TQ * HD for r in range(N_ROWS)], tile=Tile(c, 1),
                              obj_types=[o_l1_ty] * N_ROWS, names=[f"O_L1L2_{c}_{r}" for r in range(N_ROWS)],
                              depths=[1] * N_ROWS)
        for r in range(N_ROWS):
            O_l1[(c, r)] = parts[r]

    rtp, barriers = {}, {}

    def make_core_fn(i):
        tok0 = i * TQ

        def core_fn(a_in, b_in, o_out, kern, q_hi, q_lo, s_buf, st, my_rtp, barrier):
            (meta, cap, init, qk, sm, pv, fin, ptab, prep_q, prep_kv) = kern
            barrier.wait_for_value(1)
            n_pass = my_rtp[RTP_PASSES]
            n_ch = my_rtp[RTP_CHUNKS]
            for _ in range_(n_pass):
                o = o_out.acquire(1)
                e = a_in.acquire(1)
                meta(e, st)
                a_in.release(1)
                for q in range(Q_ELEMS):
                    e = a_in.acquire(1)
                    if q == i // 2:
                        cap(e, q_hi, i % 2)
                    a_in.release(1)
                if QSPLIT:
                    for q in range(Q_ELEMS):
                        e = b_in.acquire(1)
                        if q == i // 2:
                            cap(e, q_lo, i % 2)
                        b_in.release(1)
                init(o, st)
                for c in range_(n_ch):
                    k = a_in.acquire(1)
                    qk(q_hi, q_lo, k, s_buf, st, tok0, c)
                    a_in.release(1)
                    sm(s_buf, st, o, tok0, c)
                    v = b_in.acquire(1)
                    pv(s_buf, v, o, st)
                    b_in.release(1)
                fin(o, st)
                o_out.release(1)
            # prep dispatch: the block's table once (resident in s_buf), then a pass per query head and per kv head
            for _ in range_(my_rtp[RTP_TBL]):
                for e in range(CORES):
                    x = a_in.acquire(1)
                    if e == i:
                        ptab(x, s_buf)
                    a_in.release(1)
            for _ in range_(my_rtp[RTP_PQ]):
                o = o_out.acquire(1)
                for e in range(CORES):
                    x = a_in.acquire(1)
                    if e == i:
                        prep_q(x, s_buf, q_hi, q_lo, o)
                    a_in.release(1)
                o_out.release(1)
            for _ in range_(my_rtp[RTP_PKV]):
                o = o_out.acquire(1)
                for e in range(CORES):
                    xk = a_in.acquire(1)
                    xv = b_in.acquire(1)
                    if e == i:
                        prep_kv(xk, xv, s_buf, o)
                    a_in.release(1)
                    b_in.release(1)
                o_out.release(1)
            barrier.release_with_value(1)
        return core_fn

    workers = []
    for g in range(N_GROUPS):
        for i in range(CORES):
            t = core_tile(g, i)
            c, r = t.col, t.row - 2
            rtp[(g, i)] = Buffer(rtp_ty, name=f"rtp_{g}_{i}", initial_value=np.zeros(RTP_LEN, dtype=np.int32),
                                 use_write_rtp=True)
            barriers[(g, i)] = WorkerRuntimeBarrier()
            args = [A_l1[g][i], B_l1[g][i], O_l1[(c, r)].prod(), kernels,
                    Buffer(q_ty, name=f"qh_{g}_{i}"), Buffer(q_ty, name=f"ql_{g}_{i}"),
                    Buffer(s_ty, name=f"s_{g}_{i}"), Buffer(st_ty, name=f"st_{g}_{i}"),
                    rtp[(g, i)], barriers[(g, i)]]
            workers.append(Worker(make_core_fn(i), args, tile=t, stack_size=STACK))

    Q_ty = np.ndarray[(heads * qh_elems,), np.dtype[bfloat16]]
    KV_ty = np.ndarray[(max_rows * kv_ld,), np.dtype[bfloat16]]
    O_ty = np.ndarray[(T * qw,), np.dtype[np.float32]]
    TB_ty = np.ndarray[(PREP_TBL_ELEMS,), np.dtype[bfloat16]]
    qa_rows = (1 + Q_ELEMS) * LKP          # record + Q hi, in 64-wide rows
    qb_rows = Q_ELEMS * LKP

    def sequence(q, kv, o, tb, a_prods, b_prods, o_conses):
        for buf in rtp.values():
            buf[RTP_PASSES] = hpg
            buf[RTP_CHUNKS] = n_chunks
            buf[RTP_TBL] = 0
            buf[RTP_PQ] = 0
            buf[RTP_PKV] = 0
        for b in barriers.values():
            b.set(1)
        inflight: list[TaskGroup] = []
        for k in range(hpg):
            tg = TaskGroup()
            for g in range(N_GROUPS):
                h = g * hpg + k
                kvh = h // gqa
                base = h * qh_elems
                a_prods[g].fill(q, offset=base, sizes=[qa_rows, HD], strides=[HD, 1], group=tg)
                a_prods[g].fill(kv, offset=kvh * HD, sizes=[n_chunks, LKP, HD],
                                strides=[LKP * kv_ld, kv_ld, 1], group=tg)
                if QSPLIT:
                    b_prods[g].fill(q, offset=base + qa_rows * HD, sizes=[qb_rows, HD], strides=[HD, 1], group=tg)
                b_prods[g].fill(kv, offset=(kv_heads + kvh) * HD, sizes=[n_chunks, LKP, HD],
                                strides=[LKP * kv_ld, kv_ld, 1], group=tg)
                for cc in range(GROUP_COLS):
                    col = g * GROUP_COLS + cc
                    row0 = cc * N_ROWS * TQ
                    o_conses[col].drain(o, offset=row0 * qw + h * HD, sizes=[N_ROWS * TQ, HD],
                                        strides=[qw, 1], wait=True, group=tg)
            inflight.append(tg)
            if len(inflight) == MAX_INFLIGHT:
                inflight.pop(0).finish()
        for tg in inflight:
            tg.finish()

    # prep: y is the qkv GEMM's c-tiled fp32 output (seen as bf16 pairs), tb the block's table, q gets Q hi / lo in the
    # pass layout and kv (a window of the cache that starts at the block's first row) gets K rotated and V
    y_cols = 8
    y_tiles_per_col = QKV_NPAD // 128 // y_cols
    y_ty = np.ndarray[(T * QKV_NPAD * 2,), np.dtype[bfloat16]]

    def y_off(ct, half):    # bf16 offset of c-tile ct's column half (64 floats) in the c-tiled y
        return 2 * (((ct % y_cols) * y_tiles_per_col + ct // y_cols) * T * 128 + half * 64)

    def prep_sequence(q, kv, y, tb, a_prods, b_prods, o_conses):
        for buf in rtp.values():
            buf[RTP_PASSES] = 0
            buf[RTP_CHUNKS] = 0
            buf[RTP_TBL] = PREP_NTBL
            buf[RTP_PQ] = PREP_NQ if PREP_NQ >= 0 else hpg
            buf[RTP_PKV] = PREP_NKV if PREP_NKV >= 0 else PREP_KV_PER_GROUP
        for b in barriers.values():
            b.set(1)
        inflight: list[TaskGroup] = []

        prev: list[tuple[TaskGroup, TaskGroup]] = []

        def pass_group(fills, drains, first=False):
            tf, td = TaskGroup(), TaskGroup()
            if first and PREP_NTBL:
                for g in range(N_GROUPS):
                    a_prods[g].fill(tb, offset=0, sizes=[T, 2 * HD], strides=[2 * HD, 1], group=tf)
            for fill in fills:
                fill(tf)
            if PREP_PIPE and prev:
                prev[-1][1].finish()
            for drain in drains:
                drain(td)
            if PREP_PIPE and prev:
                prev.pop()[0].finish()
            if not PREP_PIPE:
                td.finish()
                tf.finish()
                return
            prev.append((tf, td))

        if PREP_NQ == 0 and PREP_NTBL:
            pass_group([], [], first=True)
        for k in range(PREP_NQ if PREP_NQ >= 0 else hpg):
            fills, drains = [], []
            for g in range(N_GROUPS):
                h = g * hpg + k
                fills.append(lambda tg, g=g, h=h: a_prods[g].fill(
                    y, offset=y_off(h // 2, h % 2), sizes=[T, 2 * HD], strides=[2 * 128, 1], group=tg))
                for cc in range(GROUP_COLS):
                    col = g * GROUP_COLS + cc
                    # the column's 4 cores, hi then lo each (2048 bf16), at their slots of the pass's Q regions
                    # a drain a core: a fourth size would be read as the repeat count
                    for r in range(N_ROWS):
                        drains.append(lambda tg, col=col, h=h, cc=cc, r=r: o_conses[col].drain(
                            q, offset=h * qh_elems + ELEM + (cc * N_ROWS + r) * (TQ * HD), sizes=[2, TQ * HD],
                            strides=[ELEM * Q_ELEMS, 1], wait=True, group=tg))
            pass_group(fills, drains, first=k == 0)
        for j in range(PREP_NKV if PREP_NKV >= 0 else PREP_KV_PER_GROUP):
            fills, drains = [], []
            for g in range(N_GROUPS):
                kvh = g * PREP_KV_PER_GROUP + j
                fills.append(lambda tg, g=g, kvh=kvh: a_prods[g].fill(
                    y, offset=y_off(qw // 128 + kvh // 2, kvh % 2), sizes=[T, 2 * HD], strides=[2 * 128, 1], group=tg))
                fills.append(lambda tg, g=g, kvh=kvh: b_prods[g].fill(
                    y, offset=y_off((qw + kv_heads * HD) // 128 + kvh // 2, kvh % 2), sizes=[T, 2 * HD],
                    strides=[2 * 128, 1], group=tg))
                for cc in range(GROUP_COLS):
                    col = g * GROUP_COLS + cc
                    for r in range(N_ROWS):
                        drains.append(lambda tg, col=col, kvh=kvh, cc=cc, r=r: o_conses[col].drain(
                            kv, offset=kvh * HD + (cc * N_ROWS + r) * TQ * kv_ld, sizes=[2, TQ, HD],
                            strides=[kv_heads * HD, kv_ld, 1], wait=True, group=tg))
            pass_group(fills, drains)
        if prev:
            prev[-1][1].finish()
            prev.pop()[0].finish()

    a_prods = [A_l3l2[g].prod(tile=Tile(g * GROUP_COLS, 0)) for g in range(N_GROUPS)]
    b_prods = [B_l3l2[g].prod(tile=Tile(g * GROUP_COLS + 1, 0)) for g in range(N_GROUPS)]
    o_conses = [O_l2l3[c].cons(tile=Tile(c, 0)) for c in range(N_GROUPS * GROUP_COLS)]
    if MODE == "prep":
        rt = Runtime(prep_sequence, [Q_ty, KV_ty, y_ty, TB_ty, a_prods, b_prods, o_conses])
    else:
        rt = Runtime(sequence, [Q_ty, KV_ty, O_ty, TB_ty, a_prods, b_prods, o_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = attn_fa
SPECIALIZE = dict(
    L=int(os.environ.get("AF_L", 2048)),
    heads=int(os.environ.get("AF_HEADS", 40)),
    kv_heads=int(os.environ.get("AF_KV_HEADS", 8)),
    kv_ld=int(os.environ.get("AF_KV_LD", 1024)),
    max_rows=int(os.environ.get("AF_MAX_ROWS", 4096)),
)
