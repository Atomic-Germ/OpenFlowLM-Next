r"""fa_wide: causal flash attention at head dims 256 and 512 for a 256-row query block at any cache position.

    O[t, h*D:(h+1)*D] = softmax(scale * Q_h K_kv(h)^T) V_kv(h)        over the keys t may see

The query rows sit at cache position pos0 (rows pos0 .. pos0 + L - 1); K and V are the layer's decode cache rows
`[K | V]` from row 0, read in place; a sliding window W keeps key j for query row r only when r - W < j <= r. pos0, W and
each pass's first key chunk are runtime parameters, and every K / V fill's offset and length is rewritten per dispatch by
the engine (stream_patch.hpp fw_table / fw_apply), so one stream serves every layer of a head dim at any position.

Topology: four groups of eight cores (two columns each), a group on one head at a time. Each core owns TQ query rows
(32 at head dim 256, 16 at 512), so a pass is 256 or 128 rows; K reaches a group through one memtile (fifo A), V through
another (fifo B), broadcast to its 8 cores in 32-key chunks, the head dim split into 64-wide pieces inside the core. The
arithmetic is dit_fa's (fa_dit.cc), re-tiled so Q, O and the key fifos fit a core at these head dims (dit_fa needs ~75 KB
of 64 at 256). A query block of QB cores' rows rides the K / V fifos ahead of the keys, QB * TQ == 32.

    FW_D=256 FW_L=256 FW_HEADS=8 FW_KV_HEADS=4 FW_Q_LD=2048 FW_KV_LD=2048 FW_V_COL=1024 python build_design.py ... <out>
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
KERNEL_SRC = str(HERE / "fa_wide.cc")

DFULL = int(os.environ.get("FW_D", 256))
assert DFULL in (256, 512), f"fa_wide: FW_D={DFULL} (256 or 512)"
LKP, DC = 32, 64
TQ = 32 if DFULL == 256 else 16
QB = LKP // TQ
NDCH = DFULL // DC
N_GROUPS, GROUP_COLS, N_ROWS = 4, 2, 4
CORES = GROUP_COLS * N_ROWS            # per group
PASS = CORES * TQ                      # 256 or 128 query rows
Q_BLOCKS = CORES // QB                 # Q blocks a pass; the first half ride A, the rest B
MAX_INFLIGHT = 2
STACK = 0x1000
# 0 heads per group, 2 the core's first row in a pass, 3 pos0, 4 window; from 8 each pass's first key chunk, then each
# pass's chunk count (8 + n_pass + p)
RTP_HEADS, RTP_ROW0, RTP_POS0, RTP_WIN, RTP_C0 = 0, 2, 3, 4, 8
MAX_PASSES = 8
RTP_LEN = RTP_C0 + 2 * MAX_PASSES
assert RTP_LEN <= 32, "the rtp buffer is 32 ints"
# the stream is built at this cache position; the engine rewrites the K / V fills and the RTP words for the real one
POS0_PLACEHOLDER = 1000
SCALE = float(os.environ.get("FW_SCALE", 0.0))   # 0: 1 / sqrt(head dim); Gemma 4's scores are unscaled (1.0)
TAU = int(os.environ.get("FW_TAU", 32))
INC = HERE.parent.parent / "include"


def chunks_of_pass(pos0: int, p: int, window: int) -> tuple[int, int]:
    """(first key chunk, chunk count) pass p reads: from the window's oldest row (or 0) through the pass's last row."""
    lo = max(0, pos0 + p * PASS - window + 1) if window else 0
    c0 = lo // LKP
    c1 = -(-(pos0 + (p + 1) * PASS) // LKP)
    return c0, c1 - c0


def _block_dims(rows, cols, blk=8):
    """Row-major [rows, cols] -> the 8x8-block column-major order the mmul reads ([col block][row block][8][8])."""
    return [(cols // blk, blk), (rows // blk, blk * cols), (blk, cols), (blk, 1)]


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def fa_wide(
    Q: In, K: In, V: In, O: Out, *,
    L: CompileTime[int], heads: CompileTime[int], kv_heads: CompileTime[int],
    q_ld: CompileTime[int], kv_ld: CompileTime[int], v_col: CompileTime[int], o_ld: CompileTime[int],
    window: CompileTime[int] = 0,
):
    n_pass = L // PASS
    assert L % PASS == 0 and 0 < n_pass <= MAX_PASSES, f"fa_wide: L={L} is not 1..{MAX_PASSES} passes of {PASS}"
    assert heads % N_GROUPS == 0 and heads % kv_heads == 0, "fa_wide: heads % 4 and heads % kv_heads"
    gqa = heads // kv_heads
    heads_local = heads // N_GROUPS

    blk_ty = np.ndarray[(LKP, DFULL), np.dtype[bfloat16]]     # L2: 32 keys (or QB cores' Q rows)
    elem_ty = np.ndarray[(LKP, DC), np.dtype[bfloat16]]       # L1: one head-dim piece of it
    q_ty = np.ndarray[(NDCH * TQ * DC,), np.dtype[bfloat16]]
    g_ty = np.ndarray[(TQ * LKP,), np.dtype[bfloat16]]
    o_l1_ty = np.ndarray[(TQ * DFULL,), np.dtype[bfloat16]]
    o_l2_ty = np.ndarray[(N_ROWS * TQ * DFULL,), np.dtype[bfloat16]]
    up_ty = np.ndarray[(32,), np.dtype[bfloat16]]
    sp_ty = np.ndarray[(TQ * 8,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(32,), np.dtype[np.int32]]               # 128 bytes, so the buffers after it stay 64-byte aligned

    from aie.utils import config as _aie_config
    flags = [f"-I{INC}", f"-I{Path(_aie_config.cxx_header_path()) / 'aie_kernels'}", f"-DFW_TQ={TQ}", f"-DFW_LKP={LKP}",
             f"-DFW_DC={DC}", f"-DFW_DFULL={DFULL}", f"-DFW_QB={QB}", f"-DFW_PASS={PASS}", f"-DFW_SCALE={SCALE}f",
             f"-DFW_TAU={TAU}", "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16", "-Wno-deprecated-declarations"]
    _ef = ExternalFunction("fw_zero_g", source_file=KERNEL_SRC, arg_types=[g_ty], compile_flags=flags)
    obj = _ef.object_file_name
    kernels = [
        Kernel("fw_capture_q", obj, [elem_ty, q_ty, np.int32, np.int32]),
        Kernel("fw_pass_init", obj, [o_l1_ty, up_ty, sp_ty]),
        Kernel("fw_zero_g", obj, [g_ty]),
        Kernel("fw_qk", obj, [q_ty, elem_ty, g_ty, np.int32]),
        Kernel("fw_pv", obj, [g_ty, elem_ty, o_l1_ty, np.int32]),
        Kernel("fw_mask", obj, [g_ty, rtp_ty, np.int32, np.int32]),
        Kernel("fw_softmax_step", obj, [g_ty, up_ty, sp_ty, o_l1_ty]),
        Kernel("fw_finalize", obj, [sp_ty, o_l1_ty]),
    ]

    def core_tile(g, i):
        return Tile(g * GROUP_COLS + i // N_ROWS, 2 + i % N_ROWS)

    A_l3l2, B_l3l2, A_l1, B_l1 = [], [], [], []
    for g in range(N_GROUPS):
        fa = ObjectFifo(blk_ty, name=f"A_L3L2_{g}", depth=2)
        fb = ObjectFifo(blk_ty, name=f"B_L3L2_{g}", depth=2)
        A_l3l2.append(fa)
        B_l3l2.append(fb)
        a2 = fa.cons().forward(tile=Tile(g * GROUP_COLS, 1), obj_type=elem_ty, depth=2,
                               name=f"A_L2L1_{g}", dims_to_stream=_block_dims(LKP, DFULL))
        b2 = fb.cons().forward(tile=Tile(g * GROUP_COLS + 1, 1), obj_type=elem_ty, depth=2,
                               name=f"B_L2L1_{g}", dims_to_stream=_block_dims(LKP, DFULL))
        A_l1.append([a2.cons(depth=2) for _ in range(CORES)])
        B_l1.append([b2.cons(depth=2) for _ in range(CORES)])

    # O: a column's 4 cores join at its memtile, which streams each part out row-major
    out_dims = [(TQ, 8), (DFULL // 8, TQ * 8), (8, 1)]
    O_l2l3, O_l1 = [], {}
    for c in range(N_GROUPS * GROUP_COLS):
        f = ObjectFifo(o_l2_ty, name=f"O_L2L3_{c}", depth=2, dims_to_stream=out_dims)
        O_l2l3.append(f)
        parts = f.prod().join([r * TQ * DFULL for r in range(N_ROWS)], tile=Tile(c, 1),
                              obj_types=[o_l1_ty] * N_ROWS, names=[f"O_L1L2_{c}_{r}" for r in range(N_ROWS)],
                              depths=[1] * N_ROWS)
        for r in range(N_ROWS):
            O_l1[(c, r)] = parts[r]

    rtp, barriers = {}, {}

    def make_core_fn(i):
        def core_fn(a_in, b_in, o_out, kern, q_saved, g_buf, up, sp, my_rtp, barrier):
            capture, init, zero_g, qk, pv, mask, step, fin = kern
            barrier.wait_for_value(1)
            for _ in range_(my_rtp[RTP_HEADS]):
                for p in range(n_pass):           # unrolled: each pass's chunk range is its own RTP word
                    o = o_out.acquire(1)
                    for blk in range(Q_BLOCKS):
                        fifo = a_in if blk < Q_BLOCKS // 2 else b_in
                        for d in range(NDCH):
                            e = fifo.acquire(1)
                            if blk == i // QB:
                                capture(e, q_saved, d, i % QB)
                            fifo.release(1)
                    init(o, up, sp)
                    for c in range_(my_rtp[RTP_C0 + n_pass + p]):
                        zero_g(g_buf)
                        for d in range(NDCH):
                            k = a_in.acquire(1)
                            qk(q_saved, k, g_buf, d)
                            a_in.release(1)
                        mask(g_buf, my_rtp, p, c)
                        step(g_buf, up, sp, o)
                        for d in range(NDCH):
                            v = b_in.acquire(1)
                            pv(g_buf, v, o, d)
                            b_in.release(1)
                    fin(sp, o)
                    o_out.release(1)
            barrier.release_with_value(1)
        return core_fn

    workers = []
    for g in range(N_GROUPS):
        for i in range(CORES):
            t = core_tile(g, i)
            c, r = t.col, t.row - 2
            rtp[(g, i)] = Buffer(rtp_ty, name=f"rtp_{g}_{i}", initial_value=np.zeros(32, dtype=np.int32),
                                 use_write_rtp=True)
            barriers[(g, i)] = WorkerRuntimeBarrier()
            args = [A_l1[g][i], B_l1[g][i], O_l1[(c, r)].prod(), kernels,
                    Buffer(q_ty, name=f"q_saved_{g}_{i}"), Buffer(g_ty, name=f"g_{g}_{i}"),
                    Buffer(up_ty, name=f"up_{g}_{i}"), Buffer(sp_ty, name=f"sp_{g}_{i}"),
                    rtp[(g, i)], barriers[(g, i)]]
            workers.append(Worker(make_core_fn(i), args, tile=t, stack_size=STACK))

    kv_rows = POS0_PLACEHOLDER + L + LKP
    Q_ty = np.ndarray[(L * q_ld,), np.dtype[bfloat16]]
    K_ty = np.ndarray[(kv_rows * kv_ld,), np.dtype[bfloat16]]
    O_ty = np.ndarray[(L * o_ld,), np.dtype[bfloat16]]
    rs = LKP
    while rs * kv_ld > (1 << 21):
        rs //= 2
    plan = [chunks_of_pass(POS0_PLACEHOLDER, p, window) for p in range(n_pass)]

    def sequence(q, k, v, o, a_prods, b_prods, o_conses):
        for (g, i), buf in rtp.items():
            buf[RTP_HEADS] = heads_local
            buf[RTP_ROW0] = i * TQ
            buf[RTP_POS0] = POS0_PLACEHOLDER
            buf[RTP_WIN] = window
            for p, (c0, n) in enumerate(plan):
                buf[RTP_C0 + p] = c0
                buf[RTP_C0 + n_pass + p] = n
        for b in barriers.values():
            b.set(1)
        inflight: list[TaskGroup] = []
        for hl in range(heads_local):
            for p in range(n_pass):
                c0, n = plan[p]
                tg = TaskGroup()
                for g in range(N_GROUPS):
                    h = g * heads_local + hl
                    kvh = h // gqa
                    qbase = h * DFULL + p * PASS * q_ld
                    half = PASS // 2
                    a_prods[g].fill(q, offset=qbase, sizes=[half, DFULL], strides=[q_ld, 1], group=tg)
                    b_prods[g].fill(q, offset=qbase + half * q_ld, sizes=[half, DFULL], strides=[q_ld, 1], group=tg)
                    rows = n * LKP
                    a_prods[g].fill(k, offset=c0 * LKP * kv_ld + kvh * DFULL, sizes=[rows // rs, rs, DFULL],
                                    strides=[rs * kv_ld, kv_ld, 1], group=tg)
                    b_prods[g].fill(v, offset=c0 * LKP * kv_ld + v_col + kvh * DFULL, sizes=[rows // rs, rs, DFULL],
                                    strides=[rs * kv_ld, kv_ld, 1], group=tg)
                    for cc in range(GROUP_COLS):
                        row0 = p * PASS + cc * N_ROWS * TQ
                        o_conses[g * GROUP_COLS + cc].drain(o, offset=h * DFULL + row0 * o_ld,
                                                            sizes=[N_ROWS * TQ, DFULL], strides=[o_ld, 1],
                                                            wait=True, group=tg)
                inflight.append(tg)
                if len(inflight) == MAX_INFLIGHT:
                    inflight.pop(0).finish()
        for tg in inflight:
            tg.finish()

    a_prods = [A_l3l2[g].prod(tile=Tile(g * GROUP_COLS, 0)) for g in range(N_GROUPS)]
    b_prods = [B_l3l2[g].prod(tile=Tile(g * GROUP_COLS + 1, 0)) for g in range(N_GROUPS)]
    o_conses = [O_l2l3[c].cons(tile=Tile(c, 0)) for c in range(N_GROUPS * GROUP_COLS)]
    rt = Runtime(sequence, [Q_ty, K_ty, K_ty, O_ty, a_prods, b_prods, o_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = fa_wide
SPECIALIZE = dict(
    L=int(os.environ.get("FW_L", 256)),
    heads=int(os.environ.get("FW_HEADS", 8)),
    kv_heads=int(os.environ.get("FW_KV_HEADS", 4)),
    q_ld=int(os.environ.get("FW_Q_LD", 0)) or int(os.environ.get("FW_HEADS", 8)) * DFULL,
    kv_ld=int(os.environ.get("FW_KV_LD", 0)) or 2 * int(os.environ.get("FW_KV_HEADS", 4)) * DFULL,
    v_col=int(os.environ.get("FW_V_COL", 0)) or int(os.environ.get("FW_KV_HEADS", 4)) * DFULL,
    o_ld=int(os.environ.get("FW_O_LD", 0)) or int(os.environ.get("FW_HEADS", 8)) * DFULL,
    window=int(os.environ.get("FW_WINDOW", 0)),
)
