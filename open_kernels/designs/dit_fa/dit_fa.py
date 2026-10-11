r"""dit_fa: flash attention at head dim 128 for diffusion transformers (and their text encoder).

    O[t, h*D:(h+1)*D] = softmax(Q_h K_kv(h)^T / sqrt(D)) V_kv(h),   D = 128

Q, K, V and O are bf16, token-major ([tokens, ld] with head h at columns
col + h*D), which is what the QKV GEMM writes and the out-projection reads, so
no transpose is needed on either side. Q/K/V may be three views of ONE fused
buffer (pass it three times with different column offsets).

Topology (one xclbin; every shape is an instruction stream over it):

  * The 32 cores form two groups of 16 (columns 0-3 and 4-7). A group works on
    one head at a time; each of its cores owns TQ = 32 query rows, so a group
    covers QROWS_PASS = 512 query rows per pass and walks every key.
  * K and V reach a group through one memtile each, broadcast to all 16 cores
    (fifo A carries K, fifo B carries V). The Q tiles of a pass ride the same
    two broadcasts ahead of the keys -- 8 cores' worth on A, 8 on B -- and each
    core keeps only its own rows (whisper_fa's selective capture).
  * L3 -> memtile moves whole [64, 128] row blocks (2-D/3-D patterns: a shim BD's
    outermost dimension wraps silently above 64, and splitting a stream over
    several fills per pass hung); the memtile emits each block as its two
    [64, 64] head-dim chunks, block-transposed for the mmul.
  * No cascade: each core finishes its own online softmax, so there is no merge
    (upstream MLIR-AIR's head-spatial and temporal-causal designs drop it for
    the same reason).
  * The head dim is split into 64-wide chunks inside the core: Q is captured as
    two [32, 64] A tiles, K and V arrive as [64 keys, 64] pieces, and the
    output accumulator is the whole [32, 128] slab -- the acquired output
    element itself, so it costs no extra L1.

Why not whisper_fa's cascade topology: whisper_fa measured ~1.9 TFLOPS at its
own shape (20 x 1536^2 x 64, 2026-09-26, turbo), latency-bound on 60 serialized
6-chunk iterations with a full drain between each; it re-streams K and V once
per 256 query rows. Here a pass is 512 rows x all keys (72 chunks at 4608
tokens) and two passes are in flight.

L1 per core: A and B fifos 2 x 8 KB each, the output 8 KB, q_saved 8 KB,
G 4 KB, ~1 KB of state and the kernels' .bss, and the stack.

Runtime parameters (per core, RTP_*): heads per group, passes per head, key
chunks, valid key length (text-encoder padding), causal flag (text encoder),
the core's first query row within a pass. Shape constraints: tokens % 512,
heads % 2, keys % 64.

    DF_L=4608 DF_HEADS=24 python build_design.py designs/dit_fa/dit_fa.py <out>
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
KERNEL_SRC = str(HERE / "fa_dit.cc")

TQ, LKP, DC, DFULL = 32, 64, 64, 128
NDCH = DFULL // DC
N_GROUPS, GROUP_COLS, N_ROWS = 2, 4, 4
CORES = GROUP_COLS * N_ROWS            # per group
QROWS_PASS = CORES * TQ                # 512
Q_ON_A = CORES // 2                    # Q tiles on fifo A; the rest ride B
MAX_INFLIGHT = 2                       # passes queued ahead
STACK = 0x1000                         # softmax_step with FA_EXP_FIX spills a 3 KB frame

RTP_HEADS, RTP_PASSES, RTP_CHUNKS, RTP_VALID, RTP_CAUSAL, RTP_ROW0 = range(6)
RTP_LEN = 8
EXP_FIX = int(os.environ.get("DF_EXP_FIX", 0))   # fa_dit.cc FA_EXP_FIX (a build-wide choice)
TAU = int(os.environ.get("DF_TAU", 32))          # fa_dit.cc FA_TAU, the lazy-rescale threshold
_EXPT = os.environ.get("DF_EXPT", "").split(",")  # ablation builds only


def check_shape(L: int, heads: int, kv_heads: int, lk: int | None = None) -> str | None:
    lk = L if lk is None else lk
    if L % QROWS_PASS or lk % LKP or heads % N_GROUPS or heads % kv_heads:
        return (f"dit_fa needs tokens % {QROWS_PASS}, keys % {LKP}, heads % {N_GROUPS}, "
                f"heads % kv_heads (got L={L} lk={lk} heads={heads} kv_heads={kv_heads})")
    return None


def _block_dims(rows, cols, blk=8):
    """Row-major [rows, cols] -> the 8x8-block column-major order the mmul reads
    ([col block][row block][8][8]). whisper_fa's _block_dims (attempt 21's order)."""
    return [(cols // blk, blk), (rows // blk, blk * cols), (blk, cols), (blk, 1)]


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def dit_fa(
    Q: In, K: In, V: In, O: Out, *,
    L: CompileTime[int], heads: CompileTime[int], kv_heads: CompileTime[int] = 0,
    lk: CompileTime[int] = 0,
    q_ld: CompileTime[int] = 0, k_ld: CompileTime[int] = 0, v_ld: CompileTime[int] = 0,
    o_ld: CompileTime[int] = 0,
    q_col: CompileTime[int] = 0, k_col: CompileTime[int] = 0, v_col: CompileTime[int] = 0,
    o_col: CompileTime[int] = 0, o_interleave: CompileTime[int] = 0,
    causal: CompileTime[int] = 0, valid_len: CompileTime[int] = 0,
):
    kv_heads = kv_heads or heads
    lk = lk or L
    q_ld = q_ld or heads * DFULL
    o_ld = o_ld or heads * DFULL
    k_ld = k_ld or kv_heads * DFULL
    v_ld = v_ld or kv_heads * DFULL
    valid_len = valid_len or lk
    why = check_shape(L, heads, kv_heads, lk)
    assert why is None, why

    blk_ty = np.ndarray[(LKP, DFULL), np.dtype[bfloat16]]     # L2: 64 keys (or 2 cores' Q rows)
    elem_ty = np.ndarray[(LKP, DC), np.dtype[bfloat16]]       # L1: one head-dim chunk of it
    q_ty = np.ndarray[(NDCH * TQ * DC,), np.dtype[bfloat16]]
    g_ty = np.ndarray[(TQ * LKP,), np.dtype[bfloat16]]
    o_l1_ty = np.ndarray[(TQ * DFULL,), np.dtype[bfloat16]]
    o_l2_ty = np.ndarray[(N_ROWS * TQ * DFULL,), np.dtype[bfloat16]]
    up_ty = np.ndarray[(TQ,), np.dtype[bfloat16]]
    sp_ty = np.ndarray[(TQ * 8,), np.dtype[np.float32]]      # l, replicated per lane
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]

    flags = [f"-DFA_TQ={TQ}", f"-DFA_LKP={LKP}", f"-DFA_DC={DC}", f"-DFA_DFULL={DFULL}",
             f"-DFA_QROWS_PASS={QROWS_PASS}", f"-DFA_EXP_FIX={EXP_FIX}", f"-DFA_TAU={TAU}",
             "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
             "-Wno-deprecated-declarations"]
    # One ExternalFunction compiles the file; every call goes through a plain Kernel
    # sibling (whisper_fa's note: ExternalFunction's arg check rejects acquired bf16
    # elements).
    _ef = ExternalFunction("zero_g", source_file=KERNEL_SRC, arg_types=[g_ty],
                           compile_flags=flags)
    obj = _ef.object_file_name
    k_capture = Kernel("capture_q", obj, [elem_ty, q_ty, np.int32, np.int32])
    k_init = Kernel("pass_init", obj, [o_l1_ty, up_ty, sp_ty])
    k_zero_g = Kernel("zero_g", obj, [g_ty])
    k_qk = Kernel("qk", obj, [q_ty, elem_ty, g_ty, np.int32])
    k_pv = Kernel("pv", obj, [g_ty, elem_ty, o_l1_ty, np.int32])
    k_mask = Kernel("apply_mask", obj, [g_ty, rtp_ty, np.int32, np.int32])
    k_step = Kernel("softmax_step", obj, [g_ty, up_ty, sp_ty, o_l1_ty])
    k_fin = Kernel("finalize", obj, [sp_ty, o_l1_ty])
    kernels = [k_capture, k_init, k_zero_g, k_qk, k_pv, k_mask, k_step, k_fin]

    def core_tile(g, i):
        return Tile(g * GROUP_COLS + i // N_ROWS, 2 + i % N_ROWS)

    # A (Q tiles 0-7, then K) and B (Q tiles 8-15, then V): L3 -> memtile -> the
    # group's 16 cores, block-transposed on the way out of the memtile.
    A_l3l2, B_l3l2, A_l1, B_l1 = [], [], [], []
    for g in range(N_GROUPS):
        fa = ObjectFifo(blk_ty, name=f"A_L3L2_{g}", depth=2)
        fb = ObjectFifo(blk_ty, name=f"B_L3L2_{g}", depth=2)
        A_l3l2.append(fa)
        B_l3l2.append(fb)
        # _block_dims over the whole [64, 128] block is [d chunk][d block][key block]
        # [8][8]: the two chunks leave back to back, each already in mmul order.
        a2 = fa.cons().forward(tile=Tile(g * GROUP_COLS, 1), obj_type=elem_ty, depth=2,
                               name=f"A_L2L1_{g}", dims_to_stream=_block_dims(LKP, DFULL))
        b2 = fb.cons().forward(tile=Tile(g * GROUP_COLS + 1, 1), obj_type=elem_ty, depth=2,
                               name=f"B_L2L1_{g}", dims_to_stream=_block_dims(LKP, DFULL))
        A_l1.append([a2.cons(depth=2) for _ in range(CORES)])
        B_l1.append([b2.cons(depth=2) for _ in range(CORES)])

    # O: a column's 4 cores (32 rows each, block layout) join at its memtile, which
    # streams each part out row-major (a join's output dims apply per part), so the
    # column's 128 rows leave in order.
    out_dims = [(TQ, 8), (DFULL // 8, TQ * 8), (8, 1)]
    O_l2l3, O_l1 = [], {}
    for c in range(N_GROUPS * GROUP_COLS):
        f = ObjectFifo(o_l2_ty, name=f"O_L2L3_{c}", depth=2, dims_to_stream=out_dims)
        O_l2l3.append(f)
        parts = f.prod().join([r * TQ * DFULL for r in range(N_ROWS)], tile=Tile(c, 1),
                              obj_types=[o_l1_ty] * N_ROWS,
                              names=[f"O_L1L2_{c}_{r}" for r in range(N_ROWS)],
                              depths=[1] * N_ROWS)
        for r in range(N_ROWS):
            O_l1[(c, r)] = parts[r]

    rtp = {}
    barriers = {}

    def make_core_fn(i):
        def core_fn(a_in, b_in, o_out, kern, q_saved, g_buf, up, sp, my_rtp, barrier):
            (capture, init, zero_g, qk, pv, mask, step, fin) = kern
            barrier.wait_for_value(1)
            n_heads = my_rtp[RTP_HEADS]
            n_pass = my_rtp[RTP_PASSES]
            n_chunks = my_rtp[RTP_CHUNKS]
            for _ in range_(n_heads):
                for p in range_(n_pass):
                    o = o_out.acquire(1)
                    # Q: a block is 2 cores' rows; its chunks arrive d-chunk by d-chunk.
                    for pair in range(CORES // 2):
                        fifo = a_in if pair < Q_ON_A // 2 else b_in
                        for d in range(NDCH):
                            e = fifo.acquire(1)
                            if pair == i // 2:
                                capture(e, q_saved, d, i % 2)
                            fifo.release(1)
                    init(o, up, sp)
                    for c in range_(n_chunks):
                        zero_g(g_buf)
                        for d in range(NDCH):
                            k = a_in.acquire(1)
                            if "noqk" not in _EXPT:
                                qk(q_saved, k, g_buf, d)
                            a_in.release(1)
                        mask(g_buf, my_rtp, p, c)
                        if "nosm" not in _EXPT:
                            step(g_buf, up, sp, o)
                        for d in range(NDCH):
                            v = b_in.acquire(1)
                            if "nopv" not in _EXPT:
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
            rtp[(g, i)] = Buffer(rtp_ty, name=f"rtp_{g}_{i}",
                                 initial_value=np.zeros(RTP_LEN, dtype=np.int32),
                                 use_write_rtp=True)
            barriers[(g, i)] = WorkerRuntimeBarrier()
            args = [A_l1[g][i], B_l1[g][i], O_l1[(c, r)].prod(), kernels,
                    Buffer(q_ty, name=f"q_saved_{g}_{i}"), Buffer(g_ty, name=f"g_{g}_{i}"),
                    Buffer(up_ty, name=f"up_{g}_{i}"), Buffer(sp_ty, name=f"sp_{g}_{i}"),
                    rtp[(g, i)], barriers[(g, i)]]
            workers.append(Worker(make_core_fn(i), args, tile=t, stack_size=STACK))

    heads_local = heads // N_GROUPS
    n_pass = L // QROWS_PASS
    n_chunks = lk // LKP
    gqa = heads // kv_heads

    Q_ty = np.ndarray[(L * q_ld,), np.dtype[bfloat16]]
    K_ty = np.ndarray[(lk * k_ld,), np.dtype[bfloat16]]
    V_ty = np.ndarray[(lk * v_ld,), np.dtype[bfloat16]]
    O_ty = np.ndarray[(L * o_ld,), np.dtype[bfloat16]]

    def row_split(ld):
        """Rows per block of a K/V fill: a divisor of LKP whose stride fits a BD (2^21 bf16)."""
        rs = LKP
        while rs * ld > (1 << 21):
            rs //= 2
        return rs

    rs_k, rs_v = row_split(k_ld), row_split(v_ld)

    def sequence(q, k, v, o, a_prods, b_prods, o_conses):
        for (g, i), buf in rtp.items():
            buf[RTP_HEADS] = heads_local
            buf[RTP_PASSES] = n_pass
            buf[RTP_CHUNKS] = n_chunks
            buf[RTP_VALID] = valid_len
            buf[RTP_CAUSAL] = causal
            buf[RTP_ROW0] = i * TQ
        for b in barriers.values():
            b.set(1)

        inflight: list[TaskGroup] = []
        for hl in range(heads_local):
            for p in range(n_pass):
                tg = TaskGroup()
                for g in range(N_GROUPS):
                    h = g * heads_local + hl
                    kvh = h // gqa
                    qbase = q_col + h * DFULL + p * QROWS_PASS * q_ld
                    for prods, first in ((a_prods, 0), (b_prods, Q_ON_A)):
                        prods[g].fill(q, offset=qbase + first * TQ * q_ld,
                                      sizes=[Q_ON_A * TQ, DFULL], strides=[q_ld, 1], group=tg)
                    # the keys as [lk/rs, rs, 128] row blocks: a BD stride is at most
                    # 2^20 words, so rs rows of a wide fused buffer must stay under it
                    # (the fifo only sees rows in order; the grouping is free)
                    a_prods[g].fill(k, offset=k_col + kvh * DFULL,
                                    sizes=[lk // rs_k, rs_k, DFULL],
                                    strides=[rs_k * k_ld, k_ld, 1], group=tg)
                    b_prods[g].fill(v, offset=v_col + kvh * DFULL,
                                    sizes=[lk // rs_v, rs_v, DFULL],
                                    strides=[rs_v * v_ld, v_ld, 1], group=tg)
                    for cc in range(GROUP_COLS):
                        col = g * GROUP_COLS + cc
                        row0 = p * QROWS_PASS + cc * N_ROWS * TQ
                        if o_interleave:   # head h's dims d at o_col + 256h + 128(d//64) + d%64
                            o_conses[col].drain(o, offset=o_col + 2 * h * DFULL + row0 * o_ld,
                                                sizes=[N_ROWS * TQ, 2, DFULL // 2],
                                                strides=[o_ld, DFULL, 1], wait=True, group=tg)
                            continue
                        o_conses[col].drain(o, offset=o_col + h * DFULL + row0 * o_ld,
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
    rt = Runtime(sequence, [Q_ty, K_ty, V_ty, O_ty, a_prods, b_prods, o_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# build_design.py convention
DESIGN = dit_fa
SPECIALIZE = dict(
    L=int(os.environ.get("DF_L", 4608)),
    heads=int(os.environ.get("DF_HEADS", 24)),
    kv_heads=int(os.environ.get("DF_KV_HEADS", 0)),
    causal=int(os.environ.get("DF_CAUSAL", 0)),
    valid_len=int(os.environ.get("DF_VALID_LEN", 0)),
    # Layout (0 = the default token-major [L, heads*128] per tensor). Q/K/V in one fused
    # buffer: DF_QKV_LD = its row stride, DF_K_COL / DF_V_COL their column offsets.
    q_ld=int(os.environ.get("DF_QKV_LD", 0)),
    k_ld=int(os.environ.get("DF_QKV_LD", 0)),
    v_ld=int(os.environ.get("DF_QKV_LD", 0)),
    k_col=int(os.environ.get("DF_K_COL", 0)),
    v_col=int(os.environ.get("DF_V_COL", 0)),
    o_ld=int(os.environ.get("DF_O_LD", 0)),
    o_col=int(os.environ.get("DF_O_COL", 0)),
    # O written 64-of-128 (each head's two halves 128 columns apart): the layout
    # dit_gemm's gathered A reads, for a single block's out-projection input.
    o_interleave=int(os.environ.get("DF_O_INTERLEAVE", 0)),
)
