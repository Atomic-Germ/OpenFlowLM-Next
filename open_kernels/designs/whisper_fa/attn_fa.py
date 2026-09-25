# attn_fa.py -- Whisper-encoder flash attention, ported from AMD's MLIR-AIR
# attn_npu2.py/.cc (Xilinx/mlir-air, programming_examples/flash_attention/
# kernel_fusion_based at e91630a, MIT; the kernels modified for Whisper -- see attn_npu2.cc's own header and
# open_kernels/PROVENANCE.md) to IRON on this project's pinned mlir-aie
# (1.4.2.dev16, Peano 21), using aie.iron.dataflow.CascadeFlow.
#
# DESIGN/SPECIALIZE below follow this repo's build_design.py convention
# (designs/whisper_gemm/whisper_gemm.py is the other example): `flash_attn`
# is the @iron.jit design, specialised at the FIXED shape
# src/open_whisper/fa_attention.hpp documents (H=20, dk=dv=64, lq=lk=1536,
# valid_len=1500) unless overridden by the FA_* environment variables below
# (debugging/smoke-shape builds only -- the shipped kernel set always builds
# the production shape). `export_whisper_kernels.py --no-fa` skips it.
#
# Full derivation -- six rounds of hardware bisection, the three bugs found,
# and the byte-identical-vs-AIR hardware comparison at this exact production
# shape on real Whisper layers 0/15/31 -- is NpuEmbeddings (sibling repo)
# tasks/0181-fa-iron-port/TASK.md. This file is that task's round6/
# fa_iron_final.py, unmodified at the `flash_attn` body (only this header and
# the DESIGN/SPECIALIZE tail differ).
#
# ---------------------------------------------------------------------------
# Mapping from the AIR design (attn_npu2.py, non-causal / Whisper path):
#
#   physical grid: H (num_heads_per_unroll) head-groups run concurrently as
#   H*NQ physical COLUMNS (NQ=4 q-tile columns per head), NS=4 cascade
#   stages as physical ROWS 2..5 (the 4 compute rows npu2 has per column).
#   col = head_local * NQ + tx           (tx = q-tile index, 0..NQ-1)
#   row = 2 + ty                         (ty = cascade stage, 0..NS-1)
#
#   cascade direction: ty=NS-1 (bottom, row 5, "put_only") -> ... ->
#   ty=0 (top, row 2, "get_only" + normalise + drain), exactly the
#   row3->row2->row1->row0 shape of
#   programming_examples/basic/matrix_multiplication/cascade/cascade.py.
#   Each row role in that file (get_only / put_get / put_only) is one
#   Python closure instantiated per tile -- IRON specialises per-Worker at
#   compile time, so (unlike AIR, which emits ONE core body reused across
#   the whole herd and branches at runtime on `ty`/`tx` via ops.branch) no
#   runtime branch on stage/role is needed here at all.
#
#   Whisper's own config (non-causal, dv_chunks==1, dk_chunks==1):
#   causal=False so the whole `ctr` counter / apply_causal_mask machinery
#   in the AIR source is DEAD for this shape and is dropped entirely.
#   apply_length_mask (valid_len=1500) is the only mask, unconditional.
#
# ---------------------------------------------------------------------------
# THIRD ATTEMPT (this file): deviations #1 and #2 below are RETRACTED.
# Reading pass_057_after_cse.mlir (H=2 AND the production H=20 dump) answered
# the open question #2 named: AIR's mem tile column c is NOT "this q-tile
# column's own relay" -- it is cascade STAGE c's ONE shim feed, broadcast
# (one aie.dma_bd, several destination flows) to row 2+c across all NQ
# physical q-tile columns of a head. Q rides the SAME broadcast channel as K
# (attn_npu2.py's own `qkin[s].put(q...)` then `qkin[s].put(k...)`), sent NQ
# times (once per q-tile) with each column keeping only the one addressed to
# it -- exactly AIR's "broadcast full Q, keep the tx-th tile" that deviation
# #1 below used to avoid. The earlier "route Q direct + split K/V per column"
# shape is what overflowed the mem tile's port budget in the first place
# (compile_attempt9/10.log): 3 independent per-column ObjectFifos times NS
# stages is 3-in/9-out on ONE mem tile against a 3-in/3-out actual need once
# it is structured as broadcast-per-stage instead of split-per-column. See
# the QK/V ObjectFifo construction below for the corrected topology.
#
#  1. RETRACTED. Q is now broadcast exactly as AIR does (NQ times per
#     stage, each reaching all 4 physical columns; each core keeps only
#     qt == its own tx). IRON's per-Worker specialisation still means the
#     "keep" decision is a compile-time Python `if`, not AIR's runtime
#     `ops.branch(tx == qt)` -- that part of the original claim holds.
#  2. RETRACTED. K and V are now sent ONCE per (head, cascade-stage) and
#     broadcast to all NQ=4 columns of that head via ObjectFifo.forward()'s
#     multi-consumer semantics (multiple `.cons()` calls on one forwarded
#     fifo IS the broadcast primitive -- no split() involved). This is the
#     4x reduction in K/V shim traffic deviation #2 said AIR might have and
#     did not attempt; the ground truth shows AIR takes it.
#  3. RETRACTED (as a side effect of fixing #1/#2). Q and K are now
#     genuinely ONE ObjectFifo per (h, s) -- not two independent fifos
#     pinned to the same channel number, which is what this port did
#     before deviation #1 was removed. This is now the SAME shape as
#     attn_npu2.cc's `qkin`/`qk2l1`: Q and K interleave through one
#     physical channel because they are one logical stream, matching AIR
#     exactly. Per compute tile, 2 DMA in (QK broadcast channel 0, V
#     broadcast channel 1), 1 DMA out on row 2 only -- trap 3b's budget,
#     confirmed against pass_057_after_cse.mlir's own per-tile flow count.
# ---------------------------------------------------------------------------
#
# Names like "attempt 21", "fa_iron4" or "compile_attempt10.log" in the comments below
# refer to the debugging rounds archived in the sibling NpuEmbeddings repository,
# tasks/0181-fa-iron-port/ (round1/ .. round6/, each with its NOTES.txt).

import argparse
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import (
    Buffer,
    CascadeFlow,
    CompileTime,
    ExternalFunction,
    In,
    Kernel,
    ObjectFifo,
    str_to_dtype,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name

# Absolute path (not a bare filename): export_whisper_kernels.py's build_fa() invokes
# build_design.py with cwd=open_kernels/, not this directory, so a relative source_file
# would resolve against the wrong directory. attn_cascade_wrap.cc's own #include
# "attn_npu2.cc" / attn_npu2.cc's own #include "zero.cc" are quote-includes and are
# unaffected -- those always resolve relative to the INCLUDING file's own directory,
# which is this one either way.
KERNEL_SRC = str(Path(__file__).resolve().parent / "attn_cascade_wrap.cc")
# attn_cascade_wrap.cc #includes attn_npu2.cc VERBATIM (byte-identical copy in
# this directory) and adds only cascade_get3/cascade_put3 -- the two-hop
# get_scd/put_mcd wrappers this port needs because IRON's CascadeFlow only
# declares the topology edge; the transfer itself must come from the linked
# kernel (unlike AIR, which emits it as inline aie.put_cascade/get_cascade
# MLIR ops around calls to attn_npu2.cc's otherwise-unmodified kernels --
# confirmed by reading pass_057_after_cse.mlir). Verified standalone this
# session: the same-width v32bfloat16<->v16int32 / v16float<->v16int32
# bitcast compiles clean under our pinned Peano 21/aie2p and emits real
# `vmov mcd,x0` / `vmov x0,scd` cascade-port instructions
# (cascade_probe.cc/.o, llvm-objdump -d), not a numeric conversion.
BF16 = bfloat16  # raw ml_dtypes class -- used for the L3 Runtime tensor
# types (Kernel._validate_arg is not on that path).
F32 = np.float32


def _block_dims(rows, cols, blk=8):
    """DMA access-pattern transform for the M=8 MMUL block-transposed
    layout every 2-D bf16 tile in attn_npu2.cc lives in (see max_g_bf16's
    comment in the .cc: "column-major 8x8 tiled"). Ported from
    attn_npu2.py's `.reshape(rows//M, M, cols//M, M).transpose(2, 0, 1, 3)`
    into the (size, stride) pair-list StreamDims form
    matrix_multiplication/cascade/cascade.py's `a_dims`/`b_dims` already
    use for the identical MMUL block shape -- same formula, m=rows, k=cols,
    r=s=blk.
    """
    assert rows % blk == 0 and cols % blk == 0
    # ATTEMPT 21 FIX: the first two entries were swapped relative to this
    # docstring's own cited derivation. Ground-truthed against
    # fa_mlir_aie/debug_run/air_project/debug_ir/pass_057_after_cse.mlir
    # (H=2 oracle): the mem-tile MM2S BD that feeds a compute tile's K/Q/V
    # fill reads `sizes=[8,64,8] strides=[8,64,1]` from a mem-tile buffer
    # that is otherwise a PLAIN row-major copy of the L3 tile (its own
    # S2MM-from-shim BD has no sizes/strides at all -- lines 1963-2008,
    # buf239/buf231, K/Q's and V's channels, identically). Decoded:
    # address(a,b,c) = a*8 + b*64 + c, sizes [8,64,8] in stream order
    # (a outer, b mid, c inner), with a=col_blk, b=row (undecomposed --
    # blk*(rows//blk)=rows merges row_blk+row_in into one dim here),
    # c=col_in. That is exactly this docstring's own cited
    # `.reshape(rows//M,M,cols//M,M).transpose(2,0,1,3)` flattened, which
    # puts cb (cols//blk, stride=blk) OUTERMOST, then rb (rows//blk,
    # stride=blk*cols) -- i.e. dim0/dim1 as coded were the REVERSE of what
    # the cited transpose produces. The (size,stride) pairs themselves are
    # each individually correct; only their ORDER (dim0 vs dim1) was wrong,
    # which is exactly why this passed every OOB/byte-count check (same
    # address SET, same bijection) and only showed up as scrambled/
    # overflowing arithmetic downstream (matmul_a_b_bf16, matmul_g_b_bf16),
    # never as a compile or placement error. fa_iron4/NOTES.txt attempt 21.
    return [
        (cols // blk, blk),
        (rows // blk, blk * cols),
        (blk, cols),
        (blk, 1),
    ]


def _out_block_dims(rows, cols, blk=8):
    """DMA access-pattern transform for draining an M=8 block-tiled L1
    result tile back to a linear L3 layout.

    ATTEMPT 22 FIX (fa_iron5, round 2): this formula was NEVER updated when
    attempt 21 fixed _block_dims's axis order, and it turns out it needed the
    SAME fix. Ground-truthed two ways this round:

    1. A hardware LAYOUT PROBE (fa_iron5/01_probe_k_layout.py,
       03_probe_v_layout.py) confirmed attempt-21-fixed _block_dims's fill
       order is EXACTLY what the DMA engines produce for K and V (byte-exact
       on real hardware against a pure-numpy simulation, 8/8 head/tx
       combinations each).
    2. Re-running fa_iron4/attempt16.xclbin (Q copy-through, pre-attempt-21
       _block_dims) with a RAMP instead of random Q data confirmed it is a
       TRUE identity round-trip on hardware, not a coincidence of random
       data. But attempt16.xclbin was built with the OLD (pre-attempt-21)
       _block_dims, which puts row_block outermost -- the exact opposite
       axis order from what attn_npu2.cc's matmul_vectorized_2x2_mmul
       ACTUALLY writes to its C output (re-derived directly from the .cc
       pointer arithmetic: address(row_block=z, col_block=j) =
       size_C*(z + j*rowA), i.e. col_block is OUTER (stride rowA*size_C),
       row_block is INNER (stride size_C) -- confirmed by a from-scratch
       numpy simulation of that exact loop, matching this file's kernel
       verbatim: see fa_iron5/06_attempt22_fix_out_block_dims.py's own
       verification script).

    So the OLD _out_block_dims correctly inverts the OLD (pre-attempt-21)
    _block_dims -- which is why attempt 16's Q pass-through test (built with
    the OLD _block_dims) showed byte-identical results and "exonerated" this
    drain leg. But that never tested what THIS file needs: the matmul
    kernel's OWN C-write convention, which matches attempt-21-fixed
    _block_dims's col_block-outer order, not the old row_block-outer order.
    _out_block_dims was simply never touched to match.

    Verified (fa_iron5, pure numpy, no hardware needed for this part):
      - the OLD formula composed with attempt-21-fixed _block_dims does NOT
        round-trip to identity (3584/4096 elements differ, arange test).
      - this NEW formula, applied to a buffer laid out exactly as the
        matmul's own store loop writes it (independently reconstructed from
        attn_npu2.cc lines ~121-134), DOES correctly recover natural
        row-major order -- for both the square (64,64) smoke shape and a
        non-square (32,64) sanity check.
      - this NEW formula round-trips to identity when composed with
        attempt-21-fixed _block_dims (the Q/K/V fill this design now uses).

    New axis order: [row_block, row_in, col_block, col_in] (unchanged from
    before), but now with STRIDES matching the matmul's col_block-outer
    physical convention: row_block stride=blk*blk (not blk*cols), col_block
    stride=rows*blk (not blk*blk).
    """
    assert rows % blk == 0 and cols % blk == 0
    return [
        (rows // blk, blk * blk),
        (blk, blk),
        (cols // blk, rows * blk),
        (blk, 1),
    ]


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def flash_attn(
    Q: In,
    K: In,
    V: In,
    GP: Out,
    *,
    lq: CompileTime[int],
    lk: CompileTime[int],
    lqp: CompileTime[int],
    lkp: CompileTime[int],
    dk: CompileTime[int],
    dv: CompileTime[int],
    num_heads: CompileTime[int],
    num_heads_per_unroll: CompileTime[int],
    num_cascade_stages: CompileTime[int],
    valid_len: CompileTime[int],
):
    NQ = 4  # q-tile columns per head-group -- fixed, matches attn_npu2.py main()
    NS = num_cascade_stages
    H = num_heads_per_unroll
    assert H * NQ <= 8, "H * NQ must fit in npu2's 8 columns"
    assert NS <= 4, "NS must fit npu2's 4 compute rows per column"
    assert num_heads % H == 0
    num_head_groups = num_heads // H

    assert lq % lqp == 0
    assert lk % lkp == 0
    assert lk % (lkp * NS) == 0
    tile_size_q = lqp // NQ  # == attn_npu2.cc's `lqp` macro
    assert dk == lkp, "this port assumes dk_chunks == 1 (dk == lkp)"
    assert dv == lkp, "this port assumes dv_chunks == 1 (dv == lkp)"
    num_lq_iters = lq // lqp
    num_chunks = lk // lkp
    chunks_per_stage = num_chunks // NS
    lk_per_stage = lkp * chunks_per_stage
    g_flat = tile_size_q * lkp

    # L1 budget check (CLAUDE.md trap 3), worst case (row 2, s=0: has BOTH
    # the GP_l1l2 output leg and gp_c merge scratch). Fixed from an earlier
    # version of this port that summed ELEMENT counts and called them bytes
    # (off by the bf16 dtype's 2 B/elem) and did not count the QK_l1/V_l1/
    # GP_l1l2 fifo buffers at all -- compile_attempt12.log's real allocation
    # dump (~84 KB at one core) is what caught both mistakes; the fifo legs
    # are now depth=1 (see the QK_l1/GP_l1l2 comments) specifically so this
    # sum fits. AIR's own analytic estimate for this exact shape, "~49-50
    # KiB of 63" (task report, Part 5), lines up with a bottom/mid row
    # (no GP_l1l2 leg) at these single-buffered sizes.
    BF = 2  # bytes/elem, bf16
    per_core_bytes = BF * (
        tile_size_q * lkp  # QK_l1 (Q-broadcast/K, depth=1)
        + lkp * dv  # V_l1 (depth=1)
        + tile_size_q * dv  # GP_l1l2 (depth=1; only on row 2, counted anyway)
        + tile_size_q * lkp  # q_saved
        + tile_size_q * lkp  # g
        + tile_size_q * dv  # gp
        + tile_size_q * dv  # gp_c (merge scratch; present on ty != NS-1)
    )
    assert per_core_bytes < 64512, f"L1 budget exceeded: {per_core_bytes} B"

    # ------------------------------------------------------------- kernels
    # Compile attn_npu2.cc ONCE (bound to one symbol); every other symbol
    # is a sibling Kernel against the same .o, matching
    # kernels.linalg.cascade_mm's own pattern (one ExternalFunction + many
    # Kernel siblings sharing `object_file_name`).
    compile_flags = [
        f"-Dlqp={tile_size_q}",
        f"-Dlkp={lkp}",
        f"-Ddk={dk}",
        f"-Ddk_full={dk}",
        f"-Ddv={dv}",
        f"-Ddv_full={dv}",
        f"-Dvalid_len={valid_len}",
        "-DFP32_STATE=1",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        "-DROUND_CONV_EVEN",
    ]

    def bf(n):
        return np.ndarray[(n,), np.dtype[BF16]]

    def f32(n):
        return np.ndarray[(n,), np.dtype[F32]]

    # 2-D, matching what ObjectFifo.forward()/split() hand back from
    # acquire() (ObjectFifo element shape, not the flat pointer
    # attn_npu2.cc's C signature actually takes -- shape is IRON-side
    # bookkeeping only, trap 8, but Kernel._validate_arg enforces it, so it
    # must agree with what acquire() returns rather than with the .cc
    # signature).
    qk_ty = np.ndarray[(tile_size_q, lkp), np.dtype[BF16]]
    v_ty = np.ndarray[(lkp, dv), np.dtype[BF16]]
    g_ty = np.ndarray[(tile_size_q, lkp), np.dtype[BF16]]
    gp_ty = np.ndarray[(tile_size_q, dv), np.dtype[BF16]]
    state_bf_ty = np.ndarray[(tile_size_q,), np.dtype[BF16]]
    state_f32_ty = np.ndarray[(tile_size_q,), np.dtype[F32]]

    # Bind ONE symbol via ExternalFunction to trigger compilation of the
    # whole translation unit; every actual call site -- INCLUDING this same
    # symbol -- goes through a plain Kernel sibling instead of the EF
    # itself. Reason (found empirically, compile_attempt1-4.log):
    # ExternalFunction.__call__ overrides BaseKernel.__call__ with a strict
    # `_validate_arg` that compares an ACQUIRED ObjectFifo element's
    # `.dtype` (always IRON's canonical short string, e.g. "bf16", per
    # `np_ndarray_type_get_dtype` / `ObjectFifoHandle.dtype`) against
    # whatever raw Python object `np.dtype[...]` was given for arg_types
    # (e.g. the class `ml_dtypes.bfloat16`, str()'ing as
    # "<class 'ml_dtypes.bfloat16'>", or `np.dtype(bfloat16)`, str()'ing as
    # "bfloat16") -- neither ever equals the string "bf16", so every call
    # passing an acquired bf16 element into the EF itself raises, no matter
    # which BF16 alias is used. Plain `Kernel` does not override
    # `__call__` (no `_validate_arg` at all), which is why
    # `kernels.linalg.cascade_mm`'s siblings (`.zero`/`.put_only`/etc.) work
    # -- cascade.py's own top-level int-dtype EF call happens to dodge this
    # because its dtypes (`i16`/`i32`) round-trip through `str()` cleanly;
    # bf16 does not. Filed for the report, not fixed upstream here.
    _copy_tile_ef = ExternalFunction(
        "copy_tile",
        source_file=KERNEL_SRC,
        arg_types=[qk_ty, qk_ty],
        compile_flags=compile_flags,
    )
    obj = _copy_tile_ef.object_file_name
    copy_tile = Kernel("copy_tile", obj, [qk_ty, qk_ty])

    matmul_a_b = Kernel("matmul_a_b_bf16", obj, [qk_ty, qk_ty, g_ty])
    matmul_g_b = Kernel("matmul_g_b_bf16", obj, [g_ty, v_ty, gp_ty])
    zero_fill_gp = Kernel("zero_fill_gp_bf16", obj, [gp_ty])
    zero_fill_g = Kernel("zero_fill_g_bf16", obj, [g_ty])
    neg_inf_fill_up = Kernel("neg_inf_fill_up_bf16", obj, [state_bf_ty])
    maximum_up_u = Kernel("maximum_up_u_bf16", obj, [state_bf_ty, state_bf_ty])
    exp_up_minus_u = Kernel(
        "exp_up_minus_u", obj, [state_bf_ty, state_bf_ty, state_bf_ty]
    )
    mul_r_gp = Kernel("mul_r_gp", obj, [state_bf_ty, gp_ty])
    add_gp_g = Kernel("add_gp_g", obj, [gp_ty, gp_ty])
    vector_copy = Kernel(
        "vector_copy_32elems", obj, [np.int32, state_bf_ty, state_bf_ty]
    )
    apply_length_mask = Kernel("apply_length_mask", obj, [g_ty, np.int32])
    fused_softmax = Kernel(
        "fused_softmax", obj, [g_ty, state_bf_ty, state_bf_ty, state_bf_ty]
    )

    zero_fill_sp_f32 = Kernel("zero_fill_sp_f32", obj, [state_f32_ty])
    widen_bf16_to_f32 = Kernel("widen_bf16_to_f32", obj, [state_bf_ty, state_f32_ty])
    vector_copy_f32 = Kernel(
        "vector_copy_32elems_f32", obj, [np.int32, state_f32_ty, state_f32_ty]
    )
    accum_sp_r_s_f32 = Kernel(
        "accum_sp_r_s_f32", obj, [state_f32_ty, state_bf_ty, state_f32_ty]
    )
    div_gp_sp_f32 = Kernel("div_gp_sp_f32", obj, [state_f32_ty, gp_ty])

    # Cascade-port transfer -- see attn_cascade_wrap.cc / the KERNEL_SRC
    # comment above. Order matches attn_npu2.py's merge(): gp/gp_c (bf16),
    # up/up_c (bf16), sp/sp_c (f32 under FP32_STATE).
    cascade_get3 = Kernel(
        "cascade_get3", obj, [gp_ty, state_bf_ty, state_f32_ty]
    )
    cascade_put3 = Kernel(
        "cascade_put3", obj, [gp_ty, state_bf_ty, state_f32_ty]
    )

    # Program.resolve_program() only resolves a Kernel/ExternalFunction if it
    # appears (possibly nested) in SOME Worker's fn_args -- a plain Python
    # closure reference is invisible to that walk (confirmed empirically,
    # compile_attempt5.log: "Kernel must be resolved before it can be
    # called" on copy_tile, called only via closure). fn_args nesting is
    # preserved for registration purposes while core_fn still receives it as
    # one structure (Worker.__init__'s own comment), so bundling every
    # kernel into one list and appending it once per Worker registers all of
    # them without changing any call site inside local_compute/merge/
    # make_*_fn, which keep using the closure-captured names directly.
    ALL_KERNELS = [
        copy_tile, matmul_a_b, matmul_g_b, zero_fill_gp, zero_fill_g,
        neg_inf_fill_up, maximum_up_u, exp_up_minus_u, mul_r_gp, add_gp_g,
        vector_copy, apply_length_mask, fused_softmax, zero_fill_sp_f32,
        widen_bf16_to_f32, vector_copy_f32, accum_sp_r_s_f32, div_gp_sp_f32,
        cascade_get3, cascade_put3,
    ]

    # -------------------------------------------------------------- tensors
    # Flat L3 tensors, matching attn_npu2.py's own q_flat/k_flat/v_flat/gp_flat
    # (num_kv_heads == num_heads: Whisper's encoder attention has no GQA).
    Q_ty = np.ndarray[(num_heads * lq * dk,), np.dtype[BF16]]
    K_ty = np.ndarray[(num_heads * lk * dk,), np.dtype[BF16]]
    V_ty = np.ndarray[(num_heads * lk * dv,), np.dtype[BF16]]
    GP_ty = np.ndarray[(num_heads * lq * dv,), np.dtype[BF16]]

    n_cols = H * NQ

    # ------------------------------------------------ QK (broadcast per stage)
    # Ground truth (pass_057_after_cse.mlir, H=2 AND the production H=20 dump)
    # is NOT "one mem tile per physical column, split into NS stage-fifos" --
    # that was this port's first attempt and it blows the mem tile's 3-in/3-out
    # budget the moment NS>1 (compile_attempt9/10.log: "tile (0,1) requires 1
    # input/4 output ... only 4 input/1 output available"). AIR's actual
    # topology, read straight off the flow list:
    #
    #   aie.flow(%shim_noc_tile_S_0, DMA:0, %mem_tile_S_1, DMA:0)      [fill]
    #   aie.flow(%mem_tile_S_1, DMA:0, %tile_0_R, DMA:0)               \
    #   aie.flow(%mem_tile_S_1, DMA:0, %tile_1_R, DMA:0)                broadcast,
    #   aie.flow(%mem_tile_S_1, DMA:0, %tile_2_R, DMA:0)                same BD,
    #   aie.flow(%mem_tile_S_1, DMA:0, %tile_3_R, DMA:0)               /
    #
    # i.e. mem tile COLUMN S (== cascade stage s, within a head's own H*NQ
    # column block) is the ONE shim feed for that stage, and it BROADCASTS
    # (one aie.dma_bd, four destination flows -- not four separate transfers)
    # to row R = 2+s across all NQ=4 physical q-tile columns of that head.
    # That is 1 mem tile in-port / 1 out-port for this leg (the "1 input/4
    # output" in the earlier error IS the broadcast -- 4 flow edges sharing
    # one DMA channel), not 1-in/NQ-out physical ports; the ORIGINAL bug was
    # splitting into NS *separate* channels on ONE mem tile column instead of
    # using NS separate mem tile COLUMNS with ONE broadcast channel each.
    #
    # Q and K share this same channel (attn_npu2.py: `qkin[s].put(q...)` then
    # `qkin[s].put(k...)`, per stage), and -- critically -- Q is *itself*
    # broadcast NQ times, once per q-tile, with each physical column keeping
    # only the broadcast addressed to it ("Q selective capture: receive all
    # NQ * dk_chunks sends, keep the one this column owns", attn_npu2.py herd
    # body). This is what lets ONE broadcast fifo carry all of Q AND K for a
    # stage without a separate direct-routed Q channel (this port's original
    # deviation #1, now removed to match ground truth exactly). Per (h, s)
    # pair (h = local head 0..H-1, s = cascade stage 0..NS-1): one L3->L2
    # ObjectFifo, forwarded (not split) through mem tile Tile(h*NQ+s, 1), then
    # consumed NQ times (once per physical q-tile column) via forward()'s
    # returned ObjectFifo's own multi-consumer broadcast semantics
    # (ObjectFifo.cons() "may have multiple consumers" -- each gets the same
    # bytes, which IS the hardware broadcast).
    QK_l3l2 = [
        [
            ObjectFifo(qk_ty, name=f"QK_L3L2_h{h}_s{s}", depth=2)
            for s in range(NS)
        ]
        for h in range(H)
    ]
    QK_fwd = [
        [
            QK_l3l2[h][s]
            .cons()
            .forward(
                tile=Tile(h * NQ + s, 1),
                obj_type=qk_ty,
                name=f"QK_L2L1_h{h}_s{s}",
                dims_to_stream=_block_dims(lkp, dk),
            )
            for s in range(NS)
        ]
        for h in range(H)
    ]
    # QK_l1[h][s][tx]: the physical-column-tx consumer of stage s's broadcast,
    # at row 2+s, column h*NQ+tx -- NQ handles per (h, s), all reading the
    # SAME bytes (Q selective-capture + shared K), channel 0 (trap 3b: 2 in
    # per compute core, V takes the other).
    # depth=1 (not the ObjectFifo default of 2): compile_attempt12.log hit
    # "allocated buffers exceeded available memory" at ~84 KB/core once
    # placement got far enough to allocate real buffers -- double-buffering
    # QK_l1 + V_l1 + GP_l1l2 on top of the six ~8 KiB persistent buffers the
    # trap-3 budget above already counts (q_saved/g/gp/gp_c) blows well past
    # 63 KiB. AIR's own L1 allocs (`qk`/`v_l1` in attn_npu2.py) are single,
    # not ObjectFifo-managed double buffers, and "L1 estimated ~49-50 KiB of
    # 63" (task report, Part 5) only reconciles against the real per-buffer
    # byte counts (8192 B here, not the element counts the trap-3 assert
    # above uses) if these three legs are single-buffered too -- matching
    # AIR exactly costs pipelining overlap on this leg, not correctness.
    QK_l1 = [
        [
            [
                QK_fwd[h][s].cons(tile=Tile(h * NQ + tx, 2 + s), channel=0, depth=1)
                for tx in range(NQ)
            ]
            for s in range(NS)
        ]
        for h in range(H)
    ]

    # -------------------------------------------------- V (broadcast, shared)
    # Same broadcast shape as QK, channel 1: ground truth's V2L1 channel is
    # ALSO `broadcast_shape=[H, 1, NQ]` (attn_npu2.py) -- V does not depend on
    # the q-tile, so every physical column of a head genuinely needs the SAME
    # K/V range, and this removes deviation #2's 4x-redundant V shim traffic
    # (each of the 4 q-tile columns previously pulled its own private copy).
    V_l3l2 = [
        [
            ObjectFifo(v_ty, name=f"V_L3L2_h{h}_s{s}", depth=2)
            for s in range(NS)
        ]
        for h in range(H)
    ]
    V_fwd = [
        [
            V_l3l2[h][s]
            .cons()
            .forward(
                tile=Tile(h * NQ + s, 1),
                obj_type=v_ty,
                name=f"V_L2L1_h{h}_s{s}",
                dims_to_stream=_block_dims(lkp, dv),
            )
            for s in range(NS)
        ]
        for h in range(H)
    ]
    V_l1 = [
        [
            [
                V_fwd[h][s].cons(tile=Tile(h * NQ + tx, 2 + s), channel=1, depth=1)
                for tx in range(NQ)
            ]
            for s in range(NS)
        ]
        for h in range(H)
    ]

    # -------------------------------------------------------------- GP out
    # Row 0 (top, ty==0) only: L1 -> mem -> L3.
    # depth=1 on the ObjectFifo itself: its default depth is what prod()
    # (the compute-tile L1 buffer, budget-critical) inherits; the mem-tile
    # relay leg is overridden back to depth=2 below since it costs mem-tile
    # memory, not compute-tile L1 (see the QK_l1 comment above).
    GP_l1l2 = [
        ObjectFifo(
            np.ndarray[(tile_size_q, dv), np.dtype[BF16]],
            name=f"GP_L1L2_{c}",
            depth=1,
        )
        for c in range(n_cols)
    ]
    GP_l2l3 = [
        GP_l1l2[c]
        .cons()
        .forward(
            tile=Tile(c, 1),
            depth=2,
            obj_type=np.ndarray[(tile_size_q * dv,), np.dtype[BF16]],
            name=f"GP_L2L3_{c}",
            dims_to_stream=_out_block_dims(tile_size_q, dv),
        )
        for c in range(n_cols)
    ]

    # ------------------------------------------------------------ workers
    #
    # Per-stage local compute: capture Q once, then loop chunks_per_stage
    # times over this stage's K/V range accumulating the online-softmax
    # state (gp, up, sp) exactly as attn_npu2.py's h.body does (the causal
    # branch, apply_mask, and window/dv_chunks machinery are all dropped --
    # dead code for Whisper's non-causal, dv_chunks==1 shape). Returns the
    # three state buffers so the caller (bot/mid/top) decides what happens
    # at the cascade boundary -- merge() in attn_npu2.py is exactly that
    # split between "compute this stage's local partial" and "combine with
    # the neighbour's partial", which maps directly onto cascade.py's
    # get_only/put_only/put_get three-way role split.
    # Per-core Buffers cannot be created INSIDE local_compute/merge (as an
    # earlier version of this file did): a Buffer must reach some Worker's
    # fn_args (flattened) for Program.resolve_program() to place/resolve it
    # -- same discovery mechanism as ALL_KERNELS, same failure mode when
    # skipped (NotResolvedError: "class not resolved", compile_attempt6.log,
    # on a Buffer created inside a closure and never fn_args'd). Buffers
    # must therefore be created at Worker-construction time (outer scope,
    # before core_fn ever runs) and threaded in as parameters -- exactly
    # cascade.py's own pattern (`c_buf = Buffer(...); Worker(_row_mid_fn,
    # [..., c_buf, ...])`), just with more state per tile here. `lb`/`mb`
    # below are plain lists in a FIXED order, unpacked by position; a list
    # (not a dict) because fn_args nesting only flattens
    # lists/tuples for registration (Worker.__init__'s own comment).
    def make_local_bufs(c, s):
        return [
            Buffer(qk_ty, name=f"q_saved_{c}_{s}"),
            Buffer(g_ty, name=f"g_{c}_{s}"),
            Buffer(gp_ty, name=f"gp_{c}_{s}"),
            Buffer(state_bf_ty, name=f"up_{c}_{s}"),
            Buffer(state_f32_ty, name=f"sp_{c}_{s}"),
            Buffer(state_bf_ty, name=f"s_tmp_{c}_{s}"),
            Buffer(state_bf_ty, name=f"r_tmp_{c}_{s}"),
            Buffer(state_f32_ty, name=f"s_tmp_f32_{c}_{s}"),
        ]

    def make_merge_bufs(c, s):
        return [
            Buffer(gp_ty, name=f"gp_c_{c}_{s}"),
            Buffer(state_bf_ty, name=f"up_c_{c}_{s}"),
            Buffer(state_f32_ty, name=f"sp_c_{c}_{s}"),
            Buffer(state_bf_ty, name=f"up_s_{c}_{s}"),
            Buffer(state_bf_ty, name=f"rc_{c}_{s}"),
            Buffer(state_bf_ty, name=f"rl_{c}_{s}"),
            Buffer(state_f32_ty, name=f"st_{c}_{s}"),
        ]

    def local_compute_one(own_tx, s, qk_in, v_in, lb):
        """ROUND 5 FIX (attempt 43): this used to be `local_compute`, with the
        `for _ in range_(num_head_groups): for _ in range_(num_lq_iters):`
        loop INSIDE it, and merge()/cascade_put3/the top drain all OUTSIDE --
        i.e. called ONCE per Worker invocation, after this whole nested loop
        finished. That is a genuine host/device loop-structure mismatch:
        `sequence()` below issues ONE TaskGroup + ONE `wait=True` GP drain
        PER lx (num_lq_iters of them, each to a DIFFERENT output offset --
        `out_off` includes `lx * (lqp*dv)`, so each lx is a DISTINCT slice
        of the output, not an accumulation), but the device only executed a
        single acquire/merge/drain sequence for ALL lx values combined, so
        gp/up/sp held only the LAST lx's partial (zero_fill_gp/sp/up reset
        them at the top of every lx, discarding all prior lx results before
        they were ever merged or drained) and the top worker issued exactly
        ONE gp_out.acquire/release, not `num_lq_iters` of them.
        At num_lq_iters=1 the mismatch is invisible (1 fill round == 1
        drain). At num_lq_iters>=2 -- independent of cascade depth NS, and
        of column count -- the runtime's tg.finish() after lx=0 blocks
        forever on a GP drain the device will not produce until it has also
        consumed lx=1's K/V fills, which the runtime has not issued yet
        (blocked in the same wait): ERT_CMD_STATE_TIMEOUT. Round 5's
        bisection (attempts 38-42) isolated this exactly: NS=3 clean,
        NS=4/H=1(4 cols) hangs, NS=4/chunks_per_stage=1(iters=1) clean,
        NS=4/chunks_per_stage=2/iters=1 clean, NS=2/chunks_per_stage=2/
        iters=2 ALSO hangs -- num_lq_iters>=2 is the sole discriminator, not
        NS, not chunks_per_stage, not column count.

        FIX: this function now does exactly ONE (Q-capture, zero-fill,
        chunks_per_stage-loop) pass -- one (ly,lx) iteration's worth -- and
        the ly/lx loop plus merge()/cascade_put3/drain move to each
        make_*_fn below, so a merge+cascade+drain happens once PER lx,
        matching the runtime's per-lx TaskGroup+drain exactly. This is also
        a correctness fix: previously only the last lx's result could ever
        reach the cascade or the output at all.
        """
        q_saved, g, gp, up, sp, s_tmp, r_tmp, s_tmp_f32 = lb

        # Q selective capture (attn_npu2.py herd body, verbatim mechanism):
        # qk_in is this stage's broadcast fifo -- NQ Q elements (one per
        # physical q-tile column; every column receives all NQ of them,
        # since it is a true hardware broadcast) followed by
        # chunks_per_stage K elements (shared as-is, no selection needed).
        # Every column must acquire+release all NQ Q elements to keep the
        # stream position in sync with the mem tile's single BD; only the
        # one addressed to THIS column (qt == own_tx) is copied into
        # q_saved. Plain Python `range(NQ)` (own_tx is a compile-time int --
        # IRON specialises a distinct core body per Worker/column), no
        # runtime branch, matching attn_npu2.py's own unrolled `for qt in
        # range(NQ):` at this same spot (it is the ONE loop in that file's
        # herd body NOT wrapped in `air.sequential`).
        for qt in range(NQ):
            qe = qk_in.acquire(1)
            if qt == own_tx:
                copy_tile(qe, q_saved)
            qk_in.release(1)

        zero_fill_gp(gp)
        zero_fill_sp_f32(sp)
        neg_inf_fill_up(up)

        # K/V: chunks_per_stage acquires against the SAME (shared,
        # broadcast) fifos -- unchanged cadence from before the broadcast
        # restructuring.
        for chunk in range_(chunks_per_stage):
            ke = qk_in.acquire(1)
            ve = v_in.acquire(1)
            zero_fill_g(g)
            matmul_a_b(q_saved, ke, g)

            kv_block = s * chunks_per_stage + chunk
            apply_length_mask(g, kv_block)
            fused_softmax(g, up, s_tmp, r_tmp)
            mul_r_gp(r_tmp, gp)
            matmul_g_b(g, ve, gp)
            widen_bf16_to_f32(s_tmp, s_tmp_f32)
            accum_sp_r_s_f32(sp, r_tmp, s_tmp_f32)
            vector_copy_f32(0, s_tmp_f32, sp)
            qk_in.release(1)
            v_in.release(1)

        return gp, up, sp

    def merge(s, gp, up, sp, mb):
        """Fold the neighbour's cascade partial into (gp, up, sp) in place.

        Mirrors attn_npu2.py's merge(): get the neighbour's (gp_c, up_c,
        sp_c) off the cascade, rescale both partials to the combined max,
        add, and return (gp_c, sp_c) as the merged result -- same buffer
        names/roles as the AIR source, so the arithmetic kernel sequence is
        a direct transcription.
        """
        gp_c, up_c, sp_c, up_s, rc, rl, st = mb

        cascade_get3(gp_c, up_c, sp_c)
        vector_copy(0, up, up_s)
        maximum_up_u(up_c, up)
        exp_up_minus_u(up_c, up, rc)
        exp_up_minus_u(up_s, up, rl)
        mul_r_gp(rc, gp_c)
        mul_r_gp(rl, gp)
        add_gp_g(gp, gp_c)
        zero_fill_sp_f32(st)
        accum_sp_r_s_f32(sp_c, rc, st)
        accum_sp_r_s_f32(sp, rl, st)
        vector_copy_f32(0, st, sp_c)
        return gp_c, sp_c

    # ROUND 5 FIX (attempt 43): the `for _ in range_(num_head_groups): for _
    # in range_(num_lq_iters):` loop moved HERE (one level out of
    # local_compute_one), wrapping the full local-compute + merge + cascade
    # sequence so each (ly, lx) gets its OWN merge/cascade_put3/drain, in
    # lockstep with `sequence()`'s own per-lx TaskGroup + wait=True drain.
    # See local_compute_one's docstring for why the old structure hung at
    # num_lq_iters >= 2 regardless of NS.
    def make_bot_fn(tx, s):
        """ty == NS - 1 (physical row 5): put_only -- no neighbour yet."""

        def fn(qk_in, v_in, _kernels, lb):
            for _ in range_(num_head_groups):
                for _ in range_(num_lq_iters):
                    gp, up, sp = local_compute_one(tx, s, qk_in, v_in, lb)
                    cascade_put3(gp, up, sp)

        return fn

    def make_mid_fn(tx, s):
        """0 < ty < NS - 1: put_get -- merge, forward the merged partial."""

        def fn(qk_in, v_in, _kernels, lb, mb):
            for _ in range_(num_head_groups):
                for _ in range_(num_lq_iters):
                    gp, up, sp = local_compute_one(tx, s, qk_in, v_in, lb)
                    gp_c, sp_c = merge(s, gp, up, sp, mb)
                    cascade_put3(gp_c, up, sp_c)

        return fn

    def make_top_fn(tx, s):
        """ty == 0 (physical row 2): get_only -- merge, normalise, drain."""

        def fn(qk_in, v_in, gp_out, _kernels, lb, mb):
            for _ in range_(num_head_groups):
                for _ in range_(num_lq_iters):
                    gp, up, sp = local_compute_one(tx, s, qk_in, v_in, lb)
                    gp_c, sp_c = merge(s, gp, up, sp, mb)
                    div_gp_sp_f32(sp_c, gp_c)
                    elem = gp_out.acquire(1)
                    # A scalar per-element Python loop here would lower to one
                    # store instruction per element (4096 for this shape) --
                    # CLAUDE.md trap 5/10, both program-memory (trap 9) and
                    # cycle cost. copy_tile is a vectorised 32-lane bf16 copy;
                    # its compile-time size (lqp*dk in attn_npu2.cc, i.e.
                    # tile_size_q*dk in this port's naming) matches
                    # tile_size_q*dv only because dk == dv == lkp is asserted
                    # above -- if that assumption is ever relaxed this call
                    # needs its own appropriately-sized copy kernel instead of
                    # reusing copy_tile by coincidence.
                    copy_tile(gp_c, elem)
                    gp_out.release(1)

        return fn

    workers = [[[None] * NS for _ in range(NQ)] for _ in range(H)]
    for h in range(H):
        for tx in range(NQ):
            c = h * NQ + tx
            for s in range(NS):
                fn_args = [QK_l1[h][s][tx], V_l1[h][s][tx]]
                lb = make_local_bufs(c, s)
                if s == NS - 1:
                    fn = make_bot_fn(tx, s)
                    fn_args += [ALL_KERNELS, lb]
                elif s == 0:
                    fn = make_top_fn(tx, s)
                    mb = make_merge_bufs(c, s)
                    fn_args.append(GP_l1l2[c].prod())
                    fn_args += [ALL_KERNELS, lb, mb]
                else:
                    fn = make_mid_fn(tx, s)
                    mb = make_merge_bufs(c, s)
                    fn_args += [ALL_KERNELS, lb, mb]
                # ROUND 4 ATTEMPT 37: same stack_size=2048 fix as attempt
                # 33/round-4-step-1, applied to the COMPLETE full design
                # (attempt 22's, WITH div_gp_sp_f32 -- unlike the round-3
                # debug ladder scripts (attempts 29-32, 21_attempt30...),
                # which never re-added the final normalisation dropped by
                # fa_iron4 attempt 17's debug bypass).
                workers[h][tx][s] = Worker(fn, fn_args, tile=Tile(c, 2 + s), stack_size=2048)

    # Cascade edges: ty = NS-1 (bottom, row 2+NS-1) -> ... -> ty = 0 (top,
    # row 2). Same direction/shape as cascade.py's row (n_aie_rows-1) -> row
    # 0 chain, just row-origin-shifted by 2 (rows 0/1 are shim/mem here).
    # Per physical column (unchanged by the broadcast restructuring -- the
    # cascade runs down a q-tile column, orthogonal to the QK/V broadcast
    # which runs across columns within a row).
    for h in range(H):
        for tx in range(NQ):
            for s in range(NS - 1, 0, -1):
                CascadeFlow(workers[h][tx][s], workers[h][tx][s - 1])

    flat_workers = [
        workers[h][tx][s] for h in range(H) for tx in range(NQ) for s in range(NS)
    ]

    # ------------------------------------------------------------ runtime
    # QK/V producers are now per (h, s) -- one shim feed per cascade stage,
    # per local head, matching the mem tile broadcast topology above. GP
    # stays per physical column (its drain path did not change).
    QK_prods = [
        [QK_l3l2[h][s].prod(tile=Tile(h * NQ + s, 0)) for s in range(NS)]
        for h in range(H)
    ]
    V_prods = [
        [V_l3l2[h][s].prod(tile=Tile(h * NQ + s, 0)) for s in range(NS)]
        for h in range(H)
    ]
    GP_conses = [GP_l2l3[c].cons(tile=Tile(c, 0)) for c in range(n_cols)]

    def sequence(Q, K, V, GP, qk_hs, v_hs, gp_hs):
        # RuntimeData (Q/K/V/GP here) is not Python-sliceable (compile_attempt7
        # .log: "'RuntimeData' object is not subscriptable") -- fill()/drain()
        # take offset/sizes/strides (or an equivalent `tap=`) directly instead.
        # Looped over every (ly, lx) grid iteration to match local_compute's
        # own `range_(num_head_groups): range_(num_lq_iters)` nesting -- at
        # this smoke-test shape num_head_groups == num_lq_iters == 1 so the
        # loop is invisible, but production (num_head_groups=10,
        # num_lq_iters=6) needs the full 60 iterations' worth of fills or the
        # workers hang waiting on fifo elements nothing ever sends (K/V are
        # deliberately RE-SENT every lx, matching attn_npu2.py's own
        # per-launch-iteration K/V puts -- see the module docstring).
        #
        # qk_hs[h][s] / v_hs[h][s] are now per (head, cascade-stage) --
        # ONE fill feeds the mem tile broadcast that reaches all NQ physical
        # q-tile columns of head h at once (deviations #1 and #2 from the
        # module header are RETRACTED: this is 4x less Q and V shim traffic
        # than either of this port's earlier attempts, and matches AIR's
        # own numbers exactly). Q is sent as ONE fill covering all NQ
        # q-tiles (contiguous in L3, stride tile_size_q*dk between them) --
        # the mem tile's ping-pong buffer streams them one at a time to the
        # NQ downstream acquire()s in local_compute's Q-capture loop, same
        # idiom this file already used for K's chunks_per_stage elements.
        for ly in range(num_head_groups):
            for lx in range(num_lq_iters):
                tg = TaskGroup()
                for h in range(H):
                    head_idx = ly * H + h
                    q_base = head_idx * (lq * dk) + lx * (lqp * dk)
                    k_base = head_idx * (lk * dk)
                    v_base = head_idx * (lk * dv)
                    for s in range(NS):
                        qk_hs[h][s].fill(
                            Q, offset=q_base, sizes=[NQ, tile_size_q, dk],
                            strides=[tile_size_q * dk, dk, 1], group=tg,
                        )
                        qk_hs[h][s].fill(
                            K,
                            offset=k_base + s * chunks_per_stage * lkp * dk,
                            sizes=[chunks_per_stage, lkp, dk],
                            strides=[lkp * dk, dk, 1], group=tg,
                        )
                        v_hs[h][s].fill(
                            V,
                            offset=v_base + s * chunks_per_stage * lkp * dv,
                            sizes=[chunks_per_stage, lkp, dv],
                            strides=[lkp * dv, dv, 1], group=tg,
                        )
                    for tx in range(NQ):
                        c = h * NQ + tx
                        out_off = (
                            head_idx * (lq * dv)
                            + lx * (lqp * dv)
                            + tx * (tile_size_q * dv)
                        )
                        gp_hs[c].drain(
                            GP, offset=out_off, sizes=[tile_size_q * dv],
                            strides=[1], wait=True, group=tg,
                        )
                tg.finish()

    rt = Runtime(
        sequence,
        [Q_ty, K_ty, V_ty, GP_ty, QK_prods, V_prods, GP_conses],
    )

    return Program(
        iron.get_current_device(), rt, workers=flat_workers
    ).resolve_program()


# ---------------------------------------------------------------- export
# build_design.py's convention (see designs/whisper_gemm/whisper_gemm.py):
# a design module exposes DESIGN (an @iron.jit callable) and SPECIALIZE (its
# CompileTime kwargs). Production shape by default -- the fixed shape
# src/open_whisper/fa_attention.hpp documents and fa_guards.hpp's
# check_fa_geometry enforces at load -- with FA_* environment overrides for
# a debugging/smoke-shape build only (trap 7e: every value here is a literal
# or a literal read once from the environment, never derived).
DESIGN = flash_attn
SPECIALIZE = dict(
    lq=int(os.environ.get("FA_LQ", 1536)),
    lk=int(os.environ.get("FA_LK", 1536)),
    lqp=int(os.environ.get("FA_LQP", 256)),
    lkp=int(os.environ.get("FA_LKP", 64)),
    dk=int(os.environ.get("FA_DK", 64)),
    dv=int(os.environ.get("FA_DV", 64)),
    num_heads=int(os.environ.get("FA_NUM_HEADS", 20)),
    num_heads_per_unroll=int(os.environ.get("FA_HEADS_PER_UNROLL", 2)),
    num_cascade_stages=int(os.environ.get("FA_CASCADE_STAGES", 4)),
    valid_len=int(os.environ.get("FA_VALID_LEN", 1500)),
)


def _device_for(dev_str, n_aie_cols):
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


def main():
    p = argparse.ArgumentParser(
        description="IRON/CascadeFlow port of AMD's attn_npu2.py (Route B)"
    )
    p.add_argument("--lq", type=int, default=256)
    p.add_argument("--lk", type=int, default=256)
    p.add_argument("--lqp", type=int, default=256)
    p.add_argument("--lkp", type=int, default=64)
    p.add_argument("--dk", type=int, default=64)
    p.add_argument("--dv", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--num-heads-per-unroll", type=int, default=2)
    p.add_argument("--num-cascade-stages", type=int, default=2)
    p.add_argument("--valid-len", type=int, default=256)
    p.add_argument("--dev", type=str, default="npu2")
    p.add_argument("--xclbin-path", type=str, default="fa_iron.xclbin")
    p.add_argument("--insts-path", type=str, default="fa_iron.insts.bin")
    opts = p.parse_args()

    iron.set_current_device(_device_for(opts.dev, opts.num_heads_per_unroll * 4))

    program = flash_attn.specialize(
        lq=opts.lq,
        lk=opts.lk,
        lqp=opts.lqp,
        lkp=opts.lkp,
        dk=opts.dk,
        dv=opts.dv,
        num_heads=opts.num_heads,
        num_heads_per_unroll=opts.num_heads_per_unroll,
        num_cascade_stages=opts.num_cascade_stages,
        valid_len=opts.valid_len,
    )
    program.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)
    print("Compilation complete.")


if __name__ == "__main__":
    main()
