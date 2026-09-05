# NpuEmbeddings -- M7: export the FOUR GEMM shapes as ONE xclbin + four
# instruction streams (tasks/0032).
# SPDX-License-Identifier: MIT
#
# tasks/0030 proved the mechanism (RTP loop bounds, exact results, zero switch
# cost); tasks/0031 measured what it is worth (~2.3 ms per design switch); and
# with LayerNorm, softmax and GELU on the host (tasks/0032), the encode's NPU
# work is 24 GEMM dispatches -- so ONE static design serving all four shapes
# makes every switch disappear.
#
# The export builds each shape with gemm_pretiled(rtp=True), verifies the four
# final.xclbin files are byte-identical modulo UUID metadata (the 0029 check --
# anything beyond ~80 differing bytes is real divergence and the export
# REFUSES), and emits:
#
#   gemm_rtp/final.xclbin              the shared static configuration
#   gemm_rtp/insts.bin                 the largest tier's qkv (slot 0)
#   gemm_rtp/insts_<shape>_b<batch>.bin every (shape, batch) stream
#   gemm_rtp/design.json               per-stream metadata the C++ parser reads
#
# BATCH TIERS (0037). M enters the static design ONLY through the loop bound
# `n_tiles_per_core`, which is a runtime parameter, so a batch-4 stream and a
# batch-128 stream share the same xclbin -- measured, 67-69 differing bytes,
# the UUID footprint (experiments/m7-switch-cost/batch_share_probe.py).
# Exporting several tiers lets a server RIGHT-SIZE each request: four texts run
# a four-sequence encode instead of padding to 128, and switching tiers costs
# nothing because it is the same context.
#
# Env: iron env WITH iron_env.ps1 dot-sourced.
# Usage:
#   python tools\export_gemm_rtp.py --batch 128 --cols 8 --out runtime\artifacts_b128il

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments" / "m5-pretiled-gemm"))
sys.path.insert(0, str(REPO / "tools"))
# RELOCATABLE: this script is also SHIPPED FLAT, as one directory of
# sibling files, into OpenFlowLM-Next's npu_offload/gemm_rtp/ (tasks/0156,
# T63). Its own directory therefore comes first on the path, and the repo
# layout below is the fallback -- so the same file works in both places
# and the sync stays a dumb copy rather than a transformation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path.home() / ".npu" / "cache"

import aie.iron as iron                              # noqa: E402
from aie.iron.device import from_name                # noqa: E402
from gemm_pretiled import pretiled_array             # noqa: E402
from npue import gemm_b_layout, layout_hash          # noqa: E402
from toolchain_provenance import write_toolchain_json  # noqa: E402

# THE SEQUENCE LENGTH, and the only thing in this file that decides it.
#
# It reaches a design through exactly one expression -- `M = batch * seq` in
# shapes_for() below -- and nowhere else. The GEMMs themselves never see it:
# an instruction stream knows M, K and N, so `batch 128 x seq 64` and
# `batch 16 x seq 512` are the SAME M = 8192 and the same arithmetic. The
# runtime inverts the split at load time (`batch = design.M / design.seq`,
# main.cpp), which is why seq has to be RECORDED in design.json rather than
# inferred -- two different splits of one M are indistinguishable afterwards.
#
# So this is a HOST-SIDE SLICING CONVENTION over a fixed row count, not a
# hardware limit, and --seq costs a re-export rather than a redesign.
#
# It is a flag rather than an edited constant for a specific reason: `M` is a
# CompileTime[int] KEYWORD ARGUMENT to pretiled_array(), so the JIT cache key
# derives from it and a changed seq produces a genuinely new build. CLAUDE.md
# trap 7d -- iron.jit's cache key never inspecting a generator's module
# globals -- would have applied had shapes_for() been a generator reading SEQ
# via LOAD_GLOBAL. It is not: it is a plain helper that computes M and passes
# it down as an argument. Read out of gemm_pretiled.py's signature, not
# assumed.
#
# WHAT ELSE HAS TO AGREE, before exporting a longer design:
#   * the runtime requires `seq % 8 == 0` (set_design_seq, main.cpp)
#   * the container must carry at least `seq` position embeddings -- the
#     packer's --max-seq, default 256. set_design_seq REFUSES seq >
#     max_seq_len rather than indexing past the table, so that one is already
#     loud; it just fires at load rather than here.
#   * attention runs on the HOST and is O(seq^2). F3 prices it at 2-5% of the
#     work AT SEQ 64. Nothing here predicts what it costs at 512, and no
#     measurement in this repo covers it -- that is a hardware question, and
#     until it is traced the throughput of a long-seq design is unknown
#     rather than assumed-proportional.
DEFAULT_SEQ = 64
STREAM_ORDER = ["qkv", "attn_out", "ffn_up", "ffn_down"]


def shapes_for(batch, hidden=384, intermediate=None, gated=False,
               qkv_n=None, seq=DEFAULT_SEQ):
    # NPUE-M13 (tasks/0069): the FFN width used to be hardcoded `4 * hidden`.
    # That is true of every BERT-family model this project ships, and it is a
    # property of those checkpoints rather than of the architecture -- so it was
    # an assumption wearing a constant's clothes. A GATED FFN (SwiGLU/GeGLU)
    # breaks it twice over: `ffn_up` must emit BOTH halves, N = 2*intermediate,
    # while `ffn_down` still consumes only one, K = intermediate.
    #
    # nomic-embed-text-v1.5: hidden 768, intermediate 3072 (so 4*h happens to
    # hold), gated -> ffn_up N = 6144. Note that 6144 crosses the C-drain guard
    # threshold that tasks/0068 found half-wired; do not build this without that
    # fix in gemm_pretiled.py.
    #
    # NPUE-M13 (tasks/0074): and `qkv` N used to be hardcoded `3 * hidden`, for
    # the same reason and with the same lifetime -- true of every model that
    # has MHA with `num_key_value_heads == num_attention_heads`, false the
    # moment one arrives with MQA/GQA. EmbeddingGemma-300M has ONE KV head at
    # head_dim 256, so its fused Q|K|V is 1280 wide, and it is PADDED to 1536
    # to make `N % (n * n_aie_cols) == 0` hold at tile_n=48 (the packer's
    # gemma_qkv_blocks() owns that arithmetic; this only has to agree with it).
    M, h = batch * seq, hidden
    f = 4 * h if intermediate is None else intermediate
    return {
        "qkv":      dict(M=M, K=h, N=3 * h if qkv_n is None else qkv_n),
        "attn_out": dict(M=M, K=h, N=h),
        "ffn_up":   dict(M=M, K=h, N=2 * f if gated else f),
        "ffn_down": dict(M=M, K=f, N=h),
    }


def core_columns(d):
    import re
    m = d / "input_with_addresses.mlir"
    if not m.exists():
        return None
    tiles = re.findall(r"aie\.tile\((\d+),\s*(\d+)\)",
                       m.read_text(encoding="utf-8", errors="ignore"))
    cols = {int(c) for c, r in tiles if int(r) >= 2}
    return len(cols) if cols else None


def markers_for(shape, m, k, n, c_dtype="f32", a_dtype="bf16"):
    """Strings that identify THIS design in the JIT cache.

    The C element type is part of the identity and must be, because
    `--c-bf16` and the fp32 default differ in the cache only by it.

    MATCH THE ORDERED SIGNATURE, NOT THREE LOOSE MEMREF STRINGS. The old form
    listed `memref<M*K xbf16>`, `memref<K*N xbf16>` and `memref<M*N xf32>` and
    asked only whether each appeared SOMEWHERE in the module. With fp32 C the
    `xf32` suffix happened to keep them apart. With bf16 C it does not, and the
    collision is exact rather than theoretical (tasks/0045):

        ffn_up   [8192,  384, 1536]  -> M*K=3145728  K*N=589824  M*N=12582912
        ffn_down [8192, 1536,  384]  -> M*K=12582912 K*N=589824  M*N=3145728

    Same three numbers, all now `bf16`, so the two shapes became
    indistinguishable and `purge()` for ffn_down DELETED ffn_up's build
    mid-export. It surfaced as a FileNotFoundError on a missing final.xclbin,
    which is luck: had ffn_up been built second it would have shipped the wrong
    instruction stream.

    `aie.runtime_sequence(%arg0: ..., %arg1: ..., %arg2: ...)` binds each size
    to an ARGUMENT POSITION, so A, B and C cannot trade places whatever their
    element types are.

    TILE-GEOMETRY MARKER, mlir-aie 1.4.x FORMAT (tasks/0060, T22). Up to and
    including 1.3.4, the sequence-body `aie.dma_bd` op printed each access-
    pattern dimension as a bracket-tuple (`<size = k, stride = n>`), and the
    second marker below matched that literally against B's innermost tiled
    dimension. The 1.4.x MLIR pretty-printer replaced that with a flat
    `sizes = [...] strides = [...]` pair of arrays for `aie.dma_bd` specif-
    ically -- `<size = N, stride = M>` bracket-tuples survive ONLY in
    `aie.objectfifo`'s `dimensionsToStream` attribute, a different op, so the
    old substring silently stopped matching anything in a freshly built
    `aie.mlir` (confirmed by grepping a fresh cache dir: 0 hits for the old
    form, `markers_for` always finding 0 cache candidates).

    Confirmed directly (all four production shapes, m=64/k=64/n=48, cols=2):
    B's (`%arg1`) `aie.dma_bd` always ends its access pattern with the tile
    dims as the LAST TWO entries of `sizes`, immediately followed by the
    `strides` array --
        sizes = [.., .., 64, 48] strides = [.., .., 48, 1]
    -- exactly twice per build (the ping/pong pair), in every one of qkv,
    attn_out, ffn_up and ffn_down, and nowhere else in the file (A's and C's
    `aie.dma_bd` end their `sizes` in different values). So `f"{k}, {n}]
    strides = ["` is the direct translation of the old `<size=k, stride=n>`
    marker into the new textual form: same two numbers, same adjacency
    requirement, just spelled the way 1.4.x's printer spells it.
    """
    # A and B carry the OPERAND type, which stopped being bf16 in tasks/0078.
    # Leaving `xbf16` hardcoded here would have made the cache search look for
    # a design that does not exist and report "0 cache candidates after purge"
    # -- the same silent-miss shape tasks/0060 hit when the MLIR printer
    # changed format under an unchanged marker string.
    M, K, N = shape["M"], shape["K"], shape["N"]
    return [f"aie.runtime_sequence(%arg0: memref<{M * K}x{a_dtype}>, "
            f"%arg1: memref<{K * N}x{a_dtype}>, "
            f"%arg2: memref<{M * N}x{c_dtype}>)",
            f"{k}, {n}] strides = [",
            'sym_name = "rtp_0_0"']


def find_cache(markers, cols, what):
    hits = []
    for d in CACHE.iterdir():
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists() and (d / "final.xclbin").exists()
                and (d / "insts.bin").exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if not all(x in text for x in markers):
            continue
        if core_columns(d) != cols:
            continue
        hits.append(d)
    if len(hits) != 1:
        raise SystemExit(f"{what}: {len(hits)} cache candidates after purge -- "
                         f"expected exactly 1")
    return hits[0]


def purge(markers, cols, what):
    n = 0
    for d in list(CACHE.iterdir()):
        mlir = d / "aie.mlir"
        if not (d.is_dir() and mlir.exists()):
            continue
        text = mlir.read_text(encoding="utf-8", errors="ignore")
        if all(x in text for x in markers) and core_columns(d) in (cols, None):
            shutil.rmtree(d)
            n += 1
    if n:
        print(f"  {what}: purged {n} cache candidate(s)")


def xclbin_identical_mod_uuid(a: bytes, b: bytes):
    """Do these two xclbins carry the SAME static configuration?

    0029's version was `size equal and <= 80 differing bytes`, the 80 fitted
    to one observed pair (67 bytes: a 16-byte UUID at 0x1a0, its 32-char hex
    rendering in the build metadata, a second partition UUID, and four
    single-byte fragments).

    THE BOUND WAS FITTED TO ONE SAMPLE AND IT IS TOO TIGHT (tasks/0152). A
    legitimate bge-base export was refused at **81** bytes, and the diff
    showed why: BOTH UUIDs get a 32-character hex rendering, not just the
    first, so the metadata footprint alone runs to ~110 bytes and whether a
    given pair lands under 80 depends on how many hex digits two random UUIDs
    happen to share. Measured across eight legitimate pairs: 70-81 bytes in
    11-13 runs, every run <= 32 bytes.

    So the test is STRUCTURAL now rather than a count. What a real divergence
    looks like is qualitatively different: the AIE_PARTITION section carries
    the core ELFs and the whole DMA/lock topology, so two genuinely different
    static configurations differ in contiguous kilobytes. Metadata differs in
    short, isolated runs. The rule is therefore:

      * identical size, and
      * no differing run longer than 32 bytes (a UUID is 16, its hex
        rendering 32), and
      * at most 32 runs and 192 bytes in total -- generous against the
        measured 13 runs / 81 bytes, and still three orders of magnitude
        below any real configuration change.

    The detail string reports the run structure, not just the total, so the
    two failure modes are distinguishable in a log written months ago.
    """
    if len(a) != len(b):
        return False, f"sizes differ: {len(a)} vs {len(b)}"
    runs = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    total = sum(e - s + 1 for s, e in runs)
    longest = max((e - s + 1 for s, e in runs), default=0)
    detail = (f"{total} differing bytes in {len(runs)} runs, "
              f"longest {longest} B")
    ok = longest <= 32 and len(runs) <= 32 and total <= 192
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runtime" / "artifacts"))
    ap.add_argument("--batch", type=int, default=128,
                    help="largest tier; also the buffer sizing")
    ap.add_argument("--batches", default=None,
                    help="comma-separated batch tiers, e.g. 4,16,32,128. "
                         "Defaults to just --batch.")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--intermediate", type=int, default=None,
                    help="FFN width. Defaults to 4*hidden, which every "
                         "BERT-family model here happens to satisfy; pass it "
                         "explicitly for anything else.")
    ap.add_argument("--qkv-n", type=int, default=None,
                    help="width of the fused qkv operand. Defaults to "
                         "3*hidden, which holds whenever num_key_value_heads "
                         "== num_attention_heads; MQA/GQA models need it "
                         "stated (EmbeddingGemma-300M: 1536, tasks/0074).")
    ap.add_argument("--gated-ffn", action="store_true",
                    help="ffn_up emits BOTH halves of a gated FFN "
                         "(N = 2*intermediate), as SwiGLU/GeGLU need, while "
                         "ffn_down still takes K = intermediate. tasks/0069.")
    ap.add_argument("--seq", type=int, default=DEFAULT_SEQ,
                    help="sequence length this design is built for. Enters "
                         "only as M = batch*seq, so it trades against --batch "
                         "at constant array work: --batch 16 --seq 512 builds "
                         "the same M=8192 as the shipped --batch 128 --seq 64. "
                         "Must be a multiple of 8, and the .npue must carry at "
                         "least this many position embeddings (pack_npue.py "
                         "--max-seq, default 256). Host attention is O(seq^2) "
                         "and unmeasured above 64 -- see DEFAULT_SEQ above.")
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-n", type=int, default=48)
    # NPUE-M9 (tasks/0045): narrow C to bf16 on the core, after the fp32 K
    # reduction, so the C DMA and the host readback move half the bytes.
    # tasks/0044 measured that readback at 18.8% of a MiniLM encode.
    # THE DATAPATH FLAG (tasks/0077, 0078). bf16 stays the default and stays
    # supported; int8 is an alternative artifact set the runtime selects by
    # reading `a_dtype` out of design.json.
    ap.add_argument("--int8", action="store_true",
                    help="build the int8 MMAC datapath (i8 operands, int32 "
                         "accumulator) instead of bf16. Measured at 5.5-7.7x "
                         "the bf16 datapath on all four production shapes and "
                         "BIT-EXACT (tasks/0077). Needs an int8 container: "
                         "tools/pack_npue.py --int8.")
    ap.add_argument("--c-bf16", action="store_true",
                    help="GEMM emits bf16 C (fp32 accumulate, one round at "
                         "the end). Halves C transport; the runtime reads the "
                         "dtype from design.json.")
    # RESEARCH FLAG (T23). The bfp16-emulated MMAC datapath is 2.9x of array
    # GEMM time (tasks/0049) at 1-cos ~3.5e-03 -- it FAILED the 2e-03 gate in
    # 0026 and MTEB (0035) is the authority on reopening it. This flag exists
    # so that MTEB measurement can be taken; it is NOT a production mode.
    # Cache note: bfp16 and plain-bf16 builds are AMBIGUOUS in aie.mlir (same
    # buffer dtypes, same sym names) -- correctness rests entirely on
    # purge-before-build (tasks/0030, fifth fail-open), which removes every
    # matching candidate before each fresh build. Re-export the plain set
    # after using this, for the same reason.
    ap.add_argument("--emulate-bfp16", action="store_true",
                    help="RESEARCH: build the GEMM on the bfp16-emulated "
                         "MMAC datapath (T23 accuracy measurement)")
    # T61-2 (tasks/0152). How many row blocks the runtime sequence keeps in
    # flight between TCT barriers: tb_max_n_rows//2 per ping-pong half, one
    # tg.finish() per half. 4 (the shipped value) means 2 row blocks per group
    # and 16 barriers per dispatch at M = 8192.
    #
    # It reaches the design as a CompileTime keyword argument -- trap 7e's
    # rule -- so a changed value is a genuinely different JIT cache entry, and
    # purge-before-build still removes the previous one because the markers
    # are identical between the two.
    # T61-2 (tasks/0152). Issue the next row-block group BEFORE awaiting the
    # previous one, so the shim's queues are never empty across a barrier.
    # Same descriptor peak (two groups outstanding), same bytes, same order.
    ap.add_argument("--tg-depth", type=int, default=2,
                    help="how many row-block halves the runtime sequence "
                         "keeps outstanding. 1 is the pre-tasks/0152 "
                         "schedule: it awaits each half immediately after "
                         "issuing it, so nothing is ever in flight across a "
                         "barrier. 2 (the default) issues half n+1 first, "
                         "for 1.034-1.141x of array time on the seven "
                         "catalogue models, bit-identical. 3 compiles and "
                         "then TIMES OUT on hardware -- do not ship it.")
    ap.add_argument("--tb-rows", type=int, default=4,
                    help="row blocks in flight between the sequence's TCT "
                         "barriers, x2 (default 4 = 2 per ping-pong half). "
                         "Raising it trades shim BDs for fewer barriers; "
                         "tasks/0152 measures what that is worth.")
    args = ap.parse_args()
    tiers = ([int(x) for x in args.batches.split(",")] if args.batches
             else [args.batch])
    tiers = sorted(set(tiers))
    for b in tiers:
        if b % 4:
            raise SystemExit(f"batch {b}: must be a multiple of 4")
    if max(tiers) != args.batch:
        raise SystemExit(f"--batch {args.batch} must be the largest tier "
                         f"(tiers are {tiers})")

    # The runtime's own rule on seq (set_design_seq in main.cpp), checked HERE
    # so a bad value costs a second rather than a full export followed by a
    # load-time refusal.
    if args.seq <= 0 or args.seq % 8:
        raise SystemExit(f"--seq {args.seq}: must be positive and a multiple "
                         f"of 8 (the runtime refuses anything else)")
    # M % (m * n_aie_rows) == 0 with 4 rows. gemm_pretiled asserts this, but it
    # asserts it about M, and someone who just set --seq is thinking in
    # sequences -- so name the two numbers that produced the M.
    for b in tiers:
        if (b * args.seq) % (args.m * 4):
            raise SystemExit(
                f"--seq {args.seq} x batch tier {b} gives M = {b * args.seq}, "
                f"which is not a multiple of m*rows = {args.m * 4}. Pick a "
                f"tier or a seq whose product divides it.")
    if args.seq != DEFAULT_SEQ:
        print(f"  seq        {args.seq} (default {DEFAULT_SEQ}). M = "
              f"batch*seq, so tiers {tiers} give "
              f"M {[b * args.seq for b in tiers]}.")
        print("             Host attention is O(seq^2) and this repo has no "
              "measurement above seq 64. Treat this design's throughput as "
              "unknown until it is traced (CLAUDE.md rules 1 and 6).")

    iron.set_current_device(from_name("npu2", n_cols=None))

    # Build every (shape, tier). The identity check then covers BOTH axes:
    # if any of them diverged, the whole one-context story is false and the
    # export refuses rather than shipping an artifact that lies about it.
    # THE DATAPATH, decided once and written into design.json (tasks/0078).
    #
    # int8 is not a variant of the bf16 path, it is a different MMAC datapath:
    # `mac_dims` (8,8,8) against bf16's (4,8,8), an int32 accumulator with NO
    # rounding in the reduction, and 5.5-7.7x the throughput measured on all
    # four production shapes (tasks/0077). Both stay available -- the runtime
    # reads which one an artifact set is, exactly as it already reads
    # `c_dtype`, so a wrong pairing refuses instead of reading the right number
    # of bytes in the wrong format.
    if args.int8:
        if args.emulate_bfp16:
            raise SystemExit("--int8 and --emulate-bfp16 are both datapath "
                             "choices; pick one")
        # --int8 --c-bf16 COMPOSE, and the earlier refusal here was wrong.
        # It read the two flags as rival answers to "what leaves the core",
        # but they answer different questions: --int8 picks the MMAC datapath
        # and the accumulator, --c-bf16 picks the TRANSPORT width of a result
        # that has already been reduced. tasks/0080 measured the pairing at
        # 1.333x on the four production dispatches -- larger than the +4.9% the
        # same flag bought on bf16, because int8 moved the GEMM back into the
        # traffic-bound regime (0010's model at R2 0.987, against the
        # iteration-bound 0048 model that governs bf16). Accuracy is free: the
        # int32 accumulator's low bits sit under the int8 quantisation noise,
        # measured at 1.178e-03 -> 1.161e-03 with `npuembed --sim-c-bf16`
        # BEFORE the kernel was written.
        a_str, acc_str, a_np = "i8", "i32", np.int8
        c_np = bfloat16 if args.c_bf16 else np.int32
        c_marker = "bf16" if args.c_bf16 else "i32"
        c_bytes_out = 2 if args.c_bf16 else 4
    else:
        a_str, acc_str, a_np = "bf16", "f32", bfloat16
        c_np = bfloat16 if args.c_bf16 else np.float32
        c_marker = "bf16" if args.c_bf16 else "f32"
        c_bytes_out = 2 if args.c_bf16 else 4

    dirs = {}
    for b in tiers:
        shapes_b = shapes_for(b, args.hidden, args.intermediate, args.gated_ffn,
                              args.qkv_n, args.seq)
        for name in STREAM_ORDER:
            sh = shapes_b[name]
            mk = markers_for(sh, args.m, args.k, args.n, c_marker, a_str)
            purge(mk, args.cols, f"{name}@b{b}")
            M, K, N = sh["M"], sh["K"], sh["N"]
            A = iron.zeros((M, K), dtype=a_np, device="npu")
            B = iron.zeros((K, N), dtype=a_np, device="npu")
            C = iron.zeros(M * N, dtype=c_np, device="npu")
            pretiled_array(A, B, C, M=M, K=K, N=N, m=args.m, k=args.k,
                           n=args.n, n_aie_cols=args.cols,
                           dtype_in_str=a_str, dtype_out_str=acc_str,
                           emulate_bf16_mmul_with_bfp16=args.emulate_bfp16,
                           pretiled=True, trace_config=None, rtp=True,
                           c_bf16=args.c_bf16,
                           tb_max_n_rows=args.tb_rows,
                           tg_depth=args.tg_depth)
            dirs[(name, b)] = find_cache(mk, args.cols, f"{name}@b{b}")
            print(f"  b{b:<4} {name:<9} {str([M, K, N]):>20} -> "
                  f"{dirs[(name, b)].name}")

    ref_key = ("qkv", max(tiers))
    base = (dirs[ref_key] / "final.xclbin").read_bytes()
    for key, d in dirs.items():
        if key == ref_key:
            continue
        ok, detail = xclbin_identical_mod_uuid(base, (d / "final.xclbin").read_bytes())
        print(f"  identity {ref_key[0]}@b{ref_key[1]} vs "
              f"{key[0]}@b{key[1]:<4} {detail}  {'OK' if ok else 'DIVERGED'}")
        if not ok:
            raise SystemExit(
                "static configurations diverged -- the streams do NOT share "
                "an xclbin, refusing to export a lying artifact")

    out = Path(args.out) / "gemm_rtp"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("insts_*.bin"):
        f.unlink()                        # never leave a stale tier behind
    shutil.copy(dirs[ref_key] / "final.xclbin", out / "final.xclbin")
    shutil.copy(dirs[ref_key] / "insts.bin", out / "insts.bin")

    # Slots 1..N in load order; slot 0 is insts.bin and is never bound to an
    # op, so the mapping stays explicit rather than depending on which stream
    # happened to be copied first.
    slot = 0
    stream_meta = []
    for b in tiers:
        for name in STREAM_ORDER:
            slot += 1
            fn = f"insts_{name}_b{b}.bin"
            shutil.copy(dirs[(name, b)] / "insts.bin", out / fn)
            sh = shapes_for(b, args.hidden, args.intermediate, args.gated_ffn,
                              args.qkv_n, args.seq)[name]
            stream_meta.append({"op": name, "batch": b, "slot": slot,
                                "file": fn, "M": sh["M"], "K": sh["K"],
                                "N": sh["N"],
                                "src": dirs[(name, b)].name})

    biggest = shapes_for(max(tiers), args.hidden, args.intermediate,
                        args.gated_ffn, args.qkv_n, args.seq)
    c_bytes = c_bytes_out
    a_bytes = np.dtype(a_np).itemsize
    b_layout = gemm_b_layout(args.k, args.n,
                             dtype="I8" if args.int8 else "BF16")
    meta = {
        "name": "gemm_rtp", "kind": "gemm_rtp", "kernel": "MLIR_AIE",
        "M": biggest["qkv"]["M"],        # batch inference in the runtime
        "buffers": [max(sh["M"] * sh["K"] * a_bytes for sh in biggest.values()),
                    max(sh["K"] * sh["N"] * a_bytes for sh in biggest.values()),
                    max(sh["M"] * sh["N"] * c_bytes for sh in biggest.values())],
        # The runtime must READ this, never assume it. An artifact that is
        # bf16 and says nothing looks exactly like an fp32 one to a parser
        # that defaults -- the eighth fail-open in CLAUDE.md is a literal that
        # should have been data.
        "c_dtype": c_marker,
        # The A/B operand type. Absent in every artifact exported before
        # tasks/0078, and every one of those is bf16 -- so the runtime reads
        # silence as "bf16" rather than defaulting blindly.
        "a_dtype": a_str,
        # THE MMAC DATAPATH (tasks/0104, T23). True when this design's matmul
        # was built with emulate_bf16_mmul_with_bfp16 -- a DIFFERENT MMAC
        # precision on the SAME bf16-shaped operands, so it changes neither
        # a_dtype/c_dtype nor b_layout_hash (gemm_b_layout() only sees
        # dtype="BF16" either way, tasks/0080's own comment on int8 makes the
        # same point about geometry not being enough once there is more than
        # one datapath). Without this field a bfp16 design and a plain-bf16
        # design at the same geometry are byte-for-byte indistinguishable to
        # design_fits(), and bge-small -- the one model T23 did NOT clear for
        # bfp16 -- shares MiniLM's hidden-384 geometry. Absent (every export
        # before this field existed) means false, which is correct: every one
        # of those builds predates --emulate-bfp16 ever landing in a shipped
        # artifact.
        "emulate_bfp16": bool(args.emulate_bfp16),
        "b_layout_hash": layout_hash(b_layout),
        "b_layout": b_layout,
        "cols": args.cols, "batch": max(tiers), "tiers": tiers,
        # The sequence length these designs were compiled for. It is a
        # property of the DESIGN, not of the model: the container's
        # max_seq_len is how many position embeddings were packed (256),
        # which is a different and larger number.
        "seq": args.seq,
        # NPUE-M13 (tasks/0069, thread T31). The geometry this design was built
        # FOR, stated rather than inferred. `design_fits()` used to ask only
        # whether `hidden` appeared as some "K" in this file -- which is true of
        # any design at the same width, whatever its FFN looks like. nomic's K
        # set {768, 3072} is IDENTICAL to bge-base's while its gated ffn_up is
        # N=6144 against bge-base's 3072, so that check passes and the runtime
        # would dispatch a stream built for half the output width, silently.
        # With these three keys the match can be exact.
        "hidden": args.hidden,
        "intermediate": (4 * args.hidden if args.intermediate is None
                         else args.intermediate),
        "gated_ffn": args.gated_ffn,
        # tasks/0074: qkv's width joins the other three as DATA. design_fits()
        # derived it as 3*hidden, which is exactly the shape of the T31
        # fail-open one field to the left -- an MQA model's fused qkv is
        # narrower, and a design built for 2304 would silently serve a model
        # needing 1536.
        "qkv_n": biggest["qkv"]["N"],
        "tile": {"m": args.m, "k": args.k, "n": args.n},
        # THE RUNTIME SEQUENCE'S BARRIER SCHEDULE (T61-2, tasks/0152).
        #
        # These two change NOTHING that any other field in this file can see:
        # same geometry, same tile, same datapath, same b_layout_hash, same
        # final.xclbin. They change only insts.bin -- which is precisely the
        # shape of trap 7c, a build artifact identifiable by nothing visible.
        # Recorded so a measurement can name the schedule it measured, and so
        # a directory that predates the fields reads as the shipped schedule
        # (which it is: tb_max_n_rows was a constant 4 and there was no
        # pipelined variant to be).
        "tb_max_n_rows": args.tb_rows,
        "tg_depth": args.tg_depth,
        "streams": stream_meta,
    }
    (out / "design.json").write_text(json.dumps(meta, indent=2),
                                     encoding="utf-8")
    # T39 (tasks/0106): record which toolchain built this. Sidecar, not a
    # key in design.json above -- see toolchain_provenance.py's header.
    tc = write_toolchain_json(out)
    print(f"  toolchain  mlir_aie {tc['mlir_aie_version']}, "
          f"peano {tc['peano_version']}, "
          f"mlir-aie HEAD {tc['mlir_aie_git_head']}")
    print(f"\n  wrote {out} -- ONE xclbin, {len(stream_meta)} streams "
          f"({len(STREAM_ORDER)} shapes x {len(tiers)} batch tiers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
