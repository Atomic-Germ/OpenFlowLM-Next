r"""Build the open Whisper engine's kernel set (issue #72): ONE xclbin carrying an
instruction stream per encoder GEMM shape, plus fa/ (the bidirectional FlashAttention
kernel, NpuEmbeddings task 0181), plus the files the engine reads beside them.

    . C:\dev\mlir-aie\iron_env.ps1          # or: source ~/ironenv142/bin/activate
    python open_kernels/export_whisper_kernels.py [--out DIR] [--only qkv,o] [--force] [--no-fa]

By default the set is written to src/xclbins/Whisper-V3-Turbo-NPU2/open_kernels/ -- the
same place the other open engines' exporters write theirs, which the build tree's
xclbins junction and the install step already cover, so the engine finds it with no
configuration (src/common/whisper/whisper_engine_select.cpp, find_open_kernels).

Each GEMM stream is designs/whisper_gemm/whisper_gemm.py specialised to one (M, K, N) and
built by build_design.py in its own directory. The set is only valid if every stream's
final.xclbin is the SAME static configuration -- the instruction streams are swapped over
one hardware context, so a stream whose core program or DMA topology differed would run
against the wrong one. That is checked, not assumed: fc1 and xkv drain C one row block at
a time (tb_n_rows = 1) and conv2 has K = 3840, so they are exactly the streams that could
diverge. The comparison is npu_offload/gemm_rtp's xclbin_identical_mod_uuid, the one the
embedding exporter already uses.

fa/ is a SEPARATE hardware context (its own xclbin, its own fixed shape -- H=20, dk=dv=64,
lq=lk=1536, valid_len=1500 -- src/open_whisper/fa_attention.hpp's contract), built from
designs/whisper_fa/attn_fa.py the same way, by the same build_design.py, with the same
pinned toolchain -- so the whole kernel set, GEMM and attention alike, now comes from our
own IRON source and never needs a kernel built elsewhere (e.g. AMD's MLIR-AIR toolchain)
copied in. `--no-fa` skips it (the engine then falls back to host attention, OW_ATTN=auto).
It is built into a temporary staging directory and only renamed to <out>/fa/ once every
file in it exists and is verified -- a failed or partial fa build must never leave a
directory at <out>/fa/ that OW_ATTN=auto could pick up and dispatch against.

Output (DIR):
    final.xclbin, insts.bin (= the first GEMM stream), insts_<stream>.bin
    design.json          what open_npue's npu::Design parses (buffers sized for the
                         largest stream), plus the per-stream table
    toolchain.json       which mlir-aie / Peano / git HEAD built it (T39)
    whisper_kernels.json the marker the engine's kernel lookup accepts, with the model
                         geometry it was built for (hf_config_check)
    build/<stream>/      each GEMM stream's own build, kept for the identity evidence
    fa/air.xclbin, fa/air.insts.bin, fa/fa.json   the FlashAttention kernel (see above)
    build/fa/            its own build, kept for the identity evidence
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GEMM_RTP = HERE.parent / "npu_offload" / "gemm_rtp"
sys.path.insert(0, str(GEMM_RTP))
sys.path.insert(0, str(HERE / "designs" / "whisper_gemm"))
sys.path.insert(0, str(HERE / "designs" / "whisper_fa"))

from npue import gemm_b_layout, layout_hash  # noqa: E402
from toolchain_provenance import write_toolchain_json  # noqa: E402

DESIGN = HERE / "designs" / "whisper_gemm" / "whisper_gemm.py"
FA_DESIGN = HERE / "designs" / "whisper_fa" / "attn_fa.py"
FA_SOURCES = ["attn_fa.py", "attn_npu2.cc", "attn_cascade_wrap.cc", "zero.cc"]
FORMAT = "oflm-open-whisper-kernels-v1"
# The directory name src/model_list.json gives whisper-v3:turbo; the engine looks for
# <xclbins root>/xclbins/<this>/open_kernels (Whisper_Config::model_name).
MODEL_NAME = "Whisper-V3-Turbo-NPU2"

# whisper_gemm.py is the single source of the shapes and knobs; read them from it rather
# than restating them (a second copy is a chance to drift).
import whisper_gemm as wg  # noqa: E402

# FA_SHAPE is NOT read from attn_fa.py's own SPECIALIZE dict: that dict is perturbable by
# FA_* environment variables (a debugging/smoke-shape convenience -- attn_fa.py's own
# README), so importing it here would let a developer's ambient shell silently change what
# this exporter believes it shipped. This exporter owns the production shape it ships,
# the same way build_stream() below owns each GEMM stream's (M, K, N) rather than reading
# WG_M/K/N from the environment -- and sets exactly these values in the subprocess env,
# never inheriting FA_* from whatever the caller's shell happens to have (mirrors
# WG_BFP16's own comment in build_stream: "SET here, never inherited"). Matches
# attn_fa.py's own SPECIALIZE defaults and src/open_whisper/fa_attention.hpp's fixed-shape
# contract; a change to either must be made in both, the same duplication
# whisper_kernels.json's HF_CONFIG_CHECK below already accepts against config.json.
FA_SHAPE = {
    "lq": 1536, "lk": 1536, "lqp": 256, "lkp": 64, "dk": 64, "dv": 64,
    "num_heads": 20, "num_heads_per_unroll": 2, "num_cascade_stages": 4,
    "valid_len": 1500,
}

HF_CONFIG_CHECK = {"model_type": "whisper", "d_model": 1280, "encoder_layers": 32,
                   "decoder_layers": 4, "encoder_attention_heads": 20,
                   "decoder_attention_heads": 20, "encoder_ffn_dim": 5120,
                   "decoder_ffn_dim": 5120, "num_mel_bins": 128,
                   "max_source_positions": 1500, "max_target_positions": 448,
                   "vocab_size": 51866}


def xclbin_identical_mod_uuid(a: bytes, b: bytes):
    # Imported lazily: export_gemm_rtp pulls in IRON at module level.
    from export_gemm_rtp import xclbin_identical_mod_uuid as same
    return same(a, b)


def build_stream(name: str, out: Path, force: bool, bfp16: bool) -> Path:
    M, K, N = wg.STREAMS[name]
    bdir = out / "build" / name
    if not force and (bdir / "final.xclbin").is_file() and (bdir / "insts.bin").is_file():
        stamp = bdir / "shape.json"
        if _read_stamp(stamp) == {"M": M, "K": K, "N": N, "bfp16": bfp16}:
            print(f"  {name:6s} {M}x{K}x{N}  (kept)")
            return bdir
    # WG_BFP16 is SET here, never inherited: an exporter that let the environment
    # decide its datapath would record whatever this function was written to assume
    # (see the emulate_bfp16 note below), which is how the two sets became
    # indistinguishable in the first place.
    env = dict(os.environ, WG_M=str(M), WG_K=str(K), WG_N=str(N),
               WG_BFP16="1" if bfp16 else "0")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "build_design.py"), str(DESIGN), str(bdir)],
                       env=env, cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0 or "BUILD_OK" not in r.stdout:
        sys.stdout.write(r.stdout[-4000:])
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(f"build of stream {name} failed (exit {r.returncode})")
    (bdir / "shape.json").write_text(
        json.dumps({"M": M, "K": K, "N": N, "bfp16": bfp16}))
    print(f"  {name:6s} {M}x{K}x{N}  built in {time.time() - t0:.0f} s")
    return bdir


def _git_blob_hash(repo_root: Path, path: Path) -> str:
    # git hash-object needs no commit -- it hashes the file's CURRENT bytes on disk,
    # which is what "the design source that actually built this" means while this very
    # tree is uncommitted. "unavailable" (never a guess) if git or the file is missing.
    try:
        rel = str(path.relative_to(repo_root)).replace("\\", "/")
        r = subprocess.run(["git", "hash-object", rel], cwd=str(repo_root),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unavailable"
    except Exception:
        return "unavailable"


def _read_stamp(path: Path):
    # A missing, truncated or otherwise unreadable stamp means "rebuild", never a crash --
    # the stamp only decides whether a build may be kept.
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fa_stamp() -> dict:
    # What a kept fa build must match: the shape AND the bytes of every source that went
    # into it. A shape-only stamp would keep a stale build after an edit to attn_npu2.cc
    # or attn_fa.py -- the "stale binary fails open" family (NpuEmbeddings CLAUDE.md traps
    # 7c/7d): the kernel set would carry the old kernel while fa.json's blob hashes named
    # the new sources.
    # And the toolchain that compiles them: after an mlir-aie or Peano change a kept build
    # would otherwise ship next to a fa.json naming the NEW versions.
    import importlib.metadata as md

    def _ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except Exception:
            return "unavailable"

    fa_dir = HERE / "designs" / "whisper_fa"
    return {"shape": FA_SHAPE,
            "sources": {f: hashlib.sha256((fa_dir / f).read_bytes()).hexdigest() for f in FA_SOURCES},
            "toolchain": {"mlir_aie": _ver("mlir_aie"), "llvm-aie": _ver("llvm-aie")}}


def _remove_stale_fa(out: Path) -> bool:
    """--no-fa must leave a GEMM-only/host-attention set: OW_ATTN=auto discovers a kernel
    purely from <out>/fa/'s presence (fa_kernel_present() in encoder.cpp), so a `fa/` left
    over from an earlier `--fa` build of the same --out would still be picked up and used,
    silently defeating --no-fa (PR #111 review). <out>/build/fa (the build cache) is left
    alone -- it is never discovered by the engine and keeping it avoids re-paying the ~45 s
    IRON build if --fa is turned back on later.

    Removed atomically: rename fa/ out of the way first, then delete the renamed copy, so
    the canonical path either has a complete fa/ or none at all -- never a directory
    mid-rmtree that still satisfies the engine's file-presence check.
    """
    final_dir = out / "fa"
    if not final_dir.exists():
        return False
    stale = out / ".fa_stale"
    shutil.rmtree(stale, ignore_errors=True)
    final_dir.rename(stale)
    shutil.rmtree(stale, ignore_errors=True)
    return True


def build_fa(out: Path, force: bool) -> Path:
    """Build designs/whisper_fa/attn_fa.py at the fixed production shape (FA_SHAPE) and
    stage it into <out>/fa/ (air.xclbin, air.insts.bin, fa.json) -- see the module
    docstring for why this is staged rather than built directly into <out>/fa/.
    """
    bdir = out / "build" / "fa"
    if not force and (bdir / "final.xclbin").is_file() and (bdir / "insts.bin").is_file():
        stamp = bdir / "shape.json"
        if _read_stamp(stamp) == _fa_stamp():
            print("  fa      (kept)")
        else:
            _run_fa_build(bdir)
    else:
        _run_fa_build(bdir)

    # Stage every file fa/ needs in a directory nothing reads yet, then become <out>/fa/
    # in one rename -- a build failure above already raised SystemExit before this point,
    # so nothing here runs unless final.xclbin/insts.bin exist and are the shape recorded
    # in shape.json; this staging step is about atomicity of the RENAME itself, not about
    # catching a build failure a second time.
    staging = out / ".fa_staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shutil.copyfile(bdir / "final.xclbin", staging / "air.xclbin")
    shutil.copyfile(bdir / "insts.bin", staging / "air.insts.bin")

    tc = write_toolchain_json(staging)  # also leaves toolchain.json in fa/, same as design.json's
    fa_dir = HERE / "designs" / "whisper_fa"
    fa_json = dict(FA_SHAPE)
    fa_json.update({
        "heads": FA_SHAPE["num_heads"],
        # fp32_state / emulate_bfp16 are not flags of this exporter -- attn_fa.py's
        # flash_attn() compiles FP32_STATE and AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
        # unconditionally (see its compile_flags list), so True is the only value this
        # design ever builds, not a default being assumed.
        "fp32_state": True,
        "emulate_bfp16": True,
        "mlir_aie_version": tc["mlir_aie_version"],
        "peano_version": tc["peano_version"],
        "design_source": {
            "path": "open_kernels/designs/whisper_fa",
            "blob_sha1": {f: _git_blob_hash(HERE.parent, fa_dir / f) for f in FA_SOURCES},
            "mlir_aie_git_head": tc["mlir_aie_git_head"],
        },
    })
    (staging / "fa.json").write_text(json.dumps(fa_json, indent=2) + "\n", encoding="utf-8")

    final_dir = out / "fa"
    shutil.rmtree(final_dir, ignore_errors=True)
    staging.rename(final_dir)
    print(f"  fa      H={FA_SHAPE['num_heads']} dk=dv={FA_SHAPE['dk']} "
          f"lq=lk={FA_SHAPE['lq']} valid_len={FA_SHAPE['valid_len']}  -> {final_dir}")
    return final_dir


def _run_fa_build(bdir: Path) -> None:
    # FA_* is SET here, never inherited -- same reasoning as build_stream's WG_BFP16: an
    # exporter that let the environment decide the shape it ships would record whatever
    # this function was written to assume, not what actually got built.
    env = dict(os.environ)
    for k in ("FA_LQ", "FA_LK", "FA_LQP", "FA_LKP", "FA_DK", "FA_DV", "FA_NUM_HEADS",
             "FA_HEADS_PER_UNROLL", "FA_CASCADE_STAGES", "FA_VALID_LEN"):
        env.pop(k, None)
    env.update({
        "FA_LQ": str(FA_SHAPE["lq"]), "FA_LK": str(FA_SHAPE["lk"]),
        "FA_LQP": str(FA_SHAPE["lqp"]), "FA_LKP": str(FA_SHAPE["lkp"]),
        "FA_DK": str(FA_SHAPE["dk"]), "FA_DV": str(FA_SHAPE["dv"]),
        "FA_NUM_HEADS": str(FA_SHAPE["num_heads"]),
        "FA_HEADS_PER_UNROLL": str(FA_SHAPE["num_heads_per_unroll"]),
        "FA_CASCADE_STAGES": str(FA_SHAPE["num_cascade_stages"]),
        "FA_VALID_LEN": str(FA_SHAPE["valid_len"]),
    })
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "build_design.py"), str(FA_DESIGN), str(bdir)],
                       env=env, cwd=str(HERE), capture_output=True, text=True)
    if r.returncode != 0 or "BUILD_OK" not in r.stdout:
        sys.stdout.write(r.stdout[-4000:])
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(f"build of fa failed (exit {r.returncode})")
    (bdir / "shape.json").write_text(json.dumps(_fa_stamp()))
    print(f"  fa      built in {time.time() - t0:.0f} s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="destination (default src/xclbins/<--model-name>/open_kernels)")
    ap.add_argument("--model-name", default=MODEL_NAME,
                    help="the model directory name the server resolves the tag to "
                         "(default %(default)s, whisper-v3:turbo in src/model_list.json)")
    ap.add_argument("--only", default=None, help="comma-separated stream names (debugging)")
    ap.add_argument("--force", action="store_true", help="rebuild streams already built")
    # Default changed 2026-09-23 (task 0180 Parts 9/11/15): the golden-token-path gate
    # this help text used to cite (2 of 6 paths, enc.out cosine 0.99943 -> 0.99303) was
    # measured under the legacy decode protocol, whose 16-token watchdog truncations
    # dominate the noise floor (round 1, p = 0.044). Under the hf protocol, on the WER
    # gate (1200 utterances, LibriSpeech + FLEURS, 9 languages) that actually gates this
    # engine, bfp16 is statistically indistinguishable from bf16 (H4 vs H0: 8/16 texts
    # differ favourably, sign test not significant) and worth 1.71x on the array /
    # 5.67 RTFx end to end -- so it is now the default datapath. --no-emulate-bfp16
    # builds the plain bf16 matmul on the fp32 vector unit instead (measured slower).
    ap.add_argument("--emulate-bfp16", dest="emulate_bfp16", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="compile the bf16 matmul onto the MMAC unit via bfp16 emulation "
                         "(default: on). See tasks/0180 Parts 9/11/15 in NpuEmbeddings for "
                         "the WER evidence; --no-emulate-bfp16 builds plain bf16 instead.")
    # Default changed 2026-09-24 (NpuEmbeddings task 0181): the FlashAttention kernel used
    # to have no build path in this repository at all -- fa/ had to be produced by AMD's
    # MLIR-AIR toolchain (route A) and copied in by hand. designs/whisper_fa/attn_fa.py is
    # now our own IRON port of that kernel, verified byte-identical to AIR's own build at
    # production shape (0181's TASK.md), so the normal kernel-set build makes it too.
    # --no-fa skips it (the engine then falls back to host attention, OW_ATTN=auto).
    ap.add_argument("--fa", dest="build_fa", default=True, action=argparse.BooleanOptionalAction,
                    help="also build the FlashAttention kernel into <out>/fa/ (default: on). "
                         "--no-fa skips it.")
    args = ap.parse_args()

    names = list(wg.STREAMS) if not args.only else args.only.split(",")
    unknown = [n for n in names if n not in wg.STREAMS]
    if unknown:
        raise SystemExit(f"unknown stream(s) {unknown}; known: {list(wg.STREAMS)}")
    out = (args.out if args.out is not None
           else HERE.parent / "src" / "xclbins" / args.model_name / "open_kernels").resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"whisper kernel set -> {out}")
    dirs = {n: build_stream(n, out, args.force, args.emulate_bfp16) for n in names}

    ref_name = names[0]
    ref = (dirs[ref_name] / "final.xclbin").read_bytes()
    for n in names[1:]:
        ok, detail = xclbin_identical_mod_uuid(ref, (dirs[n] / "final.xclbin").read_bytes())
        print(f"  xclbin {n:6s} vs {ref_name}: {'same' if ok else 'DIFFERENT'} ({detail})")
        if not ok:
            raise SystemExit(f"stream {n}'s xclbin is a different static configuration from "
                             f"{ref_name}'s; they cannot share one hardware context")

    shutil.copyfile(dirs[ref_name] / "final.xclbin", out / "final.xclbin")
    shutil.copyfile(dirs[ref_name] / "insts.bin", out / "insts.bin")
    streams = []
    for slot, n in enumerate(names):
        fn = f"insts_{n}.bin"
        shutil.copyfile(dirs[n] / "insts.bin", out / fn)
        M, K, N = wg.STREAMS[n]
        streams.append({"op": n, "slot": slot, "file": fn, "M": M, "K": K, "N": N})

    shapes = [wg.STREAMS[n] for n in names]
    b_layout = gemm_b_layout(wg.K_TILE, wg.N_TILE, dtype="BF16")
    # Key order matters: npu::Design's reader takes the FIRST occurrence of a key, so the
    # top-level M/K/N must precede the per-stream table.
    meta = {
        "name": "whisper_gemm", "kind": "gemm_rtp", "kernel": "MLIR_AIE",
        "M": max(s[0] for s in shapes), "K": max(s[1] for s in shapes),
        "N": max(s[2] for s in shapes),
        "buffers": [max(M * K * 2 for M, K, _ in shapes),
                    max(K * N * 2 for _, K, N in shapes),
                    max(M * N * 4 for M, _, N in shapes)],
        # THE VALUE THAT WAS BUILT, never the one this exporter assumes. It was a
        # hardcoded False until 2026-09-22, so a bfp16 set declared itself bf16 --
        # two sets that differ in 2 of 6 golden token paths were indistinguishable
        # by their own metadata, which is the failure design.json exists to prevent.
        "c_dtype": "f32", "a_dtype": "bf16", "emulate_bfp16": bool(args.emulate_bfp16),
        "b_layout_hash": layout_hash(b_layout), "b_layout": b_layout,
        "cols": wg.N_COLS,
        "tile": {"m": wg.M_TILE, "k": wg.K_TILE, "n": wg.N_TILE},
        "tb_max_n_rows": 4, "tg_depth": wg.TG_DEPTH,
        "streams": streams,
    }
    (out / "design.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    tc = write_toolchain_json(out)

    fa_built = False
    if args.build_fa:
        build_fa(out, args.force)
        fa_built = True
    else:
        removed = _remove_stale_fa(out)
        print("  fa      skipped (--no-fa)" + (", removed stale fa/" if removed else ""))

    marker = {"format": FORMAT, "design": "design.json",
              "streams": [s["op"] for s in streams],
              "complete": names == list(wg.STREAMS),
              "emulate_bfp16": bool(args.emulate_bfp16),
              "fa": fa_built,
              "hf_config_check": HF_CONFIG_CHECK}
    (out / "whisper_kernels.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(f"  toolchain  mlir_aie {tc['mlir_aie_version']}, peano {tc['peano_version']}")
    print(f"wrote {out}: one xclbin, {len(streams)} streams"
          + ("" if marker["complete"] else "  (INCOMPLETE: --only)")
          + (", fa/" if fa_built else ", NO fa/ (--no-fa)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
