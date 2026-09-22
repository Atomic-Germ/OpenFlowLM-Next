r"""Build the open Whisper engine's kernel set (issue #72): ONE xclbin carrying an
instruction stream per encoder GEMM shape, plus the files the engine reads beside it.

    . C:\dev\mlir-aie\iron_env.ps1          # or: source ~/ironenv142/bin/activate
    python open_kernels/export_whisper_kernels.py --out DIR [--only qkv,o] [--force]

Each stream is designs/whisper_gemm/whisper_gemm.py specialised to one (M, K, N) and built
by build_design.py in its own directory. The set is only valid if every stream's
final.xclbin is the SAME static configuration -- the instruction streams are swapped over
one hardware context, so a stream whose core program or DMA topology differed would run
against the wrong one. That is checked, not assumed: fc1 and xkv drain C one row block at
a time (tb_n_rows = 1) and conv2 has K = 3840, so they are exactly the streams that could
diverge. The comparison is npu_offload/gemm_rtp's xclbin_identical_mod_uuid, the one the
embedding exporter already uses.

Output (DIR):
    final.xclbin, insts.bin (= the first stream), insts_<stream>.bin
    design.json          what open_npue's npu::Design parses (buffers sized for the
                         largest stream), plus the per-stream table
    toolchain.json       which mlir-aie / Peano / git HEAD built it (T39)
    whisper_kernels.json the marker the engine's kernel lookup accepts, with the model
                         geometry it was built for (hf_config_check)
    build/<stream>/      each stream's own build, kept for the identity evidence
"""

from __future__ import annotations

import argparse
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

from npue import gemm_b_layout, layout_hash  # noqa: E402
from toolchain_provenance import write_toolchain_json  # noqa: E402

DESIGN = HERE / "designs" / "whisper_gemm" / "whisper_gemm.py"
FORMAT = "oflm-open-whisper-kernels-v1"

# whisper_gemm.py is the single source of the shapes and knobs; read them from it rather
# than restating them (a second copy is a chance to drift).
import whisper_gemm as wg  # noqa: E402

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
        if stamp.is_file() and json.loads(stamp.read_text()) == {
                "M": M, "K": K, "N": N, "bfp16": bfp16}:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", default=None, help="comma-separated stream names (debugging)")
    ap.add_argument("--force", action="store_true", help="rebuild streams already built")
    ap.add_argument("--emulate-bfp16", action="store_true",
                    help="compile the bf16 matmul onto the MMAC unit via bfp16 emulation. "
                         "NOT the shipped datapath: measured 1.71x on the array and "
                         "1.16x on the encoder, and it costs 2 of 6 golden token paths "
                         "(enc.out cosine 0.99943 -> 0.99303). See specs/open-whisper.")
    args = ap.parse_args()

    names = list(wg.STREAMS) if not args.only else args.only.split(",")
    unknown = [n for n in names if n not in wg.STREAMS]
    if unknown:
        raise SystemExit(f"unknown stream(s) {unknown}; known: {list(wg.STREAMS)}")
    out = args.out.resolve()
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
    marker = {"format": FORMAT, "design": "design.json",
              "streams": [s["op"] for s in streams],
              "complete": names == list(wg.STREAMS),
              "emulate_bfp16": bool(args.emulate_bfp16),
              "hf_config_check": HF_CONFIG_CHECK}
    (out / "whisper_kernels.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    print(f"  toolchain  mlir_aie {tc['mlir_aie_version']}, peano {tc['peano_version']}")
    print(f"wrote {out}: one xclbin, {len(streams)} streams"
          + ("" if marker["complete"] else "  (INCOMPLETE: --only)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
