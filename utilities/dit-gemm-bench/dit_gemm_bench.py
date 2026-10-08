r"""dit_gemm_bench: the whole-array GEMM at diffusion-transformer shapes.

Phase 0 of .claude/plans/image-diffusion-npu-analysis.md: before any diffusion
work, measure what npu_offload/gemm_rtp/gemm_pretiled.py -- the GEMM Whisper's
encoder, block attention and the BERT embedders already ship -- sustains at the
shapes a DiT denoise step is made of (M = 4096 image tokens at 1024x1024).

Every datapath the design already has, with the flags the shipped sets use
(rtp=True, tg_depth=2 by default):

    bf16        bf16 x bf16, fp32 C             whisper_gemm / attn_block today
    bf16-cbf16  bf16 x bf16, fp32 acc, bf16 C   halves C transport
    bfp16       bf16 emulated on the bfp16 MMAC, bf16 C
    int8        i8 x i8, i32 acc, bf16 C        the BERT --int8 datapath

Timing is start->wait wall clock per dispatch (mlir-aie's npu_time), the same
observation open_kernels/harness prints -- a dispatch figure, not a traced
kernel-cycle claim. Correctness is checked on a sample of rows against float64.

Run from a shell where C:\dev\mlir-aie\iron_env.ps1 has been dot-sourced:

    python utilities\dit-gemm-bench\dit_gemm_bench.py                 # everything
    python utilities\dit-gemm-bench\dit_gemm_bench.py --shapes o --paths bf16
    python utilities\dit-gemm-bench\dit_gemm_bench.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "npu_offload" / "gemm_rtp"))

import aie.iron as iron  # noqa: E402
from aie.iron import kernels, str_to_dtype  # noqa: E402
from aie.iron.device import from_name  # noqa: E402
from aie.utils.benchmark import run_iters  # noqa: E402

from gemm_pretiled import pretiled_array  # noqa: E402
from npue import tile_b  # noqa: E402

# (M, K, N). M = 4096 is a 1024^2 image through a 16x-per-token latent.
SHAPES = {
    # control: attn_block's recorded 0.95 ms / 2.2 TFLOPS (designs/attn_block/README.md)
    "ctrl_s2048": (2048, 256, 2048),
    # FLUX.2 klein 4B, hidden 3072, SwiGLU hidden 9216
    "o":          (4096, 3072, 3072),
    "qkv":        (4096, 3072, 9216),
    "mlp_down":   (4096, 9216, 3072),
    # Qwen-Image-2.1, hidden 4096, mlp 12288
    "q21_mlp_up": (4096, 4096, 12288),
}

# name -> (dtype_in, dtype_out, emulate_bfp16, c_bf16, (m, k, n))
PATHS = {
    "bf16":       ("bf16", "f32", False, False, (64, 64, 32)),
    "bf16-cbf16": ("bf16", "f32", False, True,  (64, 64, 32)),
    "bfp16":      ("bf16", "f32", True,  True,  (64, 64, 32)),
    "int8":       ("i8",   "i32", False, True,  (64, 64, 64)),
    # Wider n. If the core's two 32-bit input streams are the limit, a core
    # sustains at most 4*m*n/(m+n) MAC/cycle on 2-byte operands, so n 32 -> 48
    # should buy ~1.28x on bf16 and bfp16 alike.
    "bf16-n48":   ("bf16", "f32", False, True,  (64, 64, 48)),
    "bfp16-n48":  ("bf16", "f32", True,  True,  (64, 64, 48)),
}

N_COLS = 8
CHECK_ROWS = 128


def set_pmode(mode: str) -> None:
    rc = subprocess.run([r"C:\Windows\System32\AMD\xrt-smi.exe", "configure", "--pmode", mode],
                        cwd=r"C:\Windows\System32\AMD", capture_output=True).returncode
    print(f"NPU power mode: {mode}{'' if rc == 0 else ' (xrt-smi failed; mode unchanged)'}")


def operands(M, K, N, dtype_in, rng):
    if dtype_in == "i8":
        a = rng.integers(-127, 128, size=(M, K)).astype(np.int8)
        b = rng.integers(-127, 128, size=(K, N)).astype(np.int8)
    else:
        a = rng.standard_normal((M, K), dtype=np.float32).astype(bfloat16)
        b = (rng.standard_normal((K, N), dtype=np.float32) / np.sqrt(K)).astype(bfloat16)
    return a, b


def bench(name, shape, path, iters, warmup, rng, tg_depth=2):
    M, K, N = shape
    dtype_in, dtype_out, emulate, c_bf16, (m, k, n) = PATHS[path]
    if M % (m * 4) or K % k or N % (n * N_COLS):
        return dict(shape=name, path=path, skipped=f"does not tile by ({m * 4},{k},{n * N_COLS})")
    dt_in, dt_out = str_to_dtype(dtype_in), str_to_dtype(dtype_out)
    dt_c = str_to_dtype("bf16") if c_bf16 else dt_out

    a_np, b_np = operands(M, K, N, dtype_in, rng)
    A = iron.zeros((M, K), dtype=dt_in, device="npu")
    B = iron.zeros((K, N), dtype=dt_in, device="npu")
    C = iron.zeros(M * N, dtype=dt_c, device="npu")
    A[:] = a_np
    # Pre-tile B with the same tile_b() the shipped packers use.
    _, s, t = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=dt_in, output_dtype=dt_out,
                         b_col_maj=False, c_col_maj=False, use_chess=False,
                         emulate_bf16_mmul_with_bfp16=emulate, vectorized=True).mac_dims
    uview = {1: np.uint8, 2: np.uint16}[np.dtype(dt_in).itemsize]
    B[:] = tile_b(b_np.view(uview), k, n, s, t, order="k,n").view(dt_in).reshape(K, N)

    kw = dict(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=N_COLS,
              dtype_in_str=dtype_in, dtype_out_str=dtype_out,
              emulate_bf16_mmul_with_bfp16=emulate, rtp=True, tg_depth=tg_depth,
              c_bf16=c_bf16, trace_config=None)

    t0 = time.time()
    try:
        res = run_iters(pretiled_array, A, B, C, warmup=warmup, iters=iters, **kw)
    except Exception as e:  # compile or run failure is a result, not a crash
        first = next((ln for ln in str(e).splitlines() if ln.strip()), repr(e))
        return dict(shape=name, path=path, M=M, K=K, N=N, failed=first[:300])
    wall = time.time() - t0

    rows = rng.choice(M, size=min(CHECK_ROWS, M), replace=False)
    got = C.numpy().reshape(M, N)[rows].astype(np.float64)
    ref = a_np[rows].astype(np.float64) @ b_np.astype(np.float64)
    rel_fro = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    tol = 5e-2 if emulate else (1e-2 if c_bf16 else 5e-3)

    flop = 2.0 * M * K * N
    npu = res.npu
    out = dict(shape=name, path=path, M=M, K=K, N=N, tile=[m, k, n], tg_depth=tg_depth,
               npu_avg_ms=npu.avg_us / 1e3, npu_min_ms=npu.min_us / 1e3,
               tflops_min=flop / (npu.min_us * 1e-6) / 1e12,
               tflops_avg=flop / (npu.avg_us * 1e-6) / 1e12,
               rel_fro=rel_fro, tol=tol, pass_=rel_fro <= tol,
               build_and_run_s=wall, iters=iters)
    # DDR traffic of this design: A re-read per 256-wide N block, B per 256-row M block.
    in_sz, c_sz = np.dtype(dt_in).itemsize, np.dtype(dt_c).itemsize
    traffic = (M * K * in_sz * (N // (n * N_COLS)) + K * N * in_sz * (M // (m * 4))
               + M * N * c_sz)
    out["ddr_gb"] = traffic / 1e9
    out["ddr_gbps_at_min"] = traffic / (npu.min_us * 1e-6) / 1e9
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shapes", default=",".join(SHAPES), help=f"comma list of {list(SHAPES)}")
    ap.add_argument("--paths", default=",".join(PATHS), help=f"comma list of {list(PATHS)}")
    ap.add_argument("--tg-depth", type=int, default=2, help="task groups in flight; 2 is whisper_gemm's shipped value")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--pmode", default="turbo", help="set before, restored to 'performance' after; 'none' leaves it")
    ap.add_argument("--out", default=None, help="write results as JSON")
    args = ap.parse_args()

    iron.set_current_device(from_name("npu2", n_cols=None))
    rng = np.random.default_rng(7)
    if args.pmode != "none":
        set_pmode(args.pmode)

    results = []
    try:
        for path in args.paths.split(","):
            for name in args.shapes.split(","):
                r = bench(name, SHAPES[name], path, args.iters, args.warmup, rng, args.tg_depth)
                results.append(r)
                if "npu_min_ms" in r:
                    print(f"{path:<11} {name:<11} {r['M']}x{r['K']}x{r['N']:<6} "
                          f"min {r['npu_min_ms']:8.2f} ms  avg {r['npu_avg_ms']:8.2f} ms  "
                          f"{r['tflops_min']:6.2f} TFLOPS  ddr {r['ddr_gbps_at_min']:5.1f} GB/s  "
                          f"relfro {r['rel_fro']:.1e} {'PASS' if r['pass_'] else 'FAIL'}", flush=True)
                else:
                    print(f"{path:<11} {name:<11} {r.get('failed') or r.get('skipped')}", flush=True)
                if args.out:
                    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        if args.pmode != "none":
            set_pmode("performance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
