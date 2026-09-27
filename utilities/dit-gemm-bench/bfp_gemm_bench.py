r"""bfp_gemm_bench: mlir-aie's bf16 x bfp16 whole-array GEMMs at diffusion-transformer shapes.

Both designs take A as bf16 activations and B as bfp16ebs8 weights (one shared
exponent per 8 values along K, 1.125 bytes/value) and write C as bf16:

    wam  whole_array_mixed (wam/): the symmetric 64x64x64 design, stock
         aie2p/mm_bfp_mixed.cc kernel. mlir-aie CI runs it on npu2 with Peano.
    atbs the ATB dataflow below with the stock kernel on quarter tiles
         (atb/mm_atb_stock.cc) -- ATB's reuse without its chess-tuned kernel.
    atb  asymmetric tile buffering (atb/), 128x64x128 L1 tile. Upstream reports
         24.3 TFLOPS at 4096x4096x2048 on a Ryzen AI 9 HX 370 -- with chess,
         which is closed-source. Built with Peano it currently returns
         NaN/garbage (see atb/n32_core_atb.py); its timings mean nothing until
         that is fixed.

Each run builds the design, packs B the way that design's upstream host test
does, runs it through open_kernels/harness's run_kernel.exe, and checks C
against float64 on a sample of rows. Two error figures:

    vs_bf16   against A @ B with B exactly as given (what a model sees)
    vs_bfp    against A @ decode(bfp16(B)) (the kernel's own arithmetic)

Run from a shell where C:\dev\mlir-aie\iron_env.ps1 has been dot-sourced:

    python utilities\dit-gemm-bench\bfp_gemm_bench.py --design wam --shape 4096x3072x9216
    python utilities\dit-gemm-bench\bfp_gemm_bench.py --design atb --shape 4096x4096x2048 --data int
"""

from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HARNESS = REPO / "open_kernels" / "harness" / "out" / "run_kernel.exe"
sys.path.insert(0, str(HERE / "atb"))
sys.path.insert(0, str(HERE / "wam"))


# ---------------------------------------------------------------- bfp16ebs8

def bfp16ebs8_encode(x: np.ndarray, rounding: str = "trunc") -> np.ndarray:
    """float32 [n*8] -> uint8 [n*9]: per block of 8, one shared exponent then 8
    two's-complement int8 mantissas. "trunc" is helper.h's floatToBfp16 (the
    hardware's own conversion rounds the same way); "nearest" rounds half away
    from zero, which the device decodes identically -- the choice is the host's."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).reshape(-1, 8)
    exp = ((bits >> 23) & 0xFF).astype(np.int64)
    max_exp = exp.max(axis=1, keepdims=True)
    mant = (bits & 0x7FFFFF).astype(np.int64) | np.where(exp != 0, 1 << 23, 0)
    signed = np.where(bits >> 31 == 1, -mant, mant)
    shift = 17 + np.minimum(max_exp - exp, 31)
    if rounding == "nearest":
        half = np.left_shift(np.int64(1), shift - 1)
        vals = np.floor_divide(signed + half, np.left_shift(np.int64(1), shift))
        vals = np.clip(vals, -128, 127)
    else:
        vals = np.floor_divide(signed, np.left_shift(np.int64(1), shift))
    out = np.empty((bits.shape[0], 9), dtype=np.uint8)
    out[:, 0] = max_exp[:, 0].astype(np.uint8)
    out[:, 1:] = vals.astype(np.int8).view(np.uint8)
    return out.reshape(-1)


def bfp16ebs8_decode(b: np.ndarray) -> np.ndarray:
    blk = b.reshape(-1, 9)
    scale = np.ldexp(1.0, blk[:, :1].astype(np.int64) - 127) / 64.0
    return (blk[:, 1:].view(np.int8).astype(np.float64) * scale).reshape(-1)


# ------------------------------------------------------------ design: atb

ATB_TILE = (128, 64, 128)   # the in-tree microkernel's tile; not overridable


def atb_layout(b: np.ndarray) -> np.ndarray:
    """[K, N] -> gemm_atb_layout.h's layout_transpose_L1_1x2_8x8block order:
    L1 tiles column-major; inside a tile, 1x2 super-blocks of 8x8 column-major
    sub-blocks. Consecutive runs of 8 are 8 K-values of one output column."""
    _, k, n = ATB_TILE
    K, N = b.shape
    t = b.reshape(K // k, k // 8, 8, N // n, n // 16, 2, 8)
    #          [L1r,   sbr,   row, L1c,  sbc2,   bis, col]
    return t.transpose(3, 0, 4, 1, 5, 6, 2).reshape(-1)


def atb_unlayout(flat: np.ndarray, K: int, N: int) -> np.ndarray:
    _, k, n = ATB_TILE
    t = flat.reshape(N // n, K // k, n // 16, k // 8, 2, 8, 8)
    return t.transpose(1, 3, 6, 0, 2, 4, 5).reshape(K, N)


def atb_pack(b: np.ndarray, rounding: str, k: int):
    K, N = b.shape
    packed = bfp16ebs8_encode(atb_layout(b), rounding)
    return packed, atb_unlayout(bfp16ebs8_decode(packed), K, N)


def atb_check(M, K, N, tile):
    m, _, n = ATB_TILE
    if M % (m * 4) or K % 512 or N % (n * 8) or ((M // m) * (N // n)) % 128:
        return "atb needs M%512, K%512, N%1024 and (M/128)*(N/128)%128 == 0"
    return None


def atb_build(M, K, N, out: Path, args):
    from n32_core_atb import n32_core_gemm
    m, k, n = ATB_TILE
    n32_core_gemm.specialize(M=M, K=K, N=N, m=m, k=k, n=n, use_chess=args.chess,
                             stack_size=args.stack).compile(
        xclbin_path=str(out / "final.xclbin"), inst_path=str(out / "insts.bin"))


# ----------------------------------------------------------- design: atbs
# The ATB dataflow driving mlir-aie's stock bf16 x bfp16 kernel (atb/mm_atb_stock.cc).

def atbs_layout(b: np.ndarray) -> np.ndarray:
    """[K, N] -> the stock kernel's per-L1-tile B order, L1 tiles column-major as
    ATB's B fill walks them: for each 128-wide n tile, for each 64-deep k tile,
    [n-block 16][k-block 8][8 n rows][8 k values] -- a bfp block is 8 k values."""
    _, k, n = ATB_TILE
    K, N = b.shape
    t = b.reshape(K // k, k // 8, 8, N // n, n // 8, 8)
    #          [L1r,   kb,   kin, L1c,  nb,     nin]
    return t.transpose(3, 0, 4, 1, 5, 2).reshape(-1)


def atbs_unlayout(flat: np.ndarray, K: int, N: int) -> np.ndarray:
    _, k, n = ATB_TILE
    t = flat.reshape(N // n, K // k, n // 8, k // 8, 8, 8)
    return t.transpose(1, 3, 5, 0, 2, 4).reshape(K, N)


def atbs_pack(b: np.ndarray, rounding: str, k: int):
    K, N = b.shape
    packed = bfp16ebs8_encode(atbs_layout(b), rounding)
    return packed, atbs_unlayout(bfp16ebs8_decode(packed), K, N)


def atbs_build(M, K, N, out: Path, args):
    from n32_core_atb import n32_core_gemm
    m, k, n = ATB_TILE
    n32_core_gemm.specialize(M=M, K=K, N=N, m=m, k=k, n=n, use_chess=False,
                             stack_size=args.stack, kernel="stock").compile(
        xclbin_path=str(out / "final.xclbin"), inst_path=str(out / "insts.bin"))


# ------------------------------------------------------------ design: wam

WAM_TILE = (64, 64, 64)   # upstream default; --tile overrides (m%16, k%8, n%16)


def wam_pack(b: np.ndarray, rounding: str, k: int):
    """mixed_test.cpp: B is column-major (B^T [N, K] row-major), bfp-encoded in
    blocks of 8 along K, then shuffleMatrixForBfp16ebs8(K, N, k, N): within each
    k-wide column strip, every 8-row x 8-value sub-tile becomes 72 contiguous
    bytes, sub-tiles in row-major order over the strip."""
    K, N = b.shape
    enc = bfp16ebs8_encode(np.ascontiguousarray(b.T), rounding).reshape(N // 8, 8, K // k, k // 8, 9)
    #                                                         [sy,   i,  strip, sx,   j]
    packed = enc.transpose(2, 0, 3, 1, 4).reshape(K // k, N, k // 8 * 9).transpose(1, 0, 2).reshape(-1)
    b_q = bfp16ebs8_decode(enc.reshape(-1)).reshape(N, K).T   # decode the un-shuffled form
    return np.ascontiguousarray(packed), b_q


def wam_check(M, K, N, tile):
    m, k, n = tile
    if M % (m * 4) or K % k or N % (n * 8):
        return f"wam needs M%{m * 4}, K%{k}, N%{n * 8}"
    return None


def wam_build(M, K, N, out: Path, args):
    from whole_array_mixed import whole_array_mixed
    m, k, n = args.tile
    whole_array_mixed.specialize(M=M, K=K, N=N, m=m, k=k, n=n, n_aie_cols=8).compile(
        xclbin_path=str(out / "final.xclbin"), inst_path=str(out / "insts.bin"))


DESIGNS = {
    "atb": (atb_check, atb_build, atb_pack),
    "atbs": (atb_check, atbs_build, atbs_pack),
    "wam": (wam_check, wam_build, wam_pack),
}


# ------------------------------------------------------------------ driver

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--design", choices=sorted(DESIGNS), default="wam")
    ap.add_argument("--shape", default="4096x3072x9216", help="MxKxN")
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--tile", default=None, help="wam: per-core m,k,n (default 64,64,64)")
    ap.add_argument("--chess", action="store_true", help="atb: build with chess (closed-source; must be installed)")
    ap.add_argument("--stack", type=lambda v: int(v, 0), default=0xF00,
                    help="atb: per-core stack bytes (Peano's matmul frame is 0xE40)")
    ap.add_argument("--rounding", choices=["trunc", "nearest"], default="trunc",
                    help="how the host rounds B into bfp16")
    ap.add_argument("--data", choices=["normal", "int"], default="normal",
                    help="int = upstream's test data, integers in [-4, 3], exact in bfp16")
    ap.add_argument("--out", default=None, help="work dir (default: build_<design>_<shape> beside this script)")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--check-rows", type=int, default=256)
    args = ap.parse_args()

    check, build, pack = DESIGNS[args.design]
    M, K, N = (int(v) for v in args.shape.lower().split("x"))
    if args.design in ("atb", "atbs"):
        args.tile = ATB_TILE
    else:
        args.tile = tuple(int(v) for v in args.tile.split(",")) if args.tile else WAM_TILE
    if (why := check(M, K, N, args.tile)):
        sys.exit(why)
    tag = f"{args.design}{'_chess' if args.chess else ''}_t{'x'.join(map(str, args.tile))}"
    work = Path(args.out) if args.out else HERE / f"build_{tag}_{M}x{K}x{N}"
    work.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        import aie.iron as iron
        from aie.iron.device import from_name
        iron.set_current_device(from_name("npu2", n_cols=None))
        print(f"building {tag} {M}x{K}x{N} ...", flush=True)
        # Always a fresh project (open_kernels/build_design.py's rule): aiecc keeps
        # copies of kernel sources and objects in final.prj and skips recompiling
        # them, so an edited kernel is otherwise silently ignored.
        shutil.rmtree(work / "final.prj", ignore_errors=True)
        build(M, K, N, work, args)

    rng = np.random.default_rng(7)
    if args.data == "int":
        a = rng.integers(-4, 4, size=(M, K)).astype(np.float32).astype(bfloat16)
        b = rng.integers(-4, 4, size=(K, N)).astype(np.float32)
    else:
        a = rng.standard_normal((M, K), dtype=np.float32).astype(bfloat16)
        b = (rng.standard_normal((K, N), dtype=np.float32) / np.sqrt(K)).astype(bfloat16).astype(np.float32)
    b_packed, b_q = pack(b, args.rounding, args.tile[1])
    assert b_packed.size == K * N * 9 // 8
    a.tofile(work / "a.bin")
    b_packed.tofile(work / "b.bin")

    c_bytes = M * N * 2
    cfg = ["device", "xclbin G final.xclbin", "kernelx k G insts.bin",
           f"buf a {M * K * 2} a.bin", f"buf b {b_packed.size} b.bin", f"buf c {c_bytes}"]
    cfg += ["run k a b c"] * args.runs
    cfg += [f"dump c c.bin {c_bytes}", ""]
    (work / "run.cfg").write_text("\n".join(cfg))

    p = subprocess.run([str(HARNESS), "run.cfg"], cwd=work, capture_output=True, text=True)
    times = [float(m.group(2)) for m in re.finditer(r"^run \S+ \[\d+ bufs\] -> state (\d+) \((\d+\.\d+) ms\)",
                                                    p.stdout, re.M) if m.group(1) == "4"]
    if p.returncode != 0 or len(times) != args.runs:
        print(p.stdout[-2000:], p.stderr[-2000:])
        sys.exit(f"harness failed (exit {p.returncode}, {len(times)}/{args.runs} runs completed)")

    with np.errstate(invalid="ignore", over="ignore"):
        c = np.fromfile(work / "c.bin", dtype=bfloat16).reshape(M, N).astype(np.float64)
        rows = rng.choice(M, size=min(args.check_rows, M), replace=False)
        a64 = a[rows].astype(np.float64)
        err = lambda ref: float(np.linalg.norm(c[rows] - ref) / np.linalg.norm(ref))
        e_bf16, e_bfp = err(a64 @ b.astype(np.float64)), err(a64 @ b_q)
    nonfinite = float((~np.isfinite(c)).mean())

    warm = times[1:] or times
    tmin, tmed = min(warm), statistics.median(warm)
    flop = 2.0 * M * K * N
    print(f"{tag:<18} {M}x{K}x{N:<6} cold {times[0]:7.2f} ms  min {tmin:7.2f} ms  med {tmed:7.2f} ms  "
          f"{flop / (tmin * 1e-3) / 1e12:6.2f} TFLOPS (min)  {flop / (tmed * 1e-3) / 1e12:6.2f} (med)  "
          f"rel_fro vs_bf16 {e_bf16:.2e}  vs_bfp {e_bfp:.2e}  nonfinite {nonfinite:.1%}  "
          f"[{args.data}, {args.rounding}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
