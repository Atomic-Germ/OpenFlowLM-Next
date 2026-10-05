r"""Read the hardware's bf16 -> bfp16 activation rounding out of the production t2 GEMM, and
check host roundings against it (workstream B2 of the Bonsai 2 round-2 plan).

The bfp16 GEMM (GQP_BFP16, mm.cc's AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16) converts each
8 x 8 activation sub-tile, transposed so a block is 8 consecutive k of one token, with
to_v64bfp16ebs8: one shared 8-bit exponent and an 8-bit mantissa per value. With an identity
weight (W[n][k] = 1 iff k == n, exact in t2 and in bfp16) every output is one converted
activation, y[t][n] = bfp16(x[t][k = n]), so the device's rounding can be compared value by
value with a host model.

    python bfp_probe.py gen   [--build build_n5120_k6144_t256_t2ky]
    <run_kernel.exe> run_bfpid.cfg            (npu lock; probe.py's driver)
    python bfp_probe.py check

gen writes w_bfpid_t2.bin (t2 chunks at the 2176 B production stride), x_bfpid.bin (tile_x at
k 128) and run_bfpid.cfg; check reads y_bfpid.bin ([T, N] fp32, a GQP_YT build).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from q4_1_pack import pack_q4_1_pool, random_q4_1_blocks  # noqa: E402
from t2_pack import t2_from_q4_1_pool  # noqa: E402

N, K, T = 5120, 6144, 256
T2_BODY = 2176
DRIVER = Path(r"C:\code\openflowlm-next\open_kernels\harness\out\run_kernel.exe")


def make_x(rng: np.random.Generator) -> np.ndarray:
    """[T, K] bf16 bits. Blocks of 8 k get a spread of exponents (0..12 below the block max) so
    small values lose bits to the shared exponent, random signs, and a share of exact ties
    (the dropped bits exactly half an ulp of the kept mantissa) to tell the rounding modes apart."""
    sign = rng.integers(0, 2, (T, K), dtype=np.uint16) << 15
    base_e = rng.integers(110, 140, (T, K // 8, 1), dtype=np.int32)          # the block's scale
    drop = rng.integers(0, 13, (T, K // 8, 8), dtype=np.int32)
    e = np.clip(base_e - drop, 1, 254).reshape(T, K).astype(np.uint16)
    frac = rng.integers(0, 128, (T, K), dtype=np.uint16)
    bits = sign | (e << 7) | frac
    # every 5th block: force exact ties for a block whose max exponent is E, at exponent E - d
    # the mantissa keeps 8 - d - 1 fraction bits (if the model is right); set the next bit, clear the rest
    for t in range(T):
        for kb in range(0, K // 8, 5):
            blk = bits[t, kb * 8:(kb + 1) * 8]
            E = int(((blk >> 7) & 0xFF).max())
            blk[0] = (blk[0] & 0x807F) | (E << 7)                 # the block max sits at k 0
            for i in range(1, 8):
                d = int(rng.integers(1, 7))
                keep = max(0, 6 - d)                      # fraction bits that survive a 7-bit magnitude
                f = (int(rng.integers(0, 1 << keep)) << (7 - keep)) if keep else 0
                f |= 1 << (6 - keep) if keep < 7 else 0
                blk[i] = (blk[i] & 0x8000) | ((E - d) << 7) | (f & 0x7F)
            bits[t, kb * 8:(kb + 1) * 8] = blk
    return bits


def tile_x(bits: np.ndarray, tk: int = 128) -> np.ndarray:
    """block_host.cpp tile_x: [T, K] -> [K/tk][T/32][tk/8][4][8 s][8 t] (k, n order, 8x8 MAC sub-tiles)."""
    x = bits.reshape(T // 32, 4, 8, K // tk, tk // 8, 8)          # [nb, ti, t, kb, si, s]
    return np.ascontiguousarray(x.transpose(3, 0, 4, 1, 5, 2)).reshape(-1)


def gen(build: str) -> None:
    rng = np.random.default_rng(7)
    nb = K // 32
    blocks = random_q4_1_blocks(N, K, rng)
    d = np.full((N, nb), 1.0, np.float16)
    q = np.ones((N, nb, 32), np.uint8)
    for n in range(N):                                        # W[n][n] = 1: q = 2 (d * 2 - d)
        q[n, n // 32, n % 32] = 2
    blocks[..., 0:2] = d.view(np.uint8).reshape(N, nb, 2)
    blocks[..., 2:4] = (-d).view(np.uint8).reshape(N, nb, 2)
    blocks[..., 4:20] = q[..., :16] | (q[..., 16:] << 4)
    t2 = t2_from_q4_1_pool(pack_q4_1_pool(blocks, rs=2)).reshape(-1, 2560)[:, :T2_BODY]
    (HERE / "w_bfpid_t2.bin").write_bytes(np.ascontiguousarray(t2).tobytes())
    bits = make_x(rng)
    bits.astype(np.uint16).tofile(HERE / "x_bfpid_bits.bin")
    xt = tile_x(bits)
    (HERE / "x_bfpid.bin").write_bytes(xt.astype(np.uint16).tobytes())
    yb = N * T * 4
    cfg = ["device", f"xclbin G {build}/final.xclbin", f"kernelx k G {build}/insts.bin",
           f"buf w {t2.size} w_bfpid_t2.bin", f"buf x {xt.size * 2} x_bfpid.bin", f"buf y {yb}",
           "run k w x y", "run k w x y", f"dump y y_bfpid.bin {yb}", ""]
    (HERE / "run_bfpid.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"wrote w_bfpid_t2.bin ({t2.size} B), x_bfpid.bin, run_bfpid.cfg -> {build}")


def bf16_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def host_bfp16(bits: np.ndarray, mode: str, mbits: int = 8) -> np.ndarray:
    """One model of to_v64bfp16ebs8 on [T, K] bf16 bits, blocks of 8 along k: E = the block's
    largest exponent (+1 if `mode` ends in '+'), each value's significand shifted to E with
    mbits - 1 magnitude bits and a sign, rounded by `mode` (rne, rna = half away, trunc = toward
    zero, floor), saturated. Returns the represented values as float64."""
    x = bf16_to_f32(bits).astype(np.float64).reshape(T, K // 8, 8)
    e = ((bits.reshape(T, K // 8, 8) >> 7) & 0xFF).astype(np.int64)
    E = e.max(axis=2, keepdims=True)
    bump = mode.endswith("+")
    m = mode.rstrip("+")
    # the max element's significand 1.fffffff = [128, 256) * 2^(E - 127 - 7); keep mbits - 1 bits
    lsb = np.ldexp(1.0, (E - 127 - 7 + (8 - (mbits - 1)) + (1 if bump else 0)).astype(np.int64))
    v = x / lsb
    if m == "rne":
        r = np.round(v)                                   # numpy rounds half to even
    elif m == "rna":
        r = np.sign(v) * np.floor(np.abs(v) + 0.5)
    elif m == "trunc":
        r = np.trunc(v)
    elif m == "floor":
        r = np.floor(v)
    else:
        raise ValueError(m)
    lim = (1 << (mbits - 1)) - 1
    r = np.clip(r, -lim - 1, lim)
    return (r * lsb).reshape(T, K)


def check() -> None:
    bits = np.fromfile(HERE / "x_bfpid_bits.bin", np.uint16).reshape(T, K)
    y = np.fromfile(HERE / "y_bfpid.bin", np.float32).reshape(T, N).astype(np.float64)
    xs = bits[:, :N]
    exact = bf16_to_f32(xs).astype(np.float64)
    print(f"device == the bf16 input exactly: {np.mean(y == exact) * 100:.2f} % of values")
    for mode in ("rne", "rna", "trunc", "floor", "rne+", "rna+", "trunc+"):
        h = host_bfp16(bits, mode)[:, :N]
        ok = y == h
        print(f"host {mode:7s}: {ok.mean() * 100:8.4f} % identical, {np.sum(~ok)} differ")
        if 0 < np.sum(~ok) < 20 or (mode == "rne" and np.sum(~ok)):
            idx = np.argwhere(~ok)[:5]
            for t, n in idx:
                blk = bits[t, n // 8 * 8:n // 8 * 8 + 8]
                print(f"   t {t} k {n}: x {exact[t, n]:.6g} dev {y[t, n]:.8g} host {h[t, n]:.8g} "
                      f"block exps {list(((blk >> 7) & 0xFF).astype(int))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("gen", "check"))
    ap.add_argument("--build", default="build_n5120_k6144_t256_t2ky")
    a = ap.parse_args()
    gen(a.build) if a.what == "gen" else check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
