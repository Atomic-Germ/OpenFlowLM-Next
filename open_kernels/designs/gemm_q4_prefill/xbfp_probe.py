r"""GQP_XBFP against the production bf16-activation GEMM on one shape (workstream B2): same t2
weights, same activations -- bf16 tiles for the production build, host-made bfp16 tiles for
the XBFP build -- timed interleaved and compared bit for bit.

    python xbfp_probe.py gen
    python xbfp_probe.py run [--rounds 4]        (npu lock; timing lock for numbers you report)
    python xbfp_probe.py check

The host's bf16 -> bfp16ebs8 conversion is ../bfp_cvt/bfp16_host.py's, the model bfp_cvt's
hardware probe confirmed byte for byte; with it the two builds must give the same y.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent / "bfp_cvt"))
sys.path.insert(0, str(HERE.parents[1] / "harness"))
from q4_1_pack import pack_q4_1_pool, random_q4_1_blocks  # noqa: E402
from t2_pack import t2_from_q4_1_pool  # noqa: E402
from bfp16_host import bf16_to_bfp16_v64  # noqa: E402
from bench import RUN  # noqa: E402

N, K, T, KT, TN = 5120, 6144, 256, 128, 32
DRIVER = Path(r"C:\code\openflowlm-next\open_kernels\harness\out\run_kernel.exe")
BUILDS = {"t2ky": "build_n5120_k6144_t256_t2ky", "xbfp": "build_n5120_k6144_t256_xbfp",
          # timing-only ablations (GQP_NULL_MM / GQP_NULL_DEQUANT; their y is garbage)
          "t2ky_nmnd": "build_n5120_k6144_t256_t2ky_nmnd", "xbfp_nm": "build_n5120_k6144_t256_xbfp_nm",
          "xbfp_nd": "build_n5120_k6144_t256_xbfp_nd", "xbfp_nmnd": "build_n5120_k6144_t256_xbfp_nmnd"}


def tile_bf16(bits: np.ndarray) -> np.ndarray:
    """tile_x at k 128: [K/kt][T/32][kt/8 si][4 ti][8 s][8 t]"""
    x = bits.reshape(T // TN, 4, 8, K // KT, KT // 8, 8)           # [nb, ti, t, kb, si, s]
    return np.ascontiguousarray(x.transpose(3, 0, 4, 1, 5, 2)).reshape(-1)


def tile_bfp(bits: np.ndarray) -> np.ndarray:
    """the XBFP activation: [K/kt][T/32][token block pair j/2][kt/8 k blocks i][j % 2] 72-byte
    block vectors (gemm_bfp_mm.cc's B order), each the conversion of 64 lanes [8 tokens t][8 k kk]
    = x[32 nb + 8 j + t][kt kb + 8 i + kk]"""
    x = bits.reshape(T // TN, 2, 2, 8, K // KT, KT // 8, 8)        # [nb, jp, jm, t, kb, i, kk]
    lanes = np.ascontiguousarray(x.transpose(4, 0, 1, 5, 2, 3, 6)).reshape(-1, 64)
    return bf16_to_bfp16_v64(lanes).reshape(-1)


def gen() -> None:
    rng = np.random.default_rng(5)
    nb = K // 32
    blocks = random_q4_1_blocks(N, K, rng)
    d = np.repeat((rng.random((N, nb // 4), np.float32) * 0.02 + 1e-3).astype(np.float16), 4, axis=1)
    q = rng.integers(0, 3, (N, nb, 32), dtype=np.uint8)
    blocks[..., 0:2] = d.view(np.uint8).reshape(N, nb, 2)
    blocks[..., 2:4] = (-d).view(np.uint8).reshape(N, nb, 2)
    blocks[..., 4:20] = q[..., :16] | (q[..., 16:] << 4)
    w = t2_from_q4_1_pool(pack_q4_1_pool(blocks, rs=2)).reshape(-1, 2560)[:, :2176]
    (HERE / "w_xbfp_t2.bin").write_bytes(np.ascontiguousarray(w).tobytes())
    # activations shaped like the engine's: Hadamard-rotated rows are close to Gaussian
    x = rng.standard_normal((T, K)).astype(np.float32)
    bits = (x.view(np.uint32) + 0x7FFF + ((x.view(np.uint32) >> 16) & 1)) >> 16
    bits = bits.astype(np.uint16)
    (HERE / "x_xbfp_bf16.bin").write_bytes(tile_bf16(bits).tobytes())
    (HERE / "x_xbfp_bfp.bin").write_bytes(tile_bfp(bits).tobytes())
    yb = N * T * 4
    for v, b in BUILDS.items():
        xf = "x_xbfp_bfp.bin" if v.startswith("xbfp") else "x_xbfp_bf16.bin"
        xs = (HERE / xf).stat().st_size
        cfg = ["device", f"xclbin G {b}/final.xclbin", f"kernelx k G {b}/insts.bin", f"buf w {w.size} w_xbfp_t2.bin",
               f"buf x {xs} {xf}", f"buf y {yb}"] + ["run k w x y"] * 8 + [f"dump y y_xbfp_{v}.bin {yb}", ""]
        (HERE / f"run_xbfp_{v}.cfg").write_text("\n".join(cfg), newline="\n")
    print("wrote w_xbfp_t2.bin, x_xbfp_{bf16,bfp}.bin, run_xbfp_{t2ky,xbfp}.cfg")


def run(rounds: int, variants: list[str]) -> None:
    times = {v: [] for v in variants}
    for r in range(rounds):
        line = []
        for v in variants:
            p = subprocess.run([str(DRIVER), f"run_xbfp_{v}.cfg"], cwd=HERE, capture_output=True, text=True)
            ts = [float(m.group(3)) for m in map(RUN.match, p.stdout.splitlines()) if m]
            bad = [l for l in p.stdout.splitlines() if RUN.match(l) and int(RUN.match(l).group(2)) != 4]
            if p.returncode or not ts or bad:
                print(f"  {v}: exit {p.returncode} {bad[:2]}\n{p.stdout[-400:]}")
            ts = ts[1:] or ts
            if ts:
                times[v].append(min(ts))
                line.append(f"{v} {min(ts):.3f}")
        print(f"round {r}: " + "  ".join(line), flush=True)
    for v, ts in times.items():
        if ts:
            print(f"{v:5s} min {min(ts):.3f} ms  median of round minima {statistics.median(ts):.3f} ms")


def check() -> None:
    a = np.fromfile(HERE / "y_xbfp_t2ky.bin", np.uint32)
    b = np.fromfile(HERE / "y_xbfp_xbfp.bin", np.uint32)
    same = a == b
    print(f"y: {same.mean() * 100:.4f} % of {a.size} values bit-identical")
    if not same.all():
        fa, fb = a.view(np.float32).astype(np.float64), b.view(np.float32).astype(np.float64)
        rel = np.linalg.norm(fa - fb) / np.linalg.norm(fa)
        i = np.argwhere(~same)[:5].ravel()
        print(f"  rel_fro {rel:.3e}; first differing [t, n]: {[(int(j) // N, int(j) % N) for j in i]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("gen", "run", "check"))
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--variants", nargs="*", default=["t2ky", "xbfp"])
    a = ap.parse_args()
    {"gen": gen, "run": lambda: run(a.rounds, a.variants), "check": check}[a.what]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
