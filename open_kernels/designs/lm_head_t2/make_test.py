r"""Test vectors for lm_head_t2: random ternary weights packed as the head's t2 pool (std_perm's
64-row band order, unpadded 2176 B chunks -- what pools.cpp `t2_perm` writes with chunk_bytes
2176), one bf16 activation, and the fp64 reference of W @ (H(x)/32 per 1024 block) with the
rotated input rounded to bf16, as the prep feeds the table (designs/gemv_t2 probe 0a's rule).

    python make_test.py [--n 248320] [--k 5120] [--out DIR] [--runs 20]

Writes w.bin, x.bin, ref.bin and run.cfg / run_null.cfg (absolute paths) into DIR (default:
here). The build the cfg names is designs/lm_head_t2/build_<n>_k<k> (the export's build dir) and,
for run_null.cfg, build_<n>_k<k>_null (LMHEAD_NULL=1, DMA only). Then:
    run_kernel.exe run.cfg  ->  y.bin;   python make_test.py --compare [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from t2_pack import QK, random_pq2_codes  # noqa: E402

CH = 2176


def fwht_blocks(x: np.ndarray, block: int = 1024) -> np.ndarray:
    """Normalized Sylvester Walsh-Hadamard per `block` of x, fp64."""
    y = x.astype(np.float64).reshape(-1, block).copy()
    h = 1
    while h < block:
        y = y.reshape(-1, block // (2 * h), 2, h)
        y = np.stack([y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]], axis=2)
        h *= 2
    return y.reshape(-1) / np.sqrt(block)


def pack_head(codes: np.ndarray, s: np.ndarray) -> np.ndarray:
    """[n, k] ternary -> the head's t2 pool, vectorised: band b, chunk c = 2 * ktile + half."""
    n, k = codes.shape
    nkt = k // 256
    s16 = s.astype(np.float32).astype(bfloat16).view(np.uint16)                 # [n, k/128]
    # [bands, half, 32 r, ktiles, 256 k] -> chunk order [bands, ktile, half]
    q = codes.reshape(n // 64, 2, 32, nkt, 256).transpose(0, 3, 1, 2, 4)         # [b, kt, h, r, k]
    # r = 8j + p, k = 8 k8 + kk  ->  byte [k8, kk, p] holds field j
    q = q.reshape(n // 64, nkt, 2, 4, 8, 32, 8).transpose(0, 1, 2, 5, 6, 4, 3)    # [b, kt, h, k8, kk, p, j]
    code = (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).astype(np.uint8)
    sc = s16.reshape(n // 64, 2, 32, nkt, 2).transpose(0, 3, 1, 4, 2)            # [b, kt, h, g, r]
    out = np.empty((n // 64, nkt, 2, CH), np.uint8)
    out[..., :128] = np.ascontiguousarray(sc).view(np.uint8).reshape(n // 64, nkt, 2, 128)
    out[..., 128:] = code.reshape(n // 64, nkt, 2, 2048)
    return out.reshape(-1)


def reference(codes: np.ndarray, s: np.ndarray, xr: np.ndarray, rows: int = 4096) -> np.ndarray:
    y = np.empty(codes.shape[0], np.float64)
    xf = xr.astype(np.float64)
    for r0 in range(0, codes.shape[0], rows):
        c = codes[r0:r0 + rows].astype(np.float64)
        sc = np.repeat(s[r0:r0 + rows].astype(np.float32).astype(bfloat16).astype(np.float64), QK, axis=1)
        y[r0:r0 + rows] = (c * sc - sc) @ xf
    return y.astype(np.float32)


def compare(out: Path, tag: str) -> int:
    ref = np.fromfile(out / "ref.bin", np.float32).astype(np.float64)
    got = np.fromfile(out / f"y{tag}.bin", np.float32).astype(np.float64)
    if tag == "_null":
        print(f"null: zeros={not got.any()}")
        return 0
    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    ok = cos > 0.9999999 and rel < 1e-4 and np.isfinite(got).all()
    print(f"{'PASS' if ok else 'FAIL'} n={len(ref)} cos={cos:.9f} maxrel={rel:.2e} "
          f"argmax {int(np.argmax(got))} vs {int(np.argmax(ref))}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=248320)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = a.out.resolve()
    if a.compare:
        return compare(out, a.tag)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    codes, s = random_pq2_codes(a.n, a.k, rng)
    x = rng.standard_normal(a.k).astype(np.float32).astype(bfloat16)
    xr = fwht_blocks(x.astype(np.float64)).astype(np.float32).astype(bfloat16)
    w = pack_head(codes, s)
    ref = reference(codes, s, xr)
    (out / "w.bin").write_bytes(w.tobytes())
    (out / "x.bin").write_bytes(x.tobytes())
    (out / "ref.bin").write_bytes(ref.tobytes())
    for tag, sfx in (("", ""), ("_null", "_null")):
        b = (HERE / f"build_{a.n}_k{a.k}{sfx}").as_posix()
        cfg = ["device", f"xclbin G {b}/final.xclbin", f"kernelx k G {b}/insts.bin",
               f"buf w {w.nbytes} {(out / 'w.bin').as_posix()}", f"buf x {x.nbytes} {(out / 'x.bin').as_posix()}",
               f"buf y {ref.nbytes}"] + ["run k w x y"] * a.runs + [f"dump y {(out / f'y{tag}.bin').as_posix()} {ref.nbytes}", ""]
        (out / f"run{tag}.cfg").write_text("\n".join(cfg), newline="\n")
    print(f"N={a.n} K={a.k}: w {w.nbytes} B, ref absmax {np.abs(ref).max():.4g} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
