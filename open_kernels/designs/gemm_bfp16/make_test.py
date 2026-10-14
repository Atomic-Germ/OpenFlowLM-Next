r"""gemm_bfp16 harness fixture and check.

    python make_test.py make    --build <build dir> --out <dir> --M 256 --K 2560 --N 4096 [--m 64 --n 128] [--repeat 5]
    run_kernel.exe <dir>\run.cfg
    python make_test.py compare --out <dir>

make: random Gaussian activations and weights, both packed bfp16; writes run.cfg and meta.json.
compare: the dumped C against (a) the exact product of the decoded operands (kernel correctness: fp32 sum order only)
and (b) the product of the original floats (the bfp16 rounding of both operands, which is the route's accuracy).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pack  # noqa: E402


def make(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((a.M, a.K), dtype=np.float32)
    w = (rng.standard_normal((a.N, a.K), dtype=np.float32) * 0.05).astype(np.float32)
    pack.pack_a(x, a.m, a.k).tofile(out / "a.bin")
    pack.pack_b(w, a.n, a.k).tofile(out / "b.bin")
    np.save(out / "x.npy", x)
    np.save(out / "w.npy", w)
    json.dump(dict(M=a.M, K=a.K, N=a.N, m=a.m, k=a.k, n=a.n), open(out / "meta.json", "w"))
    build = Path(a.build).resolve()
    cfg = ["device", f"xclbin G {build / 'final.xclbin'}", f"kernelx k G {build / 'insts.bin'}",
           f"buf a {a.M * a.K // 8 * 9} a.bin", f"buf b {a.N * a.K // 8 * 9} b.bin", f"buf c {a.M * a.N * 4}"]
    cfg += ["run k b a c"] * a.repeat + [f"dump c c.bin {a.M * a.N * 4}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))


def compare(a) -> int:
    out = Path(a.out)
    meta = json.load(open(out / "meta.json"))
    M, K, N, m, k, n = (meta[x] for x in "MKNmkn")
    x, w = np.load(out / "x.npy"), np.load(out / "w.npy")
    c = pack.untile_c(np.fromfile(out / "c.bin", dtype=np.float32), M, N, m, n)
    xq = pack.unpack_a(np.fromfile(out / "a.bin", dtype=np.uint8), M, K, m, k).astype(np.float64)
    wq = pack.unpack_b(np.fromfile(out / "b.bin", dtype=np.uint8), N, K, n, k).astype(np.float64)
    ref_q = xq @ wq.T
    ref = x.astype(np.float64) @ w.astype(np.float64).T

    def rel(y, r):
        return np.linalg.norm(y - r) / np.linalg.norm(r)

    print(f"kernel vs decoded-operand product: rel_fro {rel(c, ref_q):.3e}  maxabs {np.abs(c - ref_q).max():.3e}")
    print(f"kernel vs exact float product:     rel_fro {rel(c, ref):.3e}")
    ok = rel(c, ref_q) < 1e-4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "compare"])
    ap.add_argument("--build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--K", type=int, default=2560)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()
    if a.mode == "make":
        make(a)
        return 0
    return compare(a)


if __name__ == "__main__":
    sys.exit(main())
