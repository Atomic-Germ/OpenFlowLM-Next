r"""gemm_bfp16 harness fixture and check.

    python make_test.py make    --build <build dir> --out <dir> --M 256 --K 2560 --N 4096 [--m 64 --n 128] [--repeat 5]
    run_kernel.exe <dir>\run.cfg
    python make_test.py compare --out <dir>

make: random Gaussian activations and weights, both packed bfp16; writes run.cfg and meta.json.
compare: the dumped C against (a) the exact product of the decoded operands (kernel correctness: fp32 sum order only)
and (b) the product of the original floats (the bfp16 rounding of both operands, which is the route's accuracy).

    python make_test.py make-glu    --build <glu build> --h-build <h build> --out <dir> [--K 2560 --ff 8192 --N 3072]
    python make_test.py compare-glu --out <dir>

make-glu / compare-glu (OPEN-PREFILL-BFP16-FFN): the gate|up GEMM with the SwiGLU epilogue into the H operand, then
the down GEMM reading H, in one hardware context. H against silu(g) * u of the decoded operands encoded by the host
(the bytes, and the decoded error beside the encode's own), and y against the decoded-H x decoded-down product.
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


def make_glu(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    M, K, ff, N = a.M, a.K, a.ff, a.N
    x = rng.standard_normal((M, K), dtype=np.float32)
    gate = (rng.standard_normal((ff, K), dtype=np.float32) * 0.05).astype(np.float32)
    up = (rng.standard_normal((ff, K), dtype=np.float32) * 0.05).astype(np.float32)
    down = (rng.standard_normal((N, ff), dtype=np.float32) * 0.02).astype(np.float32)
    pack.pack_a(x).tofile(out / "a.bin")
    pack.pack_glu_b(gate, up).tofile(out / "bgu.bin")
    pack.pack_b(down).tofile(out / "bd.bin")
    json.dump(dict(M=M, K=K, ff=ff, N=N), open(out / "meta.json", "w"))
    g, h = Path(a.build).resolve(), Path(a.h_build).resolve()
    h_bytes = (M // 64) * (2 * ff // 128) * pack.H_SLOT_WORDS * 4
    cfg = ["device", f"xclbin G {g / 'final.xclbin'}", f"kernelx kg G {g / 'insts.bin'}", f"kernelx kd G {h / 'insts.bin'}",
           f"buf a {M * K // 8 * 9} a.bin", f"buf bgu {2 * ff * K // 8 * 9} bgu.bin", f"buf bd {N * ff // 8 * 9} bd.bin",
           f"buf h {h_bytes}", f"buf y {M * N * 4}"]
    cfg += ["run kg bgu a h"] * a.repeat + ["run kd bd h y"] * a.repeat
    cfg += [f"dump h h.bin {h_bytes}", f"dump y y.bin {M * N * 4}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))


def compare_glu(a) -> int:
    out = Path(a.out)
    meta = json.load(open(out / "meta.json"))
    M, K, ff, N = (meta[x] for x in ("M", "K", "ff", "N"))
    xq = pack.unpack_a(np.fromfile(out / "a.bin", dtype=np.uint8), M, K).astype(np.float64)
    wgu = pack.unpack_b(np.fromfile(out / "bgu.bin", dtype=np.uint8), 2 * ff, K).astype(np.float64)
    rows = pack.glu_row(np.arange(ff), ff)
    g, u = xq @ wgu[rows].T, xq @ wgu[rows + 8].T
    h_ref = g / (1.0 + np.exp(-g)) * u
    hw = np.fromfile(out / "h.bin", dtype=np.uint32)
    hk = np.empty((M, ff))
    same = total = 0
    for kc in range(ff // 64):
        for r in range(M // 64):
            tile = pack.h_tile(hw, kc, r)
            ref = pack.pack_a(h_ref[64 * r:64 * r + 64, 64 * kc:64 * kc + 64].astype(np.float32))
            same += int((tile == ref).sum())
            total += tile.size
            hk[64 * r:64 * r + 64, 64 * kc:64 * kc + 64] = pack.unpack_a(tile, 64, 64)
    h_enc = pack.unpack_a(pack.pack_a(h_ref.astype(np.float32)), M, ff).astype(np.float64)

    def rel(y, r):
        return np.linalg.norm(y - r) / np.linalg.norm(r)

    print(f"H bytes equal to the host encode of silu(g) * u: {same / total:.5f}")
    print(f"H decoded vs exact h: rel_fro {rel(hk, h_ref):.3e} (the host encode alone: {rel(h_enc, h_ref):.3e})")
    dq = pack.unpack_b(np.fromfile(out / "bd.bin", dtype=np.uint8), N, ff).astype(np.float64)
    y = pack.untile_c(np.fromfile(out / "y.bin", dtype=np.float32), M, N)
    print(f"down(H) vs decoded-H x decoded-down: rel_fro {rel(y, hk @ dq.T):.3e}")
    ok = rel(y, hk @ dq.T) < 1e-4 and rel(hk, h_ref) < 1.05 * rel(h_enc, h_ref) and same / total > 0.99
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "compare", "make-glu", "compare-glu"])
    ap.add_argument("--build")
    ap.add_argument("--h-build")
    ap.add_argument("--ff", type=int, default=8192)
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
    if a.mode == "make-glu":
        make_glu(a)
        return 0
    return compare_glu(a) if a.mode == "compare-glu" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
