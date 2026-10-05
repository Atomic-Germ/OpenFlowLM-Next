r"""Test vectors for dit_gemm at one shape, for open_kernels/harness's run_kernel.

    python make_test.py --kernels <dir> --stream <name> --out <testdir> [--seed S]
    python make_test.py --M 512 --K 3072 --N 18432 --epi --out <testdir>          (writes layout.json)
        DG_M=.. DG_K=.. DG_N=.. DG_LAYOUT=<layout.json contents> build_design.py dit_gemm.py <build>
    python make_test.py --M 512 --K 3072 --N 18432 --epi --out <testdir> --build <build>

<dir> holds final.xclbin and insts_<name>.bin (export_dit_kernels.py's layout) plus
dit_kernels.json, which gives the stream's M, K, N. Writes into <testdir>:
    a_<name>.bin     A, bf16 [M, K] row-major, N(0, 1)
    b_<name>.bin     B packed by pack.pack_b from bf16 N(0, 1/K) [K, N]
    ref_<name>.npz   rows sampled every 8th, their float64 products against the bf16 B
                     ("exact") and against the B the kernel actually multiplies by ("kernel")
    run_<name>.cfg   device, xclbin, the stream, three runs, dump of C

--epi tests the SwiGLU epilogue: B's N columns are gate (first half) and up (second half),
packed in the epilogue's interleaved order (pack.interleave_swiglu); the reference is
silu(A B_gate) * (A B_up) [M, N/2], which lands in the first 64 columns of every 128 of C
(compare.py reads it out that way).

Then (Windows, from anywhere):
    open_kernels\harness\out\run_kernel.exe <testdir>\run_<name>.cfg
    python compare.py <testdir> <name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pack import interleave_swiglu, pack_b, unpack_b  # noqa: E402

ROW_STRIDE = 8


def silu(x):
    return x / (1 + np.exp(-x))


def epi_layout(M: int, N: int) -> dict:
    """The --epi stream's DG_LAYOUT: every column group gets the epilogue."""
    return {"epi": {"first_cb": 0}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels")
    ap.add_argument("--stream")
    ap.add_argument("--build", help="a single-stream build dir (final.xclbin, insts.bin)")
    ap.add_argument("--M", type=int)
    ap.add_argument("--K", type=int)
    ap.add_argument("--N", type=int)
    ap.add_argument("--epi", action="store_true")
    ap.add_argument("--gather", action="store_true",
                    help="A read 64-of-128 out of an [M, 2K] buffer (layout a_gather)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.kernels:
        kdir = Path(args.kernels).resolve()
        meta = json.loads((kdir / "dit_kernels.json").read_text())
        st = meta["streams"][args.stream]
        if st.get("layout"):
            raise SystemExit(f"stream {args.stream} has a layout {st['layout']}; this harness "
                             "allocates compact A/B/C buffers and a plain-product reference, "
                             "so it cannot test it")
        M, K, N = (st[x] for x in ("M", "K", "N"))
        name = args.stream
        xclbin, insts = kdir / "final.xclbin", kdir / f"insts_{name}.bin"
    else:
        M, K, N = args.M, args.K, args.N
        name = "epi" if args.epi else "gather" if args.gather else "t"
        if args.epi:
            (out / "layout.json").write_text(json.dumps(epi_layout(M, N)))
        if args.gather:
            (out / "layout.json").write_text(json.dumps({"a_gather": True}))
        if not args.build:
            print(f"wrote {out / 'layout.json'}; build with DG_M={M} DG_K={K} DG_N={N} DG_LAYOUT=...")
            return 0
        b_dir = Path(args.build).resolve()
        xclbin, insts = b_dir / "final.xclbin", b_dir / "insts.bin"

    rng = np.random.default_rng(args.seed)
    a = rng.standard_normal((M, K), dtype=np.float32).astype(bfloat16)
    b = (rng.standard_normal((K, N), dtype=np.float32) / np.sqrt(K)).astype(bfloat16)
    rows = np.arange(0, M, ROW_STRIDE)
    a64 = a[rows].astype(np.float64)
    if args.epi:
        packed = pack_b(interleave_swiglu(b.astype(np.float32)))
        h = N // 2
        ex = a64 @ b.astype(np.float64)
        exact = silu(ex[:, :h]) * ex[:, h:]
        ck = a64 @ unpack_b(packed, K, N)                  # interleaved column order
        ck = ck.reshape(len(rows), N // 128, 2, 64)
        kernel = (silu(ck[:, :, 0]) * ck[:, :, 1]).reshape(len(rows), h)
        n_out = h
    else:
        packed = pack_b(b)
        exact, kernel = a64 @ b.astype(np.float64), a64 @ unpack_b(packed, K, N)
        n_out = N
    np.savez(out / f"ref_{name}.npz", rows=rows, M=M, N=n_out, exact=exact, kernel=kernel,
             epi=args.epi)
    if args.gather:   # the kernel must read only the first 64 of every 128 columns
        phys = rng.standard_normal((M, K // 64, 2, 64), dtype=np.float32).astype(bfloat16)
        phys[:, :, 0] = a.reshape(M, K // 64, 64)
        phys.tofile(out / f"a_{name}.bin")
    else:
        a.tofile(out / f"a_{name}.bin")
    packed.tofile(out / f"b_{name}.bin")

    c_bytes = M * N * 2
    cfg = ["device",
           f"xclbin G {xclbin}",
           f"kernelx k G {insts}",
           f"buf a {M * K * 2 * (2 if args.gather else 1)} a_{name}.bin",
           f"buf b {packed.size} b_{name}.bin",
           f"buf c {c_bytes}"]
    cfg += ["run k a b c"] * args.runs
    cfg += [f"dump c c_{name}.bin {c_bytes}", ""]
    (out / f"run_{name}.cfg").write_text("\n".join(cfg))
    print(f"{name}: {M}x{K}x{N}{' (SwiGLU epilogue)' if args.epi else ''}, "
          f"{2 * M * K * N / 1e9:.1f} GFLOP -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
