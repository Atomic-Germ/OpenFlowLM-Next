r"""Test vectors for gemv_q4: y[N] = W[N,K] @ x[K] from pool-order q4_1 chunks,
with the fp64 reference computed from the SAME bytes the kernel streams.

    python make_test.py [--region R] [--source synthetic|captured] [--runs N] [--x ...]

Default source is `synthetic`: random GGUF-style Q4_1 blocks packed to pool
order by ../../q4_1_pack.py (no model, no captured buffers). `captured` slices
the region out of FLM's captured layer-0 pool (phlegm's original fixture; set
GEMV_POOL to its path) — useful only as a cross-check that synthetic and real
bytes exercise the kernel the same way.

Regions (byte offsets are pools.rs's; RS = band row split, see gemv_q4.h):
    qkv, z, share_up, share_gate, share_down          standard layout, RS=2
    exp_up, exp_gate     expert E's 4 stripes (128 rows x 2048 each), RS=4
    exp_down             expert E's down [2048, 512], RS=4

Writes w_<tag>.bin, x_<tag>.bin (bf16[K]), ref_<tag>.bin (f32[N]) and
run_<tag>.cfg next to this file, with paths relative to this directory (both
open_kernels/harness/run_kernel and phlegm's driver resolve them when run with
this directory as cwd).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from q4_1_pack import CH, pack_q4_1_pool, pool_reference, random_q4_1_blocks  # noqa: E402

S = 32 * CH                       # one expert stripe (128 rows x 2048)
REGIONS = {                       # name: (pool offset, N, K, RS)
    "qkv":        (505_282_560, 8192, 2048, 2),
    "z":          (515_768_320, 4096, 2048, 2),
    "share_up":   (503_316_480, 512, 2048, 2),
    "share_gate": (503_971_840, 512, 2048, 2),
    "share_down": (504_627_200, 2048, 512, 2),
    "exp_up":     (None, 512, 2048, 4),
    "exp_gate":   (None, 512, 2048, 4),
    "exp_down":   (None, 2048, 512, 4),
}


def captured_bytes(f, region: str, expert: int, bands_cap: int | None):
    """-> w_bytes for the region from the captured pool blob."""
    if region in ("exp_up", "exp_gate"):
        parts = []
        for kk in range(4):
            f.seek((8 * expert + 2 * kk + (1 if region == "exp_gate" else 0)) * S)
            parts.append(f.read(S))
        return np.frombuffer(b"".join(parts), np.uint8), 512
    if region == "exp_down":
        f.seek(335_544_320 + expert * 655_360)
        return np.frombuffer(f.read(655_360), np.uint8), 2048
    off, n, k, rs = REGIONS[region]
    if bands_cap:
        n = bands_cap * 32 * rs
    nbytes = n * k * CH // (32 * 256)
    f.seek(off)
    w = np.frombuffer(f.read(nbytes), np.uint8)
    assert len(w) == nbytes
    return w, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="qkv", choices=sorted(REGIONS))
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "captured"])
    ap.add_argument("--expert", type=int, default=0, help="captured only")
    ap.add_argument("--bands", type=int, default=None, help="cap bands (standard regions)")
    ap.add_argument("--x", default="random", help="random | ones | onehot:K | act:FILE")
    ap.add_argument("--runs", type=int, default=2, help="`run` lines in the cfg (timing)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    _, n, k, rs = REGIONS[a.region]
    if a.source == "captured":
        sys.path.insert(0, str(HERE.parents[1]))
        import fixture_paths as FX
        pool = Path(os.environ["GEMV_POOL"]) if os.environ.get("GEMV_POOL") else FX.caps("m0d/blob_536870912_836fd8e49f35a0b6.bin")
        with pool.open("rb") as f:
            w, n = captured_bytes(f, a.region, a.expert, a.bands)
    else:
        if a.bands:
            n = a.bands * 32 * rs
        blocks = random_q4_1_blocks(n, k, np.random.default_rng(a.seed))
        w = pack_q4_1_pool(blocks, rs)
    nbytes = len(w)

    if a.x == "ones":
        x = np.ones(k, np.float32).astype(bfloat16)
    elif a.x.startswith("onehot"):
        x = np.zeros(k, np.float32)
        x[int(a.x.split(":")[1])] = 1.0
        x = x.astype(bfloat16)
    elif a.x.startswith("act:"):
        x = np.fromfile(a.x[4:], np.uint16)[:k].view(bfloat16)
    else:
        x = np.random.default_rng(a.seed).standard_normal(k).astype(np.float32).astype(bfloat16)

    ref = pool_reference(w, x, n, k, rs)
    tag = a.region if not a.region.startswith("exp_") else f"{a.region}{a.expert}"
    if a.source == "captured":
        tag += "_cap"
    (HERE / f"w_{tag}.bin").write_bytes(w.tobytes())
    (HERE / f"x_{tag}.bin").write_bytes(x.tobytes())
    (HERE / f"ref_{tag}.bin").write_bytes(ref.tobytes())
    build = a.region                     # builds are per shape, shared across experts/sources
    cfg = ["device",
           f"xclbin G build_{build}/final.xclbin",
           f"kernelx k G build_{build}/insts.bin",
           f"buf w {nbytes} w_{tag}.bin",
           f"buf x {x.nbytes} x_{tag}.bin",
           f"buf y {ref.nbytes}"]
    cfg += ["run k w x y"] * a.runs
    cfg += [f"dump y y_{tag}.bin {ref.nbytes}", ""]
    (HERE / f"run_{tag}.cfg").write_text("\n".join(cfg), newline="\n")
    cores = min(8, n // (32 * rs))
    print(f"{tag}: N={n} K={k} RS={rs} w={nbytes} B x={x.nbytes} B ref={ref.nbytes} B "
          f"ref[:4]={ref[:4]} absmax={np.abs(ref).max():.4g}")
    print(f"build: GEMV_N={n} GEMV_K={k} GEMV_RS={rs} GEMV_CORES={cores} "
          f"python build_design.py designs/gemv_q4/gemv_q4.py designs/gemv_q4/build_{build}")
    print(f"run:   (cd designs/gemv_q4 && run_kernel run_{tag}.cfg && python compare.py {tag})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
