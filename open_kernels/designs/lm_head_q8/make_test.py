r"""Test vectors for lm_head_q8 from FLM's captured lm_head pool
($OPEN_KERNELS_CAPS/m0d/000127.bo, or LMHEAD_POOL=<file>; verified byte-exact
against our builder in pools.rs).

    python make_test.py [--bands B] [--x random|ones|onehot:K|act:FILE]

Writes w_<tag>.bin (first B bands of pool-order q8 chunks), x_<tag>.bin,
ref_<tag>.bin (f32, fp64 reference from the same bytes), run_<tag>.cfg (paths
relative to this directory).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import fixture_paths as FX  # noqa: E402
CH = 8704
K = 2048
PER_BAND = 32
BAND_ROWS = 128
N_ALL = 248320


def dequant_chunks(b: np.ndarray) -> np.ndarray:
    """(n, 8704) pool chunks -> f32 (n, 32 rows, 256 k)  (q4nx.rs dequant_q8_rows)."""
    n = b.shape[0]
    d = b[:, :512].copy().view(bfloat16).astype(np.float32).reshape(n, 8, 32)      # [kb, r]
    codes = b[:, 512:].copy().view(np.int8).reshape(n, 2, 256, 16)                 # [rb, k, r16]
    w = codes.transpose(0, 1, 3, 2).reshape(n, 32, 256).astype(np.float32)          # [row, k]
    dd = np.repeat(d.transpose(0, 2, 1), 32, axis=2)                                # [row, k]
    return w * dd


def reference(w_bytes: np.ndarray, x: np.ndarray, n: int, batch: int = 2048) -> np.ndarray:
    nch = len(w_bytes) // CH
    chunks = w_bytes[: nch * CH].reshape(nch, CH)
    y = np.zeros(n, np.float64)
    xf = x.astype(np.float64)
    c = np.arange(nch)
    band, ci = np.divmod(c, PER_BAND)
    rows0 = BAND_ROWS * band + 32 * (ci % 4)
    cols0 = 256 * (ci // 4)
    for lo in range(0, nch, batch):
        hi = min(lo + batch, nch)
        w = dequant_chunks(chunks[lo:hi]).astype(np.float64)                        # (b, 32, 256)
        xs = xf[cols0[lo:hi, None] + np.arange(256)[None, :]]                       # (b, 256)
        part = np.einsum("brk,bk->br", w, xs)                                       # (b, 32)
        rows = rows0[lo:hi, None] + np.arange(32)[None, :]
        np.add.at(y, rows, part)
    return y.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=None, help="first B bands only (default: all 1940)")
    ap.add_argument("--x", default="random")
    a = ap.parse_args()
    bands = a.bands or N_ALL // BAND_ROWS
    n = bands * BAND_ROWS
    nbytes = bands * PER_BAND * CH
    tag = "full" if not a.bands else f"b{bands}"
    pool = Path(os.environ["LMHEAD_POOL"]) if os.environ.get("LMHEAD_POOL") else FX.caps("m0d/000127.bo")
    with pool.open("rb") as f:
        w = np.frombuffer(f.read(nbytes), np.uint8)
    assert len(w) == nbytes

    if a.x == "ones":
        x = np.ones(K, np.float32).astype(bfloat16)
    elif a.x.startswith("onehot"):
        x = np.zeros(K, np.float32)
        x[int(a.x.split(":")[1])] = 1.0
        x = x.astype(bfloat16)
    elif a.x.startswith("act:"):
        x = np.fromfile(a.x[4:], np.uint16)[:K].view(bfloat16)
    else:
        x = np.random.default_rng(0).standard_normal(K).astype(np.float32).astype(bfloat16)

    ref = reference(w, x, n)
    (HERE / f"w_{tag}.bin").write_bytes(w.tobytes())
    (HERE / f"x_{tag}.bin").write_bytes(x.tobytes())
    (HERE / f"ref_{tag}.bin").write_bytes(ref.tobytes())
    d = "."
    cfg = "\n".join([
        "device",
        f"xclbin G {d}/build_{tag}/final.xclbin",
        f"kernelx k G {d}/build_{tag}/insts.bin",
        f"buf w {nbytes} {d}/w_{tag}.bin",
        f"buf x {x.nbytes} {d}/x_{tag}.bin",
        f"buf y {ref.nbytes}",
        "run k w x y",
        "run k w x y",
        "run k w x y",
        f"dump y {d}/y_{tag}.bin {ref.nbytes}",
        "",
    ])
    (HERE / f"run_{tag}.cfg").write_text(cfg, newline="\n")
    print(f"{tag}: N={n} bands={bands} w={nbytes} B ref absmax={np.abs(ref).max():.4g} ref[:4]={ref[:4]}")
    print(f"build: LMHEAD_N={n} python build_design.py designs/lm_head_q8/lm_head_q8.py designs/lm_head_q8/build_{tag}")
    print(f"run:   open-qwen-npu npu designs/lm_head_q8/run_{tag}.cfg ; python compare.py {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
