r"""Test vectors for lm_head_q4: logits[N] = W[N, K] @ x[K] from pool-order
q4_1 chunks in the 64-row-band layout the lm_head_q4 design streams.

The band law IS q4_1_pack's standard one at rs=2 (per_band = 2*K/256 chunks per
64-row band, rows0 = 64*band + 32*(ci % rs), cols0 = 256*(ci // rs)); the
lm_head's uneven split of the bands over the cores is taps, not bytes.

    python make_test.py [--layout q4nx|gguf] [--k K] [--bands B] [--x ...]

Writes w_<tag>.bin (pool bytes), x_<tag>.bin (bf16[K]), ref_<tag>.bin (f32, the
fp64 reference FROM THE SAME bytes), run_<tag>.cfg, and prints the build, run
and compare commands. Paths in the cfg are relative to this directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import q4_1_pack as P4
import gguf_pool as GG

PER_CALL = 2          # chunks per weight element (lm_head_q4.py PER_CALL)


def deq_chunk(b: np.ndarray, layout: str) -> np.ndarray:
    """One pool chunk -> f32[32 rows, 256 k], the W the kernel computes with:
    the scales NARROWED to bf16 (the f32 build's RNE in-flight convert narrows
    the GGUF f32 scales to the same numbers, gemv_q4.h load_scale; the q4nx
    chunk holds bf16 outright)."""
    if layout == "gguf":
        d = b[0:1024].view(np.float32).astype(bfloat16).astype(np.float32)
        m = b[1024:2048].view(np.float32).astype(bfloat16).astype(np.float32)
        qb = b[2048:]
    else:
        d = b[0:512].view(bfloat16).astype(np.float32)
        m = b[512:1024].view(bfloat16).astype(np.float32)
        qb = b[1024:]
    nib = np.empty(8192, np.uint8)
    nib[0::2] = qb & 0xF
    nib[1::2] = qb >> 4
    n3 = nib.reshape(2, 256, 16)                          # [rb, k, r16]
    w = np.empty((32, 256), np.float32)
    kb = np.arange(256) // 32
    for rb in range(2):
        r = rb * 16 + np.arange(16)
        codes = n3[rb].T.astype(np.float32)               # [16 rows, 256 k]
        w[r] = codes * d[kb[None, :] * 32 + r[:, None]] + m[kb[None, :] * 32 + r[:, None]]
    return w


def reference(pool: bytes, x: np.ndarray, n: int, k: int, layout: str) -> np.ndarray:
    """y = W @ x in fp64 from the SAME pool bytes the kernel streams."""
    ch = P4.CH if layout == "q4nx" else GG.CH_Q4
    nch, _, rows0, cols0 = P4.chunk_geometry(n, k, 2)
    chunks = np.frombuffer(pool, np.uint8).reshape(nch, ch)
    xf = x.astype(np.float64)
    y = np.zeros(n, np.float64)
    for c in range(nch):
        r0, c0 = int(rows0[c]), int(cols0[c])
        y[r0:r0 + 32] += deq_chunk(chunks[c], layout).astype(np.float64) @ xf[c0:c0 + 256]
    return y.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="q4nx", choices=["q4nx", "gguf"],
                    help="q4nx: bf16-scale chunks (q4_1_pack), the .q4nx container; "
                         "gguf: the direct-GGUF f32-scale chunks (gguf_pool), built with "
                         "LMHEAD_SCALES_F32=1")
    ap.add_argument("--k", type=int, default=2560, help="the head's K (the hidden size)")
    ap.add_argument("--bands", type=int, default=64, help="64-row bands (rows = bands * 64)")
    ap.add_argument("--x", default="random", help="random | ones | onehot:K")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    k, bands = a.k, a.bands
    n = bands * 64
    assert k % 256 == 0 and n % 64 == 0
    tag = f"k{k}_b{bands}" + ("_gguf" if a.layout == "gguf" else "")
    rng = np.random.default_rng(a.seed)

    if a.layout == "gguf":
        blocks = GG.random_q4_blocks(n, k, rng)
        w = GG.pack_q4_pool(blocks, 2)
    else:
        blocks = P4.random_q4_1_blocks(n, k, rng)
        w = P4.pack_q4_1_pool(blocks, 2)

    if a.x == "ones":
        x = np.ones(k, np.float32).astype(bfloat16)
    elif a.x.startswith("onehot"):
        x = np.zeros(k, np.float32)
        x[int(a.x.split(":")[1])] = 1.0
        x = x.astype(bfloat16)
    else:
        x = np.random.default_rng(0).standard_normal(k).astype(np.float32).astype(bfloat16)

    ref = reference(w, x, n, k, a.layout)
    (HERE / f"w_{tag}.bin").write_bytes(bytes(w))
    (HERE / f"x_{tag}.bin").write_bytes(x.tobytes())
    (HERE / f"ref_{tag}.bin").write_bytes(ref.tobytes())
    d = "."
    cfg = "\n".join([
        "device",
        f"xclbin G {d}/build_{tag}/final.xclbin",
        f"kernelx k G {d}/build_{tag}/insts.bin",
        f"buf w {len(w)} {d}/w_{tag}.bin",
        f"buf x {x.nbytes} {d}/x_{tag}.bin",
        f"buf y {ref.nbytes}",
        "run k w x y",
        "run k w x y",
        "run k w x y",
        f"dump y {d}/y_{tag}.bin {ref.nbytes}",
        "",
    ])
    (HERE / f"run_{tag}.cfg").write_text(cfg, newline="\n")
    env = f"LMHEAD_N={n} LMHEAD_K={k}"
    if a.layout == "gguf":
        env += " LMHEAD_SCALES_F32=1"
    bands_split = [bands // 8 + (1 if c < bands % 8 else 0) for c in range(8)]
    print(f"{tag}: N={n} bands={bands} ({bands_split[0]} x 6 + {bands_split[1]} x 2 over 8 cores) "
          f"K={k} w={len(w)} B ref absmax={np.abs(ref).max():.4g} ref[:4]={ref[:4]}")
    print(f"build:   {env} python build_design.py designs/lm_head_q4/lm_head_q4.py "
          f"designs/lm_head_q4/build_{tag}")
    print(f"run:     ironvenv harness: open_kernels/harness/build/run_kernel "
          f"designs/lm_head_q4/run_{tag}.cfg (cwd designs/lm_head_q4)")
    print(f"compare: python designs/lm_head_q4/compare.py {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
