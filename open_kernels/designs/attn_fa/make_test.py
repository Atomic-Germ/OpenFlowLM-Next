r"""Test vectors for attn_fa at Granite 4.2 3B's attention shape (40 query heads, 8 kv heads, head dim 64):
one 256-token block at position p0 against the L = p0 + 256 cached rows before and including it.

    python make_test.py --build <build dir> --out <dir> [--L 512] [--qk-scale 1] [--seed 0]
    run_kernel.exe <dir>\run.cfg
    python compare.py <dir>

Writes q.bin (the kernel's Q buffer: per head the record with p0, then Q hi and Q lo), kv.bin (a KV cache
of AF_MAX_ROWS rows, K | V per row, bf16), ref.bin (fp64 causal attention as f32 [256, 2560]) and run.cfg.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
T, HD, ELEM, Q_ELEMS = 256, 64, 64 * 64, 4


def bf(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32).astype(bfloat16)


def q_buffer(q: np.ndarray, p0: int, qsplit: bool = True) -> np.ndarray:
    """q [T, heads, 64] fp32, already scaled by log2(e)/sqrt(64), -> the kernel's Q buffer (bf16): per head the
    record, then for each core (32 tokens) its Q^T [64 dims][32 tokens], hi then (all cores) lo."""
    heads = q.shape[1]
    per = ELEM * (1 + Q_ELEMS * (2 if qsplit else 1))
    out = np.zeros((heads, per), np.uint16)
    hi = bf(q)
    lo = bf(q - hi.astype(np.float32))

    def cores_t(x):   # [T, 64] -> [8 cores][64 dims][32 tokens]
        return x.reshape(T // 32, 32, HD).transpose(0, 2, 1).reshape(-1)

    for h in range(heads):
        rec = np.zeros(ELEM, np.uint16)
        rec[:2] = np.array([p0], np.int32).view(np.uint16)
        out[h, :ELEM] = rec
        out[h, ELEM:ELEM * (1 + Q_ELEMS)] = cores_t(hi[:, h, :]).view(np.uint16)
        if qsplit:
            out[h, ELEM * (1 + Q_ELEMS):] = cores_t(lo[:, h, :]).view(np.uint16)
    return out.reshape(-1)


def reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, p0: int, gqa: int) -> np.ndarray:
    """fp64 causal attention; q [T, heads, 64] in natural units (1/sqrt(64) applied), k/v [rows, kvh, 64]."""
    heads = q.shape[1]
    out = np.zeros((T, heads, HD))
    for h in range(heads):
        kh, vh = k[:, h // gqa, :].astype(np.float64), v[:, h // gqa, :].astype(np.float64)
        s = q[:, h, :].astype(np.float64) @ kh.T
        pos = p0 + np.arange(T)
        s[np.arange(kh.shape[0])[None, :] > pos[:, None]] = -np.inf
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        out[:, h, :] = (p / p.sum(1, keepdims=True)) @ vh
    return out.reshape(T, heads * HD)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--L", type=int, default=512)
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=4096)
    ap.add_argument("--qk-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-qsplit", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    p0 = a.L - T
    kvh, heads = a.kv_heads, a.heads
    k = bf(rng.standard_normal((a.max_rows, kvh, HD)) * np.sqrt(a.qk_scale))
    v = bf(rng.standard_normal((a.max_rows, kvh, HD)))
    k[0, :, :] = bf(k[0].astype(np.float32) * 3.0)      # an attention sink: row 0 dominates some scores
    q = (rng.standard_normal((T, heads, HD)) * np.sqrt(a.qk_scale) / 8.0).astype(np.float32)
    kv = np.concatenate([k.reshape(a.max_rows, -1), v.reshape(a.max_rows, -1)], 1)
    ref = reference(q, k[:a.L].astype(np.float32), v[:a.L].astype(np.float32), p0, heads // kvh)
    qb = q_buffer(q * np.float32(np.log2(np.e)), p0, not a.no_qsplit)
    qb.tofile(a.out / "q.bin")
    kv.view(np.uint16).tofile(a.out / "kv.bin")
    ref.astype(np.float32).tofile(a.out / "ref.bin")
    o_bytes = T * heads * HD * 4
    b = a.build.resolve()
    cfg = "\n".join(["device", f"xclbin G {b / 'final.xclbin'}", f"kernelx k G {b / 'insts.bin'}",
                     f"buf q {qb.size * 2} q.bin", f"buf kv {kv.size * 2} kv.bin", f"buf o {o_bytes}",
                     *(["run k q kv o"] * 8), f"dump o o.bin {o_bytes}", ""])
    (a.out / "run.cfg").write_text(cfg)
    flop = 4 * heads * T * a.L * HD
    print(f"L {a.L} p0 {p0}: Q {qb.size * 2} B, KV {kv.size * 2} B, O {o_bytes} B, {flop / 1e9:.2f} GFLOP (full rectangle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
