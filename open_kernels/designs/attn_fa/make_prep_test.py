r"""Test vectors for attn_fa's prep dispatch (OPEN-PREFILL-GLUE-PREP) at Granite 4.2 3B's shape.

    python make_prep_test.py make --build <prep build dir> --out <dir> [--p0 0]
    run_kernel.exe <dir>\run.cfg
    python make_prep_test.py compare --out <dir>

make writes y.bin (the qkv GEMM's c-tiled fp32 output of a 256-token block), tb.bin (cos | sin per token), q.bin (the Q
buffer: records with p0, the rest zero), kv.bin (a 256-row cache window, zero) and the numpy reference of what the host
did (rope, scale, bf16 hi / lo, bf16 cache rows). compare scores q.bin and kv.bin after the run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_test import bf, q_buffer  # noqa: E402

T, HD, HEADS, KVH, NPAD = 256, 64, 40, 8, 4096
QW, KVW, ELEM = HEADS * HD, KVH * HD, 4096
QH = ELEM * 9
KV_LD = 2 * KVW
TBL = 2 * 32


def c_tile(y: np.ndarray) -> np.ndarray:
    """y [256, n] natural -> the GEMM's c-tiled order: tile ct at ((ct % 8) * (NPAD / 1024) + ct // 8) blocks of [256][128]."""
    yp = np.zeros((T, NPAD), np.float32)
    yp[:, :y.shape[1]] = y
    out = np.zeros(T * NPAD, np.float32)
    per_col = NPAD // 128 // 8
    for ct in range(NPAD // 128):
        blk = ((ct % 8) * per_col + ct // 8) * T * 128
        out[blk:blk + T * 128] = yp[:, ct * 128:(ct + 1) * 128].reshape(-1)
    return out


def table(p0: int, theta: float = 10000.0) -> np.ndarray:
    inv = theta ** (-np.arange(32, dtype=np.float64) / 32.0)
    ang = (p0 + np.arange(T, dtype=np.float64))[:, None] * inv[None, :]
    return np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32)


def rope(x: np.ndarray, tb: np.ndarray) -> np.ndarray:
    """x [T, heads, 64] fp32, the host's half-split rotation in fp32."""
    c, s = tb[:, None, :32], tb[:, None, 32:]
    x0, x1 = x[..., :32], x[..., 32:]
    return np.concatenate([x0 * c - x1 * s, x1 * c + x0 * s], -1).astype(np.float32)


def expected(y: np.ndarray, tb: np.ndarray, p0: int):
    q = rope(y[:, :QW].reshape(T, HEADS, HD), tb) * np.float32(0.125 * 1.4426950408889634)
    k = rope(y[:, QW:QW + KVW].reshape(T, KVH, HD), tb)
    v = y[:, QW + KVW:QW + 2 * KVW]
    kv = np.concatenate([bf(k).reshape(T, -1), bf(v)], 1)
    return q, kv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "compare"])
    ap.add_argument("--build", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--p0", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    out = a.out
    if a.cmd == "make":
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(a.seed)
        y = (rng.standard_normal((T, QW + 2 * KVW)) * a.scale).astype(np.float32)
        tb = table(a.p0)
        c_tile(y).tofile(out / "y.bin")
        tb.tofile(out / "tb.bin")
        qb = np.zeros(HEADS * QH, np.uint16)
        for h in range(HEADS):
            qb[h * QH:h * QH + 2] = np.array([a.p0], np.int32).view(np.uint16)
        qb.tofile(out / "q.bin")
        np.zeros(T * KV_LD, np.uint16).tofile(out / "kv.bin")
        b = a.build.resolve()
        cfg = "\n".join(["device", f"xclbin G {b / 'final.xclbin'}", f"kernelx k G {b / 'insts.bin'}",
                         f"buf q {qb.size * 2} q.bin", f"buf kv {T * KV_LD * 2} kv.bin", f"buf y {T * NPAD * 4} y.bin",
                         f"buf tb {tb.size * 4} tb.bin", "run k q kv y tb", f"dump q q_out.bin {qb.size * 2}",
                         f"dump kv kv_out.bin {T * KV_LD * 2}", ""])
        (out / "run.cfg").write_text(cfg)
        print(f"y {T * NPAD * 4} B, table {tb.size * 4} B, Q buffer {qb.size * 2} B, p0 {a.p0}")
        return 0
    y = np.fromfile(out / "y.bin", np.float32)
    per_col = NPAD // 128 // 8
    nat = np.zeros((T, NPAD), np.float32)
    for ct in range(NPAD // 128):
        blk = ((ct % 8) * per_col + ct // 8) * T * 128
        nat[:, ct * 128:(ct + 1) * 128] = y[blk:blk + T * 128].reshape(T, 128)
    tb = np.fromfile(out / "tb.bin", np.float32).reshape(T, 64)
    q_ref, kv_ref = expected(nat, tb, 0)
    p0 = int(np.fromfile(out / "q.bin", np.int32, 1)[0])
    qo = np.fromfile(out / "q_out.bin", np.uint16).reshape(HEADS, QH)
    qr = q_buffer(q_ref, p0).reshape(HEADS, QH)
    rec_ok = bool((qo[:, :2] == qr[:, :2]).all())
    qo_p, qr_p = qo[:, ELEM:], qr[:, ELEM:]
    eq = float((qo_p == qr_p).mean())

    def decode(buf):    # hi + lo back to fp32 [heads, 8 cores, 64 dims, 32 tokens]
        hi = buf[:, :ELEM * 4].view(bfloat16).astype(np.float32)
        lo = buf[:, ELEM * 4:].view(bfloat16).astype(np.float32)
        return (hi + lo).reshape(HEADS, 8, 64, 32)

    dq, dr = decode(qo_p), decode(qr_p)
    rel_q = float(np.linalg.norm(dq - dr) / np.linalg.norm(dr))
    per_head = np.linalg.norm((dq - dr).reshape(HEADS, -1), axis=1) / np.linalg.norm(dr.reshape(HEADS, -1), axis=1)
    kvo = np.fromfile(out / "kv_out.bin", np.uint16).reshape(T, KV_LD)
    kvr = kv_ref.view(np.uint16)
    kv_eq = float((kvo == kvr).mean())
    kd = kvo.view(bfloat16).astype(np.float32) - kvr.view(bfloat16).astype(np.float32)
    rel_kv = float(np.linalg.norm(kd) / np.linalg.norm(kvr.view(bfloat16).astype(np.float32)))
    print(f"Q: record intact {rec_ok}, bytes equal {eq:.4%}, (hi + lo) rel_fro {rel_q:.3e}, worst head "
          f"h{int(per_head.argmax())} {per_head.max():.2e}")
    print(f"KV: bytes equal {kv_eq:.4%}, rel_fro {rel_kv:.3e}, K equal {float((kvo[:, :KVW] == kvr[:, :KVW]).mean()):.4%}, "
          f"V equal {float((kvo[:, KVW:] == kvr[:, KVW:]).mean()):.4%}")
    ok = rec_ok and rel_q < 1e-4 and rel_kv < 3e-3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
