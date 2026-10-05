r"""Test vectors for dit_fa, for open_kernels/harness's run_kernel.

    python make_test.py --xclbin <x> --insts <i> --out <testdir> --L 4608 --heads 24
                        [--kv-heads H] [--causal] [--valid-len N] [--qk-scale S] [--seed S]
    python make_test.py --kernels <set> --stream <name> --out <testdir>     (an exported set)

Q, K, V are token-major [L, heads*128] (K/V [L, kv_heads*128]), N(0, 1) scaled by
--qk-scale (Q and K both, so scores scale by its square: 1 gives a flat softmax,
3 a peaked one). Writes q.bin k.bin v.bin, ref.npz (the inputs, for compare.py) and
run.cfg (three runs, dump of O).

Then (Windows):
    open_kernels\harness\out\run_kernel.exe <testdir>\run.cfg
    python compare.py <testdir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

D = 128


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin")
    ap.add_argument("--insts")
    ap.add_argument("--kernels")
    ap.add_argument("--stream")
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--kv-heads", type=int, default=0)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--valid-len", type=int, default=0)
    ap.add_argument("--qk-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()

    if a.kernels:
        kdir = Path(a.kernels).resolve()
        s = json.loads((kdir / "dit_fa.json").read_text())["streams"][a.stream]
        if s.get("layout"):
            raise SystemExit(f"stream {a.stream} has a strided layout {s['layout']}; this harness "
                             "allocates compact Q/K/V/O buffers and cannot test it")
        xclbin, insts = kdir / "final.xclbin", kdir / f"insts_{a.stream}.bin"
        L, heads, kvh = s["L"], s["heads"], s.get("kv_heads", s["heads"])
        causal, valid = bool(s.get("causal", 0)), s.get("valid_len", 0)
    else:
        xclbin, insts = Path(a.xclbin).resolve(), Path(a.insts).resolve()
        L, heads, kvh = a.L, a.heads, a.kv_heads or a.heads
        causal, valid = a.causal, a.valid_len
    valid = valid or L

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    q = (rng.standard_normal((L, heads * D), dtype=np.float32) * a.qk_scale).astype(bfloat16)
    k = (rng.standard_normal((L, kvh * D), dtype=np.float32) * a.qk_scale).astype(bfloat16)
    v = rng.standard_normal((L, kvh * D), dtype=np.float32).astype(bfloat16)
    for n, x in (("q", q), ("k", k), ("v", v)):
        x.tofile(out / f"{n}.bin")
    np.savez(out / "ref.npz", q=q.view(np.uint16), k=k.view(np.uint16), v=v.view(np.uint16),
             L=L, heads=heads, kv_heads=kvh, causal=causal, valid_len=valid)

    ob = L * heads * D * 2
    cfg = ["device", f"xclbin G {xclbin}", f"kernelx k G {insts}",
           f"buf q {q.nbytes} q.bin", f"buf k {k.nbytes} k.bin", f"buf v {v.nbytes} v.bin",
           f"buf o {ob}"]
    cfg += ["run k q k v o"] * a.runs + [f"dump o o.bin {ob}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))
    gflop = 4 * heads * L * L * D / 1e9
    print(f"L={L} heads={heads}/{kvh} causal={causal} valid={valid}: {gflop:.1f} GFLOP -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
