#!/usr/bin/env python3
"""Prepare/compare deterministic standalone AB hardware fixtures (no CPU fallback).

Run prepare, then open_kernels/harness/out/run_kernel <build-dir>/ab.cfg,
then compare. This tests repeated AB dispatches, NOT recurrent DeltaNet state.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "open_kernels"))
from recipes import pack
from recipes.wide_deltanet import WideDeltaNet


class RawTensor:
    def __init__(self, data):
        self.data = data

    def raw(self, name):
        return self.data


def prepare(out, g):
    rng = np.random.default_rng(593)
    weights = rng.normal(0, 0.015, (2, g.value_heads, g.hidden)).astype(bfloat16)
    packed = []
    for w in weights:
        dst = np.zeros(g.ab_banks * g.hidden * 32 * 2, np.uint8)
        pack.apply_op(dict(op="transpose_banked", tensor="w", rows=g.value_heads,
                           cols=g.hidden, elem=2, dst=0), RawTensor(w.tobytes()), 0, dst)
        packed.append(dst.reshape(g.ab_banks, -1))
    small = np.zeros(1024, np.float32)
    # The container stores A=-exp(A_log), not A_log itself.
    small[:g.value_heads] = -np.exp(rng.uniform(-1, 1, g.value_heads))
    small[g.value_heads:2 * g.value_heads] = rng.uniform(-0.5, 0.5, g.value_heads)
    side = b"".join(packed[0][bank].tobytes() + packed[1][bank].tobytes() + small.tobytes()
                    for bank in range(g.ab_banks))
    assert len(side) == g.side_bytes
    (out / "side.bin").write_bytes(side)
    xs = list(rng.normal(0, 0.6, (3, g.hidden)).astype(bfloat16))
    for index in (0, 2047, 2048, g.hidden - 1):
        x = np.zeros(g.hidden, bfloat16)
        x[index] = 1
        xs.append(x)
    # Poison output before EVERY run; missing tails or a stale previous run must fail.
    np.full(4 * g.value_heads, np.nan, np.float32).tofile(out / "poison.bin")
    cfg = ["device", "xclbin ab final.xclbin", "kernelx ab ab insts.bin",
           f"buf side {g.side_bytes} side.bin", f"buf xn {g.xn_chunks * 4096}",
           f"buf result {g.result_bytes}"]
    for token, x in enumerate(xs):
        padded = np.zeros(g.xn_chunks * 2048, bfloat16)
        padded[:g.hidden] = x
        padded.tofile(out / f"xn{token}.bin")
        alpha, beta_logits = weights.astype(np.float64) @ x.astype(np.float64)
        decay = np.exp(small[:g.value_heads].astype(np.float64) *
                       np.logaddexp(0, alpha + small[g.value_heads:2 * g.value_heads]))
        beta = 1 / (1 + np.exp(-beta_logits))
        np.stack((alpha, beta_logits, decay, beta)).tofile(out / f"ref{token}.bin")
        cfg += [f"load xn xn{token}.bin", "load result poison.bin", "run ab side xn result",
                f"dump result result{token}.bin {g.result_bytes}"]
        (out / f"result{token}.bin").unlink(missing_ok=True)
    (out / "ab.cfg").write_text("\n".join(cfg) + "\n")
    metadata = {"geometry": vars(g), "tokens": len(xs), "seed": 593,
                "reference": "float64 math over identical bf16 inputs and f32 A/dt_bias",
                "recurrent_state_tested": False,
                "packages": {n: importlib.metadata.version(n) for n in ("numpy", "mlir-aie", "llvm-aie")},
                "sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                           for name in ("final.xclbin", "insts.bin", "side.bin")}}
    (out / "ab-fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "ab-results.json").unlink(missing_ok=True)
    print(f"Prepared {len(xs)} AB dispatches at H={g.hidden}: {out / 'ab.cfg'}")


def compare(out, g):
    metadata = json.loads((out / "ab-fixture.json").read_text())
    if metadata["geometry"] != vars(g):
        raise ValueError("fixture geometry does not match requested build")
    metrics = []
    for token in range(metadata["tokens"]):
        got = np.fromfile(out / f"result{token}.bin", np.float32).reshape(4, g.value_heads).astype(np.float64)
        ref = np.fromfile(out / f"ref{token}.bin", np.float64).reshape(4, g.value_heads)
        for name, actual, expected in zip(("alpha", "beta_logits", "decay", "beta"), got, ref):
            relative = float(np.max(np.abs(actual - expected)) / (np.max(np.abs(expected)) + 1e-30))
            cosine = float(actual @ expected / (np.linalg.norm(actual) * np.linalg.norm(expected) + 1e-30))
            good = bool(np.isfinite(actual).all() and relative < 1e-4 and cosine > 0.99999)
            metrics.append(dict(token=token, field=name, passed=good, maxrel=relative, cosine=cosine))
            print(f"{'PASS' if good else 'FAIL'} token={token} {name}: maxrel={relative:.3e} cos={cosine:.10f}")
    ok = all(m["passed"] for m in metrics)
    (out / "ab-results.json").write_text(json.dumps(dict(passed=ok, metrics=metrics), indent=2) + "\n")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compare"))
    p.add_argument("--hidden", type=int, required=True, choices=(2560, 5120))
    p.add_argument("--build-dir", type=Path)
    args = p.parse_args()
    g = WideDeltaNet(args.hidden, 16, 48, 128, 128)
    out = args.build_dir or ROOT / f"open_kernels/designs/wide_deltanet/build_ab_h{g.hidden}"
    if args.stage == "prepare":
        prepare(out, g)
        return 0
    return compare(out, g)


if __name__ == "__main__":
    sys.exit(main())
