#!/usr/bin/env python3
"""Numerical acceptance of the actual layer_x streamed activation/Q4 projection probe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "open_kernels"))
from q4_1_pack import pack_q4_1_pool, pool_reference, random_q4_1_blocks
from wide_deltanet_reference import metric

GUARD = bytes([0xA5]) * 64


def prepare(out):
    meta = json.loads((out / "probe-toolchain.json").read_text())
    k, n = meta["projection_k"], meta["projection_n"]
    rng = np.random.default_rng(2961)
    pool = pack_q4_1_pool(random_q4_1_blocks(n, k, rng), 2)
    pool.tofile(out / "weights.bin")
    inputs = list(rng.normal(0, .6, (2, k)).astype(bfloat16))
    inputs += [np.ones(k, bfloat16), np.zeros(k, bfloat16)]
    for index in sorted({2047, 2048, 4095, 4096, k - 1}):
        if index >= k:
            continue
        x = np.zeros(k, bfloat16)
        x[index] = 1
        inputs.append(x)
    chunks = (k + 2047) // 2048
    (out / "poison.bin").write_bytes(np.full(n, np.nan, np.float32).tobytes() + GUARD)
    cfg = ["device", "xclbin p final.xclbin", "kernelx p p insts.bin",
           f"buf w {pool.nbytes} weights.bin", f"buf x {chunks * 4096}", f"buf y {n * 4 + len(GUARD)}"]
    for i, x in enumerate(inputs):
        # A partial last element must not consume padding beyond logical K.
        padded = np.full(chunks * 2048, np.nan, bfloat16)
        padded[:k] = x
        padded.tofile(out / f"x{i}.bin")
        pool_reference(pool, x, n, k, 2).tofile(out / f"ref{i}.bin")
        cfg += [f"load x x{i}.bin", "load y poison.bin", "run p w x y", f"dump y got{i}.bin {n * 4 + len(GUARD)}"]
        (out / f"got{i}.bin").unlink(missing_ok=True)
    (out / "projection.cfg").write_text("\n".join(cfg) + "\n")
    metadata = dict(k=k, n=n, inputs=len(inputs), seed=2961,
                    sha256={f: hashlib.sha256((out / f).read_bytes()).hexdigest()
                            for f in ("final.xclbin", "insts.bin", "weights.bin", "projection.cfg")})
    (out / "projection-fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "projection-results.json").unlink(missing_ok=True)
    print(f"Prepared {len(inputs)} Q4 projection inputs, N={n}, K={k}")


def compare(out):
    meta = json.loads((out / "projection-fixture.json").read_text())
    for f, digest in meta["sha256"].items():
        if hashlib.sha256((out / f).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{f} changed since fixture generation")
    results = []
    for i in range(meta["inputs"]):
        raw = (out / f"got{i}.bin").read_bytes()
        size = meta["n"] * 4
        if len(raw) != size + len(GUARD) or raw[size:] != GUARD:
            raise ValueError(f"input {i}: wrong output size or damaged canary")
        got = np.frombuffer(raw[:size], np.float32)
        ref = np.fromfile(out / f"ref{i}.bin", np.float32)
        m = metric(got, ref, .9999999)
        results.append(dict(input=i, **m))
        print(f"{'PASS' if m['passed'] else 'FAIL'} input={i}: {m}")
    ok = all(m["passed"] for m in results)
    (out / "projection-results.json").write_text(json.dumps(dict(passed=ok, checks=results), indent=2) + "\n")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compare"))
    p.add_argument("--build-dir", type=Path, required=True)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.build_dir)
        return 0
    return compare(args.build_dir)


if __name__ == "__main__":
    sys.exit(main())
