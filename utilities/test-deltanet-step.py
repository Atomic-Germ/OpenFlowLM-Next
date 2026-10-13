#!/usr/bin/env python3
"""Synthetic recurrence regression for a built dn_step, without captured/closed fixtures."""
import argparse
import json
from pathlib import Path

import numpy as np

from wide_deltanet_reference import metric, step_reference


def prepare(out, heads, tokens):
    if heads <= 0 or tokens < 2:
        raise ValueError("positive heads and multiple tokens required")
    rng = np.random.default_rng(1907)
    state = rng.normal(0, .05, (heads, 128, 128)).astype(np.float32)
    state.tofile(out / "state-init.bin")
    np.full_like(state, np.nan).tofile(out / "state-poison.bin")
    np.full((heads, 128), np.nan, np.float32).tofile(out / "output-poison.bin")
    cfg = ["device", "xclbin step final.xclbin", "kernelx step step insts.bin",
           f"buf s {state.nbytes} state-init.bin", f"buf so {state.nbytes}",
           f"buf vec {heads * 512 * 4}", f"buf o {heads * 128 * 4}"]
    for token in range(tokens):
        vec = np.zeros((heads, 512), np.float32)
        for off in (0, 128):
            x = rng.normal(0, 1, (heads, 128))
            vec[:, off:off + 128] = x / np.linalg.norm(x, axis=-1, keepdims=True)
        vec[:, 256:384] = rng.normal(0, .5, (heads, 128))
        vec[:, 384] = rng.uniform(.9, 1, heads)
        vec[:, 385] = rng.uniform(.1, .9, heads)
        vec.tofile(out / f"vec{token}.bin")
        state, output = step_reference(state, vec.astype(np.float64))
        state = state.astype(np.float32)
        state.tofile(out / f"ref-state{token}.bin")
        output.tofile(out / f"ref-output{token}.bin")
        cfg += [f"load vec vec{token}.bin", "load so state-poison.bin", "load o output-poison.bin",
                "run step s vec so o", f"dump so got-state{token}.bin {state.nbytes}",
                f"dump o got-output{token}.bin {heads * 128 * 4}", f"copy s 0 so 0 {state.nbytes}"]
        for name in ("state", "output"):
            (out / f"got-{name}{token}.bin").unlink(missing_ok=True)
    (out / "step.cfg").write_text("\n".join(cfg) + "\n")
    (out / "step-fixture.json").write_text(json.dumps(dict(heads=heads, tokens=tokens, seed=1907)) + "\n")
    (out / "step-results.json").unlink(missing_ok=True)


def compare(out, heads):
    meta = json.loads((out / "step-fixture.json").read_text())
    if meta["heads"] != heads:
        raise ValueError("head count mismatch")
    results = []
    for token in range(meta["tokens"]):
        for name, dtype in (("state", np.float32), ("output", np.float64)):
            got = np.fromfile(out / f"got-{name}{token}.bin", np.float32)
            ref = np.fromfile(out / f"ref-{name}{token}.bin", dtype)
            m = metric(got, ref, .9999999)
            results.append(dict(token=token, field=name, **m))
            print(f"{'PASS' if m['passed'] else 'FAIL'} {token} {name}: {m}")
    ok = all(m["passed"] for m in results)
    (out / "step-results.json").write_text(json.dumps(dict(passed=ok, checks=results), indent=2) + "\n")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compare"))
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--tokens", type=int, default=4)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.build_dir, args.heads, args.tokens)
        return 0
    return compare(args.build_dir, args.heads)


if __name__ == "__main__":
    raise SystemExit(main())
