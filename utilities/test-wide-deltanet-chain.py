#!/usr/bin/env python3
"""Synthetic NPU acceptance: AB -> conv/records -> persistent DeltaNet recurrence.

prepare writes inputs/reference/cfg; run the open XRT harness; compare checks
every token/head plus output canaries. The harness performs only BO copies
between kernels. Reference arrays are never read by the hardware program.
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
DESIGN = ROOT / "open_kernels/designs/wide_deltanet"
sys.path.insert(0, str(ROOT / "open_kernels"))
from recipes import pack
from recipes.wide_deltanet import WideDeltaNet
from wide_deltanet_reference import ab_reference, glue_reference, step_reference, metric

HEADS, KEY_HEADS, HD, NCH = 48, 16, 128, 10240
SIZES = dict(ab=4 * HEADS * 4, conv=3 * NCH * 2, vec=HEADS * 512 * 4,
             state=HEADS * HD * HD * 4, output=HEADS * HD * 4)
GUARD = bytes([0xA5]) * 64


class RawTensor:
    def __init__(self, data):
        self.data = data

    def raw(self, name):
        return self.data


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out, hidden, tokens):
    if tokens < 2:
        raise ValueError("acceptance requires multiple tokens")
    out.mkdir(parents=True, exist_ok=True)
    g = WideDeltaNet(hidden, 16, HEADS, HD, HD)
    rng = np.random.default_rng(20260924)
    weights = rng.normal(0, .015, (2, HEADS, hidden)).astype(bfloat16)
    banks = []
    for w in weights:
        dst = np.zeros(2 * hidden * 32 * 2, np.uint8)
        pack.apply_op(dict(op="transpose_banked", tensor="w", rows=HEADS, cols=hidden,
                           elem=2, dst=0), RawTensor(w.tobytes()), 0, dst)
        banks.append(dst.reshape(2, -1))
    small = np.zeros(1024, np.float32)
    # Near-unit decay retains substantial state between tokens.
    small[:HEADS] = -rng.uniform(.01, .08, HEADS)
    small[HEADS:2 * HEADS] = rng.uniform(-2.5, -1.5, HEADS)
    side = b"".join(banks[0][i].tobytes() + banks[1][i].tobytes() + small.tobytes() for i in range(2))
    (out / "ab-side.bin").write_bytes(side)
    convw = rng.normal(0, .25, (4, NCH)).astype(bfloat16)
    conv_side = bytes(4096) + convw.reshape(4, 10, 1024).transpose(1, 0, 2).tobytes()
    (out / "conv-side.bin").write_bytes(conv_side)
    cfg = ["device"]
    kernels = {"ab": DESIGN / f"build_ab_h{hidden}", "glue": DESIGN / "build_glue",
               "step": DESIGN / "build_step"}
    artifacts = {}
    for name, directory in kernels.items():
        for file in ("final.xclbin", "insts.bin"):
            artifacts[str(directory / file)] = sha(directory / file)
        cfg += [f"xclbin {name} {directory}/final.xclbin",
                f"kernelx {name} {name} {directory}/insts.bin"]
    cfg += [f"buf abs {len(side)} ab-side.bin", f"buf xn {g.xn_chunks * 4096}",
            f"buf side {len(conv_side)} conv-side.bin", f"buf qkv {NCH * 4}",
            f"buf cs {SIZES['conv']}", f"buf s {SIZES['state']}"]
    for name, size in SIZES.items():
        poison = (np.full(size // 2, np.nan, bfloat16).tobytes() if name == "conv" else
                  np.full(size // 4, np.nan, np.float32).tobytes()) + GUARD
        (out / f"poison-{name}.bin").write_bytes(poison)
        cfg += [f"buf {name} {size + len(GUARD)}"]

    for case in ("cold", "warm"):
        cs = np.zeros((3, NCH), bfloat16)
        state = np.zeros((HEADS, HD, HD), np.float32)
        if case == "warm":
            cs[:] = rng.normal(0, .2, cs.shape).astype(bfloat16)
            state[:] = rng.normal(0, .05, state.shape)
        cs.tofile(out / f"{case}-conv-init.bin")
        state.tofile(out / f"{case}-state-init.bin")
        cfg += [f"load cs {case}-conv-init.bin", f"load s {case}-state-init.bin"]
        for token in range(tokens):
            tag = f"{case}-{token}"
            xn = rng.normal(0, .6, hidden).astype(bfloat16)
            qkv = rng.normal(0, .4, NCH).astype(np.float32)
            padded = np.zeros(g.xn_chunks * 2048, bfloat16)
            padded[:hidden] = xn
            padded.tofile(out / f"{tag}-xn.bin")
            qkv.tofile(out / f"{tag}-qkv.bin")
            ab = ab_reference(xn, weights, small[:HEADS], small[HEADS:2 * HEADS])
            cs, records = glue_reference(qkv, cs, convw, ab)
            state, output = step_reference(state, records)
            # The physical persistent state is f32 between dispatches.
            state = state.astype(np.float32)
            for name, data in dict(ab=ab, conv=cs, vec=records, state=state, output=output).items():
                data.tofile(out / f"{tag}-ref-{name}.bin")
                (out / f"{tag}-got-{name}.bin").unlink(missing_ok=True)
                cfg += [f"load {name} poison-{name}.bin"]
            cfg += [f"load xn {tag}-xn.bin", f"load qkv {tag}-qkv.bin",
                    "run ab abs xn ab", f"copy side 0 ab 0 {SIZES['ab']}",
                    "run glue side qkv cs conv vec", "run step s vec state output"]
            for name, size in SIZES.items():
                cfg += [f"dump {name} {tag}-got-{name}.bin {size + len(GUARD)}"]
            cfg += [f"copy cs 0 conv 0 {SIZES['conv']}", f"copy s 0 state 0 {SIZES['state']}"]
    (out / "chain.cfg").write_text("\n".join(cfg) + "\n")
    metadata = dict(hidden=hidden, tokens=tokens, cases=["cold", "warm"], seed=20260924,
                    packages={n: importlib.metadata.version(n) for n in ("numpy", "mlir-aie", "llvm-aie")},
                    kernels=artifacts, cfg_sha256=sha(out / "chain.cfg"))
    (out / "chain-fixture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "chain-results.json").unlink(missing_ok=True)
    print(f"Prepared H={hidden}, 2 x {tokens} persistent tokens; {out / 'chain.cfg'}")


def compare(out, hidden):
    metadata = json.loads((out / "chain-fixture.json").read_text())
    if hidden != metadata["hidden"]:
        raise ValueError("fixture hidden width mismatch")
    for path, digest in metadata["kernels"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"kernel changed after prepare: {path}")
    if sha(out / "chain.cfg") != metadata["cfg_sha256"]:
        raise ValueError("hardware program changed after prepare")
    results = []
    def check(tag, field, got, ref, cos=.99999, per_head=False):
        m = metric(got, ref, cos)
        if per_head and got.shape == ref.shape:
            # The existing deltanet/compare.py gate normalizes over the whole
            # tensor. Keep that same gate. Head-local normalization is an extra
            # diagnostic: near-zero outputs can amplify the inherited split-bf16
            # error. Preserve failures explicitly, rather than hiding them.
            heads = [metric(a, b, cos) for a, b in zip(got, ref)]
            m["head_diagnostics"] = heads
            m["head_diagnostics_passed"] = all(h["passed"] for h in heads)
        results.append(dict(tag=tag, field=field, **m))
        print(f"{'PASS' if m['passed'] else 'FAIL'} {tag} {field}: "
              f"rel={m.get('maxrel', float('nan')):.3e} cos={m.get('cosine', float('nan')):.10f}")
    for case in metadata["cases"]:
        for token in range(metadata["tokens"]):
            tag = f"{case}-{token}"
            arrays = {}
            for name, size in SIZES.items():
                data = (out / f"{tag}-got-{name}.bin").read_bytes()
                if len(data) != size + len(GUARD) or data[size:] != GUARD:
                    raise ValueError(f"{tag} {name}: missing output bytes or damaged canary")
                arrays[name] = np.frombuffer(data[:size], bfloat16 if name == "conv" else np.float32)
            ref_ab = np.fromfile(out / f"{tag}-ref-ab.bin", np.float64).reshape(4, HEADS)
            for field, got, ref in zip(("alpha", "beta_logits", "decay", "beta"),
                                       arrays["ab"].reshape(4, HEADS), ref_ab):
                check(tag, field, got, ref)
            ref_conv = np.fromfile(out / f"{tag}-ref-conv.bin", np.uint16)
            exact = np.array_equal(arrays["conv"].view(np.uint16), ref_conv)
            results.append(dict(tag=tag, field="conv_state_bits", passed=exact))
            print(f"{'PASS' if exact else 'FAIL'} {tag} conv_state_bits")
            got_vec = arrays["vec"].reshape(HEADS, 512)
            ref_vec = np.fromfile(out / f"{tag}-ref-vec.bin", np.float64).reshape(HEADS, 512)
            for field, sl in (("k", slice(0, 128)), ("q", slice(128, 256)), ("v", slice(256, 384)),
                              ("record_decay", 384), ("record_beta", 385)):
                check(tag, field, got_vec[:, sl], ref_vec[:, sl])
            padding = bool(np.all(got_vec[:, 386:] == 0))
            group = HEADS // KEY_HEADS
            grouping = all(np.array_equal(got_vec[h, :256], got_vec[(h // group) * group, :256])
                           for h in range(HEADS))
            results.append(dict(tag=tag, field="record_padding_and_grouping", passed=padding and grouping))
            for name, shape, dtype in (("state", (HEADS, HD, HD), np.float32),
                                        ("output", (HEADS, HD), np.float64)):
                ref = np.fromfile(out / f"{tag}-ref-{name}.bin", dtype).reshape(shape)
                check(tag, name, arrays[name].reshape(shape), ref, cos=.9999999, per_head=True)
    passed = all(m["passed"] for m in results)
    head_diagnostics_passed = all(m.get("head_diagnostics_passed", True) for m in results)
    (out / "chain-results.json").write_text(json.dumps(dict(
        passed=passed, head_diagnostics_passed=head_diagnostics_passed,
        normalization="whole tensor, as in designs/deltanet/compare.py", checks=results), indent=2) + "\n")
    print(f"Inherited tensor gates: {passed}; stricter head-local diagnostics: {head_diagnostics_passed}")
    return 0 if passed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "compare"))
    p.add_argument("--hidden", type=int, required=True, choices=(2560, 5120))
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = args.out or DESIGN / f"build_chain_h{args.hidden}"
    if args.stage == "prepare":
        prepare(out.resolve(), args.hidden, args.tokens)
        return 0
    return compare(out.resolve(), args.hidden)


if __name__ == "__main__":
    sys.exit(main())
