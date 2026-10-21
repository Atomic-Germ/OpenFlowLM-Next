r"""Score the attention gate's device output against the fp64 reference.

    python compare.py --out <build dir>

Reads the dumps the harness wrote (new<i>.bin, og<i>.bin), compares them with
the reference make_test.py computed, and prints PASS / FAIL per case. The exit
status is the gate: 0 means the geometry is good.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from designs.attn import attn_reference as REF  # noqa: E402

GUARD = bytes([0xA5]) * 64


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="the build directory make_test.py prepared")
    a = ap.parse_args()
    out = Path(a.out)
    meta = json.loads((out / "attention-fixture.json").read_text())
    nh, kvh, hd = meta["nh"], meta["kvh"], meta["hd"]
    new_bytes = 4 * kvh * hd
    og_bytes = 2 * nh * hd

    ok = True
    results = []
    for i, case in enumerate(meta["cases"]):
        got = {}
        for name, size, shape in (("new", new_bytes, (2, kvh, hd)), ("og", og_bytes, (nh, hd))):
            raw = (out / f"{name}{i}.bin").read_bytes()
            if len(raw) != size + len(GUARD) or raw[size:] != GUARD:
                raise SystemExit(f"{case['name']}/{name}: the dump is not the buffer (size or canary)")
            got[name] = np.frombuffer(raw[:size], bfloat16).reshape(shape)
        ref_new = np.fromfile(out / f"ref-new{i}.bin", bfloat16).reshape(2, kvh, hd)
        ref_og = np.fromfile(out / f"ref-og{i}.bin", np.float32).reshape(nh, hd)
        checks = {"k": REF.metric(got["new"][0], ref_new[0], 1e-2),
                  "v": REF.metric(got["new"][1], ref_new[1], 1e-2),
                  "og": REF.metric(got["og"], ref_og, 2e-2)}
        heads = [REF.metric(g, r, 2e-2) for g, r in zip(got["og"], ref_og)]
        # V is a plain fp32 -> bf16 conversion: it must be the exact bytes.
        v_exact = bool(np.array_equal(got["new"][1].view(np.uint16), ref_new[1].view(np.uint16)))
        passed = v_exact and all(c["passed"] for c in checks.values()) and all(h["passed"] for h in heads)
        ok &= passed
        results.append({"name": case["name"], "passed": passed, "v_exact": v_exact,
                        "checks": checks, "heads": [h["passed"] for h in heads]})
        print(f"{'PASS' if passed else 'FAIL'} {case['name']:10} "
              f"og rel={checks['og'].get('maxrel')} cos={checks['og'].get('cosine'):.7f} "
              f"k rel={checks['k'].get('maxrel')} v_exact={v_exact}")
    (out / "attention-results.json").write_text(json.dumps({"passed": ok, "results": results}, indent=2) + "\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
