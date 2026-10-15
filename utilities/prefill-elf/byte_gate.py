r"""byte_gate: prefill logits of two env variants of the same CLI and kernel set must be byte-identical.

    python byte_gate.py --model <dir> --cli <exe> --kernels <set> --lengths 512,2048 [--all-positions-at 512]
        --a OFLM_OPEN_PREFILL_PREP=1 --b OFLM_OPEN_PREFILL_PREP=1,OFLM_OPEN_PREFILL_ELF=1

Run it as one `dibs run --measure` unit (it drives the NPU). Writes the logits under a temp directory and compares sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "granite-combined-bench"))


def env_of(s: str) -> dict:
    return dict(kv.split("=", 1) for kv in s.split(",") if kv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--lengths", default="512,2048")
    ap.add_argument("--all-positions-at", type=int, default=0)
    ap.add_argument("--route", default="bfp16")
    ap.add_argument("--max-tokens", default="1")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    a = ap.parse_args()
    import bench
    tok = bench.tokenizer(Path(a.model))
    bad = 0
    with tempfile.TemporaryDirectory(prefix="byte_gate_") as tmp:
        for n in [int(x) for x in a.lengths.split(",")]:
            ids = Path(tmp) / f"ids{n}.txt"
            ids.write_text(",".join(map(str, bench.body_ids(tok, "prose", n))))
            sums = {}
            for tag, e in (("a", a.a), ("b", a.b)):
                d = Path(tmp) / f"{tag}{n}"
                d.mkdir()
                cmd = [a.cli, "--model", a.model, "--kernels", a.kernels, "--ids-file", str(ids), "--prefill-mode",
                       a.route, "--gemm-block", "--max-tokens", a.max_tokens, "--dump-logits", str(d / "lg")]
                if n == a.all_positions_at:
                    cmd.append("--prefill-logits")
                r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env_of(e)})
                files = sorted(d.glob("lg*"))
                if r.returncode or not files:
                    print(f"pp{n} {tag}: FAILED rc={r.returncode}\n{(r.stdout + r.stderr)[-1500:]}")
                    return 2
                sums[tag] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
            same = sums["a"] == sums["b"]
            bad += not same
            print(f"pp{n}: {len(sums['a'])} logit files, {'BYTE-IDENTICAL' if same else 'DIFFER'}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
