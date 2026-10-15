r"""Two-turn prefill (turn 2 appended at the position turn 1 ended) with the NPU glue on against off (OPEN-PREFILL-GLUE-PREP).

    python multiturn_check.py --cli <open_qwen36_cli.exe> --model <model dir> --kernels <set> \
        [--cases 512+300,300+400,301+100,3840+256] [--max-tokens 64] [--ref-arm bfp16:prep0] [--arm bfp16:prep1]

An arm is `<route>[:prep0|:prep1]`; the CLI prefills turn 1 and then turn 2 (`--ids2`) in one process, dumps every turn-2
position's logits and greedy-decodes --max-tokens. Real text (this repo's spec) so the contexts are not noise.
Needs the NPU: run it inside the track's slot under `dibs run`.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefill_mode_check import REPO, compare, load_positions  # noqa: E402


def text_ids(model: Path, n: int) -> list[int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    text = (REPO / "specs" / "open-engine" / "spec.md").read_text(encoding="utf-8", errors="ignore")
    ids = tok.encode(text, add_special_tokens=False).ids
    return ids[:n]


def run(cli: Path, model: Path, kernels: Path, ids1, ids2, arm: str, prefix: str, max_tokens: int):
    route, _, prep = arm.partition(":")
    env = dict(os.environ)
    for k in ("OFLM_OPEN_PREFILL_MODE", "OFLM_OPEN_PREFILL_PREP", "OFLM_OPEN_ATTN_FA"):
        env.pop(k, None)
    if route != "fast":
        env["OFLM_OPEN_PREFILL_MODE"] = route
    if prep in ("prep0", "prep1"):
        env["OFLM_OPEN_PREFILL_PREP"] = prep[-1]
    cmd = [str(cli), "--model", str(model), "--kernels", str(kernels), "--ids", ",".join(map(str, ids1)),
           "--ids2", ",".join(map(str, ids2)), "--max-tokens", str(max_tokens), "--gemm-block", "--prefill-logits",
           "--dump-logits", prefix]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise SystemExit(f"run failed (arm {arm}, exit {p.returncode}):\n{p.stderr[-2500:]}")
    toks = [int(m.group(1)) for m in re.finditer(r"^token (\d+)$", p.stdout, re.M)]
    t2 = [m for m in re.finditer(r"turn 2.*\n.*npu prep ([\d.]+)", p.stderr)]
    prep_ms = float(t2[-1].group(1)) if t2 else float("nan")
    return toks, prep_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--kernels", required=True, type=Path)
    ap.add_argument("--cases", default="512+300,300+400,301+100,3840+256")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--ref-arm", default="bfp16:prep0")
    ap.add_argument("--arm", default="bfp16:prep1")
    a = ap.parse_args()
    ok = True
    for case in a.cases.split(","):
        n1, n2 = (int(x) for x in case.split("+"))
        ids = text_ids(a.model, n1 + n2)
        if len(ids) < n1 + n2:
            ids = (ids * ((n1 + n2) // len(ids) + 1))[: n1 + n2]
        ids1, ids2 = ids[:n1], ids[n1:]
        with tempfile.TemporaryDirectory(prefix="mt_check_") as tmp:
            t = Path(tmp)
            res = {}
            for tag, arm in (("ref", a.ref_arm), ("test", a.arm)):
                res[tag] = run(a.cli, a.model, a.kernels, ids1, ids2, arm, str(t / tag), a.max_tokens)
            def pos(tag):
                return np.stack([np.fromfile(t / f"{tag}_p{i}.bin", np.float32) for i in range(n1, n1 + n2)]).astype(np.float64)
            x, y = pos("ref"), pos("test")
            byte_same = all((t / f"ref_p{i}.bin").read_bytes() == (t / f"test_p{i}.bin").read_bytes() for i in range(n1, n1 + n2))
            r = compare(x, y)
            same = res["ref"][0] == res["test"][0]
            first = next((i for i, (p, q) in enumerate(zip(res["ref"][0], res["test"][0])) if p != q), None)
            good = byte_same or (r["corr_min"] >= 0.995 and same)
            ok = ok and good
            print(f"turn 1 {n1} + turn 2 {n2} (start {n1}): {'BYTE-IDENTICAL' if byte_same else 'not byte-identical'}, corr min "
                  f"{r['corr_min']:.6f} mean {r['corr_mean']:.6f}, argmax {r['argmax']}/{r['positions']} flips at {r['flips_margin']}, "
                  f"greedy {a.max_tokens}: {'IDENTICAL' if same else 'differ at token ' + str(first)}, "
                  f"npu prep ms in turn 2 (test arm) {res['test'][1]} -> {'PASS' if good else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
