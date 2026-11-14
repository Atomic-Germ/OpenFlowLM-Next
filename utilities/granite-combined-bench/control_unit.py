r"""Run a command between two control runs, for a `dibs run --measure` unit that has to prove it was not shared.

    python control_unit.py --cli <unchanged exe> --kernels <unchanged set> --model <dir> --known-ms 900 -- <command...>

The control is pp512 on the bfp16 route with host attention (prose ids from bench.py). A sibling sharing the NPU slows it
down, so a CONTROL line more than 5 % over the known value means the unit's numbers are not to be quoted.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench import body_ids, tokenizer  # noqa: E402


def control(a, ids: Path, tag: str) -> None:
    p = subprocess.run([str(a.cli), "--model", str(a.model), "--kernels", str(a.kernels), "--ids-file", str(ids),
                        "--prefill-mode", "bfp16", "--gemm-block", "--max-tokens", "1"], capture_output=True, text=True,
                       env={**__import__("os").environ, "OFLM_OPEN_ATTN_FA": "0"})
    m = re.search(r"prefill (\d+) tokens: ([\d.]+) ms", p.stdout + p.stderr)
    ms = float(m[2]) if m else float("nan")
    off = (ms / a.known_ms - 1) * 100
    print(f"CONTROL {tag}: {ms:.0f} ms ({off:+.1f}% vs {a.known_ms:.0f}){'  SLOW' if off > 5 else ''}", flush=True)


def main() -> int:
    argv = sys.argv[1:]
    cut = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True, type=Path)
    ap.add_argument("--kernels", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--known-ms", type=float, required=True)
    a = ap.parse_args(argv[:cut])
    ids = HERE / "control_ids_512.txt"
    if not ids.exists():
        ids.write_text(",".join(map(str, body_ids(tokenizer(a.model), "prose", 512))))
    control(a, ids, "before")
    r = subprocess.run(argv[cut + 1:])
    control(a, ids, "after")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
