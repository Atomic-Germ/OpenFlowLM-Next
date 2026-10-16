r"""Uno against plain decode at depth for several (cli, kernel set) configs, interleaved per prompt, in one process tree.

    python unodepth.py --model <dir> --work <dir> --lengths 64,512,1024,2048 --kinds chat,code --tokens 96 --cfg label=<cli>@<set> [--cfg ...]

One row a (length, kind, config): Uno tok/s, plain tok/s in the same process, their ratio, tokens a cycle, IDENTICAL or DIFFERS.
Wrap in `dibs run --measure` inside controlled.py. Prompts are bench.py's chat / code prompts of the exact token count.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--lengths", default="64,512,1024,2048")
    ap.add_argument("--kinds", default="chat,code")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--cfg", action="append", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    cfgs = [(c.split("=", 1)[0], *c.split("=", 1)[1].rsplit("@", 1)) for c in a.cfg]
    tok = bench.tokenizer(a.model)
    rows, flip = [], 0
    for n in (int(x) for x in a.lengths.split(",")):
        for kind in a.kinds.split(","):
            ids = bench.chat_prompt(tok, a.model, kind, n)
            for label, cli, kern in (cfgs if flip % 2 == 0 else cfgs[::-1]):
                ns = type("A", (), {"cli": cli, "model": a.model, "kernels": kern})
                txt = bench.run_cli(ns, "fast", ids, ["--gemm-block", "--uno", str(a.tokens)], a.work)
                try:
                    r = bench.parse_uno(txt)
                except ValueError as e:
                    r = {"error": str(e)[-800:]}
                row = {"n": len(ids), "kind": kind, "cfg": label, **r}
                rows.append(row)
                if "error" in row:
                    print(f"{kind:5s} {len(ids):5d} {label}: ERROR {row['error']}", flush=True)
                else:
                    print(f"{kind:5s} {len(ids):5d} {label:8s} uno {r['uno_tok_s']:6.2f} plain {r['decode_tok_s']:6.2f} tok/s "
                          f"ratio {r['uno_tok_s'] / r['decode_tok_s']:.3f} {r['per_cycle']:.2f} tok/cycle "
                          f"{'IDENTICAL' if r['identical'] else 'DIFFERS'}", flush=True)
            flip += 1
    if a.out:
        a.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
