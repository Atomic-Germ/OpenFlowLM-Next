"""Compare two open_qwen36_cli --prefill-logits dumps position by position: argmax, top-5 set, correlation (2026-10-19)."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def positions(prefix: Path) -> dict[int, Path]:
    pat = re.compile(re.escape(prefix.name) + r"_p(\d+)\.bin$")
    return {int(m.group(1)): p for p in prefix.parent.iterdir() if (m := pat.match(p.name))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", help="dump prefix of the reference run (e.g. the sequential route)")
    ap.add_argument("b", help="dump prefix of the run under test (e.g. the block route)")
    ap.add_argument("--vocab", type=int, default=0, help="score only the first N logits (the real vocabulary)")
    a = ap.parse_args()
    pa, pb = positions(Path(a.a)), positions(Path(a.b))
    common = sorted(set(pa) & set(pb))
    if not common:
        print("no common positions")
        return 2
    am = t5 = 0
    corrs = []
    worst = None
    for p in common:
        x = np.fromfile(pa[p], np.float32).astype(np.float64)
        y = np.fromfile(pb[p], np.float32).astype(np.float64)
        if a.vocab:
            x, y = x[:a.vocab], y[:a.vocab]
        am += int(np.argmax(x) == np.argmax(y))
        t5 += int(set(np.argsort(-x)[:5]) == set(np.argsort(-y)[:5]))
        c = float(np.corrcoef(x, y)[0, 1]) if np.isfinite(y).all() else float("nan")
        corrs.append(c)
        if worst is None or not (c >= worst[1]):
            worst = (p, c)
    n = len(common)
    c = np.array(corrs)
    print(f"{n} positions: argmax {am}/{n}, top-5 set {t5}/{n}, corr min {np.nanmin(c):.7f} (position {worst[0]}) "
          f"median {np.nanmedian(c):.7f}, finite {np.isfinite(c).all()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
