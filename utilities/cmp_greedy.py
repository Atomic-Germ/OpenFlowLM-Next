"""Compare two `--dump-logits` runs position by position: argmax agreement to the
first divergence, correlation at each position, and the top-2 gap where they differ.

    python utilities/cmp_greedy.py <ref_prefix> <new_prefix> [--n 64]

Each run writes `<prefix>_t<i>.bin`, f32[vocab], one per decode position. The gate
for a kernel change that is NOT bit-exact -- a block rescale instead of one per row
changes the softmax's rounding -- is identical tokens, or a first divergence late in
the run that is a near-tie: the two kernels' top-2 within ~0.05 logits and a
correlation above 0.9999 at that position. `cmp_exact.py` reports max |diff| only,
which is the wrong instrument when the arithmetic is allowed to re-round.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load(prefix: Path, i: int, kind: str = "t") -> np.ndarray | None:
    p = Path(str(prefix) + f"_{kind}{i}.bin")
    if not p.is_file():
        return None
    return np.fromfile(p, np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--kind", default="t", choices=("t", "p"),
                    help="t = the decode loop's _t<i> dumps (greedy, each run feeds itself); "
                         "p = --prefill-logits' _p<position> dumps (teacher-forced: both runs see "
                         "the same ids, so a difference never compounds)")
    ap.add_argument("--start", type=int, default=0, help="first index (a _p dump is position-numbered)")
    a = ap.parse_args()

    first_div = None
    rows = []
    for i in range(a.start, a.start + a.n):
        r, n = load(Path(a.ref), i, a.kind), load(Path(a.new), i, a.kind)
        if r is None or n is None:
            if a.kind == "p":
                continue
            break
        if r.shape != n.shape:
            print(f"t{i}: shape {r.shape} vs {n.shape}")
            return 1
        corr = float(np.corrcoef(r.astype(np.float64), n.astype(np.float64))[0, 1])
        ar, an = int(r.argmax()), int(n.argmax())
        mx = float(np.abs(r - n).max())
        # the new run's own top-2 gap: how close the flip was on the side that moved
        o = np.argpartition(n, -2)[-2:]
        o = o[np.argsort(-n[o])]
        gap = float(n[o[0]] - n[o[1]])
        rows.append((i, ar, an, corr, mx, gap))
        if ar != an and first_div is None:
            first_div = (i, ar, an, corr, gap, float(r[ar] - r[an]))

    if not rows:
        print("no logit dumps found")
        return 1
    print(f"{'pos':>4} {'ref':>8} {'new':>8} {'corr':>12} {'max|diff|':>10} {'top2 gap':>9}")
    for i, ar, an, corr, mx, gap in rows:
        flag = "  <-- DIVERGES" if ar != an else ""
        print(f"{i:>4} {ar:>8} {an:>8} {corr:>12.7f} {mx:>10.5f} {gap:>9.5f}{flag}")
    cmin = min(r[3] for r in rows)
    print(f"\n{len(rows)} positions, corr min {cmin:.7f}")
    if a.kind == "p":
        # Teacher-forced, so each position stands alone. The block-only walk masks a
        # padding row only when the cached-row count is EVEN; a masking bug shows up as
        # the even positions being systematically worse than the odd ones.
        for par, name in ((0, "even (a padding row masked)"), (1, "odd (no padding)")):
            c = [r[3] for r in rows if r[0] % 2 == par]
            if c:
                print(f"  {name:30s} n={len(c):3d}  corr min {min(c):.7f}  median {float(np.median(c)):.7f}")
    if first_div is None:
        print(f"IDENTICAL: {len(rows)}/{len(rows)} argmax agree")
        return 0
    i, ar, an, corr, gap, refgap = first_div
    print(f"first divergence at position {i}: ref {ar} -> new {an}, corr {corr:.7f}, "
          f"new's top-2 gap {gap:.5f}, ref's gap between the two {refgap:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
