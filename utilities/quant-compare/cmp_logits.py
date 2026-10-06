r"""Compare two runs' per-position logits: how far a weight format moves the model.

    open_qwen36_cli ... --prefill-logits --dump-logits <dirA>/y --max-tokens 1    (reference)
    open_qwen36_cli ... --prefill-logits --dump-logits <dirB>/y --max-tokens 1    (candidate)
    python utilities/quant-compare/cmp_logits.py <dirA>/y <dirB>/y

Reads every `<prefix>_p<position>.bin` (f32[vocab]) present in both runs and prints, over those
positions: the worst and median logits correlation, argmax agreement (and how many of the flips
were near-ties in the reference, top-2 margin < 0.5 logits), top-5 agreement, and the mean KL
divergence of the candidate's next-token distribution from the reference's.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


def positions(prefix: Path) -> dict[int, Path]:
    pat = re.compile(re.escape(prefix.name) + r"_p(\d+)\.bin$")
    return {int(m.group(1)): p for p in prefix.parent.iterdir() if (m := pat.match(p.name))}


def main() -> int:
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    pa, pb = positions(a), positions(b)
    common = sorted(set(pa) & set(pb))
    if not common:
        sys.exit(f"no common positions ({len(pa)} in {a}, {len(pb)} in {b})")
    corr, kl, flips, near, top5 = [], [], 0, 0, 0
    for pos in common:
        x = np.fromfile(pa[pos], np.float32).astype(np.float64)
        y = np.fromfile(pb[pos], np.float32).astype(np.float64)
        corr.append(np.corrcoef(x, y)[0, 1])
        px = np.exp(x - x.max()); px /= px.sum()
        py = np.exp(y - y.max()); py /= py.sum()
        kl.append(float(np.sum(px * (np.log(px + 1e-30) - np.log(py + 1e-30)))))
        ax, ay = int(x.argmax()), int(y.argmax())
        if ax != ay:
            flips += 1
            s = np.sort(x)
            near += (s[-1] - s[-2]) < 0.5
        top5 += set(np.argsort(x)[-5:]) == set(np.argsort(y)[-5:])
    n = len(common)
    corr = np.array(corr)
    print(f"{n} positions ({common[0]}..{common[-1]})")
    print(f"  logits corr: worst {corr.min():.6f}  median {np.median(corr):.6f}  mean {corr.mean():.6f}")
    print(f"  argmax: {n - flips}/{n} agree; {flips} flips, {near} of them near-ties (ref top-2 margin < 0.5)")
    print(f"  top-5 set: {top5}/{n} identical")
    print(f"  KL(ref || cand): mean {np.mean(kl):.5f}  max {np.max(kl):.5f} nats")
    return 0


if __name__ == "__main__":
    sys.exit(main())
