"""Long-context prefill: does the tail degrade with block count?

The 255/256/257 evidence spans a block boundary only as a 1-token tail, so it
cannot answer this. A 513-token prompt at T=256 runs TWO full blocks, which is
the first measurement where block 2 carries a substantial state from block 1.

Two questions must be kept apart, because they call for different fixes:

  * system drift -- the MEDIAN defect grows with position, so every position is
    getting worse and the arithmetic or state accumulates error;
  * tail outliers -- the median is flat but individual positions are far worse,
    so something rare and input-dependent (attention over the carried KV, a
    cancellation) hits a handful of rows.

The verdict reports both separately. Averaging them together is how a real tail
regression hides inside a healthy-looking mean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CORR_GATE = 0.9999          # the project's unchanged logit correlation bound
BLOCK = 256                 # the compiled prefill block width


def correlation(got, ref, real_vocab):
    """Pearson correlation over the real vocabulary, or None if degenerate."""
    a = np.asarray(got)[:real_vocab].astype(np.float64)
    b = np.asarray(ref)[:real_vocab].astype(np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError('logit vectors differ in shape or are empty')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('non-finite logits')
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return None
    return float(a @ b / den)


def block_of(position, block=BLOCK):
    return position // block


def summarise(corr, block=BLOCK):
    """Per-block and per-quartile minima, plus the outlier inventory."""
    if not corr:
        raise ValueError('no positions scored')
    positions = sorted(corr)
    defect = {p: 1.0 - corr[p] for p in positions}
    blocks = {}
    for p in positions:
        b = block_of(p, block)
        blocks.setdefault(b, []).append(corr[p])
    quartiles = []
    span = positions[-1] + 1
    for lo in range(0, span, block):
        hi = min(lo + block, span)
        vals = [corr[p] for p in positions if lo <= p < hi]
        if vals:
            quartiles.append(dict(start=lo, end=hi - 1, n=len(vals), minimum=min(vals),
                                  median=float(np.median(vals))))
    worst = sorted(corr.items(), key=lambda kv: kv[1])[:8]
    failing = [p for p in positions if corr[p] <= CORR_GATE]
    # Which blocks the outliers live in must be judged on positions that actually
    # stand out, not on the top-N list: when the body is uniform, ties there are
    # broken arbitrarily and can drag block-0 positions into it.
    body = float(np.median([defect[p] for p in positions]))
    outliers = [p for p in positions if body > 0 and defect[p] >= 3.0 * body]
    median = float(np.median([corr[p] for p in positions]))
    p50 = float(np.percentile([defect[p] for p in positions], 50))
    p99 = float(np.percentile([defect[p] for p in positions], 99))
    p100 = float(np.max([defect[p] for p in positions]))
    # "System drift" is judged on the MEDIAN of each block: if block 2's median
    # defect is within 2x block 1's, the body is not degrading.
    block_medians = {str(b): float(np.median(v)) for b, v in sorted(blocks.items())}
    body_stable = None
    if len(block_medians) >= 2:
        first, last = sorted(blocks)[0], sorted(blocks)[-1]
        d1 = float(np.median([1.0 - c for c in blocks[first]]))
        d2 = float(np.median([1.0 - c for c in blocks[last]]))
        body_stable = bool(d2 <= 2.0 * d1) if d1 > 0 else bool(d2 == 0)
    # "Tail" is judged on the worst position relative to the body's p99.
    tail_ratio = p100 / p99 if p99 > 0 else None
    return dict(positions=len(positions), corr_gate=CORR_GATE, median_corr=median,
                block=block, blocks=len(blocks), block_medians_corr=block_medians,
                quartiles=quartiles, worst=[dict(position=p, correlation=v) for p, v in worst],
                failing=failing, defect=dict(p50=p50, p99=p99, p100=p100),
                body_stable=body_stable, tail_ratio=tail_ratio,
                # Tail outliers are a defect only if they are far off the body AND
                # the body itself is stable; otherwise it is drift reported once.
                verdict=('tail-outliers' if (tail_ratio is not None and tail_ratio >= 10.0
                                             and body_stable) else
                         'system-drift' if body_stable is False else 'clean'),
                outliers=outliers,
                tail_is_in_later_blocks=(all(block_of(p, block) > 0 for p in outliers)
                                         if outliers and len(blocks) > 1 else None))


def score(capture, reference, positions, real_vocab):
    capture, reference = Path(capture), Path(reference)
    corr = {}
    for p in positions:
        got, ref = capture / f'hybrid_p{p}.bin', reference / f'ref_p{p}.bin'
        if not got.is_file() or not ref.is_file():
            continue
        corr[p] = correlation(np.fromfile(got, dtype='<f4'), np.fromfile(ref, dtype='<f4'),
                              real_vocab)
    if not corr:
        raise ValueError('no position had both a capture and a reference')
    return corr


def run(a):
    corr = score(a.capture, a.reference, range(a.positions), a.real_vocab)
    result = dict(complete=True, model_acceptance=False,
                  capture=str(a.capture.resolve()), reference=str(a.reference.resolve()),
                  scope='long-context logit correlation for the production block route; '
                        'diagnosis only, not an acceptance row')
    result.update(summarise(corr))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"positions {result['positions']}  blocks {result['blocks']}  "
          f"median corr {result['median_corr']:.10f}")
    print(f"defect p50 {result['defect']['p50']:.3e}  p99 {result['defect']['p99']:.3e}  "
          f"p100 {result['defect']['p100']:.3e}   tail/body {result['tail_ratio']:.1f}x")
    for q in result['quartiles']:
        print(f"  P{q['start']}-{q['end']}: min {q['minimum']:.9f}  median {q['median']:.9f}")
    print('worst:', [(w['position'], round(w['correlation'], 9)) for w in result['worst']])
    print(f"failing corr<={CORR_GATE}: {result['failing']}")
    print(f"verdict: {result['verdict']}  (all worst in later blocks: "
          f"{result['tail_is_in_later_blocks']})")
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--positions', type=int, default=513)
    p.add_argument('--real-vocab', type=int, default=248077)
    p.add_argument('--out', type=Path)
    raise SystemExit(run(p.parse_args()))
