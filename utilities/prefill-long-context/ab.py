"""Paired A/B of two sequential captures: did an arithmetic change actually help?

A failing-position COUNT is not a measurement. It is a threshold on a noisy quantity,
so it can rise while every meaningful magnitude falls, and it can fall for reasons
that have nothing to do with the change under test. This tool answers the question
the count cannot:

  * is the change a systematic improvement or a coin flip? (paired t on the per
    position defect, with the sign of the effect)
  * where did the change come from -- did the improved component simply stop
    hiding a larger, anti-correlated error, or did the true error actually fall?

The second question is the one that decides whether a "regression" is a regression.
A component whose error is partly ANTI-CORRELATED with the dominant one is
CANCELLING it; removing that component makes the total worse, and the honest reading
is that the component was never the defect -- the dominant term that cancellation was
masking is. Measured on the Qwen3.8 norm fix (2026-10-31): the norm's share of the
projection error fell 28x while the total moved 3.7 %, because cosine(norm error
projected, total error) was 0.29 -- it was cancelling, not causing.

Nothing here changes a tolerance or claims model acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CORR_GATE = 0.9999     # the project's unchanged logit correlation bound
REAL_VOCAB = 248077


def correlation(got, ref, real_vocab=REAL_VOCAB):
    """Pearson correlation over the real vocabulary, or None if degenerate."""
    a = np.asarray(got)[:real_vocab].astype(np.float64)
    b = np.asarray(ref)[:real_vocab].astype(np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError('logit vectors differ in shape or are empty')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('non-finite logits')
    a, b = a - a.mean(), b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0.0:
        return None
    return float(a @ b / den)


def defect(corr):
    """1 - correlation, the quantity the gate thresholds."""
    if corr is None:
        raise ValueError('a degenerate position cannot carry a defect')
    return 1.0 - corr


def paired(rows):
    """Paired statistics of a before/after change over the same positions.

    `rows` is a list of (position, before_defect, after_defect). The paired t is on
    the per-position DIFFERENCE, which is the only sound test here: the defects
    themselves are wildly heteroscedastic (a few positions are 100x the body), so an
    unpaired comparison of means is dominated by whichever arm drew the outliers.
    """
    if not rows:
        raise ValueError('no positions to compare')
    p = np.array([r[0] for r in rows])
    a = np.array([float(r[1]) for r in rows])
    b = np.array([float(r[2]) for r in rows])
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('non-finite defect')
    if len(set(p.tolist())) != len(p):
        raise ValueError('duplicate positions')
    d = b - a
    n = len(d)
    mean = float(d.mean())
    # n == 1: the standard error is undefined, so report no statistic rather than
    # an infinity that a reader would take for a decisive result.
    stderr = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    t = float(mean / stderr) if stderr else None
    return dict(positions=n, mean_change=mean, median_change=float(np.median(d)),
                stderr=stderr, t=t, improved=int((d < 0).sum()), worse=int((d > 0).sum()),
                unchanged=int((d == 0).sum()),
                before_p50=float(np.median(a)), after_p50=float(np.median(b)),
                before_p99=float(np.percentile(a, 99)), after_p99=float(np.percentile(b, 99)),
                before_max=float(a.max()), after_max=float(b.max()),
                before_failing=int((a >= 1.0 - CORR_GATE).sum()),
                after_failing=int((b >= 1.0 - CORR_GATE).sum()),
                # A change is SYSTEMATIC only if it is large against its own scatter.
                # |t| < 2 is not evidence of anything, whatever the fail count did.
                systematic=(t is not None and abs(t) >= 2.0),
                direction=('improved' if t is not None and t <= -2.0 else
                           'worse' if t is not None and t >= 2.0 else 'inconclusive'))


def flips(rows, material=2.0):
    """Positions that crossed the gate, split by whether the crossing is MATERIAL.

    Crossing a threshold always moves further than the distance to it, so "did the
    change cause the flip" is trivially yes and worth nothing. The useful question is
    the one that decides whether a fail count means anything: did the position land
    clearly on the far side, or is it scraping past the gate?

    A position that crosses to a defect barely over the gate is a threshold artifact --
    it will cross back on the next unrelated change, so counting it as a regression is
    how a fail count moves without a cause. `material` sets how far past the gate
    counts as a real miss; 2.0 keeps a flip that is at least a factor of two over.

    Measured on the Qwen3.8 norm fix (2026-10-31): all 17 newly failing positions were
    within a factor of ~4.8 of the gate, and most within 2x, which is why the count rose
    while the paired effect was t = +0.89.
    """
    if not (material > 1.0 and np.isfinite(material)):
        raise ValueError('the materiality factor must be a finite number above 1')
    gate = 1.0 - CORR_GATE
    out = []
    for position, a, b in rows:
        crossed = (a < gate) != (b < gate)
        if not crossed:
            continue
        kind = 'now_failing' if b >= gate else 'now_passing'
        # How far onto the far side of the gate the position landed, as a multiple of
        # the gate itself. Exactly AT the gate is 1.0 in this convention.
        margin = (b if kind == 'now_failing' else a) / gate
        out.append(dict(position=position, before=a, after=b, kind=kind,
                        moved=abs(b - a), margin_over_gate=float(margin),
                        material=bool(margin >= material)))
    return out


def component(total, share):
    """How much of a total error one component accounts for, and in which direction.

    `total` and `share` are the same-shaped error vectors: the total error and the
    part of it attributable to the component under test. The sign of the cosine is the
    finding -- positive means the component adds to the total, negative means it
    cancels part of it, and a cancelling component is never the thing to fix first.
    """
    total = np.asarray(total, np.float64)
    share = np.asarray(share, np.float64)
    if total.shape != share.shape or total.ndim != 1 or not total.size:
        raise ValueError('component and total must be equal non-empty vectors')
    if not (np.isfinite(total).all() and np.isfinite(share).all()):
        raise ValueError('non-finite component/total')
    nt, ns = float(np.linalg.norm(total)), float(np.linalg.norm(share))
    if nt == 0.0 or ns == 0.0:
        return dict(total_norm=nt, share_norm=ns, cosine=None, fraction=None,
                    role='absent' if ns == 0.0 else 'only')
    cos = float(share @ total / (nt * ns))
    return dict(total_norm=nt, share_norm=ns, cosine=cos, fraction=ns / nt,
                # A share that is mostly ANTI-parallel is cancelling the rest.
                role=('cancelling' if cos < -0.1 else 'neutral' if cos <= 0.1 else 'adding'))


def score(capture, reference, positions, before, capture_name='sequential_p{n}.bin'):
    """Per-position defects for two captures against one reference.

    The reference's own file set decides which positions are comparable: a reference
    prepared for 257 positions cannot score 513. The correlation window fits BOTH
    vectors, so a short reference is caught as a preparation mismatch instead of
    being scored on a truncated window that happens to correlate well.
    """
    reference, before, capture = Path(reference), Path(before), Path(capture)
    rows = []
    for p in positions:
        ref = reference / f'ref_p{p}.bin'
        if not ref.is_file():
            continue
        r = np.fromfile(ref, dtype='<f4')
        defects = []
        for capture_dir in (before, capture):
            path = capture_dir / capture_name.format(n=p)
            if not path.is_file():
                raise FileNotFoundError(f'missing capture: {path}')
            got = np.fromfile(path, dtype='<f4')
            vocab = min(REAL_VOCAB, r.size, got.size)
            if vocab < 2:
                raise ValueError(f'position P{p} has fewer than two comparable logits')
            c = correlation(got, r, vocab)
            if c is None:
                raise ValueError(f'degenerate position P{p}')
            defects.append(defect(c))
        rows.append((p, defects[0], defects[1]))
    if not rows:
        raise ValueError('no position had a capture and a reference')
    return rows


def run(a):
    rows = score(a.capture, a.reference, range(a.positions), a.before, a.capture_name)
    result = dict(complete=True, model_acceptance=False,
                  before=str(Path(a.before).resolve()), capture=str(Path(a.capture).resolve()),
                  reference=str(Path(a.reference).resolve()),
                  corr_gate=CORR_GATE,
                  scope='paired per-position attribution of one arithmetic change on the '
                        'sequential route; diagnosis only, not an acceptance row')
    result.update(paired(rows))
    result['flips'] = flips(rows, a.material)
    result['immaterial_flips'] = [f for f in result['flips'] if not f['material']]
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"positions {result['positions']}   gate corr > {CORR_GATE}")
    print(f"  defect p50 {result['before_p50']:.3e} -> {result['after_p50']:.3e}")
    print(f"  defect p99 {result['before_p99']:.3e} -> {result['after_p99']:.3e}")
    print(f"  defect max {result['before_max']:.3e} -> {result['after_max']:.3e}")
    print(f"  failing    {result['before_failing']} -> {result['after_failing']}")
    print(f"  improved {result['improved']}  worse {result['worse']}")
    t = 'n/a' if result['t'] is None else f"{result['t']:+.2f}"
    print(f"  paired mean change {result['mean_change']:+.3e} (t = {t}) -> {result['direction']}")
    immaterial = result['immaterial_flips']
    print(f"  gate flips {len(result['flips'])}, of which {len(immaterial)} are not material "
          f"(within {a.material}x of the gate): {[f['position'] for f in immaterial]}")
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--positions', type=int, default=513)
    p.add_argument('--capture-name', default='sequential_p{n}.bin',
                   help='per-position file name inside each capture directory')
    p.add_argument('--material', type=float, default=2.0,
                   help='how far over the gate a flip must land to count as material')
    p.add_argument('--out', type=Path)
    raise SystemExit(run(p.parse_args()))
