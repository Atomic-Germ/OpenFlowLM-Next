"""Profile where the sequential prefill route loses precision, layer by layer.

Both prefill routes run the SAME audited capture with the same weights and the
same kernel set; only the schedule differs. `sequential` fails every late gate
while `hybrid` passes all of them, driven by the same embeddings -- so the pair
proves the model CAN be computed acceptably on this hardware, and localises the
deficit to the sequential route's arithmetic rather than to the weights, the
packing or the reference.

Two causes must not be confused:

  * `per-token`   -- the error is already present at position 0 and does not
                     grow with the prompt. The per-token NPU kernels compute a
                     worse result than the host arithmetic the block route uses.
  * `accumulation` -- the error ramps with position. State carried across
                     positions is degrading.

This reads only retained captures: no NPU run, no weights, no model residency.
`model_acceptance` stays false; a profile is a diagnosis, not an acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOUND = 5e-3          # the project's unchanged residual maxrel bound
ROUTES = ('sequential', 'hybrid')
HIDDEN = 5120


def rel_error(got, ref):
    """max|got-ref| over max|ref| for one row; None when the row has no scale."""
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    den = float(np.max(np.abs(ref)))
    if den == 0.0:
        return None
    return float(np.max(np.abs(got - ref)) / den)


def read_layer(path, rows, hidden=HIDDEN):
    if not path.is_file():
        raise FileNotFoundError(f'missing capture: {path}')
    flat = np.fromfile(path, dtype='<f4')
    if flat.size % hidden:
        raise ValueError(f'{path}: {flat.size} floats is not a multiple of {hidden}')
    table = flat.reshape(-1, hidden)
    if table.shape[0] < rows:
        raise ValueError(f'{path}: {table.shape[0]} rows, need {rows}')
    return table[:rows]


def summarise_route(rel, bound=BOUND):
    """first_failing_layer / worst_layer / worst from (layers x positions) maxrel."""
    table = np.asarray(rel, np.float64)
    if table.ndim != 2 or not table.size or not np.isfinite(table).all():
        raise ValueError('a finite (layers x positions) matrix is required')
    row = table.max(axis=1)                                # per-layer worst
    ok = np.array([m <= bound for m in row])
    failing = np.flatnonzero(~ok)
    worst = int(np.argmax(row))
    return dict(first_failing_layer=int(failing[0]) if failing.size else None,
                worst_layer=worst, worst=float(row[worst]),
                passing_layers=int(ok.sum()), layers=int(table.shape[0]))


def classify(rel, positions, bound=BOUND):
    """Per-tensor growth verdict per route, plus how much better the passing one is.

    `p0_over_median` compares the median error at the FIRST position with the
    route's overall median: near or above 1 means the deficit is already full
    size on the first token, which rules out history accumulation.
    """
    out = {}
    medians = {}
    for route, table in rel.items():
        table = np.asarray(table, np.float64)
        if table.ndim != 2 or table.shape[1] != len(positions):
            raise ValueError(f'{route}: expected one column per position')
        first = float(np.median(table[:, 0]))
        overall = float(np.median(table))
        last = float(np.median(table[:, -1]))
        medians[route] = overall
        if overall == 0.0:
            kind, ratio = 'exact', 0.0
        else:
            ratio = last / overall
            # A ramp reaches >=4x its median at the end; a flat deficit does not.
            kind = 'accumulation' if ratio >= 4.0 else 'per-token'
        out[route] = dict(kind=kind, median=overall, p0=first, last=last,
                          p0_over_median=(first / overall) if overall else None,
                          last_over_median=(ratio if overall else None),
                          passed=bool(overall <= bound))
    if 'sequential' in medians and 'hybrid' in medians and medians['hybrid']:
        out['ratio'] = medians['sequential'] / medians['hybrid']
    return out


def profile(capture, reference, length, layers, positions=None):
    """(layers x length) maxrel tables for both routes against the reference."""
    capture, reference = Path(capture), Path(reference)
    rows = length + 1                                  # the continuation row is ignored
    rel = {r: np.zeros((layers, length), np.float64) for r in ROUTES}
    for layer in range(layers):
        ref = read_layer(reference / f'ref_res{layer}.bin', length)
        for route in ROUTES:
            got = read_layer(capture / f'{route}_res{layer}.bin', rows)[:length]
            for p in range(length):
                value = rel_error(got[p], ref[p])
                # A row with no scale contributes no relative error; 0.0 here
                # would silently look like a perfect result.
                rel[route][layer, p] = np.nan if value is None else value
    return rel


def run(a):
    rel = profile(a.capture, a.reference, a.length, a.layers)
    verdict = classify(rel, list(range(a.length)))
    summary = {route: summarise_route(rel[route]) for route in ROUTES}
    result = dict(complete=True, model_acceptance=False,
                  capture=str(a.capture.resolve()), reference=str(a.reference.resolve()),
                  length=a.length, layers=a.layers, bound=BOUND,
                  routes=verdict, summary=summary,
                  scope='retained captures only; profile of the sequential route against '
                        'the passing hybrid route; no NPU run and no acceptance')
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for route in ROUTES:
        s, v = summary[route], verdict[route]
        print(f'{route:11s} median {v["median"]:.3e}  P0/median {v["p0_over_median"]:.2f}  '
              f'last/median {v["last_over_median"]:.2f}  {v["kind"]:12s}  '
              f'first failing L{s["first_failing_layer"]}  worst L{s["worst_layer"]} '
              f'{s["worst"]:.4f}  passing {s["passing_layers"]}/{s["layers"]}')
    print(f'sequential is {verdict["ratio"]:.1f}x worse than hybrid at the same layers')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--length', type=int, default=257)
    p.add_argument('--layers', type=int, default=64)
    p.add_argument('--out', type=Path)
    raise SystemExit(run(p.parse_args()))
