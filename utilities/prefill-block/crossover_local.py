"""Phase 27 closure item 5: where does the block loop actually become cheaper?

The published constant is 128, inherited from 27.4. 27.4 sampled 1, 8, 32, 64, 128, 192,
255 and reported "the smallest sampled length at which the padded block is no slower",
which returns the first grid point past the crossing rather than the crossing: its own
table (64 -> 1.18, 128 -> 0.72) crosses around 81. A constant chosen from a six-point grid
is a grid artefact, so it has to be re-measured on a fine grid through the shipped binary.

The measurement has two sides and each has to be forced, because the shipped policy will
not take a padded block below its own crossover:

  sequential side   OFLM_OPEN_GEMM_BLOCK=0. Every length must report the sequential route.
  block side        OFLM_OPEN_LAYER_MAJOR=0 (block loop rather than layer-major) AND
                    OFLM_OPEN_GEMM_BLOCK_MIN=<floor>, so a padded 256-wide pass really runs
                    at lengths the policy would otherwise step. With the floor at 1 the
                    tail rule pads any non-empty tail, so each length costs
                    ceil(n / 256) passes.

Each side is ONE process: the model loads once and every length is measured against the
same resident weights, with `/clear` between prompts resetting the profilers, so each
sample is that prompt alone -- no load, no decode, no carry-over.

Both routes send the SAME text per length, so the two sides must agree on the engine token
count. That count is measured, never computed from the text: the chat template's offset
depends on how its boundary merges with the first text token (8, 10 and 11 have all been
observed), so `text + K` is not a fact.

    python crossover_local.py --block-min 1        # the fine sweep, forced at both ends
    python crossover_local.py                      # the shipped policy, as a control
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oflm_drive as D  # noqa: E402

_ROUTE = re.compile(r'prefill-route (block|layer-major|sequential) n=(\d+) blocks=(\d+)'
                    r' tail=(\d+)(?: padded=(\d+) passes=(\d+))?')
_TAILSEQ = re.compile(r'prefill-route tail-sequential n=(\d+) tokens=(\d+)')
_STATUS = re.compile(r'(Total tokens|Prefill time):\s+([\d.eE+-]+)(?:\s+\((\d+)\))?\s*(\S*)')
# `re_unit` prints seconds above 1e6 us and milliseconds above 1e3 us: the unit is read
# off the line, never assumed.
_TO_MS = {'ns': 1e-6, 'us': 1e-3, 'ms': 1.0, 's': 1e3}


def parse(text: str) -> dict:
    out = dict(routes=[], chunks=[], status=[])
    for line in text.splitlines():
        m = _ROUTE.search(line)
        if m:
            out['routes'].append(dict(
                route=m.group(1), n=int(m.group(2)), blocks=int(m.group(3)),
                tail=int(m.group(4)),
                passes=int(m.group(6)) if m.group(6) else None))
            continue
        m = _TAILSEQ.search(line)
        if m and out['routes']:
            out['routes'][-1]['tail_sequential'] = int(m.group(2))
            continue
        m = _STATUS.search(line)
        if m:
            field, value, paren, unit = m.groups()
            out['status'].append(dict(field=field, value=float(value),
                                      paren=int(paren) if paren else None, unit=unit or ''))
            continue
        m = re.search(r'Prefill chunk \d+/\d+ with (\d+) tokens', line)
        if m:
            out['chunks'].append(int(m.group(1)))
    return out


def expected(n: int, *, side: str, floor: int, physical: int = 256) -> dict:
    """What the route line must say, for each SIDE of the measurement.

    The sequential side forces the switch off, so it is sequential at every length. The
    block side is run with an explicit floor, so the policy that applies is the one at
    that floor, not the shipped one.
    """
    if side == 'sequential':
        return dict(route='sequential', blocks=0)
    if n < floor:
        return dict(route='sequential', blocks=0)
    full, tail = divmod(n, physical)
    return dict(route='block', blocks=full,
                tail_sequential=tail if tail < floor else 0)


def sweep(texts: list[str], *, side: str, floor: int, gen_lim: int, ctx: int,
          timeout: float) -> dict:
    session: list[str] = []
    for text in texts:
        session += ['/clear', text, '/status']
    extra = {'OFLM_OPEN_PREFILL_ROUTE': '1'}
    if side == 'block':
        extra['OFLM_OPEN_LAYER_MAJOR'] = '0'
        extra['OFLM_OPEN_GEMM_BLOCK_MIN'] = str(floor)
    else:
        extra['OFLM_OPEN_GEMM_BLOCK'] = '0'
    rec = D.run(session, block=True, gen_lim=gen_lim, ctx=ctx, timeout=timeout,
                tag=f'cross-{side}', extra=extra)
    p = parse(rec['text'])
    if rec['rc'] != 0:
        raise SystemExit(f'{side} sweep failed (rc {rec["rc"]}):\n'
                         + '\n'.join(rec['text'].strip().splitlines()[-15:]))
    for name, got in (('route', len(p['routes'])), ('chunk', len(p['chunks']))):
        if got != len(texts):
            raise SystemExit(f'{side}: expected {len(texts)} {name} lines, saw {got}; the '
                             'CLI may have merged a line into a paste')
    prefill = [s for s in p['status'] if s['field'] == 'Prefill time']
    total = [s for s in p['status'] if s['field'] == 'Total tokens']
    if len(prefill) != len(texts):
        raise SystemExit(f'{side}: expected {len(texts)} /status blocks, saw {len(prefill)}')
    tok = D.Tokenizer()
    rows = []
    for text, chunk, rt, sample, tot in zip(texts, p['chunks'], p['routes'], prefill, total):
        want = expected(chunk, side=side, floor=floor)
        ok = rt['route'] == want['route'] and rt['blocks'] == want['blocks']
        if want['route'] == 'block' and want.get('tail_sequential') is not None:
            ok = ok and rt.get('tail_sequential', 0) == want['tail_sequential']
        # `Total tokens` is the runner's independent count of the context it built; it
        # must agree with what the prefill reported, or the two numbers are not the same
        # prompt and no timing below means anything.
        ctx_ok = tot['paren'] is None or tot['paren'] >= chunk
        rows.append(dict(text_tokens=tok.count(text), engine_tokens=chunk,
                         total_tokens=tot['paren'], context_ok=ctx_ok,
                         route=rt['route'], blocks=rt['blocks'], tail=rt['tail'],
                         passes=rt['passes'], tail_sequential=rt.get('tail_sequential', 0),
                         prefill_ms=round(sample['value'] * _TO_MS[sample['unit']], 1),
                         expected=want, route_ok=ok))
    return dict(side=side, floor=floor, rc=rec['rc'], wall_s=rec['wall_s'], rows=rows,
                command=rec['command'], env=extra)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # A fine grid around the place the two lines meet: 27.4's own numbers put it near 80,
    # so the sweep is dense there and sparse above.
    ap.add_argument('--text-tokens', type=int, nargs='+',
                    default=[48, 64, 72, 80, 88, 96, 104, 112, 120, 128])
    ap.add_argument('--block-min', type=int, default=1,
                    help='floor used to force the padded block on the block side; the '
                         'shipped policy is reproduced with 128')
    ap.add_argument('--gen-lim', type=int, default=1)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--timeout', type=float, default=7200)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    tok = D.Tokenizer()
    texts = [tok.exact_text(n) for n in a.text_tokens]
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'crossover-local-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-crossover-local', version=3, phase=27,
                  gate='27 closure item 5: block-loop crossover on a fine grid',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, requested_text_tokens=a.text_tokens, block_min=a.block_min,
                  physical_block=256, gen_lim=a.gen_lim, ctx=a.ctx,
                  timing='runner PREFILL_TIME per prompt (no load, no decode)',
                  why_forced='the shipped policy will not pad a block below its own '
                             'crossover, so the block side is forced with '
                             'OFLM_OPEN_GEMM_BLOCK_MIN to measure it there',
                  note='OFLM_OPEN_LAYER_MAJOR=0 is required: layer-major is the shipped '
                       'default route on this model and never enters the block loop')
    print(f'=== fine crossover sweep, block floor {a.block_min}, '
          f'{len(texts)} lengths ===', flush=True)

    sides = {}
    for side in ('sequential', 'block'):
        s = sweep(texts, side=side, floor=a.block_min, gen_lim=a.gen_lim, ctx=a.ctx,
                  timeout=a.timeout)
        sides[side] = s
        print(f'--- {side} side (one process, one load, wall {s["wall_s"]} s) ---',
              flush=True)
        for r in s['rows']:
            verdict = 'ok' if r['route_ok'] else 'ROUTE DIFFERS'
            print(f"  text={r['text_tokens']:4d} eng={r['engine_tokens']:4d} "
                  f"{r['route']:11s} passes={r['passes']} "
                  f"{r['prefill_ms']:9.0f} ms  {verdict}", flush=True)

    seq = {r['text_tokens']: r for r in sides['sequential']['rows']}
    blk = {r['text_tokens']: r for r in sides['block']['rows']}
    rows, bad = [], []
    for t in a.text_tokens:
        s, b = seq[t], blk[t]
        if s['engine_tokens'] != b['engine_tokens']:
            bad.append(f'text {t}: the two sides saw {s["engine_tokens"]} vs '
                       f'{b["engine_tokens"]} engine tokens')
        for name, r in (('sequential', s), ('block', b)):
            if not r['route_ok']:
                bad.append(f'text {t}: {name} side reported {r["route"]}/blocks '
                           f'{r["blocks"]}/tailseq {r["tail_sequential"]}, '
                           f'expected {r["expected"]}')
            if not r['context_ok']:
                bad.append(f'text {t}: {name} side context {r["total_tokens"]} < prefill '
                           f'{r["engine_tokens"]}')
        ratio = round(b['prefill_ms'] / s['prefill_ms'], 3) if s['prefill_ms'] else None
        # Only a point where the block side REALLY took a padded pass says anything about
        # the crossover. A point where it fell below its floor is the sequential path
        # twice, and its ratio is noise around 1.0 -- counting it was the first version's
        # mistake, and it reported a "crossover" of 104 from two sequential runs.
        decisive = b['route'] == 'block'
        rows.append(dict(text_tokens=t, engine_tokens=s['engine_tokens'],
                         sequential_ms=s['prefill_ms'], block_ms=b['prefill_ms'],
                         block_passes=b['passes'], ratio=ratio, decisive=decisive,
                         block_wins=(ratio is not None and ratio < 1.0)))
    usable = [r for r in rows if r['decisive'] and r['ratio'] is not None]
    wins = [r['engine_tokens'] for r in usable if r['block_wins']]
    measured = min(wins) if wins else None

    # A linear fit through the sequential side gives the crossing directly instead of
    # snapping it to whichever grid point happens to sit either side of it: the block cost
    # is flat, so seq(n) = block is the crossing.
    fit = None
    pts = [(r['engine_tokens'], r['sequential_ms']) for r in rows if r['engine_tokens'] >= 2]
    if len(pts) >= 2:
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        den = sum((p[0] - mx) ** 2 for p in pts)
        if den > 0:
            slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / den
            intercept = my - slope * mx
            flat = [r['block_ms'] for r in usable if r['block_passes']]
            if flat and slope > 0:
                block_cost = sorted(flat)[len(flat) // 2]     # median of the flat passes
                fit = dict(slope_ms_per_token=round(slope, 2),
                           intercept_ms=round(intercept, 1),
                           block_cost_ms=block_cost,
                           crossing_tokens=round((block_cost - intercept) / slope, 1))
    result.update(sides=sides, rows=rows, decisive_points=len(usable),
                  first_decisive_block_win=measured, linear_fit=fit,
                  grid_step_note='the crossing is a property of the two cost curves, not '
                                 'of the sampled grid; the fit gives it without snapping',
                  problems=bad)
    result['verdict'] = 'PASS' if not bad and usable else 'FAIL'

    print(f'\n{"text":>5} {"engine":>7} {"sequential":>11} {"block":>9} {"ratio":>7}  '
          f'{"decisive":>9}')
    for r in rows:
        print(f"{r['text_tokens']:5d} {r['engine_tokens']:7d} "
              f"{r['sequential_ms']:10.0f}ms {r['block_ms']:8.0f}ms "
              f"{r['ratio'] if r['ratio'] is not None else '-':>7}  "
              f"{'yes' if r['decisive'] else 'no':>9}")
    print(f'\nfirst measured block win = {measured} engine tokens')
    if fit:
        print(f"linear fit: sequential {fit['slope_ms_per_token']} ms/token "
              f"+ {fit['intercept_ms']} ms; flat block {fit['block_cost_ms']} ms "
              f"-> crossing {fit['crossing_tokens']} engine tokens")
    if bad:
        print('problems:')
        for b in bad:
            print(f'  - {b}')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
