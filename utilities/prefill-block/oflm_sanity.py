"""Phase 27 closure item 9: a small performance sanity check through the shipped binary.

This is NOT the production profile -- Phase 28 owns that. It answers one question only:
does the cost behave the way the schedule says it should, at four lengths that span one,
two and four blocks?

  * cost roughly CONSTANT in the active rows inside a block (a padded dispatch is
    full-width whatever it holds), and
  * cost roughly LINEAR in the number of blocks,

with wall, TTFT and the route the engine reports for each length. TTFT is the runner's own
`PREFILL_TIME` from `/status`, which is the prompt ingestion alone: the wall figure also
carries the one model load and the decode, so it cannot show the block structure on its own.

The block loop is forced with `OFLM_OPEN_LAYER_MAJOR=0` because layer-major is this model's
shipped default; that is a property of the route under test, not of the timings, and it is
recorded in the artefact. No new performance claim is published from this run.

    python oflm_sanity.py                  # 128 / 256 / 512 / 1024 text tokens
    python oflm_sanity.py --lengths 256 512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oflm_drive as D  # noqa: E402
import crossover_local as C  # noqa: E402

PHYSICAL = 256
# The shipped block-loop crossover. A tail shorter than this is stepped, not padded into
# another dispatch, so the pass count is NOT ceil(n / 256).
CROSSOVER = 96


def expected_passes(n: int, crossover: int = CROSSOVER, physical: int = PHYSICAL) -> int:
    """The policy's own prediction: full blocks, plus one more only if the tail earns it."""
    if n < crossover:
        return 0
    full, tail = divmod(n, physical)
    return full + (1 if tail >= crossover else 0)


def measure(n: int, *, gen_lim: int, ctx: int, timeout: float) -> dict:
    """One prompt, one route line, one /status sample."""
    tok = D.Tokenizer()
    text = tok.exact_text(n)
    extra = {'OFLM_OPEN_PREFILL_ROUTE': '1', 'OFLM_OPEN_LAYER_MAJOR': '0'}
    rec = D.run(['/clear', text, '/status'], block=True, gen_lim=gen_lim, ctx=ctx,
                timeout=timeout, tag=f'n{n}', extra=extra)
    p = C.parse(rec['text'])
    # The status block follows the prompt, so its Prefill time is this prompt's. Only the
    # timed fields carry a unit -- `Total tokens: 63 (66)` matches the same shape with an
    # empty one, and indexing the unit table with that raised instead of being ignored.
    status = {}
    for s in p['status']:
        if s['unit'] in C._TO_MS:
            status[s['field']] = s['value'] * C._TO_MS[s['unit']]
    route = p['routes'][0] if p['routes'] else {}
    eng = sum(p['chunks'])
    return dict(text_tokens=n, engine_tokens=eng,
                route=route.get('route'), blocks=route.get('blocks'),
                tail=route.get('tail'), passes=route.get('passes'),
                tail_sequential=route.get('tail_sequential', 0),
                expected_passes=expected_passes(eng),
                wall_s=rec['wall_s'], ttft_ms=round(status.get('Prefill time', -1), 1),
                rc=rec['rc'])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lengths', type=int, nargs='+', default=[128, 256, 512, 1024])
    ap.add_argument('--gen-lim', type=int, default=8)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'oflm-sanity-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-oflm-sanity', version=1, phase=27,
                  gate='27 closure item 9: performance sanity, not a production profile',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, physical_block=PHYSICAL, gen_lim=a.gen_lim, ctx=a.ctx,
                  ttft_definition='runner PREFILL_TIME per prompt (/clear resets it)',
                  not_a_claim='Phase 28 owns the production profile; nothing here is a '
                              'published performance number',
                  note='block loop forced with OFLM_OPEN_LAYER_MAJOR=0; layer-major is the '
                       'shipped default on this model')
    print('=== sanity ===', flush=True)
    rows = []
    for n in a.lengths:
        r = measure(n, gen_lim=a.gen_lim, ctx=a.ctx, timeout=a.timeout)
        rows.append(r)
        print(f"  text={n:5d} eng={r['engine_tokens']:5d} passes={r['passes']} "
              f"(expected {r['expected_passes']}) wall={r['wall_s']:7.2f}s "
              f"ttft={r['ttft_ms']:8.0f}ms route={r['route']}", flush=True)

    checks = []
    for r in rows:
        checks.append(dict(check=f"route_{r['text_tokens']}_is_the_block_loop",
                           ok=r['route'] == 'block' and r['rc'] == 0,
                           detail=dict(route=r['route'], rc=r['rc'])))
        # The pass count is the schedule's own prediction from the engine token count.
        checks.append(dict(check=f"passes_{r['text_tokens']}_match_the_prediction",
                           ok=r['passes'] == r['expected_passes'],
                           detail=dict(passes=r['passes'],
                                       predicted=r['expected_passes'])))
    # The two shape expectations. Both are ratios, so they do not depend on the machine's
    # absolute speed -- and neither is a published number.
    by = {r['text_tokens']: r for r in rows}
    if 128 in by and 256 in by:
        same_block = by[256]['ttft_ms'] / by[128]['ttft_ms']
        checks.append(dict(check='one_block_costs_about_the_same_at_128_and_256',
                           ok=0.75 <= same_block <= 1.35,
                           detail=dict(ratio=round(same_block, 3),
                                       why='both are a single padded dispatch')))
    growth = [(by[n]['passes'], by[n]['ttft_ms']) for n in a.lengths if n in by
              and by[n]['passes']]
    if len(growth) >= 2:
        per_pass = [ms / p for p, ms in growth]
        spread = max(per_pass) / min(per_pass) if min(per_pass) else None
        checks.append(dict(check='cost_is_linear_in_blocks_not_in_active_rows',
                           ok=spread is not None and spread <= 1.6,
                           detail=dict(passes=[p for p, _ in growth],
                                       ms_per_pass=[round(x, 1) for x in per_pass],
                                       spread=round(spread, 3) if spread else None,
                                       why='ms per pass must stay together as blocks are '
                                           'added; a ramp with active rows would not')))

    result['rows'] = rows
    result['checks'] = checks
    result['verdict'] = 'PASS' if all(c['ok'] for c in checks) else 'FAIL'
    for c in checks:
        print(f"  {c['check']:56s} {'PASS' if c['ok'] else 'FAIL'}", flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
