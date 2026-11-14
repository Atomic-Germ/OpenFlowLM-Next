"""Phase 28.2/28.3: matched layer-major vs block-loop A/B through the shipped binary.

Two schedules exist for the whole prompt. Neither is a synonym for "fast": the layer-major
route runs every layer once over the entire prompt (`Core::step_gemm_prompt`, with
`stage_major` overlapping the next block's GEMM behind the current host stage), and the
block loop runs one 256-wide block through all layers before the next (`Engine::prefill`'s
loop over `Core::step_gemm_block`). Which one a request gets is decided by
`Core::layer_major_ok()` -- a capability test -- plus `OFLM_OPEN_LAYER_MAJOR`.

That a route is *supported* says nothing about whether it is *preferred*, so this measures
them against each other with everything else held equal: same binary, same weights, same
prompt ids, same engine-visible token count, `paired` precision, same context length, one
machine.

## Ordering

`layer-major all lengths, then block-loop all lengths` would let sustained device load bias
whichever half ran second, so the runs are INTERLEAVED by length and the direction flips:
256 LM, 256 BL, 512 BL, 512 LM, 1024 LM, 1024 BL, 2048 BL, 2048 LM, with 96 and 128 at the
ends. The sequence actually executed is recorded.

A route cannot be switched inside one process (`OFLM_OPEN_LAYER_MAJOR` is read once in the
Core constructor), so each point is its own process. Both halves of a pair therefore pay
one model load -- which is excluded from the measurement, not hidden:

  * the number compared is the runner's own `PREFILL_TIME` from `/status`, which brackets
    prompt ingestion only (no load, no decode, no sampling);
  * `/clear` before the measured prompt resets every profiler, so the sample is that prompt
    alone;
  * wall time is recorded too, and is expected to differ by the load.

## Lengths

Engine-visible tokens, as always: 96, 128, 256, 512, 1024, 2048. The rows that decide this
are 256 and up, because below one block the block route is a padded dispatch and the
comparison is about the tail policy rather than the schedule. 4096 stays gated.

    python route_ab.py                      # the full interleaved matrix
    python route_ab.py --lengths 256 512    # smoke
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
import crossover_local as C  # noqa: E402

_ROUTE = re.compile(r'prefill-route (block|layer-major|sequential) n=(\d+)')
_STATUS = re.compile(r'(Total tokens|Total time|Decoding time|Prefill time):\s+([\d.eE+-]+)'
                     r'(?:\s+\((\d+)\))?\s*(\S*)')

# A prompt whose engine count is short of the request still has to be reported as such, so
# the same text is used for both routes of a pair and the two engine counts must match.
WARMUP_ENGINE = 32


def measure(text: str, *, route: str, gen_lim: int, ctx: int, timeout: float,
            warm: bool) -> dict:
    """One prompt, one /status sample, and the route the engine says it took."""
    extra = {'OFLM_OPEN_PREFILL_ROUTE': '1'}
    if route == 'block':
        extra['OFLM_OPEN_LAYER_MAJOR'] = '0'
    session: list[str] = ['/clear']
    if warm:
        session += [text, '/clear']
    session += [text, '/status']
    rec = D.run(session, block=True, gen_lim=gen_lim, ctx=ctx, timeout=timeout,
                tag=route, extra=extra)
    if rec['rc'] != 0:
        raise SystemExit(f'{route} failed (rc {rec["rc"]}):\n'
                         + '\n'.join(rec['text'].strip().splitlines()[-15:]))
    p = C.parse(rec['text'])
    routes = [(m.group(1), int(m.group(2))) for m in _ROUTE.finditer(rec['text'])]
    status = []
    for line in rec['stdout'].splitlines():
        m = _STATUS.search(line)
        if not m:
            continue
        field, value, paren, unit = m.groups()
        if unit in C._TO_MS:
            status.append(dict(field=field, ms=float(value) * C._TO_MS[unit]))
    prefill = [s for s in status if s['field'] == 'Prefill time']
    chunks = p['chunks']
    # With a warmup the LAST /status is the measured prompt's.
    return dict(route=route, rc=rec['rc'], wall_s=rec['wall_s'],
                engine_tokens=chunks[-1] if chunks else 0,
                all_engine_tokens=chunks, routes=routes,
                reported_route=routes[-1][0] if routes else 'none',
                prefill_ms=round(prefill[-1]['ms'], 1) if prefill else None,
                passes=max((r.get('passes') or 0) for r in p['routes']) if p['routes'] else 0)


def expected_passes(engine: int, physical: int = 256, crossover: int = 96) -> int:
    if engine < crossover:
        return 0
    full, tail = divmod(engine, physical)
    return full + (1 if tail >= crossover else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lengths', type=int, nargs='+', default=[96, 128, 256, 512, 1024, 2048])
    ap.add_argument('--gen-lim', type=int, default=1,
                    help='1 keeps decode out of the picture; TTFT does not depend on it')
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--timeout', type=float, default=7200)
    ap.add_argument('--no-warmup', action='store_true',
                    help='skip the short warm prompt (the first dispatch of each context '
                         'pays for it, so this makes the first point noisy)')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    tok = D.Tokenizer()
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'route-ab-{stamp}')
    out.mkdir(parents=True)

    # Interleaved order, direction alternating, so neither route sits entirely in one half
    # of the session. Recorded verbatim: the order is part of the evidence.
    order: list[tuple[int, str]] = []
    for i, n in enumerate(a.lengths):
        pair = ('layer-major', 'block') if i % 2 == 0 else ('block', 'layer-major')
        order += [(n, pair[0]), (n, pair[1])]

    result = dict(tool='prefill-block-route-ab', version=1, phase=28,
                  gate='28.2/28.3 matched layer-major vs block-loop A/B',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, lengths=a.lengths,
                  run_order=[f'{r}@{n}' for n, r in order],
                  ordering_note='interleaved by length, direction alternating, so a '
                                'thermal drift cannot land on one route',
                  timing='runner PREFILL_TIME per prompt; /clear resets the profilers; '
                         'wall includes the one model load and decode',
                  precision='paired', gen_lim=a.gen_lim, ctx=a.ctx,
                  physical_block=256, block_crossover=96)
    print(f'=== route A/B, {len(order)} runs, interleaved ===', flush=True)
    print('  order: ' + ', '.join(f'{r}@{n}' for n, r in order), flush=True)

    rows: list[dict] = []
    for n, route in order:
        text = tok.exact_text(n)
        r = measure(text, route=route, gen_lim=a.gen_lim, ctx=a.ctx, timeout=a.timeout,
                    warm=not a.no_warmup)
        r['requested'] = n
        r['text_tokens'] = tok.count(text)
        want = 'layer-major' if route == 'layer-major' else 'block'
        r['route_ok'] = r['reported_route'] == want
        r['passes_ok'] = (r['passes'] == expected_passes(r['engine_tokens'])
                          if route == 'block' else None)
        rows.append(r)
        print(f"  {route:11s} req={n:5d} eng={r['engine_tokens']:5d} "
              f"passes={r['passes']} prefill={r['prefill_ms']}ms wall={r['wall_s']}s "
              f"reported={r['reported_route']} {'ok' if r['route_ok'] else 'WRONG ROUTE'}",
              flush=True)

    # Pair them up. Both halves must have reached the engine as the same number of tokens or
    # the comparison is between two different prompts.
    by: dict[int, dict[str, dict]] = {}
    problems = []
    for r in rows:
        by.setdefault(r['requested'], {})[r['route']] = r
        if not r['route_ok']:
            problems.append(f"{r['route']}@{r['requested']}: the engine reported "
                            f"{r['reported_route']}")
    pairs = []
    for n in a.lengths:
        if n not in by or len(by[n]) != 2:
            continue
        lm, bl = by[n].get('layer-major'), by[n].get('block')
        if not (lm and bl):
            continue
        if lm['engine_tokens'] != bl['engine_tokens']:
            problems.append(f'{n}: routes saw {lm["engine_tokens"]} vs '
                            f'{bl["engine_tokens"]} engine tokens')
            continue
        if not (lm['prefill_ms'] and bl['prefill_ms']):
            problems.append(f'{n}: no PREFILL_TIME from one side')
            continue
        slow, fast = max(lm['prefill_ms'], bl['prefill_ms']), min(lm['prefill_ms'],
                                                                 bl['prefill_ms'])
        pairs.append(dict(engine_tokens=lm['engine_tokens'],
                          layer_major_ms=lm['prefill_ms'], block_ms=bl['prefill_ms'],
                          layer_major_tok_s=round(lm['engine_tokens'] / (lm['prefill_ms'] / 1e3), 2),
                          block_tok_s=round(bl['engine_tokens'] / (bl['prefill_ms'] / 1e3), 2),
                          winner='layer-major' if lm['prefill_ms'] < bl['prefill_ms'] else 'block',
                          speedup=round(slow / fast, 3)))
    result.update(rows=rows, pairs=pairs, problems=problems)
    result['verdict'] = 'PASS' if not problems and len(pairs) == len(a.lengths) else 'FAIL'

    print(f'\n{"engine":>7} {"layer-major":>12} {"block":>10} {"winner":>12} {"ratio":>7}')
    for p in pairs:
        print(f"{p['engine_tokens']:7d} {p['layer_major_ms']:10.0f}ms {p['block_ms']:8.0f}ms "
              f"{p['winner']:>12} {p['speedup']:>7}")
    if problems:
        print('problems:')
        for p in problems:
            print(f'  - {p}')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
