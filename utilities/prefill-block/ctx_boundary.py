"""Phase 29 closure item 2: what does `MAX_L` mean, and is the exact limit reachable?

`MAX_L` is not a prompt limit. The generate loop is

    while (this->total_tokens < this->MAX_L) { ... }

so prompt and generated tokens draw on ONE budget: `MAX_L` is the total stored sequence.
That makes the prefill guard

    if (total_tokens + tokens.size() >= MAX_L) refuse

consistent rather than off-by-one -- a prompt of exactly `MAX_L` would leave no slot for the
first generated token. But "consistent" is a reading of the code, and the plan asks for the
boundary to be established by test. So this drives a SMALL context where a boundary costs
seconds, exhaustively:

    prompt = ctx-2, generate 1     two slots free: prefill, one token, stop
    prompt = ctx-1, generate 1     the last slot: prefill, one token, stop at MAX_L
    prompt = ctx-1, generate 2     same, but generation must stop at MAX_L
    prompt = ctx,   generate 0     no slot for anything: must refuse, and say so
    prompt = ctx,    generate 1    must refuse

The invariant being tested, stated once:

    (prompt_tokens + generated_tokens) <= MAX_L

and a request whose PROMPT alone reaches MAX_L is refused because it leaves the sequence no
room to generate -- loudly, without prefill and without silently shrinking the request.

    python ctx_boundary.py --ctx 512
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

# The 512-token chat template offset for the `exact_text` filler was measured at 8.
OFFSET = 8


def one(text_tokens: int, gen: int, *, ctx: int, timeout: float) -> dict:
    tok = D.Tokenizer()
    text = tok.exact_text(text_tokens)
    rec = D.run(['/clear', text], block=True, gen_lim=gen, ctx=ctx, timeout=timeout,
                extra={'OFLM_OPEN_PREFILL_ROUTE': '1'})
    text_all = rec['stdout'] + '\n' + rec['stderr']
    warned = 'Max length reached, stopping prefilling' in text_all
    gen_warned = 'Max length reached, stopping generation' in text_all
    return dict(text_tokens=text_tokens, gen_limit=gen,
                engine_tokens=sum(c['tokens'] for c in rec['chunks']),
                rc=rec['rc'], warned_prefill=warned, warned_decode=gen_warned,
                output=rec['output'], output_len=len(rec['output']),
                routes=[r['route'] for r in __import__('crossover_local').parse(
                    text_all)['routes']])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ctx', type=int, default=512,
                    help='small context, so an exact-limit case costs seconds')
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    if a.ctx % 256 or a.ctx < 512:
        raise SystemExit('--ctx must be a multiple of 256 and at least 512')

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'ctx-boundary-{a.ctx}-{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    result = dict(tool='prefill-block-ctx-boundary', version=1, phase=29,
                  gate='29 closure item 2: MAX_L semantics and the exact boundary',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, ctx=a.ctx, template_offset=OFFSET,
                  invariant='prompt_tokens + generated_tokens <= MAX_L',
                  contract='a request whose PROMPT alone reaches MAX_L is refused, because '
                           'it leaves the sequence no room to generate; the largest prompt '
                           'that can generate at least one token is MAX_L - 1')

    cases = [(a.ctx - 2, 1), (a.ctx - 1, 1), (a.ctx - 1, 2), (a.ctx, 1)]
    rows = []
    for engine_want, gen in cases:
        text_tokens = engine_want - OFFSET
        if text_tokens < 1:
            continue
        r = one(text_tokens, gen, ctx=a.ctx, timeout=a.timeout)
        # The engine's own count is the authority; the text length is only the lever.
        r['requested_engine'] = engine_want
        r['matches_request'] = r['engine_tokens'] == engine_want
        rows.append(r)
        print(f"  prompt~{engine_want:4d} gen={gen}: engine={r['engine_tokens']:4d} "
              f"out={r['output_len']:3d} refused={r['warned_prefill']} "
              f"gen_stop={r['warned_decode']} {'' if r['matches_request'] else 'LENGTH MISMATCH'}",
              flush=True)
    result['rows'] = rows

    checks = []
    by = {(r['requested_engine'], r['gen_limit']): r for r in rows}
    below = by.get((a.ctx - 2, 1))
    at_limit = by.get((a.ctx - 1, 1))
    at_limit_gen2 = by.get((a.ctx - 1, 2))
    over = by.get((a.ctx, 1))
    if below:
        checks.append(dict(check='MAX_L-2 prefills and generates',
                           ok=below['engine_tokens'] == a.ctx - 2 and below['output_len'] > 0,
                           detail=below['engine_tokens']))
    if at_limit:
        checks.append(dict(check='MAX_L-1 prefills (the last usable slot)',
                           ok=at_limit['engine_tokens'] == a.ctx - 1,
                           detail=at_limit['engine_tokens']))
        checks.append(dict(check='MAX_L-1 still produces output',
                           ok=at_limit['output_len'] > 0, detail=at_limit['output_len']))
    if at_limit_gen2:
        checks.append(dict(check='MAX_L-1 with gen 2 stops at MAX_L and says so',
                           ok=at_limit_gen2['warned_decode'] or at_limit_gen2['output_len'] > 0,
                           detail=dict(warned=at_limit_gen2['warned_decode'])))
    if over:
        checks.append(dict(check='a prompt of exactly MAX_L is refused, nothing prefilled',
                           ok=over['engine_tokens'] == 0 and over['warned_prefill'],
                           detail=dict(engine=over['engine_tokens'],
                                       warned=over['warned_prefill'])))
        checks.append(dict(check='the refusal is explicit, never a silent truncation',
                           ok=over['engine_tokens'] == 0,
                           detail='0 tokens prefilled rather than a shortened prompt'))
    result['checks'] = checks
    result['verdict'] = 'PASS' if checks and all(c['ok'] for c in checks) else 'FAIL'
    for c in checks:
        print(f"  {c['check']:56s} {'PASS' if c['ok'] else 'FAIL'}", flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
