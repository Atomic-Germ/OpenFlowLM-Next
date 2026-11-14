"""Phase 27.6/27.8: the shipped `oflm run` across the block boundary, and request isolation.

The plan asks for 255/256/257/511/512/513 and for a prompt of more than one block. Those
lengths are the interesting ones because the physical block is 256: 255 is one block's
worth minus one (a tail with no full block), 256 is exactly one block, 257 is one block
plus a one-token tail, 511 is one short of two, 512 is two, 513 is two plus one.

Correctness here is *functional agreement between the two routes through the shipped
runtime*, not bit equality: `oflm` samples, and although the harness pins greedy
sampling, there is no requirement that the sequential and block routes agree bit for bit
-- Phase 26 showed they agree to cosine 0.999999 and produce the same token, and Phase 27
asks whether the runtime works, not whether it is bit-exact. So each length is run twice
(block on, block off) with the same greedy settings, and the model's own output text is
compared as a string. A route that crashed, silently prefilled nothing, or produced a
different first token is a FAIL, and it is named.

`--isolation` adds 27.8: in ONE process, two requests back to back. The second request's
output is compared with the same request made in a fresh process. If resident state
leaked across requests, the second one's output would differ.
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

# The chat template's fixed overhead, measured once: 255 text tokens prefilled as 266.
TEMPLATE_OFFSET = 11


def one(text_tokens: int, *, block: bool, gen_lim: int, ctx: int) -> dict:
    tok = D.Tokenizer()
    text = tok.exact_text(text_tokens)
    rec = D.run([text], block=block, gen_lim=gen_lim, ctx=ctx, tag=f'{text_tokens}')
    prefilled = D.prefilled_tokens(rec)
    return dict(text_tokens=text_tokens, expected_prefilled=text_tokens + TEMPLATE_OFFSET,
                prefilled=prefilled, chunks=rec['chunks'], rc=rec['rc'],
                wall_s=rec['wall_s'], output=rec['output'],
                # A run that reached the model prefilled the template plus the text.
                # One that did not prefills nothing and says so here rather than later.
                reached_model=prefilled >= text_tokens,
                matches_expected=(prefilled == text_tokens + TEMPLATE_OFFSET))


def matrix(lengths: list[int], *, gen_lim: int, ctx: int) -> dict:
    rows = []
    for n in lengths:
        a = one(n, block=True, gen_lim=gen_lim, ctx=ctx)
        b = one(n, block=False, gen_lim=gen_lim, ctx=ctx)
        agree = a['output'] == b['output'] and a['output'] != ''
        row = dict(text_tokens=n, block=a, sequential=b, outputs_agree=agree,
                   verdict='PASS' if (
                       a['rc'] == 0 and b['rc'] == 0 and a['reached_model']
                       and b['reached_model'] and a['matches_expected']
                       and b['matches_expected'] and agree) else 'FAIL')
        rows.append(row)
        print(f"  n={n:4d} blk {a['prefilled']:4d}tok {a['wall_s']:7.2f}s  "
              f"seq {b['prefilled']:4d}tok {b['wall_s']:7.2f}s  "
              f"agree={agree} ({a['output'][:24]!r})  {row['verdict']}", flush=True)
    bad = [r['text_tokens'] for r in rows if r['verdict'] != 'PASS']
    return dict(rows=rows, verdict='PASS' if not bad else 'FAIL', failed=bad,
                template_offset=TEMPLATE_OFFSET, gen_lim=gen_lim, ctx=ctx)


def isolation(*, text_tokens: int, gen_lim: int, ctx: int) -> dict:
    """27.8: request 2 inside the same process must match request 2 in a fresh process."""
    tok = D.Tokenizer()
    t1, t2 = tok.exact_text(text_tokens), tok.exact_text(text_tokens + 37)

    fresh2 = D.run([t2], block=True, gen_lim=gen_lim, ctx=ctx, tag='fresh2')
    same = D.run([t1, t2], block=True, gen_lim=gen_lim, ctx=ctx, tag='pair')
    # The pair run's second answer is the tail of its stdout; compare the model text of
    # each turn separately by re-running each in isolation and by splitting on the prompt.
    first = D.run([t1], block=True, gen_lim=gen_lim, ctx=ctx, tag='fresh1')
    rows = [dict(check='request2_matches_fresh', fresh=fresh2['output'],
                 paired=same['output'], verdict='PASS' if fresh2['output'] and
                 fresh2['output'] in same['output'] else 'FAIL'),
            dict(check='request1_nonempty', fresh=first['output'],
                 verdict='PASS' if first['output'] else 'FAIL'),
            dict(check='pair_prefilled_twice',
                 chunks=[c for c in same['chunks']],
                 verdict='PASS' if len(same['chunks']) >= 2 else 'FAIL')]
    bad = [r for r in rows if r['verdict'] != 'PASS']
    return dict(rows=rows, verdict='PASS' if not bad else 'FAIL',
                fresh2=fresh2, paired=same, fresh1=first)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lengths', type=int, nargs='+',
                    default=[255, 256, 257, 511, 512, 513])
    ap.add_argument('--gen-lim', type=int, default=8)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--isolation', action='store_true')
    ap.add_argument('--isolation-tokens', type=int, default=300)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'oflm-run-matrix-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-oflm-run-matrix', version=1, phase=27,
                  gate='27.6 oflm run boundary matrix / 27.8 request isolation',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp)
    print(f'=== 27.6 oflm run matrix, {a.lengths} text tokens, gen_lim {a.gen_lim} ===',
          flush=True)
    result['matrix'] = matrix(a.lengths, gen_lim=a.gen_lim, ctx=a.ctx)
    if a.isolation:
        print(f'\n=== 27.8 isolation: two requests in one process ===', flush=True)
        result['isolation'] = isolation(text_tokens=a.isolation_tokens,
                                        gen_lim=a.gen_lim, ctx=a.ctx)
        for r in result['isolation']['rows']:
            print(f"  {r['check']:26s} {r['verdict']}", flush=True)
    result['verdict'] = ('PASS' if result['matrix']['verdict'] == 'PASS' and
                         result.get('isolation', {}).get('verdict', 'PASS') == 'PASS'
                         else 'FAIL')
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"\n{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
