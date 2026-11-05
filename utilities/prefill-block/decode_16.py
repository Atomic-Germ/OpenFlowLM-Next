"""Phase 27 closure item 8: a block-prefilled prompt must decode 16 tokens correctly.

The prefill route is only half of a request. If the block path left the KV cache or the
DeltaNet state anywhere the sequential path would not, the damage shows up as the generated
tokens drifting away from the sequential oracle -- which is why the plan asks for a
multi-block prompt plus 16 greedy decode tokens, not just the first token.

Compared here, on the rebuilt shipped binary:

  * the 16 greedy token IDs, one at a time, against forced sequential;
  * the text of the two answers, as a second, human-readable opinion.

Cosine is not used as the gate: `oflm` reports no logits, so the trajectory is judged on the
tokens it actually emits, which is stricter than a cosine on a hidden row. `--max-tokens 16`
with `temp 0` and `topk 1` makes the sequence deterministic, so "agrees" means exactly equal.

The block run forces the block loop with `OFLM_OPEN_LAYER_MAJOR=0`: layer-major is this
model's shipped default and would otherwise hide the code path under test. The prompt is
long enough for several blocks (256-wide), so the decode steps inherit a cache written by
more than one pass.

    python decode_16.py                       # 700 text tokens, 16 decode tokens
    python decode_16.py --text-tokens 300 --decode 16
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


def run(text: str, *, block: bool, decode: int, ctx: int, timeout: float) -> dict:
    extra = {'OFLM_OPEN_PREFILL_ROUTE': '1'}
    if block:
        extra['OFLM_OPEN_LAYER_MAJOR'] = '0'
    rec = D.run([text], block=block, gen_lim=decode, ctx=ctx, timeout=timeout,
                tag='blk' if block else 'seq', extra=extra)
    p = C.parse(rec['text'])
    return dict(rc=rec['rc'], wall_s=rec['wall_s'], output=rec['output'],
                engine_tokens=sum(p['chunks']),
                passes=[r.get('passes') for r in p['routes']],
                routes=[r['route'] for r in p['routes']])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--text-tokens', type=int, default=700)
    ap.add_argument('--decode', type=int, default=16)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    tok = D.Tokenizer()
    text = tok.exact_text(a.text_tokens)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'decode16-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-decode-16', version=1, phase=27,
                  gate='27 closure item 8: block prefill -> 16 greedy decode tokens',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, text_tokens=a.text_tokens, decode_tokens=a.decode,
                  gate_note='token-for-token equality against forced sequential; no new '
                            'tolerance was introduced')

    print(f'=== {a.text_tokens} text tokens, then {a.decode} greedy decode tokens ===',
          flush=True)
    seq = run(text, block=False, decode=a.decode, ctx=a.ctx, timeout=a.timeout)
    blk = run(text, block=True, decode=a.decode, ctx=a.ctx, timeout=a.timeout)
    result['sequential'] = seq
    result['block'] = blk
    seq_tokens = tok._t.encode(seq['output'], add_special_tokens=False).ids \
        if seq['output'] else []
    blk_tokens = tok._t.encode(blk['output'], add_special_tokens=False).ids \
        if blk['output'] else []
    n = min(len(seq_tokens), len(blk_tokens))
    agree = sum(1 for i in range(n) if seq_tokens[i] == blk_tokens[i])
    first_diff = next((i for i in range(n) if seq_tokens[i] != blk_tokens[i]), None)

    rows = [
        dict(check='both_routes_reached_the_model',
             ok=seq['engine_tokens'] > 0 and blk['engine_tokens'] > 0,
             detail=dict(sequential=seq['engine_tokens'], block=blk['engine_tokens'])),
        dict(check='the_block_side_really_ran_blocks',
             ok=blk['routes'] == ['block'] and max(blk['passes'] or [0]) >= 2,
             detail=dict(routes=blk['routes'], passes=blk['passes'])),
        dict(check='the_sequential_side_really_was_sequential',
             ok=seq['routes'] == ['sequential'], detail=dict(routes=seq['routes'])),
        dict(check=f'all_{a.decode}_decode_tokens_agree',
             ok=bool(seq['output']) and seq['output'] == blk['output'],
             detail=dict(compared_tokens=n, agreeing=agree, first_difference=first_diff,
                         sequential_text=seq['output'], block_text=blk['output'])),
    ]
    result['rows'] = rows
    result['compared_tokens'] = n
    result['agreeing_tokens'] = agree
    result['verdict'] = 'PASS' if all(r['ok'] for r in rows) else 'FAIL'
    print(f"  sequential eng={seq['engine_tokens']} {seq['wall_s']}s", flush=True)
    print(f"  block      eng={blk['engine_tokens']} passes={blk['passes']} {blk['wall_s']}s",
          flush=True)
    print(f"  text tokens agreeing: {agree}/{n}"
          f"{'' if first_diff is None else f' (first differs at {first_diff})'}", flush=True)
    for r in rows:
        print(f"  {r['check']:44s} {'PASS' if r['ok'] else 'FAIL'}", flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
