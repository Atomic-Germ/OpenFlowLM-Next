"""Phase 31.E: decode throughput with load AND prefill excluded.

Wall time cannot answer this: a run carries the model load (~90 s) and the prefill (tens of
seconds) in front of every generated token. So the number comes from the runner's own
`DECODING_TIME` profiler, which brackets the generation loop only:

    /clear            resets every profiler
    <prompt>          prefill -- lands in PREFILL_TIME, not DECODING_TIME
    /status           prints the split

and the decode rate is `generated / decode_seconds` from that single request. `/status` also
prints the counter it divided by, so the token count is the runner's, not a guess.

Two rates are reported because they answer different questions:

    decode tok/s   the generation loop's own rate, after its first token is already out
    end-to-end     generated / (prompt wall) -- what a user of `run` sees

The first is the frozen baseline. The second is printed next to it so nobody quotes one for
the other.

    python decode_rate.py --tokens 32
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

# `re_unit` prints seconds above 1e6 us, milliseconds above 1e3 -- the unit is read, not assumed.
_TO_S = {'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1.0}
_STATUS = re.compile(r'(Total time|Decoding time|Prefill time):\s+([\d.eE+-]+)\s+(\S+)')
_DECSPEED = re.compile(r'Average decoding speed:\s+([\d.eE+-]+)\s+tokens/s')
_PREFSPEED = re.compile(r'Average prefill\s+speed:\s+([\d.eE+-]+)\s+tokens/s')
_TOTTOK = re.compile(r'Total tokens:\s+(\d+)\s+\((\d+)\)')


def one(prompt_text_tokens: int, *, tokens: int, ctx: int, timeout: float,
        extra: dict | None = None) -> dict:
    tok = D.Tokenizer()
    text = tok.exact_text(prompt_text_tokens)
    rec = D.run(['/clear', text, '/status'], block=True, gen_lim=tokens, ctx=ctx,
                timeout=timeout, tag=f'd{tokens}', extra=extra)
    out = rec['stdout']
    prof = {}
    for line in out.splitlines():
        m = _STATUS.search(line)
        if m:
            prof[m.group(1)] = round(float(m.group(2)) * _TO_S[m.group(3)], 3)
    speed = _DECSPEED.search(out)
    pspeed = _PREFSPEED.search(out)
    tots = _TOTTOK.search(out)
    decode_s = prof.get('Decoding time')
    return dict(prompt_text_tokens=prompt_text_tokens,
                engine_tokens=sum(c['tokens'] for c in rec['chunks']),
                requested_generated=tokens, rc=rec['rc'], wall_s=rec['wall_s'],
                profile=prof,
                total_context=int(tots.group(2)) if tots else None,
                generated=(int(tots.group(2)) - sum(c['tokens'] for c in rec['chunks']))
                if tots else None,
                decode_s=decode_s,
                decode_tok_s=round(tokens / decode_s, 3) if decode_s else None,
                runner_decode_tok_s=float(speed.group(1)) if speed else None,
                runner_prefill_tok_s=float(pspeed.group(1)) if pspeed else None,
                output_len=len(rec['output']), output=rec['output'])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prompt', type=int, nargs='+', default=[40, 300],
                    help='prompt TEXT token counts; one measurement each')
    ap.add_argument('--tokens', type=int, default=32)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'decode-rate-{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    result = dict(tool='prefill-block-decode-rate', version=1, phase=31,
                  gate='31.E decode throughput, load and prefill excluded',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, tokens=a.tokens, ctx=a.ctx,
                  method='runner DECODING_TIME from /status after /clear; the decode loop '
                         'only, so the load and the prefill are outside it',
                  two_rates='decode_tok_s is the loop rate; end_to_end_tok_s is what a user '
                            'of `run` sees, and they are never interchangeable')
    rows = []
    for p in a.prompt:
        r = one(p, tokens=a.tokens, ctx=a.ctx, timeout=a.timeout)
        r['end_to_end_tok_s'] = round(a.tokens / r['wall_s'], 3) if r['wall_s'] else None
        rows.append(r)
        print(f"  prompt={p:4d} eng={r['engine_tokens']:4d} gen={r['generated']} "
              f"decode={r['decode_s']}s -> {r['decode_tok_s']} tok/s "
              f"(runner {r['runner_decode_tok_s']}, e2e {r['end_to_end_tok_s']})", flush=True)
    result['rows'] = rows
    rates = [r['decode_tok_s'] for r in rows if r['decode_tok_s']]
    result['decode_tok_s_median'] = sorted(rates)[len(rates) // 2] if rates else None
    result['verdict'] = 'PASS' if rates and all(r['rc'] == 0 for r in rows) else 'FAIL'
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"\nmedian decode {result['decode_tok_s_median']} tok/s")
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
