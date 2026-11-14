"""Does `why=` say `fallback` when a block attempt failed, or does it hide it as policy?

The route line used to compute its reason from static state alone, with`fallback` as the
final `else`. A block route that threw, was rewound and was then stepped leaves exactly the
same switch, crossover and capability behind it as one the policy never chose -- so a
recovered crash could be printed as `not-preferred`, which reads as "we decided this".

This drives the shipped `oflm` through three cases and reads the engine's own line:

  none           no injection: whichever reason the policy gives, and it must not be
                 `fallback`, because nothing failed;
  before         the FIRST block throws before it writes: nothing has moved, so a reset is
                 honest and the prompt is stepped -> `fallback`;
  unrecoverable  the second block throws AND the rewind is broken: the request must fail,
                 never execute the sequential path, and never claim `fallback`.

`OFLM_OPEN_LAYER_MAJOR=1` is set so the block loop -- not layer-major -- is what runs and can
be injected into; the route under test is the block loop's exception handler.

    python prefill_fallback.py --text-tokens 400
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

_ROUTE_SEQ = re.compile(r'prefill-route sequential n=(\d+) blocks=0 tail=0 why=(\S+)')
_BLOCK_FAILED = 'block prefill failed on a set that advertises it'
_NOT_REWOUND = 'state NOT rewound'
_SEQUENTIAL_AFTER = 'then the sequential path'
_COULD_NOT_RESTORE = 'the pre-prompt state could not be restored'


def one(text: str, inject: str | None, *, ctx: int, gen: int, timeout: float) -> dict:
    extra = {
        'OFLM_OPEN_PREFILL_ROUTE': '1',
        # Both routes are legal above the crossover; this makes the BLOCK LOOP the one that
        # runs, because the block loop is what the exception handler under test belongs to.
        'OFLM_OPEN_LAYER_MAJOR': '0',
    }
    if inject is not None:
        extra['OFLM_OPEN_BLOCK_FAIL_INJECT'] = inject
    r = D.run([text], block=True, gen_lim=gen, ctx=ctx, timeout=timeout, extra=extra)
    all_text = r['stdout'] + '\n' + r['stderr']
    seq = _ROUTE_SEQ.search(all_text)
    return dict(inject=inject, rc=r['rc'],
                why=seq.group(2) if seq else None,
                engine_tokens=int(seq.group(1)) if seq else None,
                block_route_ran='prefill-route block' in all_text,
                block_failed=_BLOCK_FAILED in all_text,
                not_rewound=_NOT_REWOUND in all_text,
                hard_failure=_COULD_NOT_RESTORE in all_text,
                sequential_after=_SEQUENTIAL_AFTER in all_text,
                output_len=len(r['output']),
                route_lines=[l.strip() for l in all_text.splitlines()
                             if 'prefill-route' in l],
                fail_lines=[l.strip() for l in all_text.splitlines()
                            if 'block prefill failed' in l or 'NOT rewound' in l
                            or 'could not be restored' in l])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--text-tokens', type=int, default=400,
                    help='above the crossover, so the block route is the one attempted')
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--gen', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=2400)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'prefill-fallback-{stamp}')
    out.mkdir(parents=True, exist_ok=True)

    tok = D.Tokenizer()
    text = tok.exact_text(a.text_tokens)
    rows, checks = [], []

    # A successful block route prints no `sequential` line at all, which is the point: there
    # is no reason to give because nothing was skipped. The `none` case exists to prove the
    # word `fallback` needs an actual failure, so it is judged as "either the block route
    # succeeded outright, or a policy kept it away -- never `fallback`".
    clean = one(text, None, ctx=a.ctx, gen=a.gen, timeout=a.timeout)
    rows.append(clean)
    print(f"  none          rc={clean['rc']} why={clean['why']} "
          f"block_ran={clean['block_route_ran']} failed={clean['block_failed']}", flush=True)

    recoverable = one(text, 'before', ctx=a.ctx, gen=a.gen, timeout=a.timeout)
    rows.append(recoverable)
    print(f"  before        rc={recoverable['rc']} why={recoverable['why']} "
          f"failed={recoverable['block_failed']} seq_after={recoverable['sequential_after']}",
          flush=True)

    fatal = one(text, 'unrecoverable', ctx=a.ctx, gen=a.gen, timeout=a.timeout)
    rows.append(fatal)
    print(f"  unrecoverable rc={fatal['rc']} why={fatal['why']} "
          f"rewound={not fatal['not_rewound']}", flush=True)

    # 1. A policy decision stays a policy decision: no injection means nothing failed, so the
    #    word `fallback` would be a lie whatever the static reason is.
    checks.append(dict(
        check='no injection: nothing failed, so nothing says fallback',
        ok=(clean['why'] != 'fallback' and not clean['block_failed']
            and (clean['block_route_ran'] or clean['why'] is not None)),
        detail=dict(why=clean['why'], block_ran=clean['block_route_ran'],
                    failed=clean['block_failed'])))
    # 2. The case the report is about.
    checks.append(dict(
        check='a recovered failure reports why=fallback',
        ok=(recoverable['why'] == 'fallback' and recoverable['block_failed']),
        detail=dict(why=recoverable['why'], failed=recoverable['block_failed'])))
    checks.append(dict(
        check='and the sequential path really ran afterwards',
        ok=(recoverable['sequential_after'] and recoverable['output_len'] > 0
            and recoverable['engine_tokens'] == a.text_tokens + 8),
        detail=dict(seq_after=recoverable['sequential_after'],
                    out=recoverable['output_len'],
                    engine=recoverable['engine_tokens'])))
    # 3. An unrecoverable failure is a hard failure: it throws, it does not step, and it does
    #    not get to call itself a successful fallback.
    # Two signals are unusable here and it is worth saying why. The exit code is 0 because
    # `oflm run` keeps the session alive after a failed turn, and `output_len` counts the
    # ERROR text the CLI prints to stdout, not an answer. What must hold is that the request
    # failed loudly by name, that the sequential path never started, and that no `why=` was
    # emitted -- a fallback would have produced that line.
    checks.append(dict(
        check='an unrecoverable failure throws instead of falling back',
        ok=(fatal['hard_failure'] and fatal['why'] is None
            and not fatal['sequential_after']),
        detail=dict(hard=fatal['hard_failure'], why=fatal['why'],
                    seq_after=fatal['sequential_after'])))
    checks.append(dict(
        check='an unrecoverable failure does not report why=fallback',
        ok=fatal['why'] != 'fallback',
        detail=dict(why=fatal['why'])))

    result = dict(tool='prefill-block-prefill-fallback', version=1, phase=33,
                  gate='why=fallback for a recovered block failure, policy otherwise',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, ctx=a.ctx, text_tokens=a.text_tokens, gen_lim=a.gen,
                  rows=rows, checks=checks)
    result['verdict'] = 'PASS' if checks and all(c['ok'] for c in checks) else 'FAIL'
    for c in checks:
        print(f"  {c['check']:58s} {'PASS' if c['ok'] else 'FAIL'}", flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
