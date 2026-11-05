"""Phase 27 closure, item 7: request isolation on the shipped server, as A -> B -> A.

The plan asks for three requests in ONE server process. The interesting property is that
the first answer comes back byte-identical after a DIFFERENT request has run in between:
that is what proves no resident state leaked from one request into the next. A second
identical request (what 27.8 did) is a weaker test, because the prefix-skip path can make
it match for a reason that has nothing to do with isolation.

The comparison for B is against B in a FRESH process, so "B still works after A" is not
confused with "B is unaffected by A".

`serve` has no context-reset endpoint -- `/v1/context/reset`, `/reset` and `/v1/reset` all
404 on this build. That is recorded as `N/A` with the reason, not as a failed check:
inventing an endpoint to satisfy a checklist would be worse than saying the API has none.

    python isolation_aba.py                  # A=300 text tokens, B=200, 8 new tokens
    python isolation_aba.py --a 260 --b 130
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
import oflm_serve_check as S  # noqa: E402  the server start/ready/chat helpers

RESET_PATHS = ('/v1/context/reset', '/reset', '/v1/reset', '/v1/cache/reset')


def probe_reset(port: int) -> dict:
    """Does the server expose any reset? Recorded either way, never assumed."""
    import urllib.request
    for path in RESET_PATHS:
        req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', method='POST')
        try:
            with S._open(req, 10) as r:
                return dict(available=True, endpoint=path, status=r.status)
        except Exception:  # noqa: BLE001 -- 404 is the expected answer here
            continue
    return dict(available=False, endpoint=None,
                reason='serve exposes no context-reset endpoint on this build '
                       '(/v1/context/reset, /reset, /v1/reset, /v1/cache/reset all 404). '
                       'Recorded as N/A, not as a failed requirement.')


def one_server(port: int, a_text: str, b_text: str, *, max_tokens: int,
               timeout: float) -> dict:
    proc, _ = S.start(port, ctx=4096)
    try:
        err = S.wait_ready(port, proc, timeout=1200)
        if err:
            return dict(started=False, reason=err)
        first = S.chat(port, a_text, max_tokens=max_tokens, timeout=timeout)
        mid = S.chat(port, b_text, max_tokens=max_tokens, timeout=timeout)
        again = S.chat(port, a_text, max_tokens=max_tokens, timeout=timeout)
        return dict(started=True, reset=probe_reset(port),
                    a1=first, b=mid, a2=again)
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', type=int, default=300, help='text tokens of prompt A')
    ap.add_argument('--b', type=int, default=200, help='text tokens of prompt B')
    ap.add_argument('--max-tokens', type=int, default=8)
    ap.add_argument('--port', type=int, default=18131)
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    tok = D.Tokenizer()
    a_text, b_text = tok.exact_text(a.a), tok.exact_text(a.b)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'isolation-aba-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-isolation-aba', version=1, phase=27,
                  gate='27 closure item 7: A -> B -> A in one server process',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, a_text_tokens=a.a, b_text_tokens=a.b,
                  max_tokens=a.max_tokens, port=a.port)

    print(f'=== one server, A({a.a} text tokens) -> B({a.b}) -> A ===', flush=True)
    run = one_server(a.port, a_text, b_text, max_tokens=a.max_tokens, timeout=a.timeout)
    result['one_server'] = run
    rows: list[dict] = []
    if not run.get('started'):
        result['verdict'] = 'FAIL'
        result['rows'] = [dict(check='server_starts', ok=False, detail=run.get('reason'))]
        (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
        print(f"FAIL server did not start: {run.get('reason')}")
        return 1

    # B in a fresh process: the reference that separates "B is unaffected by A" from
    # "B happens to work after A".
    print('  fresh-process B for comparison...', flush=True)
    fresh = one_server(a.port + 1, b_text, b_text, max_tokens=a.max_tokens,
                       timeout=a.timeout)
    result['fresh_b'] = fresh

    a1, mid, a2 = run['a1'], run['b'], run['a2']
    fresh_b = fresh.get('b') if fresh.get('started') else None
    rows.append(dict(check='A_first_answered', ok=bool(a1.get('ok') and a1.get('content')),
                     detail=a1.get('content'), wall_s=a1.get('wall_s')))
    rows.append(dict(check='B_answered', ok=bool(mid.get('ok') and mid.get('content')),
                     detail=mid.get('content'), wall_s=mid.get('wall_s')))
    rows.append(dict(check='A_second_answered', ok=bool(a2.get('ok') and a2.get('content')),
                     detail=a2.get('content'), wall_s=a2.get('wall_s')))
    same_a = a1.get('ok') and a2.get('ok') and a1.get('content') == a2.get('content')
    rows.append(dict(check='A_before_B_equals_A_after_B', ok=bool(same_a),
                     detail=dict(first=a1.get('content'), second=a2.get('content')),
                     why='resident state leaked across requests if these differ'))
    same_b = (fresh_b is not None and mid.get('ok') and fresh_b.get('ok')
              and mid.get('content') == fresh_b.get('content'))
    rows.append(dict(check='B_matches_fresh_process_B', ok=bool(same_b),
                     detail=dict(after_A=mid.get('content'),
                                 fresh=fresh_b.get('content') if fresh_b else None)))
    distinct = a1.get('content') != mid.get('content')
    rows.append(dict(check='A_and_B_are_different_prompts', ok=bool(distinct),
                     why='if these matched, the isolation check would pass vacuously'))

    reset = run.get('reset') or {}
    rows.append(dict(check='after_reset', ok=None,
                     verdict='N/A',
                     detail=reset.get('reason') if not reset.get('available')
                     else f"reset endpoint {reset.get('endpoint')} answered "
                          f"{reset.get('status')}; the next request still has to be checked"))

    bad = [r for r in rows if r['ok'] is False]
    result['rows'] = rows
    result['verdict'] = 'PASS' if not bad else 'FAIL'
    for r in rows:
        mark = {True: 'PASS', False: 'FAIL', None: 'N/A '}[r['ok']]
        print(f"  {r['check']:34s} {mark}", flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
