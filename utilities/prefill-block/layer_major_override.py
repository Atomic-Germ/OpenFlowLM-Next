"""Does OFLM_OPEN_LAYER_MAJOR actually change the route, or only the log line?

This exists because the two were once different. `OFLM_OPEN_LAYER_MAJOR=1` set a boolean,
the log said layer-major was forced, and `Engine::prefill()` still took the block loop --
because `Core::layer_major_preferred()` re-decided with the automatic dense rule on top of
the operator's request. A test that greps the source for the variable cannot see that: the
assignment and the string were both there.

So this drives the shipped `oflm` and compares two independent statements of the truth:

  * the runtime's own line -- `prefill schedule: ...`
  * the route the prefill actually took -- `prefill-route ...` (OFLM_OPEN_PREFILL_ROUTE=1)

and requires them to agree, in all three operator states. A prompt above the block
crossover is used so that both routes are legal for this set: below it the block loop would
run sequential and the comparison would say nothing.

    python layer_major_override.py --text-tokens 300
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

_SCHEDULE = re.compile(r'prefill schedule: (\S+)')
_ROUTE = re.compile(r'prefill-route (block|layer-major|sequential) n=(\d+)')

# The three operator states, and what each must produce on a CAPABLE set.
CASES = (
    ('unset', None, 'block', 'the automatic rule: dense sets take the block loop'),
    ('force_on', '1', 'layer-major', 'the operator asked, and the set can run it'),
    ('force_off', '0', 'block', '0 means 0, even where layer-major would be automatic'),
)


def one(text: str, value: str | None, *, ctx: int, gen: int, timeout: float) -> dict:
    extra = {'OFLM_OPEN_PREFILL_ROUTE': '1'}
    if value is not None:
        extra['OFLM_OPEN_LAYER_MAJOR'] = value
    r = D.run([text], block=True, gen_lim=gen, ctx=ctx, timeout=timeout, extra=extra)
    all_text = r['stdout'] + '\n' + r['stderr']
    sched = _SCHEDULE.search(all_text)
    route = _ROUTE.search(all_text)
    return dict(env=value, rc=r['rc'],
                logged_schedule=sched.group(1) if sched else None,
                actual_route=route.group(1) if route else None,
                engine_tokens=int(route.group(2)) if route else None,
                schedule_line=next((l.strip() for l in all_text.splitlines()
                                    if 'prefill schedule' in l), None),
                route_line=next((l.strip() for l in all_text.splitlines()
                                 if 'prefill-route' in l), None))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--text-tokens', type=int, default=300,
                    help='above the block crossover, so both routes are legal')
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--gen', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=2400)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'layer-major-override-{stamp}')
    out.mkdir(parents=True, exist_ok=True)

    tok = D.Tokenizer()
    text = tok.exact_text(a.text_tokens)
    rows, checks = [], []
    for name, value, want, why in CASES:
        r = one(text, value, ctx=a.ctx, gen=a.gen, timeout=a.timeout)
        r.update(case=name, expected_route=want, why=why)
        rows.append(r)
        print(f"  {name:10s} env={str(value):4s} logged={r['logged_schedule']:12s} "
              f"actual={r['actual_route']:11s} {'' if r['actual_route'] == want else 'MISMATCH'}",
              flush=True)

    # The two statements must agree, or the diagnostic is lying to the operator.
    for r in rows:
        logged = r['logged_schedule']
        actual = r['actual_route']
        agree = (logged == 'block-major' and actual == 'block') or (logged == actual)
        checks.append(dict(
            check=f"{r['case']}: the log and the route agree",
            ok=agree, detail=dict(logged=logged, actual=actual)))
    # And each state must produce the route its contract promises.
    for r in rows:
        checks.append(dict(
            check=f"{r['case']}: the route is {r['expected_route']}",
            ok=r['actual_route'] == r['expected_route'],
            detail=dict(actual=r['actual_route'], why=r['why'])))
    # The bug this file exists for: force-on must not be overruled by the preference.
    forced = next(r for r in rows if r['case'] == 'force_on')
    checks.append(dict(
        check='force_on beats block_loop_preferred rather than only relabelling the log',
        ok=forced['actual_route'] == 'layer-major' and forced['logged_schedule'] == 'layer-major',
        detail=dict(actual=forced['actual_route'], logged=forced['logged_schedule'])))

    result = dict(tool='prefill-block-layer-major-override', version=1, phase=32,
                  gate='the operator override decides the schedule, not just the message',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, ctx=a.ctx, text_tokens=a.text_tokens, gen_lim=a.gen,
                  rows=rows, checks=checks)
    result['verdict'] = 'PASS' if checks and all(c['ok'] for c in checks) else 'FAIL'
    for c in checks:
        print(f"  {c['check']:64s} {'PASS' if c['ok'] else 'FAIL'}", flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
