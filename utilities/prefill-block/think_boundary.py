"""The generation headroom of `Qwen3_5VL::generate`, measured per branch.

`generate()` does not begin with a sampled token. It steps a THINK PREAMBLE first:

    thinking      think_start_id, 198                          -> 2 rows
    not thinking  think_start_id, 271, think_end_id, 271        -> 4 rows

Those rows consume context positions before any budget check can stop the loop, so the
largest prompt that can still emit a token is

    MAX_L - (preamble_rows + 1)

and not `MAX_L - 1`. A constant "leave a few slots" guess was wrong three times while
probing (2, 3, 4 and 5 each moved the observed boundary), because the preamble also has to
be counted in `total_tokens`: the counter and the engine position diverge otherwise and
the loop guard then measures a stale position.

This drives the boundary from BOTH sides at a small context, per branch, so the rule is
observed rather than derived:

    MAX_L - need       must generate
    MAX_L - need + 1   must be REFUSED, explicitly, with nothing prefilled

Which branch actually ran is read from the `OFLM_CTX_TRACE` step names (`after-198` is the
thinking preamble, `after-think-end`/`after-271b` the other), so a model whose default
flips fails here instead of quietly measuring the wrong branch.

    python think_boundary.py --ctx 512
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

# The template offset at a 512 context, measured by route_matrix.calibrate().
OFFSET = 8

# The branch constants: this is the thing being TESTED. If the engine disagrees, the run
# fails -- the harness never adopts the engine's answer and then calls it a pass.
PREAMBLE = {False: 4, True: 2}

REFUSED = 'Max length reached: this model needs'
PREFILL_REFUSED = 'Max length reached, stopping prefilling'

# `cmd_clear` ignores its arguments ([`runner.cpp:465`](../../src/runner/runner.cpp:465)),
# so `/clear <text>` clears the context and generates NOTHING. The prompt has to be its
# own session line or the whole measurement is vacuous.
THINK_STEPS = {'after-198': 'thinking', 'after-think-end': 'not-thinking'}


def branch_of(trace: list[str]) -> str | None:
    for line in trace:
        for step, name in THINK_STEPS.items():
            if f'step {step}' in line:
                return name
    return None


def one(engine_want: int, *, ctx: int, think: bool, gen: int, timeout: float) -> dict:
    tok = D.Tokenizer()
    text = tok.exact_text(engine_want - OFFSET)
    # The CLI starts this family NOT thinking (need = 5); `/think` toggles it on (need = 3).
    # `clear_context()` does not touch the toggle, so the order of the two lines is free.
    session = ['/think', '/clear', text] if think else ['/clear', text]
    rec = D.run(session, block=True, gen_lim=gen, ctx=ctx, timeout=timeout,
                extra={'OFLM_OPEN_PREFILL_ROUTE': '1', 'OFLM_CTX_TRACE': '1'})
    all_text = rec['stdout'] + '\n' + rec['stderr']
    trace = [ln.strip() for ln in all_text.splitlines() if 'ctx-trace' in ln]
    return dict(engine_want=engine_want, think=think, gen_limit=gen, rc=rec['rc'],
                engine_tokens=D.prefilled_tokens(rec),
                output_len=len(rec['output']),
                warn_prefill=PREFILL_REFUSED in all_text,
                refused=REFUSED in all_text,
                position_error='reached the context capacity' in all_text,
                internal_error='Internal error' in all_text,
                branch_observed=branch_of(trace), trace=trace)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ctx', type=int, default=512)
    ap.add_argument('--gen', type=int, default=1)
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    if a.ctx % 256 or a.ctx < 512:
        raise SystemExit('--ctx must be a multiple of 256 and at least 512')

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'think-boundary-{a.ctx}-{stamp}')
    out.mkdir(parents=True, exist_ok=True)

    rows, checks = [], []
    for think, preamble in PREAMBLE.items():
        need = preamble + 1
        last_ok, first_bad = a.ctx - need, a.ctx - need + 1
        branch = 'thinking_branch' if think else 'normal_branch'
        want_branch = 'thinking' if think else 'not-thinking'
        for engine_want in (last_ok, first_bad):
            r = one(engine_want, ctx=a.ctx, think=think, gen=a.gen, timeout=a.timeout)
            r['branch'] = branch
            rows.append(r)
            state = 'REFUSED' if r['refused'] else f'generated {r["output_len"]}'
            print(f"  {branch:16s} want={engine_want:4d} need={need} "
                  f"engine={r['engine_tokens']:4d} branch={r['branch_observed']} {state}",
                  flush=True)

        ok = next(x for x in rows if x['engine_want'] == last_ok and x['think'] == think)
        bad = next(x for x in rows if x['engine_want'] == first_bad and x['think'] == think)
        # The branch is measured from the trace, not inferred from the request: the row that
        # generates must have stepped THIS branch's preamble. The refused row cannot show a
        # preamble step (it is refused before one), so it is checked by the `need` its guard
        # reports instead -- that is the same fact, taken from the other side of the guard.
        bad_need = next((ln for ln in bad['trace'] if 'need=' in ln), '')
        checks.append(dict(
            check=f'{branch}: the generating row took the {want_branch} preamble',
            ok=ok['branch_observed'] == want_branch,
            detail=dict(observed=ok['branch_observed'])))
        checks.append(dict(
            check=f'{branch}: the refusing row was guarded with need={need}',
            ok=f'need={need} ' in bad_need + ' ' or bad_need.endswith(f'need={need}'),
            detail=bad_need))
        checks.append(dict(
            check=f'{branch}: MAX_L - {need} ({last_ok}) generates',
            ok=(ok['engine_tokens'] == last_ok and ok['output_len'] > 0
                and not ok['refused'] and not ok['position_error']),
            detail=dict(engine=ok['engine_tokens'], output_len=ok['output_len'],
                        refused=ok['refused'])))
        # A refusal is a REFUSAL, not an empty prefill: this path prefills, then declines to
        # step a preamble it has no room for. Nothing may be stepped and nothing may throw.
        checks.append(dict(
            check=f'{branch}: MAX_L - {need} + 1 ({first_bad}) is refused before stepping',
            ok=(bad['refused'] and not bad['position_error'] and not bad['internal_error']
                and bad['branch_observed'] is None),
            detail=dict(refused=bad['refused'],
                        stepped=(bad['branch_observed'] is not None),
                        position_error=bad['position_error'],
                        internal_error=bad['internal_error'])))

    result = dict(tool='prefill-block-think-boundary', version=1, phase=31,
                  gate='31.A/31.B generation headroom, both branches',
                  model_acceptance=False, binary=str(D.OFLM), kernels=str(D.KERNELS),
                  started=stamp, ctx=a.ctx, template_offset=OFFSET, gen_limit=a.gen,
                  rule='prompt + generation_preamble_rows() + 1 <= MAX_L',
                  preamble_rows=PREAMBLE, rows=rows, checks=checks)
    result['verdict'] = 'PASS' if checks and all(c['ok'] for c in checks) else 'FAIL'
    for c in checks:
        print(f"  {c['check']:62s} {'PASS' if c['ok'] else 'FAIL'}", flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
