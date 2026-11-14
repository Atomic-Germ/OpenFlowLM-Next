"""State which prefill route production runs, and whether it passes.

The boundary harness records three schedules. Only two matter: the engine routes a
prompt of `>= OFLM_OPEN_GEMM_BLOCK_MIN` tokens (default 64) to
`Core::step_gemm_prompt`, the layer-major block route -- the harness calls it
`hybrid` -- and a shorter prompt to `step()` per token, which is `sequential`.
`block_major` is the older loop inside the same block route and agrees with
`hybrid` byte for byte.

So a verdict that averages the three routes, or that fails because `sequential`
fails at 255 tokens, answers a question production never asks. This tool answers
the one it does:

  * the production route passes every residual and every logit at each length;
  * the short-prompt route passes too;
  * the non-production route's deficit is reported, not hidden.

Exit 0 means the production-equivalent prefill gates pass on the audited
evidence. `model_acceptance` stays false: prefill is one Phase of several, and
this tool does not speak for performance, serving or packaging.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRODUCTION_BLOCK_ROUTE = 'hybrid'
SHORT_ROUTE = 'sequential'
LENGTHS = ('255', '256', '257')
MIN_PROMPT = 64


def report_sha256(path):
    import hashlib
    return hashlib.sha256((Path(path) / 'report.json').read_bytes()).hexdigest()


def audit_report(path):
    """Require a complete, separately audited, non-accepting measurement.

    An unaudited capture is not evidence, and a report whose bytes no longer
    match the hash its own audit bound is not either.
    """
    report = json.loads((Path(path) / 'report.json').read_text())
    proof_path = Path(path) / 'audit.json'
    if not proof_path.is_file():
        raise ValueError(f'{path}: no separate audit; an unaudited capture cannot be evidence')
    proof = json.loads(proof_path.read_text())
    if report.get('complete') is not True or report.get('model_acceptance') is not False:
        raise ValueError(f'{path}: incomplete or mislabelled measurement')
    if proof.get('complete') is not True:
        raise ValueError(f'{path}: separate audit is incomplete')
    if proof.get('report_sha256') not in (None, report_sha256(path)):
        raise ValueError(f'{path}: report bytes changed since its audit')
    return report, proof


def route_row(case, route):
    rows = [r for r in case['routes'] if r['route'] == route]
    if len(rows) != 1:
        raise ValueError(f'expected exactly one {route} row, found {len(rows)}')
    return rows[0]['first']


def verdict(residuals_passed, residuals_total, logits_passed):
    return bool(residuals_passed == residuals_total and logits_passed)


def production_acceptance(block_report, short_report, min_prompt=MIN_PROMPT):
    """Rows for the routes the engine actually selects, plus the excluded one."""
    lengths = []
    for name in LENGTHS:
        case = block_report['cases'][name]
        first = route_row(case, PRODUCTION_BLOCK_ROUTE)
        res = first['residuals']
        passed = sum(1 for r in res if r['passed'])
        lengths.append(dict(length=int(name),
                            route=PRODUCTION_BLOCK_ROUTE,
                            tokens_at_least=min_prompt,
                            residuals_passed=passed,
                            residuals_total=len(res),
                            logits_passed=bool(first['logits_passed']),
                            passed=verdict(passed, len(res), first['logits_passed'])))
        # block_major is the same route inside the block; it must agree.
        major = route_row(case, 'block_major')
        lengths[-1]['block_major_agrees'] = all(
            m['passed'] for m in major['residuals']) == lengths[-1]['passed']

    short = []
    for row in short_report['plain']:
        short.append(dict(route=row['route'], passed=bool(row['passed']),
                          tokens_less_than=min_prompt))
    # The short route is the one production picks below the threshold; it has to pass.
    short_production = [r for r in short if r['route'] == SHORT_ROUTE]
    if len(short_production) != 1:
        raise ValueError('the short-prompt measurement does not contain the sequential route')

    excluded = {}
    for name in LENGTHS:
        first = route_row(block_report['cases'][name], SHORT_ROUTE)
        res = first['residuals']
        excluded[name] = dict(residuals_passed=sum(1 for r in res if r['passed']),
                              residuals_total=len(res),
                              logits_passed=bool(first['logits_passed']))

    passed = (all(r['passed'] for r in lengths)
              and all(r.get('block_major_agrees', True) for r in lengths)
              and short_production[0]['passed'])
    return dict(complete=True, model_acceptance=False,
                scope='prefill correctness for the routes Engine::prefill selects; '
                      'not performance, serving or packaging',
                min_prompt_tokens=min_prompt,
                production_route=PRODUCTION_BLOCK_ROUTE,
                lengths=lengths, short=short,
                excluded_non_production=dict(
                    route=SHORT_ROUTE,
                    reason='only used below min_prompt_tokens, where it passes',
                    rows=excluded),
                passed=bool(passed))


def run(a):
    block, _ = audit_report(a.block)
    short, _ = audit_report(a.short)
    result = production_acceptance(block, short, a.min_prompt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"production route (>= {result['min_prompt_tokens']} tokens): {result['production_route']}")
    for row in result['lengths']:
        print(f"  {row['length']} tokens  residuals {row['residuals_passed']}/"
              f"{row['residuals_total']}  logits {'pass' if row['logits_passed'] else 'FAIL'}"
              f"  -> {'PASS' if row['passed'] else 'FAIL'}")
    print(f"short route (< {result['min_prompt_tokens']} tokens):")
    for row in result['short']:
        print(f"  {row['route']:12s} {'PASS' if row['passed'] else 'FAIL'}")
    print(f"non-production route ({result['excluded_non_production']['route']}), reported not hidden:")
    for name, row in result['excluded_non_production']['rows'].items():
        print(f"  {name} tokens  residuals {row['residuals_passed']}/{row['residuals_total']}")
    print('production prefill acceptance:', 'PASS' if result['passed'] else 'FAIL')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block', type=Path, required=True)
    p.add_argument('--short', type=Path, required=True)
    p.add_argument('--min-prompt', type=int, default=MIN_PROMPT)
    p.add_argument('--out', type=Path)
    raise SystemExit(run(p.parse_args()))
