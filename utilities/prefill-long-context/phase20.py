"""Phase 20 acceptance: the plan's own checklist as one verdict.

Phase 20 asks for prefill correctness across `all 64 layers`, a short prefix, a
block-boundary prefix, a long prefix, continuation after prefill, reset and
repeated identical requests, KV/state identity, and final logits against an
independent reference. Those were measured by different tools over months, so
"is Phase 20 done" had no single answer -- and a missing row looked exactly like
a passing one.

This gate takes one boolean per item, from whoever measured it, and refuses to
average. An absent item is a FAILURE, never a pass by omission. It also never
claims model acceptance: prefill is one phase, and performance, serving and
packaging are separate gates that this file does not speak for.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ITEMS = ('all_layers', 'short_prefix', 'block_boundary_prefix', 'long_prefix',
         'continuation', 'reset_repeat', 'state_identity', 'independent_logits')

# What each item means, so the report cannot be read as something broader.
MEANING = {
    'all_layers': '64 layers evaluated, not a slice',
    'short_prefix': 'a prompt below the block threshold (the per-token route)',
    'block_boundary_prefix': '255/256/257 tokens, spanning the block boundary',
    'long_prefix': 'more than one full block (513+ tokens)',
    'continuation': 'the next token after prefill matches the reference',
    'reset_repeat': 'two identical requests on one resident engine agree',
    'state_identity': 'KV / recurrent state carries correctly across blocks',
    'independent_logits': 'final logits compared with an independent reference',
}


def evaluate(evidence):
    """evidence: item -> bool. Returns the verdict; never invents a pass."""
    if not isinstance(evidence, dict):
        raise ValueError('evidence must map item to bool')
    unknown = sorted(set(evidence) - set(ITEMS))
    if unknown:
        raise ValueError(f'unknown checklist item(s): {unknown}')
    items, missing, failed = {}, [], []
    for name in ITEMS:
        if name not in evidence:
            missing.append(name)
            items[name] = dict(passed=False, meaning=MEANING[name],
                              reason='no evidence supplied')
            continue
        value = evidence[name]
        if not isinstance(value, bool):
            raise ValueError(f'{name} must be a bool, got {type(value).__name__}')
        items[name] = dict(passed=value, meaning=MEANING[name],
                           reason='measured' if value else 'measured and failed')
        if not value:
            failed.append(name)
    return dict(complete=True, model_acceptance=False,
                scope='Phase 20 prefill correctness only; not performance, '
                      'serving, packaging or full-model acceptance',
                items=items, missing=missing, failed=failed,
                passed=bool(not missing and not failed))


def run(a):
    evidence = {k: getattr(a, k) for k in ITEMS if getattr(a, k) is not None}
    result = evaluate(evidence)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name in ITEMS:
        item = result['items'][name]
        print(f"  {'PASS' if item['passed'] else 'FAIL'}  {name:24s} {item['meaning']}")
    if result['missing']:
        print('  missing evidence:', result['missing'])
    print('Phase 20 prefill acceptance:', 'PASS' if result['passed'] else 'FAIL')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ITEMS:
        p.add_argument('--' + name.replace('_', '-'), dest=name,
                       type=lambda v: v.lower() in ('1', 'true', 'yes'),
                       default=None, help=MEANING[name])
    p.add_argument('--out', type=Path)
    raise SystemExit(run(p.parse_args()))
