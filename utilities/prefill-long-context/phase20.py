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

# The plan judges these per ROUTE: "If sequential or production-equivalent long-prefix
# precision fails, treat it as a blocker." A bare `true` for one of them is therefore an
# overstatement even when the production route passes -- the measured 255/256/257 run is
# exactly that case (hybrid and block_major clean, sequential 4 failures). So these two
# items take a route -> bool mapping and must name every route below.
ROUTE_ITEMS = ('block_boundary_prefix', 'long_prefix')
ROUTES = ('sequential', 'hybrid', 'block_major')

# The prompt length each route-judged item covers, and the threshold Engine::prefill
# uses to choose between them. These live here, not in the caller, so the gate cannot be
# told a route is "production" by whoever is reporting the evidence.
ITEM_LENGTH = {'block_boundary_prefix': 257, 'long_prefix': 513}
MIN_PROMPT = 64          # Engine::prefill's OFLM_OPEN_GEMM_BLOCK_MIN default


def routes_needed(length, min_prompt=MIN_PROMPT):
    """The route set PRODUCTION selects for a prompt of this length.

    Above the threshold `Engine::prefill` runs the block path and never `step()`
    ([`engine.cpp:181`]); at or below it, `step()` is the only route. Both block routes
    are required together because which one runs depends on `layer_major_ok()` (the
    kernel set) and on OFLM_OPEN_LAYER_MAJOR, so either can be the production schedule
    for a given deployment.

    `OFLM_OPEN_GEMM_BLOCK` is classified as a debug knob in generate-registry.py, not a
    user setting, so `=0` is not a supported production configuration.
    """
    if length <= 0:
        raise ValueError('a prompt length must be positive')
    return ('sequential',) if length < min_prompt else ('hybrid', 'block_major')


def route_verdict(evidence, length, min_prompt=MIN_PROMPT):
    """Pass iff every route production selects at `length` passed.

    Routes production cannot reach at this length are returned as `diagnostic`: they
    stay visible and their failures are named, but they do not gate the phase. Failing
    a phase for an unreachable path is the mirror image of the overstatement this file
    exists to prevent.
    """
    needed = routes_needed(length, min_prompt)
    missing = [r for r in needed if r not in evidence]
    if missing:
        raise ValueError(f'no evidence for the production route(s): {missing}')
    failing = [r for r in needed if not evidence[r]]
    diagnostic = {r: v for r, v in evidence.items() if r not in needed}
    return dict(passed=not failing, required_routes=list(needed),
                failing_routes=failing, diagnostic=diagnostic,
                diagnostic_failing=[r for r, v in diagnostic.items() if not v])

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


def routes_of(value, name):
    """Normalise a route-judged item: every plan route must be present and a bool.

    A typo'd route must not vanish from the evidence, and a missing route must not
    read as passing -- those are the two ways a route-qualified check silently
    degrades into the bare bool this exists to forbid.
    """
    if not isinstance(value, dict):
        raise ValueError(f'{name} is judged per route ({", ".join(ROUTES)}); '
                         f'a {type(value).__name__} cannot carry it')
    unknown = sorted(set(value) - set(ROUTES))
    if unknown:
        raise ValueError(f'{name} has unknown route(s): {unknown}')
    absent = [r for r in ROUTES if r not in value]
    if absent:
        raise ValueError(f'{name} is missing route(s): {absent}')
    for route, passed in value.items():
        if not isinstance(passed, bool):
            raise ValueError(f'{name}[{route}] must be a bool, '
                             f'got {type(passed).__name__}')
    return value


def evaluate(evidence):
    """evidence: item -> bool, or route -> bool for a route-judged item.

    Returns the verdict; never invents a pass, and never lets a route-qualified item
    pass on the strength of one route.
    """
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
        if name in ROUTE_ITEMS:
            per_route = routes_of(value, name)
            judged = route_verdict(per_route, ITEM_LENGTH[name])
            reason = ('every production route measured and passed' if judged['passed'] else
                      'production route(s) failed: ' + ', '.join(judged['failing_routes']))
            if judged['diagnostic_failing']:
                reason += ('; unreachable at this length, diagnostic only: '
                           + ', '.join(judged['diagnostic_failing']))
            items[name] = dict(meaning=MEANING[name], routes=per_route, **judged,
                               reason=reason)
            if not judged['passed']:
                failed.append(name)
            continue
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


def collect(namespace):
    """CLI namespace -> evidence. A route-judged item needs EVERY route named.

    Supplying some routes and not others is half of a measurement, and half of a
    measurement must reach `evaluate` as an error rather than as a partial dict that
    happens to be missing the route which failed.
    """
    evidence = {k: getattr(namespace, k) for k in ITEMS
                if k not in ROUTE_ITEMS and getattr(namespace, k) is not None}
    for name in ROUTE_ITEMS:
        per_route = {r: getattr(namespace, f'{name}__{r}') for r in ROUTES}
        if all(v is None for v in per_route.values()):
            continue
        evidence[name] = per_route
    return evidence


def run(a):
    evidence = a if isinstance(a, dict) else collect(a)
    result = evaluate(evidence)
    if isinstance(a, dict):
        out = None
    else:
        out = a.out
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name in ITEMS:
        item = result['items'][name]
        extra = ''
        if item.get('failing_routes'):
            extra = '  failing: ' + ', '.join(item['failing_routes'])
        print(f"  {'PASS' if item['passed'] else 'FAIL'}  {name:24s} {item['meaning']}{extra}")
    if result['missing']:
        print('  missing evidence:', result['missing'])
    print('Phase 20 prefill acceptance:', 'PASS' if result['passed'] else 'FAIL')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ITEMS:
        if name in ROUTE_ITEMS:
            # One flag per route: an item the plan judges per route cannot be given a
            # single boolean without asserting something the evidence does not say.
            for route in ROUTES:
                p.add_argument(f'--{name.replace("_", "-")}--{route.replace("_", "-")}',
                               dest=f'{name}__{route}', type=lambda v: v.lower() in ('1', 'true', 'yes'),
                               default=None, help=f'{MEANING[name]}; route {route}')
            continue
        p.add_argument('--' + name.replace('_', '-'), dest=name,
                       type=lambda v: v.lower() in ('1', 'true', 'yes'),
                       default=None, help=MEANING[name])
    p.add_argument('--out', type=Path)
    raise SystemExit(run(p.parse_args()))
