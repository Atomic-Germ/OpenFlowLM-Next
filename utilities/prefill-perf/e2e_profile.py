"""Phase 24: end-to-end prefill profile of the CURRENT implementation.

Runs the real 64-layer model through the production block route at several prompt
lengths, parses the engine's own stage breakdown, and allocates every microsecond of the
engine's total to exactly one category.

TWO DENOMINATORS, never mixed up:

  wall share   = category / process wall (the whole `open_qwen36_cli` process: model
                 load, xclbin creation, the request, teardown)
  engine share = category / `StepTiming.total_ms` (the request the engine itself timed)

The gap between them is `outside_engine` and is reported as its own row, so a wall share
and an engine share can never be read as the same number by accident.

THE ACCOUNTING MODEL, derived from src/open_qwen36/core.cpp rather than assumed:

  total   = part0 + part1 + route + lm_head + residue
  part1   = prenorm + gemm_tile + gemm_tr + tail + setup + sync + qkv_scatter + mid
  mid     = dn_conv + dn_rule + attn
  residue >= shared + state

That last line is the trap. `shared` (core.cpp ffn_block) and `state` are added to their
OWN field and not to part1, so both live INSIDE `residue`. Version 1 of this tool put
`shared` in `host_setup` and also counted the whole of `residue`, so its categories summed
to 4796.0 ms MORE than wall time at 2048 -- more than 100% of a whole run, which is the
signature of a double count rather than a timer bug. Every identity above is now
re-checked on every run and written into the JSON under `checks`.

`--reps` defaults to 1: this is an ATTRIBUTION run, not a published benchmark. The JSON
records that dispersion was not measured, because one pass cannot say anything about it.

    python e2e_profile.py --lengths 128 512 1024 2048 --out e2e-profile-v1.json
    python e2e_profile.py --from-json e2e-profile-v1.json     # recompute, no hardware
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLI = ROOT / 'Models/qwen38-27b/prefill-perf/switch-v1/build/open_qwen36_cli'
MODEL = ROOT / 'Models/qwen38-27b/Qwen3.8-27B-NPU2'
KERNELS = ROOT / 'Models/qwen38-27b/production-v2/kernels-model'
FIXTURE_DIR = ROOT / 'Models/qwen38-27b/prefill-perf/phase24'

#   gemm-prompt [0,513) t_real=513: 97340.5 ms (GEMM 68576.5, host 18030.7, per-token 0.0, lm_head 26.1)
STAGE = re.compile(r'\s*(?P<what>[\w-]+) \[(?P<n0>\d+),(?P<n1>\d+)\) t_real=(?P<t>\d+):\s+'
                   r'(?P<total>[\d.]+) ms \(GEMM (?P<gemm>[\d.]+),\s*host (?P<host>[\d.]+),'
                   r'\s*per-token (?P<route>[\d.]+),\s*lm_head (?P<lm>[\d.]+)\)')
#   mid 1.0 (dn conv 2.0, dn rule 3.0, attn 4.0), prenorm 5.0, gemm tile 6.0, gemm tr 7.0, ...
DETAIL = re.compile(r'^\s+mid (?P<mid>[\d.]+) \(dn conv (?P<dn_conv>[\d.]+), '
                    r'dn rule (?P<dn_rule>[\d.]+), attn (?P<attn>[\d.]+)\), '
                    r'prenorm (?P<prenorm>[\d.]+), gemm tile (?P<gemm_tile>[\d.]+),'
                    r' gemm tr (?P<gemm_tr>[\d.]+), tail (?P<tail>[\d.]+), shared (?P<shared>[\d.]+),'
                    r' state (?P<state>[\d.]+), sync (?P<sync>[\d.]+), setup (?P<setup>[\d.]+)'
                    r' \| moe prep (?P<moe_prep>[\d.]+), patch (?P<moe_patch>[\d.]+),'
                    r' run (?P<moe_run>[\d.]+), read (?P<moe_read>[\d.]+)'
                    r' \| qkv scatter (?P<qkv_scatter>[\d.]+) \| residue (?P<residue>-?[\d.]+)', re.M)
DISPATCH = re.compile(r'^\s+(?P<name>\S+)\s+(?P<calls>\d+) calls\s+(?P<total>[\d.]+) ms total'
                      r'\s+(?P<mean>[\d.]+) mean', re.M)

ACCOUNTING_MODEL = (
    'total = part0 + part1 + route + lm_head + residue; '
    'part1 = prenorm + gemm_tile + gemm_tr + tail + setup + sync + qkv_scatter + mid; '
    'mid = dn_conv + dn_rule + attn; residue >= shared + state (both are measured '
    'outside part1, so both sit inside residue and must not be counted twice). '
    'route nests attn on the dxB route and moe_* on an MoE route.'
)


def allocate(rec: dict, wall_ms: float) -> dict:
    """Every microsecond once, both shares side by side, and the identity checks.

    A category is exclusive: nothing inside it is also inside another. The engine's sub
    fields that nest (attn inside mid, moe_* inside route) are returned separately as
    diagnostics and are never added to the total.
    """
    t, f = rec['totals'], rec['fields']
    total, route = t['total'], t['route']
    moe = f['moe_prep'] + f['moe_patch'] + f['moe_run'] + f['moe_read']
    nested = {'attention_in_mid': f['attn'], 'moe_in_route': moe}
    ex = {
        'projection_gemm_device': t['gemm'],
        'gemm_host_tiling': f['gemm_tile'],
        'gemm_host_transpose': f['gemm_tr'],
        'deltanet_conv_and_rule': f['dn_conv'] + f['dn_rule'],
        'norm_entry': f['prenorm'],
        'residual_norm_router_host': f['tail'],
        'ffn_activation_host': f['shared'],
        'request_setup': f['setup'],
        'dispatch_sync': f['sync'],
        'qkv_scatter': f['qkv_scatter'],
        'bo_state': f['state'],
        'lm_head': t['lm'],
    }
    if route > 0:
        # Host attention (dxB) or the MoE dispatches: route_ms nests those fields, so
        # route is the exclusive number and attn / moe_* are diagnostics only.
        ex['host_route'] = route
    else:
        # The block route: attn is exclusive inside mid inside part1, and there is no MoE.
        ex['attention'] = f['attn']
        if moe > 0:
            ex['moe'] = moe
    # shared and state are inside residue by construction; take them out so residue is
    # what the engine genuinely cannot place, not what this grouping forgot.
    unattributed = f['residue'] - f['shared'] - f['state']
    ex['unattributed_engine'] = unattributed
    outside = wall_ms - total
    named_host = (f['prenorm'] + f['gemm_tile'] + f['gemm_tr'] + f['tail'] + f['setup'] +
                  f['sync'] + f['qkv_scatter'] + f['mid'])
    r2 = lambda v: round(v, 2)
    s = sum(ex.values())
    checks = {
        'wall_ms': r2(wall_ms),
        'engine_total_ms': r2(total),
        'categories_sum_ms': r2(s),
        # 1. The categories must add up to the engine's own total (a grouping error breaks
        #    this): tolerance is the 0.1 ms resolution of the printed stage lines.
        'categories_minus_engine_ms': r2(s - total),
        # 2. THE requested sanity check: exclusive categories may never exceed wall time.
        #    v1 failed exactly here, by 4796.0 ms at 2048.
        'categories_minus_wall_ms': r2(s - wall_ms),
        'within_wall': bool(s <= wall_ms * 1.0001),
        'unattributed_engine_ms': r2(unattributed),
        'outside_engine_ms': r2(outside),
        # 3. part1 really is the sum of the fields beneath it on THIS route. A non-zero
        #    delta means the run took a route whose timers nest differently (dxB rolls
        #    part1 back around attention), so the allocation would not be exact.
        'part1_identity_delta_ms': r2(named_host - t['host']),
        # 4. total really is the sum of the columns the CLI prints.
        'total_identity_delta_ms': r2(t['gemm'] + t['host'] + route + t['lm'] +
                                      f['residue'] - total),
        'nested_diagnostics': {k: r2(v) for k, v in nested.items()},
    }
    return dict(
        exclusive={k: r2(v) for k, v in ex.items()},
        checks=checks,
        share_of_wall_pct={k: r2(100 * v / wall_ms if wall_ms else 0.0) for k, v in ex.items()},
        share_of_engine_pct={k: r2(100 * v / total if total else 0.0) for k, v in ex.items()},
        outside_engine_wall_pct=r2(100 * outside / wall_ms if wall_ms else 0.0))


def parse(text: str) -> dict:
    totals = [m.groupdict() for m in STAGE.finditer(text)]
    details = [m.groupdict() for m in DETAIL.finditer(text)]
    dispatch = [m.groupdict() for m in DISPATCH.finditer(text)]
    if not totals or not details:
        return {}
    sums = {k: 0.0 for k in ('total', 'gemm', 'host', 'route', 'lm')}
    for row in totals:
        for k in sums:
            sums[k] += float(row[k])
    dsum = {}
    for d in details:
        for k, v in d.items():
            dsum[k] = dsum.get(k, 0.0) + float(v)
    disp = {}
    for d in dispatch:
        disp[d['name']] = dict(calls=int(d['calls']), ms=float(d['total']),
                               mean=float(d['mean']))
    return dict(stage_lines=len(totals), totals=sums, fields=dsum, dispatch=disp)


def recompute(path: Path) -> dict:
    """Re-derive the whole attribution from a committed JSON. No hardware, no new run.

    The JSON already holds every raw absolute value plus the process wall, so a correction
    to the grouping never costs another multi-minute sweep.
    """
    d = json.loads(path.read_text())
    for n in d['lengths']:
        rec = d['data'][str(n)]
        acc = allocate(rec, rec['process_wall_s'] * 1000.0)
        rec['allocation'] = acc
        rec['categories'] = acc['exclusive']      # older readers keep working
    d['version'] = 2
    d['accounting_model'] = ACCOUNTING_MODEL
    d['scaling'] = {}
    first = d['data'][str(d['lengths'][0])]['allocation']['exclusive']
    for k in first:
        pts = [(n, d['data'][str(n)]['allocation']['exclusive'].get(k, 0.0))
               for n in d['lengths']]
        d['scaling'][k] = dict(per_length_ms={str(n): v for n, v in pts},
                               scaling_vs_length=classify(pts))
    return d


def token_count(path: Path) -> int:
    return len([t for t in re.split(r'[,\s]+', path.read_text()) if t])


def run_one(length: int, reps: int) -> dict:
    ids = FIXTURE_DIR / f'ids{length}.txt'
    if not ids.is_file():
        raise SystemExit(f'missing {ids}')
    # The fixture must really be `length` tokens. The first version of this tool took a
    # prefix of a 513-token file for every length, so 1024 and 2048 silently measured
    # 513 tokens again -- and two identical rows in the table would have read as
    # "cost stops growing", the exact opposite of the truth.
    got = token_count(ids)
    if got != length:
        raise SystemExit(f'{ids} holds {got} tokens, not the {length} this run claims')
    env = dict(os.environ)
    # The frozen production arithmetic, explicitly: `paired`, no role override.
    env['OFLM_OPEN_PREFILL_PRECISION'] = 'paired'
    env.pop('OFLM_OPEN_PREFILL_ROLES', None)
    env.pop('OFLM_OPEN_PREFILL_ROLE_MODE', None)
    env['OFLM_OPEN_DISPATCH_LOG'] = '1'
    runs = []
    for _ in range(reps):
        t0 = time.monotonic()
        proc = subprocess.run(
            [str(CLI), '--model', str(MODEL), '--kernels', str(KERNELS),
             '--ids-file', str(ids), '--max-ctx', '4096', '--max-tokens', '1',
             '--gemm-block'],
            cwd=str(ROOT), env=env, capture_output=True, text=True)
        wall = time.monotonic() - t0
        text = proc.stdout + '\n' + proc.stderr
        rec = parse(text)
        rec['process_wall_s'] = round(wall, 2)
        rec['returncode'] = proc.returncode
        runs.append(rec)
    best = runs[0]
    best['reps'] = reps
    best['allocation'] = allocate(best, best['process_wall_s'] * 1000.0)
    best['categories'] = best['allocation']['exclusive']   # older readers keep working
    best['dispatch_total_ms'] = round(sum(d['ms'] for d in best['dispatch'].values()), 2)
    best['dispatch_calls'] = sum(d['calls'] for d in best['dispatch'].values())
    # A dispatch count that does not grow with the prompt is the signature of a run
    # that did not process the prompt. Make it a recorded fact, not a suspicion.
    best['tokens_in_run'] = got
    best['dispatch_calls_reps'] = [r['dispatch_calls'] for r in runs]
    return best


def classify(points: list) -> str:
    """Rough scaling label from the measured points. Never forced onto noise.

    `ratio` is growth-per-decade normalised to the length growth: 1.0 is O(T), 0.0 is
    fixed overhead, 2.0 is O(T^2). A first version compared only the endpoints without
    normalising, which labelled a clean 16x-for-16x O(T) term as "constant".
    """
    if len(points) < 2:
        return 'unknown'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if ys[0] <= 0 or xs[0] <= 0:
        return 'unknown'
    ratio = math.log(ys[-1] / ys[0]) / math.log(xs[-1] / xs[0])
    if ratio < 0.25:
        return 'constant/fixed overhead'
    if ratio <= 1.25:
        return 'O(T)'
    if ratio <= 1.60:
        return 'superlinear'
    return 'approximately O(T^2)'


def report(d: dict) -> int:
    """Print both share columns and the identity checks, from an in-memory profile."""
    lengths = d['lengths']
    last = str(lengths[-1])
    print()
    print(f'{"category (exclusive)":30s}' + ''.join(f'{n:>10d}' for n in lengths) +
          f'{"wall%":>9s}{"engine%":>9s}  scaling')
    for k in d['data'][last]['allocation']['exclusive']:
        row = ''.join(f'{d["data"][str(n)]["allocation"]["exclusive"].get(k, 0.0):10.1f}'
                      for n in lengths)
        aw = d['data'][last]['allocation']['share_of_wall_pct'].get(k, 0.0)
        ae = d['data'][last]['allocation']['share_of_engine_pct'].get(k, 0.0)
        print(f'{k:30s}{row}{aw:8.1f}%{ae:8.1f}%  '
              f'{d["scaling"][k]["scaling_vs_length"]}')
    print(f'{"outside_engine (process gap)":30s}'
          f'{d["data"][last]["allocation"]["checks"]["outside_engine_ms"]:10.1f}'
          + ' ' * (10 * (len(lengths) - 1)) +
          f'{d["data"][last]["allocation"]["outside_engine_wall_pct"]:8.1f}%'
          f'{"":9s}  constant-ish')
    print()
    print('denominators: wall% = category / process wall; engine% = category / engine total.')
    print('checks (2048):')
    for k, v in d['data'][last]['allocation']['checks'].items():
        print(f'  {k:32s} {v}')
    bad = []
    for n in lengths:
        c = d['data'][str(n)]['allocation']['checks']
        if not c['within_wall']:
            bad.append(f'{n}: categories exceed wall by {-c["categories_minus_wall_ms"]:.1f} ms')
        if abs(c['categories_minus_engine_ms']) > 1.0:
            bad.append(f'{n}: categories differ from engine total by '
                       f'{c["categories_minus_engine_ms"]:.1f} ms')
        if abs(c['part1_identity_delta_ms']) > 1.0:
            bad.append(f'{n}: part1 identity off by {c["part1_identity_delta_ms"]:.1f} ms')
    if bad:
        print('\nFAILED accounting checks:')
        for b in bad:
            print(f'  - {b}')
        return 1
    print('\nall accounting checks hold')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lengths', nargs='+', type=int, default=[128, 512, 1024, 2048])
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--out', type=Path, default=HERE / 'e2e-profile-v1.json')
    # Recompute an existing profile in place. The JSON holds every raw absolute value and
    # the process wall, so a grouping fix must never cost another multi-minute sweep.
    ap.add_argument('--from-json', type=Path, default=None,
                    help='recompute the allocation from a committed profile; no hardware run')
    a = ap.parse_args()

    if a.from_json:
        d = recompute(a.from_json)
        print(f'recomputed {a.from_json} (no hardware run)')
        rc = report(d)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(d, indent=1, allow_nan=False, default=float) + '\n')
        print(f'\nwrote {a.out}')
        return rc

    data = {}
    print(f'{"len":>5s} {"wall ms":>10s} {"GEMM ms":>10s} {"host ms":>10s} '
          f'{"tok/s":>8s} {"dispatches":>11s} {"calls":>6s}')
    for n in a.lengths:
        rec = run_one(n, a.reps)
        rec['prompt_tokens'] = n
        rec['ttft_s'] = rec['process_wall_s']
        rec['prompt_tok_per_s'] = round(n / rec['process_wall_s'], 2) if rec['process_wall_s'] else None
        data[n] = rec
        print(f'{n:5d} {rec["process_wall_s"] * 1000:10.1f} {rec["totals"]["gemm"]:10.1f} '
              f'{rec["totals"]["host"]:10.1f} {rec["prompt_tok_per_s"]:8.2f} '
              f'{rec["dispatch_calls"]:11d} {rec["dispatch_calls"]:6d}')

    scaling = {}
    for k in data[a.lengths[0]]['categories']:
        pts = [(n, data[n]['categories'].get(k, 0.0)) for n in a.lengths]
        scaling[k] = dict(per_length_ms={str(n): v for n, v in pts},
                          scaling_vs_length=classify(pts))

    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT),
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    result = dict(
        tool='e2e_profile', version=2,
        scope='Phase 24 attribution of the current implementation; NOT a published '
              'benchmark and NOT model acceptance',
        model_acceptance=False,
        precision_mode='paired',
        reps=a.reps,
        dispersion_measured=False,
        dispersion_note='reps=1: this pass attributes cost, it does not measure run-to-run '
                        'spread. Do not quote these as final performance numbers.',
        commit=commit,
        fixture=dict(source=str(FIXTURE_DIR),
                     note='the 128/512/2048 cases are prefixes of the same documented '
                          '513-token prompt, so length is the only variable'),
        kernels=str(KERNELS), cli=str(CLI), model=str(MODEL),
        accounting_model=ACCOUNTING_MODEL,
        denominators='wall% = category / process wall; engine% = category / engine '
                     'total_ms; categories are exclusive and sum to the engine total',
        lengths=a.lengths, data={str(n): data[n] for n in a.lengths}, scaling=scaling)
    rc = report(result)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, allow_nan=False, default=float) + '\n')
    print(f'\nwrote {a.out}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
