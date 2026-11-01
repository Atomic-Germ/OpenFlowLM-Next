"""Phase 24: end-to-end prefill profile of the CURRENT implementation.

Runs the real 64-layer model through the production block route at several prompt
lengths, parses the engine's own stage breakdown, and builds the attribution the plan
asks for: every category as an absolute time and as a share of wall, plus how each
scales between the measured lengths.

The stage breakdown is the engine's `StepTiming`, whose fields are disjoint by
construction (each is accumulated at one point in one stage), so the categories are not
double counted. `part0` is the GEMM column -- the device time the host blocked for --
and `part1` is the host half; the split of `part1` into prenorm / gemm tile / gemm
transpose / mid / tail / shared / state / sync is exactly the "what to work on next"
answer, and `residue` is whatever total minus those leaves.

`--reps` defaults to 1: this is an ATTRIBUTION run, not a published benchmark. The JSON
records that dispersion was not measured, because one pass cannot say anything about it.

    python e2e_profile.py --lengths 128 512 1024 2048 --out e2e-profile-v1.json
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

# The engine's fields, grouped into the categories the plan asks about. Every field
# appears exactly once, so the groups sum to the same total the engine reports.
GROUPS = {
    'projection_gemm': 'part0 (all four projection GEMMs, device-blocked time)',
    'ffn_and_gemm_host': 'gemm_tile + gemm_tr',
    'deltanet': 'dn_conv + dn_rule',
    'attention': 'attn',
    'norm_elementwise': 'prenorm + tail',
    'state_movement': 'state',
    'sync_bo': 'sync',
    'host_setup': 'setup + shared',
    'moe': 'moe_prep + moe_patch + moe_run + moe_read',
    'qkv_scatter': 'qkv_scatter',
    'residue': 'unattributed',
}


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


def categories(rec: dict) -> dict:
    """Absolute ms per plan category, from strictly disjoint engine fields."""
    t, f = rec['totals'], rec['fields']
    c = {
        'projection_gemm': t['gemm'],
        'ffn_and_gemm_host': f['gemm_tile'] + f['gemm_tr'],
        'deltanet': f['dn_conv'] + f['dn_rule'],
        'attention': f['attn'],
        'norm_elementwise': f['prenorm'] + f['tail'],
        'state_movement': f['state'],
        'sync_bo': f['sync'],
        'host_setup': f['setup'] + f['shared'],
        'moe': f['moe_prep'] + f['moe_patch'] + f['moe_run'] + f['moe_read'],
        'qkv_scatter': f['qkv_scatter'],
        'residue': f['residue'],
        'lm_head': t['lm'],
    }
    return {k: round(v, 2) for k, v in c.items()}


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
    best['categories'] = categories(best)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lengths', nargs='+', type=int, default=[128, 512, 1024, 2048])
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--out', type=Path, default=HERE / 'e2e-profile-v1.json')
    a = ap.parse_args()

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

    keys = list(data[a.lengths[0]]['categories'])
    scaling = {}
    for k in keys:
        pts = [(n, data[n]['categories'].get(k, 0.0)) for n in a.lengths]
        scaling[k] = dict(per_length_ms={str(n): v for n, v in pts},
                          scaling_vs_length=classify(pts))
    print()
    print(f'{"category":22s} ' + ' '.join(f'{n:>9d}' for n in a.lengths) +
          f' {"2048 share":>11s}  scaling')
    for k in keys:
        row = ' '.join(f'{data[n]["categories"].get(k, 0):9.1f}' for n in a.lengths)
        share = data[a.lengths[-1]]['categories'].get(k, 0) / max(
            1e-9, data[a.lengths[-1]]['totals']['total'])
        print(f'{k:22s} {row} {share * 100:10.1f}%  {scaling[k]["scaling_vs_length"]}')

    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT),
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    result = dict(
        tool='e2e_profile', version=1,
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
                     note='the 128/512/1024 cases are prefixes of the same documented '
                          '513-token prompt, so length is the only variable'),
        kernels=str(KERNELS), cli=str(CLI), model=str(MODEL),
        categories_definition=GROUPS,
        totals_check='groups are disjoint engine fields; their sum plus residue and '
                     'lm_head equals the engine total',
        lengths=a.lengths, data={str(n): data[n] for n in a.lengths}, scaling=scaling)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, allow_nan=False, default=float) + '\n')
    print(f'\nwrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
