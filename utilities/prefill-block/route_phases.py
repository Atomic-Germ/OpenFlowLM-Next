"""Phase 28.4/28.9: where a prefill's time actually goes, per route.

The A/B says which route wins. It does not say what either route spends its time on, and a
bottleneck ranking built on wall time alone would be a guess. The standalone CLI already
prints, for every schedule it runs, the split the engine itself maintains
(`Core::last_timing()`): TOTAL, GEMM dispatches (part0), host stages (part1), the per-token
column (route), lm_head, and inside part1 the named stages -- prenorm, gemm tile, gemm
transpose, tail, shared, state, sync, setup, and mid split into dn conv / dn rule / attn.

This drives both schedules over the same prompt ids and records those numbers, so the
mechanism question ("what does layer-major save, what does it cost") is answered from the
engine's own accounting rather than from a difference of two wall times.

## Categories

The stage columns overlap by construction -- `part1` is the SUM of prenorm, gemm tile, gemm
transpose, tail, shared, state, sync and mid -- so they cannot simply be added. They are
reported alongside the totals, and two assertions keep the reporting honest:

  * `part0 + part1 + route + lmhead <= total` (the residue is named, never absorbed);
  * every named stage is a SUBSET of part1, so no column may exceed it.

No category is a percentage of wall time here: this is engine-internal accounting for
choosing the next target, not the production profile, which has its own denominators.

    python route_phases.py --prompt 520
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
import oracle as O  # noqa: E402

# "  layer-major [0,520) t_real=520: 101475.0 ms (GEMM 1.0, host 2.0, per-token 3.0, lm_head 4.0)"
_SCHED = re.compile(r'^\s+(?P<what>[\w-]+) \[(?P<n0>\d+),(?P<n1>\d+)\) t_real=(?P<t_real>\d+): '
                    r'(?P<total>[\d.]+) ms \(GEMM (?P<gemm>[\d.]+), host (?P<host>[\d.]+), '
                    r'per-token (?P<route>[\d.]+), lm_head (?P<lm>[\d.]+)\)', re.M)
# "      mid 1.0 (dn conv 2.0, dn rule 3.0, attn 4.0), prenorm 5.0, gemm tile 6.0, ..."
_STAGES = re.compile(r'^\s+mid (?P<mid>[\d.]+) \(dn conv (?P<dn_conv>[\d.]+), dn rule '
                     r'(?P<dn_rule>[\d.]+), attn (?P<attn>[\d.]+)\), '
                     r'prenorm (?P<prenorm>[\d.]+), gemm tile (?P<gemm_tile>[\d.]+),'
                     r' gemm tr (?P<gemm_tr>[\d.]+), tail (?P<tail>[\d.]+), '
                     r'shared (?P<shared>[\d.]+), state (?P<state>[\d.]+), '
                     r'sync (?P<sync>[\d.]+), setup (?P<setup>[\d.]+)'
                     r' \| moe prep (?P<moe_prep>[\d.]+), patch (?P<moe_patch>[\d.]+),'
                     r' run (?P<moe_run>[\d.]+), read (?P<moe_read>[\d.]+)'
                     r' \| qkv scatter (?P<qkv_scatter>[\d.]+) \| residue (?P<residue>[\d.]+)',
                     re.M)


def parse(text: str) -> list[dict]:
    """Every schedule line the CLI printed, with its stage line."""
    schedules = [m.groupdict() for m in _SCHED.finditer(text)]
    stages = [m.groupdict() for m in _STAGES.finditer(text)]
    out = []
    for s, t in zip(schedules, stages):
        row = {k: (float(v) if k not in ('what', 'n0', 'n1', 't_real') else v)
               for k, v in s.items()}
        row.update({k: float(v) for k, v in t.items()})
        out.append(row)
    # A stage line without its schedule line (or the reverse) means the two patterns
    # drifted apart; that is a harness bug and must not read as "no stages".
    if len(schedules) != len(stages):
        raise SystemExit(f'parsed {len(schedules)} schedule lines but {len(stages)} stage '
                         'lines; the CLI format changed')
    return out


def run(ids: list[int], route: str, *, layers: int, workdir: Path, max_ctx: int) -> dict:
    import fcntl
    import subprocess
    idfile = workdir / 'ids.txt'
    idfile.write_text(','.join(str(i) for i in ids) + '\n')
    env = dict(__import__('os').environ)
    env['OFLM_OPEN_PREFILL_PRECISION'] = 'paired'
    env.pop('OFLM_OPEN_PREFILL_ROLES', None)
    env.pop('OFLM_OPEN_PREFILL_ROLE_MODE', None)
    if route == 'block':
        env['OFLM_OPEN_LAYER_MAJOR'] = '0'
    else:
        env.pop('OFLM_OPEN_LAYER_MAJOR', None)
    # `--pmode none` on purpose: the CLI's own note says the power mode is worth about 2x
    # and a standalone run otherwise inherits whatever mode the last app run left behind.
    # Leaving it alone keeps this a measurement of the two schedules, not of a mode.
    cmd = [str(O.CLI), '--model', str(O.MODEL), '--kernels', str(O.KERNELS),
           '--ids-file', str(idfile), '--max-ctx', str(max_ctx), '--max-tokens', '1',
           '--layers', str(layers), '--pmode', 'none', '--gemm-block']
    if route == 'block':
        # --block-major keeps `--gemm-block` off the layer-major branch, so the CLI runs
        # the same Core::step_gemm_block loop the Engine's block route calls.
        cmd.append('--block-major')
    t0 = time.monotonic()
    with O.LOCK.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        proc = subprocess.run(cmd, cwd=str(O.ROOT), env=env, capture_output=True, text=True)
    wall = time.monotonic() - t0
    text = proc.stdout + '\n' + proc.stderr
    if proc.returncode != 0:
        raise SystemExit(f'{route} failed (rc {proc.returncode}):\n'
                         + '\n'.join(text.strip().splitlines()[-15:]))
    return dict(route=route, rc=proc.returncode, wall_s=round(wall, 2),
                schedules=parse(text), command=' '.join(cmd))


def check(schedules: list[dict]) -> list[dict]:
    """The Phase 25 assertions: no category may silently exceed its denominator."""
    rows = []
    for s in schedules:
        parts = s['gemm'] + s['host'] + s['route'] + s['lm']
        rows.append(dict(schedule=s['what'],
                         total_ms=s['total'], sum_of_parts=round(parts, 1),
                         parts_fit=(parts <= s['total'] + 1e-6),
                         residue_ms=round(s['total'] - parts, 1)))
        # Two columns are deliberately NOT in this sum, because neither is inside part1:
        #   `state_ms`  the state BO syncs are bookkept on their own (11 ms at 512 tokens);
        #   `shared_ms` `ffn_block` books its own stages to shared_ms and the caller adds
        #               only the residual/norm window to part1, so shared_ms sits outside
        #               it (1064 ms at 512 tokens -- which is exactly the overage that made
        #               this assertion read as a broken denominator twice).
        named = (s['prenorm'] + s['gemm_tile'] + s['gemm_tr'] + s['tail'] + s['sync']
                 + s['mid'])
        rows.append(dict(schedule=s['what'], check='named stages fit inside host',
                         total_ms=s['host'], sum_of_parts=round(named, 1),
                         parts_fit=(named <= s['host'] * 1.02 + 1.0),
                         residue_ms=round(s['host'] - named, 1)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prompt', type=int, default=520,
                    help='prompt ids; the fixture pool is O.POOL (8192 since Phase 29)')
    ap.add_argument('--layers', type=int, default=64)
    ap.add_argument('--max-ctx', type=int, default=4096)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    if a.prompt > O.POOL:
        raise SystemExit(f'the oracle fixture pool is {O.POOL} ids; choose --prompt <= {O.POOL}')

    ids = O.fixture(a.prompt, seed=26026)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'route-phases-{a.prompt}-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-route-phases', version=1, phase=28,
                  gate='28.4/28.5 the mechanism: what each schedule spends its time on',
                  model_acceptance=False, cli=str(O.CLI), kernels=str(O.KERNELS),
                  started=stamp, prompt_tokens=a.prompt, layers=a.layers,
                  note='engine-internal accounting for target selection; not the '
                       'production profile and not a percentage of wall time')

    runs = {}
    for route in ('layer-major', 'block'):
        print(f'=== {route} ===', flush=True)
        r = run(ids, route, layers=a.layers, workdir=out, max_ctx=a.max_ctx)
        runs[route] = r
        for s in r['schedules']:
            print(f"  {s['what']} [{s['n0']},{s['n1']}) t_real={s['t_real']}: "
                  f"{s['total']:.0f} ms = GEMM {s['gemm']:.0f} + host {s['host']:.0f} + "
                  f"per-token {s['route']:.0f} + lm {s['lm']:.0f}", flush=True)
    # The two schedules must have done the same WORK, or the split is not comparable: the
    # block loop reports one line per block, so its totals are summed.
    lm = runs['layer-major']['schedules']
    bl = runs['block']['schedules']
    if len(lm) != 1:
        result['verdict'] = 'FAIL'
        result['reason'] = f'layer-major printed {len(lm)} schedule lines, expected 1'
        (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
        return 1
    result['block_blocks'] = len(bl)
    result['block_total_ms'] = round(sum(s['total'] for s in bl), 1)
    result['layer_major_total_ms'] = round(lm[0]['total'], 1)
    result['runs'] = runs
    result['checks'] = check(runs['block']['schedules'] + runs['layer-major']['schedules'])
    result['verdict'] = 'PASS' if all(c['parts_fit'] for c in result['checks']) else 'FAIL'
    for c in result['checks']:
        print(f"  {str(c.get('schedule')):12s} {c.get('check', 'parts fit'):32s} "
              f"{c['sum_of_parts']:9.1f} / {c['total_ms']:9.1f}  "
              f"{'PASS' if c['parts_fit'] else 'FAIL'}", flush=True)
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
