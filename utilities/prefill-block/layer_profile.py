"""Performance plan PR #1: where a prefill's time goes, layer by layer.

The engine keeps a running `StepTiming` over a whole prompt or block, so it can say "the
host stages cost 12 s" but not "layer 41 costs twice what layer 40 does". `OFLM_OPEN_PROFILE`
turns that same accounting into one JSON row per layer (a difference of two snapshots taken
at the layer boundaries -- no clock read inside a layer), and this script drives a run and
aggregates it.

What it answers, and in what units:

  * per-layer totals, ranked -- the layers worth looking at, and whether they are the wide
    ones or whether something is disproportionate;
  * the split inside a layer between the NPU GEMM dispatches (`gemm_ms`), the host stages
    (`host_ms`), per-token dispatches (`per_token_ms`) and the lm head;
  * the split inside the host stages (prenorm, tiling, transpose, DeltaNet conv/rule,
    attention, tails, shared expert, state syncs, buffer syncs), summed over the model;
  * the request residue -- what neither the layer rows nor the lm head account for.

The two routes are separate runs because they are different schedules: `--route block` is the
production default for this model (the block loop over 256-token blocks) and `--route
layer-major` is the alternative. `--route sequential` is decode-as-prefill, one `step()` per
token, profiled for `--steps` tokens only.

    python layer_profile.py --route block --tokens 256
    python layer_profile.py --route layer-major --tokens 256

This is engine-internal accounting for choosing the next target. It is not a throughput
claim: the wall time it prints carries the model load, and every number here is a duration
the engine attributed to a stage, not a percentage of a budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle as O  # noqa: E402

# The columns a layer row carries, in the order the report prints them.
# `wall_ms` is the layer's own steady-clock duration, taken at the module boundaries: it is
# the column that answers "which layer costs what", and the only one StepTiming cannot
# provide (its `total_ms` is written once per block, not per layer).
LAYER_COLS = ('wall_ms', 'gemm_ms', 'host_ms', 'per_token_ms', 'lmhead_ms')
HOST_COLS = ('prenorm_ms', 'gemm_tile_ms', 'gemm_tr_ms', 'mid_ms', 'dn_conv_ms', 'dn_rule_ms',
             'attn_ms', 'tail_ms', 'shared_ms', 'state_ms', 'sync_ms', 'qkv_scatter_ms',
             'moe_prep_ms', 'moe_patch_ms', 'moe_run_ms', 'moe_read_ms', 'setup_ms')
# Stage columns that are SUBSETS of another column and must not be added to the totals:
#   dn_conv / dn_rule inside mid; mid inside host; everything else is its own window.
SUBSETS = {'dn_conv_ms': 'mid_ms', 'dn_rule_ms': 'mid_ms', 'mid_ms': 'host_ms'}


def validate_profile(header, requests, layers):
    """Require complete, ordered layer coverage for every request, not just small sums."""
    if not header or header.get('version') != 1 or not requests:
        raise ValueError('missing header/requests or unsupported profile version')
    count = header.get('layers')
    if type(count) is not int or count <= 0 or len(layers) != count * len(requests):
        raise ValueError('incomplete layer coverage')
    for i, request in enumerate(requests):
        group = layers[i * count:(i + 1) * count]
        if (request.get('layers') != count or request.get('tokens', 0) <= 0
                or request.get('blocks', 0) <= 0
                or [r.get('layer') for r in group] != list(range(count))
                or any(r.get('what') != request.get('what') for r in group)):
            raise ValueError('duplicate, missing, out-of-order or mismatched layer records')
        for record in [request, *group]:
            required = ('total_ms', 'lmhead_ms') if record is request else LAYER_COLS + HOST_COLS
            if any(k not in record for k in required):
                raise ValueError('missing timing columns')
            if any(not isinstance(v, (int, float)) or not math.isfinite(v)
                   or (v < 0 and k != 'residue_ms')
                   for k, v in record.items() if k.endswith('_ms')):
                raise ValueError('invalid timing value')


def provenance(paths):
    result = {}
    for path in paths:
        with path.open('rb') as f:
            result[str(path)] = dict(bytes=path.stat().st_size,
                                     sha256=hashlib.file_digest(f, 'sha256').hexdigest())
    return result


def conditions():
    """Host observations, not a claim to measure the NPU die temperature."""
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    return dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), system=platform.platform(),
                loadavg=read('/proc/loadavg'), meminfo=read('/proc/meminfo'),
                cpu_governor=read('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
                thermal_zones=[dict(type=read(p / 'type'), temp_millicelsius=read(p / 'temp'))
                               for p in sorted(Path('/sys/class/thermal').glob('thermal_zone*'))],
                runtime_env={k: os.environ.get(k) for k in (
                    'OMP_NUM_THREADS', 'OMP_WAIT_POLICY', 'OPENBLAS_NUM_THREADS',
                    'OFLM_OPEN_SUBMIT_AHEAD', 'OFLM_OPEN_SPIN_US', 'OFLM_OPEN_STAGE_MAJOR')})


def require_power_mode(log, requested):
    """Do not silently turn a failed power-mode request into a benchmark PASS."""
    if requested == 'none':
        return 'left unchanged; actual mode requires an external observation'
    expected = f'NPU power mode: {requested}'
    if expected not in log.splitlines():
        raise ValueError('requested NPU power mode was not confirmed; inspect the retained CLI log')
    return 'CLI configure command succeeded'


def validate_route(route, requests, *, tokens, steps, max_tokens):
    expected = {'block': 'gemm-block', 'layer-major': 'gemm-prompt', 'sequential': 'step'}[route]
    prompt = [r for r in requests if r['what'] == expected]
    if route == 'sequential':
        if len(requests) != min(steps, tokens + max_tokens - 1) or len(prompt) != len(requests):
            raise ValueError('sequential profile coverage differs from requested budget')
    elif (not prompt or sum(r['tokens'] for r in prompt) != tokens
          or any(r['what'] not in (expected, 'step') for r in requests)
          or (route == 'layer-major' and len(prompt) != 1)
          or sum(r['what'] == 'step' for r in requests) != min(steps, max_tokens - 1)):
        raise ValueError(f'{route}: observed route/token coverage differs from requested route')


def read_jsonl(path: Path) -> tuple[dict | None, list[dict], list[dict]]:
    """(header, request rows, layer rows) from an OFLM_OPEN_PROFILE file."""
    header, requests, layers = None, [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f'{path}: not JSONL at {line[:60]!r}: {exc}')
        kind = rec.get('record')
        if kind == 'header':
            if header is not None or requests or layers:
                raise ValueError('duplicate or misplaced profile header')
            header = rec
        elif kind == 'request':
            if header is None or len(layers) != len(requests) * header['layers']:
                raise ValueError('request before previous layer group is complete')
            requests.append(rec)
        elif kind == 'layer':
            if not requests or len(layers) >= len(requests) * header['layers']:
                raise ValueError('layer outside a request')
            layers.append(rec)
        else:
            raise SystemExit(f'{path}: unknown record {kind!r}')
    validate_profile(header, requests, layers)
    return header, requests, layers


def run(ids: list[int], route: str, *, layers: int, max_ctx: int, max_tokens: int,
        steps: int, pmode: str, workdir: Path, timeout: float) -> dict:
    """One profiled run of the standalone CLI; returns the parsed profile."""
    idfile = workdir / 'ids.txt'
    idfile.write_text(','.join(str(i) for i in ids) + '\n')
    prof = workdir / f'profile-{route}.jsonl'
    env = dict(os.environ)
    # The accepted arithmetic, explicitly: a run under an inherited experimental mode would
    # attribute time to a route the plan is not optimizing.
    env['OFLM_OPEN_PREFILL_PRECISION'] = 'paired'
    env.pop('OFLM_OPEN_PREFILL_ROLES', None)
    env.pop('OFLM_OPEN_PREFILL_ROLE_MODE', None)
    env['OFLM_OPEN_PROFILE'] = str(prof)
    env['OFLM_OPEN_PROFILE_STEPS'] = str(steps)
    # Dense Qwen defaults to block-loop even without --block-major.
    if route == 'block':
        env['OFLM_OPEN_LAYER_MAJOR'] = '0'
    elif route == 'layer-major':
        env['OFLM_OPEN_LAYER_MAJOR'] = '1'
    else:
        env.pop('OFLM_OPEN_LAYER_MAJOR', None)
    cmd = [str(O.CLI), '--model', str(O.MODEL), '--kernels', str(O.KERNELS),
           '--ids-file', str(idfile), '--max-ctx', str(max_ctx), '--max-tokens', str(max_tokens),
           '--layers', str(layers), '--pmode', pmode]
    if route in ('block', 'layer-major'):
        cmd.append('--gemm-block')
        if route == 'block':
            cmd.append('--block-major')
    t0 = time.monotonic()
    with O.LOCK.open('a') as handle:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            proc = subprocess.run(cmd, cwd=str(O.ROOT), env=env, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(f'{route}: timed out after {timeout}s') from exc
    wall = time.monotonic() - t0
    text = proc.stdout + '\n' + proc.stderr
    (workdir / f'{route}.log').write_text(text)
    if proc.returncode != 0:
        raise SystemExit(f'{route} failed (rc {proc.returncode}):\n'
                         + '\n'.join(text.strip().splitlines()[-20:]))
    power_mode = require_power_mode(text, pmode)
    if not prof.exists():
        raise SystemExit(f'{route}: the run wrote no profile at {prof}\n'
                         + '\n'.join(text.strip().splitlines()[-10:]))
    header, requests, rows = read_jsonl(prof)
    if header['layers'] != layers:
        raise ValueError('observed layer count differs from requested count')
    validate_route(route, requests, tokens=len(ids), steps=steps, max_tokens=max_tokens)
    return dict(route=route, rc=proc.returncode, wall_s=round(wall, 2), profile=str(prof),
                power_mode=power_mode,
                header=header, requests=requests, layers=rows, command=' '.join(cmd),
                generated_ids=[int(line.split()[1]) for line in proc.stdout.splitlines()
                               if line.startswith('token ')],
                stderr_tail='\n'.join(text.strip().splitlines()[-8:]))


def summarise(run_rec: dict) -> dict:
    """Aggregate one run's layer rows: totals, the model-wide stage split, the worst layers."""
    rows, reqs = run_rec['layers'], run_rec['requests']
    validate_profile(run_rec['header'], reqs, rows)
    # Keep prefill and post-prefill step samples separate in block-route summaries.
    if run_rec['route'] != 'sequential':
        count = run_rec['header']['layers']
        groups = [(r, rows[i * count:(i + 1) * count]) for i, r in enumerate(reqs)
                  if r['what'] != 'step']
        reqs = [r for r, _ in groups]
        rows = [row for _, group in groups for row in group]
    # A block-loop run profiles one block at a time and, on a multi-block prompt, the LAST
    # block is the padded one. Rows from every block are summed for the model-wide split, and
    # the per-layer table is summed per layer index too -- so a 256-token prompt (one block)
    # and a 1024-token prompt (four) are not mistaken for each other.
    per_layer: dict[int, dict] = {}
    for r in rows:
        acc = per_layer.setdefault(r['layer'], {'layer': r['layer'], 'kind': r['kind'],
                                                'blocks': 0, **{c: 0.0 for c in
                                                                LAYER_COLS + HOST_COLS}})
        acc['blocks'] += 1
        for c in LAYER_COLS + HOST_COLS:
            acc[c] += r.get(c, 0.0)
    table = sorted(per_layer.values(), key=lambda r: -r['wall_ms'])
    total = sum(r['wall_ms'] for r in table)
    stage = {c: sum(r.get(c, 0.0) for r in table) for c in LAYER_COLS + HOST_COLS}
    request_total = sum(r['total_ms'] for r in reqs)
    lm = sum(r['lmhead_ms'] for r in reqs)
    residue = sum(r.get('residue_ms', 0.0) for r in reqs)
    # The two kinds are different layers doing different work; a per-kind mean says whether a
    # layer is slow BECAUSE of its kind or in spite of it.
    # Every column, subsets included: the kind table is a per-kind breakdown, and `mid_ms`
    # being inside `host_ms` does not make "how much mid did THIS kind cost" less useful.
    KIND_COLS = LAYER_COLS + HOST_COLS
    by_kind = {}
    for r in table:
        k = by_kind.setdefault(r['kind'], {'kind': r['kind'], 'n': 0,
                                           **{c: 0.0 for c in KIND_COLS}})
        k['n'] += 1
        for c in KIND_COLS:
            k[c] += r.get(c, 0.0)
    for k in by_kind.values():
        k['mean_wall_ms'] = k['wall_ms'] / k['n'] if k['n'] else 0.0
    return dict(
        route=run_rec['route'], wall_s=run_rec['wall_s'], profile=run_rec['profile'],
        header=run_rec['header'], requests=reqs, layers=len(table), rows=len(rows),
        block=sum(r['blocks'] for r in reqs),
        tokens=sum(r['tokens'] for r in reqs),
        stage_unaccounted_ms=request_total - lm - stage['gemm_ms'] - stage['host_ms'] - stage['per_token_ms'],
        request_total_ms=request_total, lmhead_ms=lm, residue_ms=residue,
        layer_total_ms=total, stage=stage, by_kind=sorted(by_kind.values(), key=lambda k: -k['wall_ms']),
        per_layer=table,
        top_layers=[{'layer': r['layer'], 'kind': r['kind'], 'ms': r['wall_ms'],
                     'share': r['wall_ms'] / total if total else 0.0} for r in table[:10]],
    )


def check_consistency(summary: dict) -> list[dict]:
    """The assertions that keep the report honest.

    A stage column is a window the engine opened explicitly, and some of them nest. These
    checks say what nests, so a reader cannot add `mid_ms` to `host_ms` and call it a total.
    """
    rows = []
    if not summary.get('layers'):
        return [dict(check='complete profile required', a_ms=0, b_ms=0, ok=False)]
    st = summary['stage']
    # 1. The layer walls must fit inside the request. Both are taken with the same clock, so
    #    the request total is a real denominator for this column and not a second estimate.
    covered = summary['layer_total_ms'] + summary['lmhead_ms']
    rows.append(dict(check='layer wall + lm_head fit inside the request',
                     a_ms=round(covered, 1), b_ms=round(summary['request_total_ms'], 1),
                     ok=covered <= summary['request_total_ms'] * 1.02 + 1.0))
    # 2. host_ms is the sum of the blocks it opened; the named host stages must fit inside it.
    host_named = (st['prenorm_ms'] + st['gemm_tile_ms'] + st['gemm_tr_ms'] + st['mid_ms']
                  + st['tail_ms'] + st['sync_ms'] + st['qkv_scatter_ms'])
    rows.append(dict(check='named host stages fit inside host_ms',
                     a_ms=round(host_named, 1), b_ms=round(st['host_ms'], 1),
                     ok=host_named <= st['host_ms'] * 1.05 + 1.0))
    # 3. The DeltaNet split must fit inside mid, and mid inside host.
    rows.append(dict(check='dn conv + dn rule fit inside mid_ms',
                     a_ms=round(st['dn_conv_ms'] + st['dn_rule_ms'], 1), b_ms=round(st['mid_ms'], 1),
                     ok=st['dn_conv_ms'] + st['dn_rule_ms'] <= st['mid_ms'] * 1.05 + 1.0))
    return rows


def write_csv(path: Path, summaries: list[dict]) -> None:
    """One row per (route, layer): the machine-readable companion to the JSON."""
    cols = ['route', 'layer', 'kind', 'blocks'] + list(LAYER_COLS) + list(HOST_COLS)
    with path.open('w') as f:
        f.write(','.join(cols) + '\n')
        for s in summaries:
            for r in s.get('per_layer', []):
                f.write(','.join([s['route'], str(r['layer']), r['kind'], str(r['blocks'])]
                                 + [f'{r.get(c, 0.0):.3f}' for c in LAYER_COLS + HOST_COLS]) + '\n')


def report(summary: dict, checks: list[dict]) -> None:
    st = summary.get('stage', {})
    print(f"=== {summary['route']} ===")
    if not summary.get('layers'):
        print(f"  no layer rows: {summary.get('reason', '')}")
        return
    print(f"  {summary['tokens']} tokens, {summary['block']} block(s), {summary['rows']} layer rows, "
          f"request {summary['request_total_ms']:.0f} ms, wall {summary['wall_s']:.1f} s")
    print(f"  layer wall {summary['layer_total_ms']:.0f} ms; outside layers {summary['residue_ms']:.0f} ms "
          f"(includes lm_head {summary['lmhead_ms']:.0f} ms)")
    print(f"  unattributed to stages {summary['stage_unaccounted_ms']:.0f} ms")
    print(f"  dispatch/part0 {st['gemm_ms']:.0f} ms, host/part1 {st['host_ms']:.0f} ms, "
          f"per-token {st['per_token_ms']:.0f} ms")
    print("  --- inside the host stages (mid contains dn conv/rule; host contains mid) ---")
    for c in HOST_COLS:
        if st.get(c, 0.0) <= 0.01:
            continue
        sub = f"  (of {SUBSETS[c]})" if c in SUBSETS else ""
        share = 100.0 * st[c] / st['host_ms'] if st.get('host_ms') else 0.0
        print(f"    {c:16s} {st[c]:10.1f} ms  {share:5.1f}% of host{sub}")
    print("  --- by layer kind (mean per layer) ---")
    for k in summary.get('by_kind', []):
        print(f"    {k['kind']:14s} n={k['n']:3d} wall {k['wall_ms']:9.1f} ms "
              f"({k['mean_wall_ms']:7.1f} each), gemm {k['gemm_ms']:9.1f}, host {k['host_ms']:9.1f}, "
              f"tile {k['gemm_tile_ms']:9.1f}, tr {k['gemm_tr_ms']:9.1f}, mid {k['mid_ms']:8.1f}")
    print("  --- the ten widest layers ---")
    for r in summary['top_layers']:
        print(f"    layer {r['layer']:2d} {r['kind']:14s} {r['ms']:9.1f} ms  {100 * r['share']:5.1f}%")
    for c in checks:
        print(f"  {c['check']:44s} {c['a_ms']:9.1f} / {c['b_ms']:9.1f}  "
              f"{'PASS' if c['ok'] else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--route', choices=('block', 'layer-major', 'sequential'),
                    nargs='+', default=['block'])
    ap.add_argument('--tokens', type=int, default=256, help='prompt ids')
    ap.add_argument('--layers', type=int, default=64)
    ap.add_argument('--max-ctx', type=int, default=4096)
    ap.add_argument('--max-tokens', type=int, default=1, help='decode ids after the prompt')
    ap.add_argument('--steps', type=int, default=4, help='sequential steps to profile')
    ap.add_argument('--pmode', default='performance')
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--cli', type=Path, default=O.CLI)
    ap.add_argument('--model', type=Path, default=O.MODEL)
    ap.add_argument('--kernels', type=Path, default=O.KERNELS)
    a = ap.parse_args()
    if a.tokens <= 0 or a.layers <= 0 or a.steps < 0 or a.max_tokens <= 0:
        ap.error('positive token/layer counts and nonnegative step budget required')
    O.CLI, O.MODEL, O.KERNELS = a.cli.resolve(), a.model.resolve(), a.kernels.resolve()
    if a.tokens > O.POOL:
        raise SystemExit(f'the oracle fixture pool is {O.POOL} ids; choose --tokens <= {O.POOL}')
    ids = O.fixture(a.tokens, seed=26026)

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'layer-profile-{a.tokens}-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-layer-profile', version=1,
                  gate='performance plan PR #1: per-layer attribution, not a throughput claim',
                  model_acceptance=False, cli=str(O.CLI), kernels=str(O.KERNELS),
                  started=stamp, prompt_tokens=a.tokens, layers=a.layers, pmode=a.pmode,
                  conditions_start=conditions(),
                  provenance=provenance([O.CLI, O.MODEL / 'model.q4nx', O.KERNELS / 'manifest.json',
                                         *O.KERNELS.glob('*/final.xclbin'), *O.KERNELS.glob('*/insts.bin')]),
                  note='engine-internal accounting for target selection; the wall time carries '
                       'the model load and is not a throughput figure')
    summaries, checks, runs = [], [], {}
    for route in a.route:
        r = run(ids, route, layers=a.layers, max_ctx=a.max_ctx, max_tokens=a.max_tokens,
                steps=a.steps, pmode=a.pmode, workdir=out, timeout=a.timeout)
        runs[route] = r
        if len(r['generated_ids']) != a.max_tokens:
            raise ValueError('incomplete generated trajectory')
        s = summarise(r)
        summaries.append(s)
        checks.extend(check_consistency(s))
        report(s, check_consistency(s))
    result['runs'] = {k: {kk: vv for kk, vv in v.items() if kk not in ('layers',)} for k, v in runs.items()}
    result['summary'] = summaries
    result['checks'] = checks
    result['greedy_agreement'] = all(r['generated_ids'] == next(iter(runs.values()))['generated_ids']
                                   for r in runs.values())
    ok = (all(c['ok'] for c in checks) and all(s.get('layers') for s in summaries)
          and result['greedy_agreement'])
    result['verdict'] = 'PASS' if ok else 'FAIL'
    result['conditions_end'] = conditions()
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    write_csv(out / 'layers.csv', summaries)
    print(f"\n{result['verdict']} -> {out / 'result.json'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
