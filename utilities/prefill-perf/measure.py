"""Measure the UNINSTRUMENTED production prefill path; never model acceptance.

Phase 21 asks for a wall-time budget that explains >=90% of prefill, with the
host stages separated rather than hidden inside one number. The engine already
prints that split (src/open_qwen36/cli.cpp `stages()`); this tool runs the plain
CLI, parses it, and keeps the raw logs so a separate audit can recompute every
statistic instead of trusting the summary.

Nothing here builds kernels or changes arithmetic: a capture-only CLI adds
readback and is not a benchmark, so the caller must pass the plain build.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import statistics
import subprocess
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'Models/qwen38-27b/.resident-validation.lock'
SCHEDULES = ('sequential', 'hybrid', 'block_major')
# The flags each schedule adds, from utilities/prefill-combined-cli/run.py: a
# sequential prefill is one step() per token, hybrid is the whole prompt through
# each layer (block at a time inside it), block_major forces the old
# block-at-a-time loop. --prefill-mode lean/fast is left at the default: this
# measures the production default, not a tuning knob.
EXTRA = {'sequential': [], 'hybrid': ['--gemm-block'],
         'block_major': ['--gemm-block', '--block-major']}

RESIDENT = re.compile(r'^resident after ([\d.]+) s$', re.M)
PREFILL = re.compile(r'^prefill (\d+) tokens: ([\d.]+) ms \(([\d.]+) ms/token\)$', re.M)
# "  gemm-prompt [0,256) t_real=256: 30095.8 ms (GEMM 1.0, host 29999.0, per-token 0.0, lm_head 0.0)"
STAGE = re.compile(r'^\s+(?P<what>[\w-]+) \[(?P<n0>\d+),(?P<n1>\d+)\) t_real=(?P<t_real>\d+): '
                   r'(?P<total>[\d.]+) ms \(GEMM (?P<part0>[\d.]+), host (?P<part1>[\d.]+), '
                   r'per-token (?P<route>[\d.]+), lm_head (?P<lmhead>[\d.]+)\)$', re.M)
# "      mid 1.0 (dn conv 0.5, dn rule 0.5, attn 0.0), prenorm 0.0, gemm tile 0.0, gemm tr 0.0, ..."
FINE = re.compile(r'^\s+mid (?P<mid>[\d.]+) \(dn conv (?P<dn_conv>[\d.]+), dn rule (?P<dn_rule>[\d.]+), '
                  r'attn (?P<attn>[\d.]+)\), prenorm (?P<prenorm>[\d.]+), gemm tile (?P<gemm_tile>[\d.]+),'
                  r' gemm tr (?P<gemm_tr>[\d.]+), tail (?P<tail>[\d.]+), shared (?P<shared>[\d.]+),'
                  r' state (?P<state>[\d.]+), sync (?P<sync>[\d.]+), setup (?P<setup>[\d.]+)'
                  r' \| moe prep (?P<moe_prep>[\d.]+), patch (?P<moe_patch>[\d.]+), run (?P<moe_run>[\d.]+),'
                  r' read (?P<moe_read>[\d.]+) \| qkv scatter (?P<qkv_scatter>[\d.]+) \| residue (?P<residue>[-\d.]+)$',
                  re.M)
# "      lx                  512 calls   41000.0 ms total   80.078 mean   60.100 min"
DISPATCH = re.compile(r'^\s+(?P<name>\S+) +(?P<calls>\d+) calls +(?P<ms>[\d.]+) ms total '
                      r'+(?P<mean>[\d.]+) mean +(?P<min>[\d.]+) min$', re.M)
STAGE_FULL = ('total', 'part0', 'part1', 'route', 'lmhead')
FINE_FULL = ('mid', 'dn_conv', 'dn_rule', 'attn', 'prenorm', 'gemm_tile', 'gemm_tr', 'tail', 'shared',
             'state', 'sync', 'setup', 'moe_prep', 'moe_patch', 'moe_run', 'moe_read', 'qkv_scatter',
             'residue')
COVERAGE_MIN = 0.90


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith('OFLM_OPEN_')}
    env.update(OMP_NUM_THREADS='2', OMP_WAIT_POLICY='PASSIVE', OFLM_OPEN_ATTN_BLOCK='0',
               OFLM_OPEN_LAYER_MAJOR='1', OFLM_OPEN_STAGE_MAJOR='0')
    return env


@contextmanager
def device_lock():
    """Same lock the resident validators take: never overlap another NPU job."""
    if not LOCK.is_file():
        raise RuntimeError('missing ' + str(LOCK))
    with LOCK.open('a+b') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another NPU job holds the resident-validation lock') from None
        yield


def require_device():
    text = subprocess.run(['xrt-smi', 'examine', '-r', 'platform'], capture_output=True, text=True)
    if text.returncode != 0 or 'NPU' not in text.stdout:
        raise RuntimeError('no idle NPU visible to xrt-smi')


def memory_preflight(gib=64):
    info = Path('/proc/meminfo').read_text()
    available = int(re.search(r'^MemAvailable:\s+(\d+) kB', info, re.M).group(1)) * 1024
    if available < gib * 2**30:
        raise RuntimeError(f'prefill measurement wants {gib} GiB available; have {available/2**30:.1f} GiB')
    return available


def parse(text):
    """Parse one run's stderr into the stage split; None if the run did not complete.

    The sequential route loops one `step()` per token and prints no per-stage
    line -- only the gemm-block routes do. A run without stage lines is still a
    valid TTFT measurement, but it has no budget, so `stages_available` is False
    and `coverage` is None. Reporting a coverage there would invent an
    attribution the engine never made.
    """
    prefill, resident = PREFILL.search(text), RESIDENT.search(text)
    stages = list(STAGE.finditer(text))
    if not prefill:
        return None
    fine = list(FINE.finditer(text))
    # Several stage lines appear when the prompt is split into blocks; the budget
    # must cover all of them, so sum rather than take the last.
    total = {k: sum(float(m.group(k)) for m in stages) for k in STAGE_FULL}
    detail = {k: sum(float(m.group(k)) for m in fine) for k in FINE_FULL} if fine else {}
    dispatched = {}
    for m in DISPATCH.finditer(text):
        row = dispatched.setdefault(m.group('name'),
                                    dict(calls=0, total_ms=0.0, min_ms=float('inf')))
        row['calls'] += int(m.group('calls'))
        row['total_ms'] += float(m.group('ms'))
        row['min_ms'] = min(row['min_ms'], float(m.group('min')))
    for row in dispatched.values():
        if row['min_ms'] == float('inf'):
            row['min_ms'] = 0.0
    tokens = int(prefill.group(1))
    wall = float(prefill.group(2))
    # The engine's own identity is part0 + part1 + route + lmhead + residue =
    # total (cli.cpp prints exactly that residue). So "total == wall - preproc"
    # would be a tautology; what is genuinely UNEXPLAINED is the residue inside
    # the route plus the prologue outside it.
    available = bool(stages)
    if available:
        residue = total['total'] - total['part0'] - total['part1'] - total['route'] - total['lmhead']
        preproc = wall - total['total']
        unexplained = residue + preproc
        coverage = (wall - unexplained) / wall if wall else None
    else:
        residue = preproc = unexplained = None
        coverage = None
    return dict(tokens=tokens, prefill_ms=wall, resident_s=float(resident.group(1)) if resident else None,
                stages=total, fine=detail, dispatch=dispatched,
                stages_available=available,
                route_ms_total=total['total'],
                # prompt build, embedding scatter, the host loop around the dispatches
                preproc_ms=preproc,
                residue_ms=residue,
                # the part of wall time no stage accounts for
                unexplained_ms=unexplained,
                coverage=coverage,
                tok_per_s=tokens / (wall / 1000.0) if wall else None)


def median_row(rows):
    """Median of each numeric field, with the spread kept alongside it.

    Booleans and None must survive the reduction: `stages_available` decides
    whether a run HAS a budget at all, and dropping it would turn every real row
    into "no attribution" (or worse, let a caller treat a missing flag as True).
    """
    def flat(row, prefix=''):
        out = {}
        for k, v in row.items():
            if isinstance(v, dict):
                out.update(flat(v, prefix + k + '.'))
            elif isinstance(v, bool):
                out[prefix + k] = v
            elif isinstance(v, (int, float)):
                out[prefix + k] = float(v)
            elif v is None:
                out[prefix + k] = None
        return out
    flat_rows = [flat(r) for r in rows]
    keys = sorted(set().union(*(r.keys() for r in flat_rows)))
    med, spread = {}, {}
    for k in keys:
        vals = [r[k] for r in flat_rows if k in r]
        if not vals:
            continue
        if all(isinstance(v, bool) for v in vals):
            med[k] = any(vals) and all(vals)
            spread[k] = dict(n=len(vals), true=sum(1 for v in vals if v))
            continue
        if all(v is None for v in vals):
            med[k] = None
            spread[k] = dict(n=len(vals))
            continue
        if any(v is None for v in vals):
            raise ValueError(f'mixed present/absent samples for {k}')
        med[k] = statistics.median(vals)
        spread[k] = dict(min=min(vals), max=max(vals), n=len(vals),
                         rel=(max(vals) - min(vals)) / med[k] if med[k] else None)
    return med, spread


def unflatten(flat):
    out = {}
    for k, v in flat.items():
        target, parts = out, k.split('.')
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = v
    return out


def prompt(length):
    """Synthetic in-vocab ids: prefill cost depends on the token count, not on which."""
    return [1000 + (i * 37) % 20000 for i in range(length)]


def command(cli, model, kernels, schedule, ids, max_ctx, max_tokens=1):
    # No --prefill-logits / --dump-logits: dumping logits at every prompt position
    # is not part of the production route and would inflate the measurement.
    cmd = [str(cli), '--model', str(model), '--kernels', str(kernels),
           '--ids', ','.join(map(str, ids)), '--max-ctx', str(max_ctx),
           '--max-tokens', str(max_tokens), '--decode-route', 'npu', '--pmode', 'none']
    return cmd + EXTRA[schedule]


def run_one(cli, model, kernels, schedule, length, max_ctx, workdir, tag):
    ids = prompt(length)
    log = workdir / f'{tag}.log'
    cmd = command(cli, model, kernels, schedule, ids, max_ctx)
    with log.open('w') as handle:
        done = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=environment(),
                              timeout=7200)
    parsed = parse(log.read_text())
    if done.returncode != 0 or parsed is None:
        raise RuntimeError(f'{tag} did not complete (exit {done.returncode}); see {log}')
    parsed.update(tag=tag, schedule=schedule, length=length, command=cmd,
                  log=str(log), log_sha256=digest(log))
    return parsed


def sources():
    return {str(Path(__file__).resolve()): digest(__file__)}


def run(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, model_acceptance=False, sources=sources(),
                  scope='uninstrumented production prefill; median of repeated runs with spread; '
                        'no numerical or performance-acceptance claim')
    try:
        cli = a.cli.resolve()
        model, kernels = a.model.resolve(), a.kernels.resolve()
        lengths = sorted({int(x) for x in a.lengths.split(',') if x.strip()})
        schedules = [s.strip() for s in a.schedules.split(',') if s.strip()]
        if not lengths or any(not 1 <= n <= 4096 for n in lengths):
            raise ValueError('lengths must be 1..4096')
        if not schedules or any(s not in SCHEDULES for s in schedules):
            raise ValueError(f'schedules must be drawn from {SCHEDULES}')
        if a.reps < 3:
            raise ValueError('at least three repetitions; one run is not a measurement')
        manifest = json.loads((kernels / 'manifest.json').read_text())
        max_ctx = int(manifest['max_ctx_default'])
        if max(lengths) > max_ctx:
            raise ValueError(f'{max(lengths)}-token prompt exceeds the bundle capacity {max_ctx}')
        report.update(cli=str(cli), cli_sha256=digest(cli), model=str(model), kernels=str(kernels),
                      kernels_build_key=manifest['build_key'], max_ctx=max_ctx,
                      toolchain=json.loads((kernels / 'toolchain.json').read_text()).get('mlir_aie'),
                      compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0],
                      environment=environment(), threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                      reps=a.reps, warmup=a.warmup, power_mode='none (left as found; record it yourself)')
        rows = []
        with device_lock():
            require_device()
            memory_preflight(20)
            for schedule in schedules:
                for length in lengths:
                    work = out / f'{schedule}-{length}'
                    work.mkdir(parents=True)
                    for w in range(a.warmup):
                        run_one(cli, model, kernels, schedule, length, max_ctx, work, f'warm{w}')
                    samples = [run_one(cli, model, kernels, schedule, length, max_ctx, work, f'rep{r}')
                               for r in range(a.reps)]
                    med, spread = median_row([{k: v for k, v in s.items()
                                               if k not in ('tag', 'command', 'log', 'log_sha256',
                                                            'schedule', 'length')} for s in samples])
                    row = dict(schedule=schedule, length=length,
                               median=unflatten(med), spread=spread,
                               samples=[s['tag'] for s in samples],
                               logs={s['tag']: s['log_sha256'] for s in samples})
                    rows.append(row)
                    print(schedule, length, f"{med['prefill_ms']:.0f} ms",
                          f"{med['tok_per_s']:.2f} tok/s", flush=True)
        report['rows'] = rows
        report['budget'] = budget(rows)
        report['unattributed'] = [row for row in report['budget']['rows']
                                  if row['coverage'] is None or row['coverage'] < COVERAGE_MIN]
        report.update(complete=True, passed=report['budget']['complete'])
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0 if report['budget']['complete'] else 1


def budget(rows):
    """Explain >=90% of wall time per row, and name the dominant stages."""
    per_row, complete = [], True
    for row in rows:
        med = row['median']
        wall = med.get('prefill_ms') or 0.0
        stages = med.get('stages', {})
        part0 = stages.get('part0', 0.0)
        part1 = stages.get('part1', 0.0)
        route = stages.get('route', 0.0)
        lm = stages.get('lmhead', 0.0)
        preproc = med.get('preproc_ms')
        residue = med.get('residue_ms')
        # The engine's identity is part0+part1+route+lmhead+residue = total, so
        # "preproc + total" would be a tautology. What is genuinely unexplained
        # is the prologue outside the route PLUS the route's own residue. A route
        # that prints no stage line has no budget at all -- say so, do not guess.
        if med.get('stages_available') and preproc is not None and residue is not None:
            unexplained = preproc + residue
            coverage = (wall - unexplained) / wall if wall else None
        else:
            unexplained = coverage = None
        complete &= coverage is not None and coverage >= COVERAGE_MIN
        fine = med.get('fine', {})
        named = sorted(((k, v) for k, v in fine.items() if k != 'residue'), key=lambda kv: -kv[1])[:6]
        per_row.append(dict(schedule=row['schedule'], length=row['length'], prefill_ms=wall,
                            tok_per_s=med.get('tok_per_s'), coverage=coverage,
                            host_blocked_gemm=part0, host_blocked_other=part1,
                            moe_route=route, lm_head=lm, prompt_and_embed=preproc,
                            route_residue=residue, unexplained_ms=unexplained,
                            dominant_fine=[dict(stage=k, ms=v,
                                                share=v / wall if wall else None) for k, v in named]))
    return dict(complete=bool(complete), coverage_min=COVERAGE_MIN, rows=per_row)


def compare(baseline, candidate, labels=('baseline', 'candidate')):
    """Paired A/B of two budgets: same length, one row each, delta attributed.

    Phase 22 asks where the time went when a variant changed, so the saving is
    split into the GEMM-blocked half and the rest rather than reported as one
    number. Rows must pair exactly -- an A/B across different prompt lengths, or
    two rows for one length, is not a comparison and is refused.
    """
    def index(budget_rows):
        seen = {}
        for row in budget_rows['rows']:
            key = (row['schedule'], row['length'])
            if key in seen:
                raise ValueError(f'duplicate measurement for {key}')
            seen[key] = row
        return seen

    a, b = index(baseline), index(candidate)
    if set(a) != set(b):
        raise ValueError(f'unpaired budgets: only in {labels[0]}: {sorted(set(a)-set(b))}; '
                         f'only in {labels[1]}: {sorted(set(b)-set(a))}')
    pairs = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if x['coverage'] is None or y['coverage'] is None:
            raise ValueError(f'{key} has no stage budget to attribute the delta to')
        gemm = x['host_blocked_gemm'] - y['host_blocked_gemm']
        other = x['host_blocked_other'] - y['host_blocked_other']
        wall = x['prefill_ms'] - y['prefill_ms']
        unexplained = wall - gemm - other
        pairs.append(dict(schedule=key[0], length=key[1],
                          baseline_ms=x['prefill_ms'], candidate_ms=y['prefill_ms'],
                          wall_saved_ms=wall,
                          speedup=(x['prefill_ms'] / y['prefill_ms']) if y['prefill_ms'] else None,
                          gemm_saved_ms=gemm, other_saved_ms=other,
                          unattributed_saved_ms=unexplained,
                          accounted_fraction=(gemm + other) / wall if wall else None))
    return dict(labels=list(labels), pairs=pairs)


def audit(a):
    out = a.out.resolve()
    proof = dict(complete=False, model_acceptance=False)
    try:
        report = json.loads((out / 'report.json').read_text())
        if report.get('complete') is not True or report.get('model_acceptance') is not False:
            raise ValueError('incomplete measurement report')
        if report['sources'] != sources():
            raise ValueError('measurement tool changed')
        if digest(Path(report['cli'])) != report['cli_sha256']:
            raise ValueError('measured CLI changed')
        rebuilt = []
        for row in report['rows']:
            samples = []
            for tag, sha in row['logs'].items():
                log = out / f"{row['schedule']}-{row['length']}" / f'{tag}.log'
                if digest(log) != sha:
                    raise ValueError(f'raw log changed: {log}')
                parsed = parse(log.read_text())
                if parsed is None:
                    raise ValueError(f'raw log no longer parses: {log}')
                samples.append({k: v for k, v in parsed.items()})
            med, spread = median_row([{k: v for k, v in s.items()
                                       if k not in ('tag', 'command', 'log', 'log_sha256',
                                                    'schedule', 'length')} for s in samples])
            if unflatten(med) != row['median']:
                raise ValueError(f"median changed for {row['schedule']}/{row['length']}")
            rebuilt.append(dict(schedule=row['schedule'], length=row['length'],
                                median=unflatten(med), spread=spread, samples=row['samples'],
                                logs=row['logs']))
        if budget(rebuilt) != report['budget']:
            raise ValueError('time budget changed')
        proof.update(complete=True, report_sha256=digest(out / 'report.json'),
                     rows=len(rebuilt), budget_complete=report['budget']['complete'])
    except Exception as error:
        proof['error'] = str(error)
        raise
    finally:
        (out / 'audit.json').write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    return 0 if proof.get('budget_complete') else 1


def ab(a):
    """Write the paired report from two stored measurement reports."""
    baseline = json.loads((a.baseline / 'report.json').read_text())
    candidate = json.loads((a.candidate / 'report.json').read_text())
    for name, report in (('baseline', baseline), ('candidate', candidate)):
        if report.get('complete') is not True:
            raise ValueError(f'{name} measurement is incomplete')
    result = compare(baseline['budget'], candidate['budget'],
                     labels=(a.baseline_label, a.candidate_label))
    result.update(complete=True, model_acceptance=False,
                  baseline=str(a.baseline.resolve()), candidate=str(a.candidate.resolve()),
                  baseline_sha256=digest(a.baseline / 'report.json'),
                  candidate_sha256=digest(a.candidate / 'report.json'))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for pair in result['pairs']:
        print(f"{pair['schedule']} {pair['length']}: {pair['baseline_ms']:.0f} -> "
              f"{pair['candidate_ms']:.0f} ms ({pair['speedup']:.2f}x), "
              f"GEMM half {pair['gemm_saved_ms']:.0f} ms of {pair['wall_saved_ms']:.0f} ms saved")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    q = sub.add_parser('run')
    for key in ('cli', 'model', 'kernels', 'out'):
        q.add_argument('--' + key, type=Path, required=True)
    q.add_argument('--lengths', default='1,32,128,256,512,2048')
    q.add_argument('--schedules', default=','.join(SCHEDULES))
    q.add_argument('--reps', type=int, default=5)
    q.add_argument('--warmup', type=int, default=1)
    sub.add_parser('audit').add_argument('--out', type=Path, required=True)
    c = sub.add_parser('ab', help='paired A/B of two stored reports')
    c.add_argument('--baseline', type=Path, required=True)
    c.add_argument('--candidate', type=Path, required=True)
    c.add_argument('--baseline-label', default='baseline')
    c.add_argument('--candidate-label', default='candidate')
    c.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit({'run': run, 'audit': audit, 'ab': ab}[args.action](args))
