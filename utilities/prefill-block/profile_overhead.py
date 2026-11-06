"""Measure enabled profiler overhead and optional disabled-path regression.

Warm-up pairs are excluded; at least five measured pairs alternate ON/OFF order.
ON/OFF of the same binary measures enabled overhead only. --baseline-cli adds
parent-OFF versus new-OFF for the separate disabled-path regression check.
Every measured ON decode step must have a complete 64-layer profile, and all
sampled generated trajectories must agree. No device execution-time claim is made.

    python profile_overhead.py --what decode --reps 5 --warmup 1
    python profile_overhead.py --what prefill --baseline-cli /path/to/parent/cli
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle as O  # noqa: E402
import layer_profile as P  # noqa: E402

# "decode" is the fused per-token step (one step() per generated id), "prefill" is one
# 256-token block through the block route -- the two schedules the hooks sit in.
WHAT = {
    'decode': dict(tokens=32, max_tokens=8),
    'prefill': dict(tokens=256, max_tokens=1),
}


def one(ids: list[int], what: str, profiled: bool, workdir: Path, tag: str,
        pmode: str, timeout: float, cli: Path | None = None) -> dict:
    import subprocess
    idfile = workdir / f'ids-{what}.txt'
    idfile.write_text(','.join(str(i) for i in ids) + '\n')
    env = dict(__import__('os').environ)
    env['OFLM_OPEN_PREFILL_PRECISION'] = 'paired'
    env.pop('OFLM_OPEN_PREFILL_ROLES', None)
    env.pop('OFLM_OPEN_PREFILL_ROLE_MODE', None)
    cfg = WHAT[what]
    if profiled:
        env['OFLM_OPEN_PROFILE'] = str(workdir / f'{what}-{tag}.jsonl')
        env['OFLM_OPEN_PROFILE_STEPS'] = str(len(ids) + cfg['max_tokens'] - 1)
    else:
        env.pop('OFLM_OPEN_PROFILE', None)
        env.pop('OFLM_OPEN_PROFILE_STEPS', None)
    cmd = [str(cli or O.CLI), '--model', str(O.MODEL), '--kernels', str(O.KERNELS),
           '--ids-file', str(idfile), '--max-ctx', '4096', '--max-tokens', str(cfg['max_tokens']),
           '--layers', '64', '--pmode', pmode, '--quiet', '--decode-route', 'npu']
    if what == 'prefill':
        env['OFLM_OPEN_LAYER_MAJOR'] = '0'      # the production block loop
        cmd.append('--gemm-block')
        cmd.append('--block-major')
    t0 = time.monotonic()
    with P.lock_exclusive(O.LOCK.open('a')):
        proc = subprocess.run(cmd, cwd=str(O.ROOT), env=env, capture_output=True, text=True,
                             timeout=timeout)
    wall = time.monotonic() - t0
    text = proc.stdout + '\n' + proc.stderr
    (workdir / f'{what}-{tag}.log').write_text(text)
    if proc.returncode != 0:
        raise SystemExit(f'{what}/{tag}: rc {proc.returncode}\n'
                         + '\n'.join(text.strip().splitlines()[-12:]))
    power_mode = P.require_power_mode(text, pmode)
    # The engine's own per-request figure: the prefill line or the decode loop's summary.
    engine_ms = None
    for line in text.splitlines():
        if what == 'prefill' and line.startswith('prefill '):
            engine_ms = float(line.split(':')[1].split(' ms')[0])
        elif what == 'decode' and line.startswith('decode '):
            engine_ms = float(line.split(':')[1].split(' ms/token')[0])
    if engine_ms is None:
        raise SystemExit(f'{what}/{tag}: no engine timing line\n' + text[-800:])
    profiled_decode_steps = 0
    if profiled:
        header, requests, _ = P.read_jsonl(Path(env['OFLM_OPEN_PROFILE']))
        if header['layers'] != 64:
            raise ValueError('overhead requires a full 64-layer profile')
        if what == 'decode':
            P.validate_route('sequential', requests, tokens=len(ids),
                             steps=len(ids) + cfg['max_tokens'] - 1, max_tokens=cfg['max_tokens'])
            decode = requests[len(ids):]
            if len(decode) != cfg['max_tokens'] - 1 or any(r['lmhead_ms'] <= 0 for r in decode):
                raise ValueError('measured decode steps were not profiled')
            profiled_decode_steps = len(decode)
        else:
            P.validate_route('block', requests, tokens=len(ids), steps=0, max_tokens=1)
    ids_out = [int(line.split()[1]) for line in proc.stdout.splitlines() if line.startswith('token ')]
    if len(ids_out) != cfg['max_tokens']:
        raise ValueError('incomplete generated token trajectory')
    return dict(tag=tag, profiled=profiled, generated_ids=ids_out,
                power_mode=power_mode,
                profiled_decode_steps=profiled_decode_steps,
                wall_s=round(wall, 2), engine_ms=round(engine_ms, 1),
                command=' '.join(cmd))


def verdict(off: list[float], on: list[float]) -> dict:
    """Paired comparison. The gate: the profiled run must not be slower by more than the
    spread the unprofiled runs themselves show -- otherwise the difference is not the hook."""
    if not off or len(off) != len(on) or any(not math.isfinite(x) or x <= 0 for x in off + on):
        raise ValueError('paired positive finite timing samples required')
    m_off, m_on = statistics.median(off), statistics.median(on)
    spread = (max(off) - min(off)) if len(off) > 1 else 0.0
    delta = m_on - m_off
    return dict(off_median=m_off, on_median=m_on, off=off, on=on,
                delta_ms=round(delta, 1),
                delta_pct=round(100.0 * delta / m_off, 2) if m_off else None,
                off_spread_ms=round(spread, 1), on_spread_ms=round(max(on) - min(on), 1),
                # Allow the hook the size of the run-to-run spread it is being compared against;
                # anything inside that is not attributable to instrumentation.
                ok=delta <= max(spread, 0.01 * m_off))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--what', choices=tuple(WHAT), nargs='+', default=['decode'])
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--baseline-cli', type=Path)
    ap.add_argument('--cli', type=Path, default=O.CLI)
    ap.add_argument('--model', type=Path, default=O.MODEL)
    ap.add_argument('--kernels', type=Path, default=O.KERNELS)
    ap.add_argument('--pmode', default='performance')
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    if a.reps < 5 or a.warmup < 1:
        ap.error('at least five measured pairs and one warm-up pair required')
    O.CLI, O.MODEL, O.KERNELS = a.cli.resolve(), a.model.resolve(), a.kernels.resolve()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'profile-overhead-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-profile-overhead', version=1,
                  gate='enabled profiling overhead; parent comparison is a separate optional gate',
                  model_acceptance=False, cli=str(O.CLI), kernels=str(O.KERNELS),
                  started=stamp, reps=a.reps, pmode=a.pmode,
                  conditions_start=P.conditions(),
                  method='warm-up excluded; alternating pair order; engine timing includes ON file writes',
                  disabled_path_regression='NOT MEASURED without --baseline-cli',
                  provenance=P.provenance([O.CLI, O.MODEL / 'model.q4nx', O.KERNELS / 'manifest.json',
                                         *O.KERNELS.glob('*/final.xclbin'), *O.KERNELS.glob('*/insts.bin'),
                                         *([a.baseline_cli.resolve()] if a.baseline_cli else [])]))
    rows = []
    for what in a.what:
        ids = O.fixture(WHAT[what]['tokens'], seed=26026)
        off, on, samples, parent = [], [], [], []
        for rep in range(-a.warmup, a.reps):
            order = [('off', False, O.CLI), ('on', True, O.CLI)]
            if a.baseline_cli:
                order.append(('parent', False, a.baseline_cli.resolve()))
            if rep % 2:
                order.reverse()
            pair = {}
            for label, enabled, cli in order:
                pair[label] = one(ids, what, enabled, out, f'{label}{rep}', a.pmode, a.timeout, cli)
                samples.append(pair[label] | {'warmup': rep < 0})
            if any(r['generated_ids'] != pair['off']['generated_ids'] for r in pair.values()):
                raise ValueError('ON/OFF/parent generated trajectories differ')
            if rep < 0:
                continue
            off.append(pair['off']['engine_ms'])
            on.append(pair['on']['engine_ms'])
            if 'parent' in pair:
                parent.append(pair['parent']['engine_ms'])
            print(f"  {what} rep {rep}: off {off[-1]:.1f} ms, on {on[-1]:.1f} ms", flush=True)
        v = verdict(off, on)
        if parent:
            v['disabled_path_vs_parent'] = verdict(parent, off)
        v['what'] = what
        v['samples'] = samples
        rows.append(v)
        print(f"  {what}: off median {v['off_median']:.1f} ms, on median {v['on_median']:.1f} ms "
              f"({v['delta_ms']:+.1f} ms, {v['delta_pct']:+.2f}%), "
              f"off spread {v['off_spread_ms']:.1f} ms -> {'PASS' if v['ok'] else 'FAIL'}")
    if a.baseline_cli:
        result['disabled_path_regression'] = 'PASS' if all(r['disabled_path_vs_parent']['ok'] for r in rows) else 'FAIL'
    result['rows'] = rows
    result['conditions_end'] = P.conditions()
    result['verdict'] = 'PASS' if all(r['ok'] and r.get('disabled_path_vs_parent', {}).get('ok', True) for r in rows) else 'FAIL'
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"\n{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
