"""Performance plan PR #1: the instrumentation must be free when it is off.

`OFLM_OPEN_PROFILE` adds one hook per layer (a null check) and, when it is set, one
steady-clock read per layer boundary. The acceptance criterion is that the DISABLED path
costs nothing measurable, so this script runs the same binary with the variable unset and
with it pointing at a file, alternating, and reports both distributions.

Alternating rather than "all of A, then all of B": this box's NPU clock moves with whatever
else is running (the engine's own notes on OpenMP spinning are the same effect), so a
sequence of A followed by a sequence of B measures the neighbours as much as the change.

What it does NOT do: claim the numbers are the machine's best. They are a paired comparison
of two settings on one binary, which is the only question this file asks.

    python profile_overhead.py --what decode --reps 3
    python profile_overhead.py --what prefill --reps 2
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle as O  # noqa: E402

# "decode" is the fused per-token step (one step() per generated id), "prefill" is one
# 256-token block through the block route -- the two schedules the hooks sit in.
WHAT = {
    'decode': dict(tokens=32, max_tokens=8),
    'prefill': dict(tokens=256, max_tokens=1),
}


def one(ids: list[int], what: str, profiled: bool, workdir: Path, tag: str,
        pmode: str, timeout: float) -> dict:
    import subprocess
    idfile = workdir / f'ids-{what}.txt'
    idfile.write_text(','.join(str(i) for i in ids) + '\n')
    env = dict(__import__('os').environ)
    env['OFLM_OPEN_PREFILL_PRECISION'] = 'paired'
    env.pop('OFLM_OPEN_PREFILL_ROLES', None)
    env.pop('OFLM_OPEN_PREFILL_ROLE_MODE', None)
    if profiled:
        env['OFLM_OPEN_PROFILE'] = str(workdir / f'{what}-{tag}.jsonl')
        env['OFLM_OPEN_PROFILE_STEPS'] = '2'
    else:
        env.pop('OFLM_OPEN_PROFILE', None)
        env.pop('OFLM_OPEN_PROFILE_STEPS', None)
    cfg = WHAT[what]
    cmd = [str(O.CLI), '--model', str(O.MODEL), '--kernels', str(O.KERNELS),
           '--ids-file', str(idfile), '--max-ctx', '4096', '--max-tokens', str(cfg['max_tokens']),
           '--layers', '64', '--pmode', pmode, '--quiet', '--decode-route', 'npu']
    if what == 'prefill':
        env['OFLM_OPEN_LAYER_MAJOR'] = '0'      # the production block loop
        cmd.append('--gemm-block')
        cmd.append('--block-major')
    t0 = time.monotonic()
    with O.LOCK.open('a') as handle:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX)
        proc = subprocess.run(cmd, cwd=str(O.ROOT), env=env, capture_output=True, text=True,
                             timeout=timeout)
    wall = time.monotonic() - t0
    text = proc.stdout + '\n' + proc.stderr
    if proc.returncode != 0:
        raise SystemExit(f'{what}/{tag}: rc {proc.returncode}\n'
                         + '\n'.join(text.strip().splitlines()[-12:]))
    # The engine's own per-request figure: the prefill line or the decode loop's summary.
    engine_ms = None
    for line in text.splitlines():
        if what == 'prefill' and line.startswith('prefill '):
            engine_ms = float(line.split(':')[1].split(' ms')[0])
        elif what == 'decode' and line.startswith('decode '):
            engine_ms = float(line.split(':')[1].split(' ms/token')[0])
    if engine_ms is None:
        raise SystemExit(f'{what}/{tag}: no engine timing line\n' + text[-800:])
    return dict(tag=tag, profiled=profiled, wall_s=round(wall, 2), engine_ms=round(engine_ms, 1),
                command=' '.join(cmd))


def verdict(off: list[float], on: list[float]) -> dict:
    """Paired comparison. The gate: the profiled run must not be slower by more than the
    spread the unprofiled runs themselves show -- otherwise the difference is not the hook."""
    m_off, m_on = statistics.median(off), statistics.median(on)
    spread = (max(off) - min(off)) if len(off) > 1 else 0.0
    delta = m_on - m_off
    return dict(off_median=m_off, on_median=m_on, off=off, on=on,
                delta_ms=round(delta, 1),
                delta_pct=round(100.0 * delta / m_off, 2) if m_off else None,
                off_spread_ms=round(spread, 1),
                # Allow the hook the size of the run-to-run spread it is being compared against;
                # anything inside that is not attributable to instrumentation.
                ok=delta <= max(spread, 0.01 * m_off))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--what', choices=tuple(WHAT), nargs='+', default=['decode'])
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--pmode', default='performance')
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = a.out or (HERE / 'runs' / f'profile-overhead-{stamp}')
    out.mkdir(parents=True)
    result = dict(tool='prefill-block-profile-overhead', version=1,
                  gate='performance plan PR #1: the profiler must be free when it is off',
                  model_acceptance=False, cli=str(O.CLI), kernels=str(O.KERNELS),
                  started=stamp, reps=a.reps, pmode=a.pmode,
                  method='alternating unset / set, same binary; the engine\'s own timing line')
    rows = []
    for what in a.what:
        ids = O.fixture(WHAT[what]['tokens'], seed=26026)
        off, on, samples = [], [], []
        for rep in range(a.reps):
            r = one(ids, what, False, out, f'off{rep}', a.pmode, a.timeout)
            samples.append(r)
            off.append(r['engine_ms'])
            r = one(ids, what, True, out, f'on{rep}', a.pmode, a.timeout)
            samples.append(r)
            on.append(r['engine_ms'])
            print(f"  {what} rep {rep}: off {off[-1]:.1f} ms, on {on[-1]:.1f} ms", flush=True)
        v = verdict(off, on)
        v['what'] = what
        v['samples'] = samples
        rows.append(v)
        print(f"  {what}: off median {v['off_median']:.1f} ms, on median {v['on_median']:.1f} ms "
              f"({v['delta_ms']:+.1f} ms, {v['delta_pct']:+.2f}%), "
              f"off spread {v['off_spread_ms']:.1f} ms -> {'PASS' if v['ok'] else 'FAIL'}")
    result['rows'] = rows
    result['verdict'] = 'PASS' if all(r['ok'] for r in rows) else 'FAIL'
    (out / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    print(f"\n{result['verdict']} -> {out / 'result.json'}")
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
