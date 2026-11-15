"""Alternate two Q4 prefill builds; require exact output before reporting timings.

This is an isolated submit-to-completion benchmark, not an end-to-end acceptance
gate. Run under utilities/env.sh with no competing NPU workload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REL_FRO_GATE = 5e-3


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def timings(log: str, count: int) -> list[float]:
    rows = re.findall(r'^run k \[3 bufs\] -> state (\d+) \((\S+) ms\)$', log, re.M)
    if len(rows) != count:
        raise ValueError(f'expected {count} completed dispatch records, got {len(rows)}')
    values = []
    for state, elapsed in rows:
        value = float(elapsed)
        if state != '4' or not math.isfinite(value) or value <= 0:
            raise ValueError('failed dispatch or invalid timing')
        values.append(value)
    return values


def output_metrics(got: Path, reference: Path | None, elements: int) -> dict:
    if got.stat().st_size != elements * 4:
        raise ValueError('incomplete output')
    values = np.fromfile(got, '<f4').astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite output')
    result = {'sha256': digest(got), 'elements': elements}
    if reference is not None:
        if reference.stat().st_size != elements * 4:
            raise ValueError('reference shape differs')
        ref = np.fromfile(reference, '<f4').astype(np.float64)
        if not np.isfinite(ref).all():
            raise ValueError('nonfinite reference')
        norm = float(np.linalg.norm(ref))
        error = float(np.linalg.norm(values - ref))
        relative = error / norm if norm else (0.0 if error == 0 else math.inf)
        if not math.isfinite(relative) or relative > REL_FRO_GATE:
            raise ValueError(f'kernel numerical gate failed: rel_fro={relative}')
        result['rel_fro'] = relative
    return result


def summary(pairs: list[dict]) -> dict:
    if not pairs:
        raise ValueError('measured pairs required')
    result = {}
    for arm in ('before', 'after'):
        values = [p[arm]['median_ms'] for p in pairs]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('positive finite measurements required')
        result[arm] = dict(median_ms=statistics.median(values),
                           min_ms=min(values), max_ms=max(values))
    result['change_percent'] = 100 * (result['after']['median_ms'] /
                                    result['before']['median_ms'] - 1)
    result['paired_wins'] = sum(p['after']['median_ms'] < p['before']['median_ms'] for p in pairs)
    return result


def config(kernel: Path, weights: Path, activations: Path, output: Path,
           n: int, k: int, tokens: int, runs: int) -> str:
    paths = (kernel, weights, activations, output)
    if any(any(c.isspace() for c in str(p)) for p in paths):
        raise ValueError('harness config paths must not contain whitespace')
    lines = ['device', f'xclbin G {kernel}/final.xclbin',
             f'kernelx k G {kernel}/insts.bin', f'buf w {n*k*5//8} {weights}',
             f'buf x {k*tokens*2} {activations}', f'buf y {n*tokens*4}']
    return '\n'.join(lines + ['run k w x y'] * runs +
                     [f'dump y {output} {n*tokens*4}', ''])


def conditions() -> dict:
    result = dict(platform=platform.platform(), processor=platform.processor(),
                  python=sys.version, load=os.getloadavg() if hasattr(os, 'getloadavg') else None)
    for name, path in {
        'cpu': Path('/proc/cpuinfo'),
        'governor': Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
        'memory': Path('/proc/meminfo'),
    }.items():
        if path.is_file():
            text = path.read_text()
            result[name] = (next((line for line in text.splitlines() if line.startswith('model name')), '')
                            if name == 'cpu' else text)
    result['temperatures'] = {str(p): p.read_text().strip()
                              for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')
                              if os.access(p, os.R_OK)}
    return result


def run(a: argparse.Namespace) -> None:
    # Imported only for actual hardware execution; pure tests also run on Windows.
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError('NPU benchmark requires POSIX file locking') from exc
    if a.n <= 0 or a.n % 256 or a.k <= 0 or a.k % 256 or a.tokens <= 0 or a.tokens % 256:
        raise ValueError('positive 256-aligned N, K and tokens required')
    if a.reps < 5 or a.warmup < 1 or a.runs < 1:
        raise ValueError('at least five pairs, one warmup dispatch and one measured dispatch required')
    if a.weights.stat().st_size != a.n * a.k * 5 // 8:
        raise ValueError('weight pool shape differs')
    if a.activations.stat().st_size != a.k * a.tokens * 2:
        raise ValueError('activation shape differs')
    a.out.mkdir(parents=True, exist_ok=False)
    inputs = [a.weights, a.activations, a.harness, Path(__file__).resolve()]
    if a.reference is not None:
        inputs.append(a.reference)
    for kernel in (a.before, a.after):
        inputs.extend(kernel / name for name in ('final.xclbin', 'insts.bin'))
    record = dict(complete=False, exact_outputs=False, end_to_end_acceptance=False,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  inputs={str(p): digest(p) for p in inputs}, pairs=[],
                  scope='isolated host submit-to-completion; not device-only execution',
                  conditions=conditions())
    try:
        lock_path = ROOT / 'Models/qwen38-27b/.npu.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            expected = None
            env = dict(os.environ, HARNESS_TIMEOUT_MS='30000')
            env.pop('HARNESS_KEEP_GOING', None)
            for index in range(a.reps):
                pair = {'index': index, 'conditions': conditions()}
                order = ('before', 'after') if index % 2 == 0 else ('after', 'before')
                pair['order'] = order
                for arm in order:
                    tag = f'{index:02d}-{arm}'
                    output, cfg = a.out / f'{tag}.bin', a.out / f'{tag}.cfg'
                    cfg.write_text(config(getattr(a, arm), a.weights, a.activations,
                                          output, a.n, a.k, a.tokens, a.warmup + a.runs))
                    command = [str(a.harness), str(cfg)]
                    with (a.out / f'{tag}.log').open('w') as stream:
                        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       check=True, timeout=180)
                    samples = timings((a.out / f'{tag}.log').read_text(), a.warmup + a.runs)
                    metrics = output_metrics(output, a.reference, a.n * a.tokens)
                    if expected is None:
                        expected = metrics['sha256']
                    if metrics['sha256'] != expected:
                        raise ValueError('output differs bytewise across arms or repetitions')
                    pair[arm] = dict(command=command, warmup_ms=samples[:a.warmup],
                                     measured_ms=samples[a.warmup:], output=metrics,
                                     median_ms=statistics.median(samples[a.warmup:]))
                record['pairs'].append(pair)
                print(json.dumps({k: v for k, v in pair.items() if k != 'conditions'}), flush=True)
        for path, identity in record['inputs'].items():
            if digest(Path(path)) != identity:
                raise ValueError(f'input changed during benchmark: {path}')
        record.update(complete=True, exact_outputs=True, summary=summary(record['pairs']))
    except Exception as exc:
        record['error'] = str(exc)
        raise
    finally:
        (a.out / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'weights', 'activations', 'harness', 'out'):
        parser.add_argument('--' + name, type=lambda v: Path(v).resolve(), required=True)
    parser.add_argument('--reference', type=lambda v: Path(v).resolve())
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--reps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
