"""Phase-boundary captures with live runtime provenance, for the PR #4 profiling.

One run produces:
  * a profile JSONL with layer rows (OFLM_OPEN_PROFILE),
  * a fold JSONL with exclusive host-fold windows (OFLM_OPEN_FOLD_PROFILE),
  * the CLI's own phase lines (prefill / generated-step), from which wall and
    token counts are read,
  * the loader's "resident after" line, separating model load from inference,
  * /proc CPU and wall samples, and live XRT library mappings sampled from the
    measured process itself (not a separate preflight).

Nothing here changes engine arithmetic; every knob is an opt-in measurement.

    python pipeline_capture.py --cli ... --phase prefill --tokens 256 --out ...
    python pipeline_capture.py --cli ... --phase decode --tokens 32 --max-tokens 8 --out ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT / 'Models/qwen38-27b/Qwen3.8-27B-NPU2'
KERNELS = ROOT / 'Models/qwen38-27b/production-v2/kernels-model'
LOCK = ROOT / 'Models/qwen38-27b/.npu.lock'
REAL_VOCAB = 248077
POOL = 8192


def fixture(n: int, seed: int = 26026) -> list[int]:
    if n > POOL:
        raise SystemExit(f'fixture {n} exceeds pool {POOL}')
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.choice(np.arange(2, REAL_VOCAB), size=POOL, replace=False)[:n]]


RUNTIME = re.compile(r'\S*libxrt_[A-Za-z0-9_.+-]*\.so\S*')


def sha256(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def libs_from_maps(pid: int) -> list[str]:
    try:
        text = Path(f'/proc/{pid}/maps').read_text()
    except OSError:
        return []
    return sorted({line.split(None, 5)[5] for line in text.splitlines()
                   if len(line.split(None, 5)) == 6 and RUNTIME.fullmatch(line.split(None, 5)[5])})


def process_cpu(pid: int) -> dict:
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().split()
        ticks = os.sysconf('SC_CLK_TCK')
        return dict(utime_s=int(stat[13]) / ticks, stime_s=int(stat[14]) / ticks,
                    threads=int(stat[19]))
    except (OSError, ValueError, IndexError):
        return {}


def run(cli: Path, phase: str, tokens: int, max_tokens: int, out: Path, ctx: int,
        pmode: str, profile_steps: str, extra_env: dict | None = None,
        fold_instrumentation: bool = False) -> dict:
    if max_tokens < (1 if phase == 'prefill' else 6):
        raise SystemExit('prefill needs max-tokens>=1; decode needs at least 6 generated tokens')
    out.mkdir(parents=True, exist_ok=False)
    ids = out / 'ids.txt'
    ids.write_text(','.join(str(i) for i in fixture(tokens)) + '\n')
    profile, fold = out / 'layers.jsonl', out / 'fold.jsonl'
    env = dict(os.environ, OFLM_OPEN_PREFILL_PRECISION='paired',
               OFLM_OPEN_PROFILE=str(profile), OFLM_OPEN_PROFILE_STEPS=profile_steps)
    if fold_instrumentation:
        env['OFLM_OPEN_FOLD_PROFILE'] = str(fold)
    for k in ('OFLM_OPEN_PREFILL_ROLES', 'OFLM_OPEN_PREFILL_ROLE_MODE'):
        env.pop(k, None)
    if extra_env:
        env.update(extra_env)
    cmd = [str(cli), '--model', str(MODEL), '--kernels', str(KERNELS),
           '--ids-file', str(ids), '--max-ctx', str(ctx), '--max-tokens', str(max_tokens),
           '--layers', '64', '--pmode', pmode, '--quiet']
    if phase == 'prefill':
        env['OFLM_OPEN_LAYER_MAJOR'] = '0'
        cmd += ['--gemm-block', '--block-major']
    else:
        cmd += ['--decode-route', 'npu']
    t0 = time.monotonic()
    with LOCK.open('a'):
        # The open()+lock is advisory for other drivers; the NPU itself is serial here.
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
        seen, samples, deadline = set(), [], time.monotonic() + 5400
        try:
            while proc.poll() is None:
                seen.update(libs_from_maps(proc.pid))
                samples.append(dict(at_s=round(time.monotonic() - t0, 3), **process_cpu(proc.pid)))
                if time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(cmd, 5400)
                time.sleep(0.25)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        text = proc.communicate()[0]
        rc = proc.returncode
    wall = time.monotonic() - t0
    (out / 'run.log').write_text(text)
    if rc:
        raise SystemExit(f'capture failed rc={rc}\n' + text[-2000:])
    resident = re.search(r'resident after ([\d.]+) s', text)
    prefill = re.search(r'prefill (\d+) tokens: (\d+) ms', text)
    decode = re.search(r'decode (\d+) tokens: (\d+) ms/token', text)
    steps = [dict(position=int(m.group(1)), route=m.group(2),
                  total_ms=float(m.group(3)), part0_ms=float(m.group(4)),
                  part1_ms=float(m.group(5)), lmhead_ms=float(m.group(6)),
                  dispatch_ms=float(m.group(7)))
             for m in re.finditer(r'step @(\d+) \[([a-z]+)\]: ([\d.]+) ms \(part0 ([\d.]+), '
                                  r'route [\d.]+, part1 ([\d.]+), lm_head ([\d.]+);.*?dispatches ([\d.]+)\)',
                                  text)]
    libraries = {p: dict(resolved=str(Path(p).resolve()), sha256=sha256(Path(p).resolve()))
                 for p in sorted(seen)}
    record = dict(
        tool='prefill-block-pipeline-capture', version=1, phase=phase,
        cli=dict(path=str(cli), sha256=sha256(cli)),
        model=dict(path=str(MODEL), q4nx=sha256(MODEL / 'model.q4nx')),
        kernels=dict(path=str(KERNELS), manifest=sha256(KERNELS / 'manifest.json')),
        command=' '.join(cmd), environment={k: env.get(k) for k in (
            'OFLM_OPEN_PROFILE_STEPS', 'OFLM_OPEN_LAYER_MAJOR', 'OFLM_OPEN_PREFILL_PRECISION',
            'OMP_WAIT_POLICY', 'OMP_NUM_THREADS')},
        real_prompt_tokens=tokens, physical_block_tokens=256,
        model_load_s=float(resident.group(1)) if resident else None,
        prompt_wall_ms=float(prefill.group(2)) if prefill else None,
        generated_tokens=int(decode.group(1)) if decode else None,
        generated_ms_per_token=float(decode.group(2)) if decode else None,
        generated_steps=steps, wall_s=round(wall, 3),
        cpu_samples=samples[-1] if samples else {}, cpu_sample_count=len(samples),
        runtime=dict(paths=sorted(seen), libraries=libraries),
        machine=dict(cpu=platform.processor() or platform.machine(), system=platform.platform(),
                     kernel=platform.release(), governor=Path(
                         '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read_text().strip()
                     if Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').exists() else None))
    (out / 'capture.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cli', type=Path, required=True)
    ap.add_argument('--phase', choices=('prefill', 'decode'), required=True)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--max-tokens', type=int, default=1)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--pmode', default='none')
    ap.add_argument('--profile-steps', default=None,
                    help='default: prefill block only (0) or decode:N generated steps')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fold-instrumentation', action='store_true')
    a = ap.parse_args()
    steps = a.profile_steps or ('0' if a.phase == 'prefill' else f'decode:{a.max_tokens - 1}')
    rec = run(a.cli.resolve(), a.phase, a.tokens, a.max_tokens, a.out.resolve(), a.ctx,
              a.pmode, steps, fold_instrumentation=a.fold_instrumentation)
    print(json.dumps({k: rec[k] for k in ('phase', 'real_prompt_tokens', 'model_load_s',
                                          'prompt_wall_ms', 'generated_tokens',
                                          'generated_ms_per_token')}, indent=2))
