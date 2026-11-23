"""Capture PR4A raw evidence with an exclusive lock and build-time attestation.

--build-identity is mandatory; see pipeline_evidence's module documentation for
its schema and trust boundary. No current git revision is inferred as the build.
CPU samples are whole-process observations only: phase CPU/wait is NOT MEASURED.
No phase CPU accounting, arithmetic changes, or performance report generation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import time

import pipeline_evidence as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT / 'Models/qwen38-27b/Qwen3.8-27B-NPU2'
KERNELS = ROOT / 'Models/qwen38-27b/production-v2/kernels-model'
LOCK = ROOT / 'Models/qwen38-27b/.npu.lock'
REAL_VOCAB = 248077
POOL = 8192
sha256 = E.sha256


def fixture(n: int, seed: int = 26026) -> list[int]:
    E.require(type(n) is int and 0 < n <= POOL, f'fixture tokens must be in 1..{POOL}')
    import numpy as np
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.choice(np.arange(2, REAL_VOCAB), size=POOL, replace=False)[:n]]


@contextmanager
def lock_exclusive(path):
    """Fail explicitly when advisory locking is unavailable; never run unlocked."""
    try:
        import fcntl
    except ImportError as exc:
        raise E.EvidenceError('exclusive advisory lock unavailable: fcntl is required') from exc
    with Path(path).open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, AttributeError) as exc:
            raise E.EvidenceError(f'exclusive advisory lock unavailable or busy: {path}') from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


RUNTIME = re.compile(r'libxrt_[A-Za-z0-9_.+-]*\.so(?:\.[A-Za-z0-9_.+-]+)?')


def libs_from_maps(pid):
    text = Path(f'/proc/{pid}/maps').read_text()
    paths = set()
    for line in text.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6:
            path = fields[5]
            E.require(not ('libxrt_' in path and path.endswith(' (deleted)')),
                      'live runtime mapping was deleted')
            if RUNTIME.fullmatch(Path(path).name):
                stat = Path(path).stat()
                major, minor = (int(part, 16) for part in fields[3].split(':'))
                E.require(stat.st_ino == int(fields[4]) and
                          (os.major(stat.st_dev), os.minor(stat.st_dev)) == (major, minor),
                          'live runtime mapping path was replaced')
                paths.add(path)
    return sorted(paths)


def process_cpu(pid):
    try:
        # comm can contain spaces and parentheses; fields after the final ')' are fixed.
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        ticks = os.sysconf('SC_CLK_TCK')
        return dict(utime_s=int(stat[11]) / ticks, stime_s=int(stat[12]) / ticks,
                    threads=int(stat[17]))
    except (OSError, ValueError, IndexError, AttributeError):
        return {}


def capture_process(cmd, env, log, *, timeout, executable, interval=.02):
    """Stream directly to disk (no bounded pipe), kill the group and reap on errors."""
    E.require(os.name == 'posix' and Path('/proc/self/maps').exists(),
              'live runtime provenance requires POSIX /proc; NOT MEASURED here')
    E.require(timeout > 0, 'timeout must be positive')
    started = time.monotonic()
    runtime, samples, mappings = {}, [], []
    executable_seen = False
    with Path(log).open('wb') as output:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try:
                    mapped = libs_from_maps(proc.pid)
                    live_exe = Path(f'/proc/{proc.pid}/exe')
                    if live_exe.exists():
                        stat = live_exe.stat()
                        E.require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) ==
                                  (executable['device'], executable['inode'], executable['bytes'],
                                   executable['mtime_ns']), 'live executable identity changed')
                        if not executable_seen:
                            E.require(E.sha256(live_exe) == executable['sha256'],
                                      'live executable differs from attested executable')
                            executable_seen = True
                    for path in mapped:
                        if path not in runtime:
                            runtime[path] = E.fingerprint(path)
                        else:
                            stat = Path(path).stat()
                            old = runtime[path]
                            E.require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
                                       str(Path(path).resolve())) ==
                                      (old['device'], old['inode'], old['bytes'], old['mtime_ns'],
                                       old['resolved']), f'live runtime changed: {path}')
                    if mapped:
                        mappings.append(dict(at_s=time.monotonic() - started, paths=mapped))
                    samples.append(dict(at_s=time.monotonic() - started, **process_cpu(proc.pid)))
                except (FileNotFoundError, ProcessLookupError):
                    # Exit between poll and /proc read is expected; missing evidence still fails below.
                    pass
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                try:
                    rc = proc.wait(timeout=min(interval, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Also stop descendants after an early successful parent exit: they must
            # not keep writing evidence after the capture has been hashed.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    E.require(rc == 0, f'capture failed rc={rc}; inspect retained {log}')
    E.require(executable_seen, 'live executable identity was not observed')
    live = dict(observed_pid=proc.pid, mapping_samples=mappings, before=runtime,
                after=E.snapshot(runtime))
    E.validate_runtime(live)
    return dict(wall_s=time.monotonic() - started, runtime=live, cpu_observations=samples)


def environment(profile, fold, phase, steps, route, extra_env):
    # All inherited OFLM overrides are discarded, including unknown future schedule knobs.
    inherited = {k: v for k, v in os.environ.items() if k.startswith('OFLM_')}
    env = {k: v for k, v in os.environ.items() if not k.startswith('OFLM_')}
    extra_env = extra_env or {}
    allowed = {'OMP_NUM_THREADS', 'OMP_WAIT_POLICY', 'OPENBLAS_NUM_THREADS'}
    E.require(set(extra_env) <= allowed and all(isinstance(v, str) for v in extra_env.values()),
              'unapproved profiling/scheduling environment override')
    env.update(extra_env)
    env.update(OFLM_OPEN_PREFILL_PRECISION='paired', OFLM_OPEN_PROFILE=str(profile),
               OFLM_OPEN_PROFILE_STEPS=steps,
               OFLM_OPEN_LAYER_MAJOR='1' if route == 'layer-major' else '0')
    if fold is not None:
        env['OFLM_OPEN_FOLD_PROFILE'] = str(fold)
    return env, inherited


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(cli: Path, phase: str, tokens: int, max_tokens: int, out: Path, ctx: int,
        pmode: str, profile_steps: str, extra_env: dict | None = None,
        fold_instrumentation: bool = False, *, build_identity: Path | None = None,
        model: Path = MODEL, kernels: Path = KERNELS, lock: Path = LOCK,
        layers: int = 64, route: str = 'block', timeout: float = 5400) -> dict:
    E.require(phase in ('prefill', 'decode'), 'explicit prefill/decode phase required')
    E.require(type(tokens) is int and 0 < tokens <= POOL, 'invalid prompt token count')
    E.require(type(max_tokens) is int and max_tokens >= (1 if phase == 'prefill' else 6),
              'prefill needs max-tokens>=1; decode needs at least 6 generated tokens')
    E.require(type(ctx) is int and tokens + max_tokens - 1 <= ctx, 'invalid/insufficient context')
    E.require(route in ('block', 'layer-major') and (phase == 'prefill' or route == 'block'),
              'invalid route for phase')
    E.require(not fold_instrumentation or phase == 'prefill', 'decode has no block fold schedule')
    expected_steps = '0' if phase == 'prefill' else f'decode:{max_tokens - 1}'
    E.require(profile_steps == expected_steps, 'profile budget must cover exactly the requested phase')
    E.require(pmode in ('none', 'default', 'performance', 'powersaver', 'turbo'), 'invalid power mode')
    E.require(type(timeout) in (int, float) and 0 < timeout < float('inf'), 'invalid timeout')
    cli, out, model, kernels = (Path(p).absolute() for p in (cli, out, model, kernels))
    E.require(cli.is_file() and os.access(cli, os.X_OK), 'missing required executable')
    identity = E.build_identity(build_identity, cli)
    identity_path = Path(build_identity).absolute()
    manifest_path = kernels / 'manifest.json'
    manifest = E.read_json(manifest_path)
    info = E.manifest_info(manifest, layers)
    info['real_vocab'] = manifest['spec'].get('real_vocab')
    E.require(E.positive_int(info['real_vocab']) and info['real_vocab'] == REAL_VOCAB,
              'fixture vocabulary differs from model manifest')
    def inputs():
        return E.provenance_inputs(cli, identity_path, identity, model, kernels)

    with lock_exclusive(lock):
        before = E.snapshot(inputs())
        E.validate_attested_snapshot(before, cli, identity_path, identity)
        E.require(E.read_json(manifest_path) == manifest, 'kernel manifest changed before launch')
        out.mkdir(parents=True, exist_ok=False)
        ids = out / 'ids.txt'
        ids.write_text(','.join(map(str, fixture(tokens))) + '\n')
        profile, fold = out / 'layers.jsonl', out / 'fold.jsonl'
        env, removed = environment(profile, fold if fold_instrumentation else None,
                                   phase, profile_steps, route, extra_env)
        cmd = [str(cli), '--model', str(model), '--kernels', str(kernels), '--ids-file', str(ids),
               '--max-ctx', str(ctx), '--max-tokens', str(max_tokens), '--layers', str(layers),
               '--pmode', pmode, '--quiet']
        cmd += (['--gemm-block'] + (['--block-major'] if route == 'block' else [])
                if phase == 'prefill' else ['--decode-route', 'npu'])
        invocation = dict(argv=cmd, command=shlex.join(cmd), cwd=str(ROOT), environment=env,
                          sanitized_inherited_oflm=removed, build_identity=identity,
                          before=before, phase=phase, tokens=tokens, max_tokens=max_tokens)
        write_json(out / 'invocation.json', invocation)
        input_before = E.snapshot([ids, out / 'invocation.json'])
        result = capture_process(cmd, env, out / 'run.log', timeout=timeout,
                                 executable=before[str(cli)])
        input_after = E.snapshot([ids, out / 'invocation.json'])
        E.require(input_before == input_after, 'invocation or token inputs changed during capture')
        after = E.snapshot(inputs())
        E.require(before == after, 'source/build/executable/model/kernel changed during capture')
        E.validate_runtime(result['runtime'])
        _, requests = E.validate_profile(profile, info, phase, tokens, max_tokens, route)
        if fold_instrumentation:
            E.validate_fold(fold, info, tokens, route)
        else:
            E.require(not fold.exists(), 'unexpected inherited fold instrumentation')
        text = (out / 'run.log').read_text()
        if pmode != 'none':
            E.require(f'NPU power mode: {pmode}' in text.splitlines(), 'power mode not confirmed')
        measured = E.validate_log(text, phase, tokens, max_tokens, requests, info, route)
        record = dict(tool='prefill-block-pipeline-capture', version=2, phase=phase,
                      cli=before[str(cli)], model=dict(path=str(model)), kernels=dict(path=str(kernels)),
                      manifest_path=str(manifest_path), build_identity_path=str(identity_path),
                      build_identity=identity, before=before, after=after, argv=cmd,
                      input_before=input_before, input_after=input_after,
                      command=shlex.join(cmd), cwd=str(ROOT), environment=env,
                      sanitized_inherited_oflm=removed, real_prompt_tokens=tokens,
                       manifest_block_tokens=info['block_tokens'],
                       physical_block_tokens=info['block_tokens'] if phase == 'prefill' else None,
                       max_tokens=max_tokens,
                      layers=layers, route=route, fold_instrumentation=fold_instrumentation,
                      cpu_wait='NOT MEASURED', machine=dict(system=platform.platform(),
                      kernel=platform.release(), cpu=platform.processor() or platform.machine()),
                      **result, **measured)
        record['raw_hashes'] = {p.name: E.sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
        write_json(out / 'capture.json', record)
        return record


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cli', type=Path, required=True)
    ap.add_argument('--build-identity', type=Path, required=True)
    ap.add_argument('--phase', choices=('prefill', 'decode'), required=True)
    ap.add_argument('--tokens', type=int, required=True)
    ap.add_argument('--max-tokens', type=int, required=True)
    ap.add_argument('--ctx', type=int, default=4096)
    ap.add_argument('--pmode', default='none')
    ap.add_argument('--profile-steps', required=True)
    ap.add_argument('--route', choices=('block', 'layer-major'), default='block')
    ap.add_argument('--model', type=Path, default=MODEL)
    ap.add_argument('--kernels', type=Path, default=KERNELS)
    ap.add_argument('--lock', type=Path, default=LOCK)
    ap.add_argument('--timeout', type=float, default=5400)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fold-instrumentation', action='store_true')
    a = ap.parse_args()
    rec = run(a.cli, a.phase, a.tokens, a.max_tokens, a.out, a.ctx, a.pmode,
              a.profile_steps, fold_instrumentation=a.fold_instrumentation,
              build_identity=a.build_identity, model=a.model, kernels=a.kernels,
              lock=a.lock, route=a.route, timeout=a.timeout)
    print(json.dumps({k: rec[k] for k in ('phase', 'real_prompt_tokens', 'model_load_s',
                                         'prompt_wall_ms', 'generated_tokens',
                                         'generated_ms_per_token', 'cpu_wait')}, indent=2))
