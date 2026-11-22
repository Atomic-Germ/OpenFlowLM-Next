"""Validate explicit server kernel arms, A->B->A isolation and live XRT identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oflm_drive as D  # noqa: E402
import oflm_serve_check as S  # noqa: E402
import isolation_aba as I  # noqa: E402
from runtime_provenance import runtime_paths, describe, runtime_identity

SELECTED = re.compile(r'open_qwen36: kernels (\S+) \(([^)]*)\)')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def selected_kernel(log: str) -> dict | None:
    found = SELECTED.findall(log)
    if not found:
        return None
    return dict(path=found[-1][0], how=found[-1][1])


def kernel_artifacts(directory: Path) -> dict:
    """Manifest plus the GEMM binaries the manifest names under this arm."""
    manifest = directory / 'manifest.json'
    result = dict(directory=str(directory), manifest=None, gemm={})
    if not manifest.is_file():
        result['error'] = 'no manifest.json'
        return result
    result['manifest'] = dict(bytes=manifest.stat().st_size, sha256=digest(manifest))
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or not child.name.startswith('gemm_n'):
            continue
        entry = {}
        for name in ('final.xclbin', 'insts.bin', 'insts.elf'):
            path = child / name
            entry[name] = digest(path) if path.is_file() else None
        result['gemm'][child.name] = entry
    return result


def differing_files(before: dict, after: dict) -> dict:
    names = sorted(set(before['gemm']) | set(after['gemm']))
    changed = [n for n in names if before['gemm'].get(n) != after['gemm'].get(n)]
    return dict(gemm_programs=len(names), changed=changed, unchanged=len(names) - len(changed),
                manifest_same=before['manifest'] == after['manifest'])


def matches_evidence(artifacts: dict, report: dict, arm: str) -> bool:
    expected = {case['name']: case['builds'][arm]['binaries'] for case in report['cases']}
    return (bool(expected) and set(artifacts['gemm']) == set(expected) and
            all(artifacts['gemm'][name] == hashes for name, hashes in expected.items()))


def comparison_passes(arms: dict) -> bool:
    before, after = (arms[arm] for arm in ('before', 'after'))
    runs = [arm.get(name, {}) for arm in (before, after) for name in ('main', 'fresh')]
    identities = [runtime_identity(run.get('libraries', {})) for run in runs]
    return (all(arm.get('verdict') == 'PASS' for arm in (before, after)) and
            all(identity is not None and identity == identities[0] for identity in identities) and
            before['artifacts']['manifest'] == after['artifacts']['manifest'] and
            all(before['main'].get(key, {}).get('content') == after['main'].get(key, {}).get('content')
                for key in ('a1', 'b', 'a2')))


def sample_maps(pid: int) -> str:
    try:
        return Path(f'/proc/{pid}/maps').read_text()
    except OSError:
        return ''


def one_server(port: int, kernels: Path, a_text: str, b_text: str, *, max_tokens: int,
               timeout: float, out: Path, tag: str) -> dict:
    """A->B->A plus a fresh-process B, sampling runtime maps while alive."""
    maps: dict[str, str] = {}
    extra = {'OFLM_OPEN_KERNELS_DIR': str(kernels)}
    command = [str(D.OFLM), 'serve', 'qwen3.8:27b', '--ctx-len', '4096', '-p', str(port)]
    stream = (out / f'{tag}-server.log').open('w')
    try:
        proc = subprocess.Popen(command, cwd=D.ROOT, env=D.env(block=True, extra=extra),
                                stdout=stream, stderr=subprocess.STDOUT, text=True)
    except BaseException:
        stream.close()
        raise
    run = dict(started=False, pid=proc.pid)
    try:
        err = S.wait_ready(port, proc, timeout=timeout)
        if err:
            run['reason'] = err
            return run
        maps['ready'] = sample_maps(proc.pid)
        first = S.chat(port, a_text, max_tokens=max_tokens, timeout=timeout)
        maps['after_a'] = sample_maps(proc.pid)
        mid = S.chat(port, b_text, max_tokens=max_tokens, timeout=timeout)
        maps['after_b'] = sample_maps(proc.pid)
        again = S.chat(port, a_text, max_tokens=max_tokens, timeout=timeout)
        maps['after_a2'] = sample_maps(proc.pid)
        run.update(started=True, a1=first, b=mid, a2=again, reset=I.probe_reset(port))
        return run
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        stream.close()
        log = (out / f'{tag}-server.log').read_text()
        sample = '\n'.join(maps.values())
        run.update(command=command, selection=selected_kernel(log),
                   libraries=describe(runtime_paths(sample)),
                   library_samples={k: runtime_paths(v) for k, v in maps.items()})


def run_arm(name: str, kernels: Path, out: Path, port: int, a: int, b: int,
            max_tokens: int, timeout: float) -> dict:
    tok = D.Tokenizer()
    a_text, b_text = tok.exact_text(a), tok.exact_text(b)
    artifacts = kernel_artifacts(kernels)
    folder = out / name
    folder.mkdir()
    (folder / 'artifacts.json').write_text(json.dumps(artifacts, indent=2) + '\n')
    print(f'=== {name}: {kernels} ===', flush=True)
    run = one_server(port, kernels, a_text, b_text, max_tokens=max_tokens,
                     timeout=timeout, out=folder, tag='main')
    rows: list[dict] = []
    fresh_b = None
    fresh: dict = dict(started=False)
    if run.get('started'):
        print('  fresh-process B reference...', flush=True)
        fresh = one_server(port + 1, kernels, b_text, b_text, max_tokens=max_tokens,
                           timeout=timeout, out=folder, tag='fresh')
        fresh_b = fresh.get('b') if fresh.get('started') else None
        fresh_selection = fresh.get('selection')
        rows += [dict(check='fresh_selection_is_arm', ok=bool(fresh_selection) and
                      Path(fresh_selection['path']).resolve() == kernels.resolve(),
                      detail=fresh_selection)]
        a1, mid, a2 = run['a1'], run['b'], run['a2']
        rows += [
            dict(check='selection_logged', ok=run['selection'] is not None,
                 detail=run['selection'],
                 why='the engine reports which kernel set it loaded'),
            dict(check='selection_is_arm', ok=bool(run['selection']) and
                 Path(run['selection']['path']).resolve() == kernels.resolve(),
                 detail=run['selection']),
            dict(check='A_first_answered', ok=bool(a1.get('ok') and a1.get('content')),
                 wall_s=a1.get('wall_s')),
            dict(check='B_answered', ok=bool(mid.get('ok') and mid.get('content')),
                 wall_s=mid.get('wall_s')),
            dict(check='A_second_answered', ok=bool(a2.get('ok') and a2.get('content')),
                 wall_s=a2.get('wall_s')),
            dict(check='A_before_B_equals_A_after_B',
                 ok=bool(a1.get('ok') and a2.get('ok') and a1.get('content') == a2.get('content')),
                 detail=dict(first=a1.get('content'), second=a2.get('content'))),
            dict(check='B_matches_fresh_process_B',
                 ok=bool(fresh_b is not None and mid.get('ok') and fresh_b.get('ok')
                         and mid.get('content') == fresh_b.get('content')),
                 detail=dict(after_A=mid.get('content'),
                             fresh=fresh_b.get('content') if fresh_b else None)),
            dict(check='A_and_B_are_different_prompts', ok=a1.get('content') != mid.get('content')),
            dict(check='runtime_libraries_loaded', ok=runtime_identity(run.get('libraries', {})) is not None,
                 detail=sorted(run.get('libraries') or {})),
            dict(check='fresh_runtime_libraries_loaded', ok=runtime_identity(fresh.get('libraries', {})) is not None,
                 detail=sorted(fresh.get('libraries') or {})),
            dict(check='after_reset', ok=None, verdict='N/A',
                 detail=(run.get('reset') or {}).get('reason')),
        ]
    else:
        rows.append(dict(check='server_starts', ok=False, detail=run.get('reason')))
    result = dict(arm=name, kernel_directory=str(kernels), model=D.MODEL, binary=D.OFLM,
                  binary_sha256=digest(D.OFLM), artifacts=artifacts, main=run,
                  fresh=fresh, fresh_b=fresh_b, rows=rows,
                  graded_fail=[r for r in rows if r['ok'] is False])
    result['verdict'] = 'PASS' if rows and not result['graded_fail'] else 'FAIL'
    (folder / 'result.json').write_text(json.dumps(result, indent=1, default=str) + '\n')
    for row in rows:
        mark = {True: 'PASS', False: 'FAIL', None: 'N/A '}[row['ok']]
        print(f"  {row['check']:34s} {mark}", flush=True)
    print(f"  {result['verdict']}", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--before-kernels', type=Path, required=True)
    ap.add_argument('--after-kernels', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--a', type=int, default=260)
    ap.add_argument('--b', type=int, default=130)
    ap.add_argument('--max-tokens', type=int, default=8)
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--before-port', type=int, default=18141)
    args = ap.parse_args()
    before, after = args.before_kernels.resolve(), args.after_kernels.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    evidence = json.loads((D.ROOT / 'specs/open-engine/reports/qwen38-q4-gemm-dma.json').read_text())
    for arm, directory in (('before', before), ('after', after)):
        if not matches_evidence(kernel_artifacts(directory), evidence, arm):
            ap.error(f'{arm} kernel hashes do not match the recorded evidence')
    record = dict(tool='prefill-block-server-candidate', version=1,
                  gate='candidate kernel arms pass serve A->B->A with recorded runtime maps',
                  model_acceptance=False, a_text_tokens=args.a, b_text_tokens=args.b,
                  max_tokens=args.max_tokens)
    arms = {}
    import fcntl
    D.LOCK.parent.mkdir(parents=True, exist_ok=True)
    with D.LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, (name, kernels) in enumerate((('before', before), ('after', after))):
            arms[name] = run_arm(name, kernels, args.out, args.before_port + index * 2,
                                args.a, args.b, args.max_tokens, args.timeout)
            if not matches_evidence(kernel_artifacts(kernels), evidence, name):
                arms[name]['verdict'] = 'FAIL'
                arms[name]['error'] = 'kernel artifacts changed during validation'
    record['arms'] = arms
    record['arm_separation'] = differing_files(arms['before']['artifacts'],
                                               arms['after']['artifacts'])
    control = arms['before']['main']
    candidate = arms['after']['main']
    identity = runtime_identity(control.get('libraries', {}))
    record['runtime_comparable'] = identity is not None and identity == runtime_identity(candidate.get('libraries', {}))
    record['same_answer_between_arms'] = (control.get('a1', {}).get('content') ==
                                          candidate.get('a1', {}).get('content') and
                                          control.get('b', {}).get('content') ==
                                          candidate.get('b', {}).get('content'))
    record['verdict'] = 'PASS' if comparison_passes(arms) else 'FAIL'
    (args.out / 'result.json').write_text(json.dumps(record, indent=1, default=str) + '\n')
    print(json.dumps(dict(verdict=record['verdict'],
                          arm_separation=record['arm_separation'],
                          runtime_comparable=record['runtime_comparable'],
                          same_answer_between_arms=record['same_answer_between_arms'],
                          libraries=sorted(candidate.get('libraries') or {}))))
    return 0 if record['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
