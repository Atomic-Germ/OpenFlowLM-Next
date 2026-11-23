"""Matched PR4A profiling campaign; production inference is unchanged.

Enabled samples use pipeline_capture v2. Disabled samples retain a separate
schema: no missing profile rows or labels are synthesized. Stops on any failure.
"""
import argparse
from pathlib import Path
import re
import subprocess

import pipeline_capture as C
import pipeline_evidence as E


def validate_disabled_log(text, phase, tokens, max_tokens, info):
    E.require(phase in ('prefill', 'decode'), 'unknown disabled phase')
    E.require(bool(text.splitlines()) and text.splitlines()[-1] == 'DONE' and not re.search(
        r'ERROR|TIMEOUT|NOAVAIL|DIFFERS FROM', text), 'disabled capture failed')
    observed = []
    for line in text.splitlines():
        match = (E.BLOCK if phase == 'prefill' else E.STEP).fullmatch(line)
        if match is None:
            continue
        v = match.groups()
        if phase == 'prefill':
            row = dict(zip(('total_ms', 'gemm_ms', 'host_ms', 'per_token_ms', 'lmhead_ms'),
                           map(float, v[4:])))
            row.update(what=v[0], tokens=int(v[3]))
        else:
            row = dict(zip(('total_ms', 'gemm_ms', 'per_token_ms', 'host_ms', 'lmhead_ms',
                            'embed_ms', 'patch_ms', 'dispatch_ms'), map(float, v[2:])))
        observed.append(row)
    if phase == 'prefill':
        B = info['block_tokens']
        E.require([r['tokens'] for r in observed] ==
                  [min(B, tokens - start) for start in range(0, tokens, B)],
                  'disabled block coverage mismatch')
    # These are observations from the raw log, not synthesized profiling records.
    return E.validate_log(text, phase, tokens, max_tokens, observed, info, 'block')


def conditions():
    result = {}
    for name, argv in [('npu', ['xrt-smi', 'examine']),
                       ('memory', ['free', '-b']), ('load', ['uptime'])]:
        p = subprocess.run(argv, capture_output=True, text=True, check=False)
        result[name] = dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    result['sysfs'] = {str(p): p.read_text().strip() for pattern in
                       ['/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor',
                        '/sys/class/thermal/thermal_zone*/temp',
                        '/sys/bus/pci/devices/0000:67:00.1/power/runtime_status']
                       for p in Path('/').glob(pattern.lstrip('/')) if p.is_file()}
    return result


def disabled(a, phase, tokens, max_tokens, out):
    cli = a.cli.absolute()
    identity = E.build_identity(a.build_identity, cli)
    def inputs():
        return E.provenance_inputs(cli, a.build_identity.absolute(), identity,
                                   C.MODEL, a.kernels.absolute())
    with C.lock_exclusive(C.LOCK):
        before = E.snapshot(inputs())
        E.validate_attested_snapshot(before, cli, a.build_identity.absolute(), identity)
        out.mkdir()
        ids = out / 'ids.txt'
        ids.write_text(','.join(map(str, C.fixture(tokens))) + '\n')
        env, removed = C.environment(out / 'unused', None, phase, '0', 'block', None)
        for k in ('OFLM_OPEN_PROFILE', 'OFLM_OPEN_PROFILE_STEPS'):
            env.pop(k, None)
        cmd = [str(cli), '--model', str(C.MODEL), '--kernels', str(a.kernels.absolute()),
               '--ids-file', str(ids), '--max-ctx', '4096', '--max-tokens', str(max_tokens),
               '--layers', '64', '--pmode', 'none', '--quiet']
        cmd += ['--gemm-block', '--block-major'] if phase == 'prefill' else ['--decode-route', 'npu']
        invocation = dict(argv=cmd, environment=env, cwd=str(C.ROOT), before=before,
                          build_identity=identity, sanitized_inherited_oflm=removed)
        C.write_json(out / 'invocation.json', invocation)
        input_before = E.snapshot([ids, out / 'invocation.json'])
        start = conditions()
        result = C.capture_process(cmd, env, out / 'run.log', timeout=1800,
                                   executable=before[str(cli)])
        E.require(before == E.snapshot(inputs()), 'disabled capture inputs changed')
        E.require(input_before == E.snapshot(input_before), 'disabled invocation changed')
        text = (out / 'run.log').read_text()
        info = E.manifest_info(E.read_json(a.kernels / 'manifest.json'), 64)
        info['real_vocab'] = C.REAL_VOCAB
        measured = validate_disabled_log(text, phase, tokens, max_tokens, info)
        record = dict(schema='pr4a-disabled-capture-v1', phase=phase,
                      real_prompt_tokens=tokens, **measured,
                      conditions_start=start, conditions_end=conditions(),
                      after=E.snapshot(inputs()), input_before=input_before,
                      **invocation, **result)
        record['raw_hashes'] = {p.name: E.sha256(p) for p in out.iterdir() if p.is_file()}
        C.write_json(out / 'capture.json', record)
        return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cli', type=Path, required=True)
    ap.add_argument('--build-identity', type=Path, required=True)
    ap.add_argument('--kernels', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out = a.out.absolute()
    a.out.mkdir(exist_ok=False)
    samples = []
    for phase, tokens, max_tokens in [('prefill', 256, 1), ('decode', 32, 33)]:
        for rep in range(-1, 5):
            pair = {}
            order = ['off', 'on'] if rep % 2 == 0 else ['on', 'off']
            for arm in order:
                out = a.out / f'{phase}-{rep}-{arm}'
                if arm == 'on':
                    start = conditions()
                    rec = C.run(a.cli, phase, tokens, max_tokens, out, 4096, 'none',
                                '0' if phase == 'prefill' else f'decode:{max_tokens - 1}',
                                fold_instrumentation=phase == 'prefill',
                                build_identity=a.build_identity, kernels=a.kernels)
                    audit = E.audit_capture(out)
                    E.require(audit['valid'], str(audit['errors']))
                    sidecars = a.out / 'conditions'
                    sidecars.mkdir(exist_ok=True)
                    C.write_json(sidecars / f'{out.name}.json', dict(start=start, end=conditions()))
                else:
                    rec = disabled(a, phase, tokens, max_tokens, out)
                pair[arm] = rec
                print(phase, rep, arm, rec['prompt_wall_ms'],
                      rec.get('generated_ms_per_token'), flush=True)
                samples.append(dict(phase=phase, rep=rep, warmup=rep < 0, arm=arm,
                                    capture=str(out), capture_sha256=E.sha256(out / 'capture.json')))
                C.write_json(a.out / 'selection.json', samples)
            E.require(pair['on']['generated_ids'] == pair['off']['generated_ids'],
                      'profiling changed generated trajectory')


if __name__ == '__main__':
    main()
