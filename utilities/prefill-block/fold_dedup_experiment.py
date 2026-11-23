"""Fixed PR4B hardware correctness gate, followed only explicitly by timing.

No production code changes. A failed gate aborts; captures are never overwritten.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess

import pipeline_capture as C
import pipeline_evidence as E
import pipeline_campaign as P

BASE = C.ROOT / 'Models/qwen38-27b/pr4b'
KERNELS = C.ROOT / 'Models/qwen38-27b/pr4a-hardware/kernels-model'
THREADS = dict(OMP_NUM_THREADS='8', OMP_WAIT_POLICY='PASSIVE', OPENBLAS_NUM_THREADS='1')


def preflight():
    mem = Path('/proc/meminfo').read_text()
    E.require(int(re.search(r'MemAvailable:\s+(\d+)', mem)[1]) * 1024 >= 35 * 1024**3,
              'less than 35 GiB available')
    E.require(shutil.which('fuser'), 'fuser required for device exclusivity')
    devices = list(Path('/dev/accel').glob('accel*'))
    E.require(devices, 'no accelerator device')
    result = subprocess.run(['fuser', *map(str, devices)], capture_output=True, text=True)
    E.require(result.returncode == 1 and not result.stdout.strip(),
              'another NPU client or failed device check: ' + result.stdout + result.stderr)


def correctness_capture(n, arm, repeat=1):
    build = BASE / ('control' if arm == 'original' else 'candidate')
    cli = build / 'build/open_qwen36_cli'
    identity_path = build / 'identity.json'
    identity = E.build_identity(identity_path, cli)
    out = BASE / 'correctness' / f'{n}-{arm}-repeat{repeat}'
    manifest = E.read_json(KERNELS / 'manifest.json')
    def inputs():
        return E.provenance_inputs(cli, identity_path, identity, C.MODEL, KERNELS)
    with C.lock_exclusive(C.LOCK):
        preflight()
        before = E.snapshot(inputs())
        E.validate_attested_snapshot(before, cli, identity_path, identity)
        out.mkdir(parents=True, exist_ok=False)
        ids = out / 'ids.txt'
        ids.write_text(','.join(map(str, C.fixture(n))) + '\n')
        env, removed = C.environment(out / 'unused', None, 'prefill', '0', 'block', THREADS,
                                     dedup_input_split=arm == 'on')
        env.pop('OFLM_OPEN_PROFILE')
        env.pop('OFLM_OPEN_PROFILE_STEPS')
        cmd = [str(cli), '--model', str(C.MODEL), '--kernels', str(KERNELS),
               '--ids-file', str(ids), '--max-ctx', '4096', '--max-tokens', '17',
               '--layers', '64', '--pmode', 'none', '--quiet', '--gemm-block',
               '--block-major', '--decode-route', 'npu', '--dump-logits', str(out / 'logits'),
               '--repeat', str(repeat)]
        scope = []
        for layer in (0, 3, 62, 63):
            kind = manifest['layers'][layer]
            layout = manifest['layer_types'][kind]['buffers']['state']
            size = layout['bytes'] if layout['kind'] == 'linear' else layout['row'] * (n + 16)
            path = out / f'state-{layer}.bin'
            cmd += ['--dump-state', f'{layer}:0:{size}:{path}']
            scope.append(dict(layer=layer, offset=0, bytes=size, layout=layout))
        invocation = dict(argv=cmd, environment=env, sanitized_inherited_oflm=removed,
                          before=before, build_identity=identity, state_scope=scope,
                          repeat=repeat, real_prompt_tokens=n)
        C.write_json(out / 'invocation.json', invocation)
        input_before = E.snapshot([ids, out / 'invocation.json'])
        conditions = P.conditions()
        result = C.capture_process(cmd, env, out / 'run.log', timeout=5400,
                                   executable=before[str(cli)])
        E.require(input_before == E.snapshot(input_before), 'invocation changed')
        after = E.snapshot(inputs())
        E.require(before == after, 'correctness provenance changed')
        text = (out / 'run.log').read_text()
        E.require(text.splitlines()[-1] == 'DONE' and not re.search(
            r'ERROR|TIMEOUT|NOAVAIL|DIFFERS FROM', text), 'correctness run failed')
        ids_out = [int(x) for x in re.findall(r'^token (\d+)$', text, re.M)]
        E.require(len(ids_out) == 17 * repeat, 'incomplete greedy trajectory')
        if repeat == 2:
            E.require('request 2 REPRODUCED the first' in text and ids_out[:17] == ids_out[17:],
                      'resident trajectory failed')
        binaries = {p.name: dict(bytes=p.stat().st_size, sha256=E.sha256(p))
                    for p in out.glob('*.bin')}
        expected = {f'logits_t{i}.bin' for i in range(17)} | {f'logits_p{n-1}.bin'}
        expected |= {f'state-{s["layer"]}.bin' for s in scope}
        E.require(set(binaries) == expected, 'missing/extra dump buffers')
        vocab = manifest['spec']['vocab']
        for name, entry in binaries.items():
            if name.startswith('logits'):
                E.require(entry['bytes'] == vocab * 4, 'incomplete logits buffer')
        for s in scope:
            E.require(binaries[f'state-{s["layer"]}.bin']['bytes'] == s['bytes'], 'short state')
        record = dict(**invocation, **result, after=after, generated_ids=ids_out,
                      binaries=binaries, conditions_start=conditions, conditions_end=P.conditions())
        record['raw_hashes'] = {p.name: E.sha256(p) for p in out.iterdir() if p.is_file()}
        C.write_json(out / 'capture.json', record)
        print(out.name, 'CAPTURED', flush=True)
        return record


def equal(a, b):
    E.require(a['binaries'] == b['binaries'], 'byte-exact logits/state gate failed')
    E.require(a['generated_ids'][:17] == b['generated_ids'][:17], 'greedy IDs differ')
    E.require(a['runtime']['before'] == b['runtime']['before'], 'mapped XRT differs')


def correctness():
    results = []
    for n in (255, 256, 257):
        off = correctness_capture(n, 'off')
        on = correctness_capture(n, 'on')
        equal(off, on)
        results.append(dict(tokens=n, off_on_exact=True))
        if n == 256:
            equal(off, correctness_capture(n, 'original'))
        for arm, fresh in [('off', off), ('on', on)]:
            equal(fresh, correctness_capture(n, arm, 2))
        C.write_json(BASE / 'correctness-progress.json', results)
    C.write_json(BASE / 'correctness-passed.json', dict(passed=True, cases=results,
        state_repeat_scope='first request only; final repeat logits equal fresh; IDs checked per request'))


def timing():
    E.require(E.read_json(BASE / 'correctness-passed.json')['passed'], 'correctness not passed')
    out = BASE / 'campaign'
    out.mkdir(exist_ok=False)
    C.write_json(out / 'policy.json', dict(power='none unchanged; NPU governor unverifiable',
        threads=THREADS, profiling='layer both arms; fold both prefill arms',
        primary_target_percent=1.0, repetitions=5, excluded_warmup_pairs=1,
        initial_conditions=P.conditions()))
    selection = []
    for phase, n, max_tokens in [('prefill', 256, 1), ('prefill', 512, 1),
                                  ('prefill', 1024, 1), ('decode', 32, 33)]:
        for rep in range(-1, 5):
            pair = {}
            for arm in (['off', 'on'] if rep % 2 == 0 else ['on', 'off']):
                dest = out / f'{phase}-{n}-{rep}-{arm}'
                start = P.conditions()
                rec = C.run(BASE / 'candidate/build/open_qwen36_cli', phase, n, max_tokens,
                    dest, 4096, 'none', '0' if phase == 'prefill' else 'decode:32',
                    extra_env=THREADS, fold_instrumentation=phase == 'prefill',
                    build_identity=BASE / 'candidate/identity.json', kernels=KERNELS,
                    dedup_input_split=arm == 'on', launch_check=preflight)
                audit = E.audit_capture(dest)
                E.require(audit['valid'], str(audit['errors']))
                sidecars = out / 'conditions'
                sidecars.mkdir(exist_ok=True)
                C.write_json(sidecars / f'{dest.name}.json', dict(start=start, end=P.conditions()))
                pair[arm] = rec
                selection.append(dict(phase=phase, tokens=n, rep=rep, warmup=rep < 0,
                    arm=arm, capture=str(dest), sha256=E.sha256(dest / 'capture.json')))
                C.write_json(out / 'selection.json', selection)
                print(dest.name, rec['prompt_wall_ms'], rec['generated_ms_per_token'], flush=True)
            E.require(pair['off']['generated_ids'] == pair['on']['generated_ids'], 'IDs changed')
            E.require(pair['off']['runtime']['before'] == pair['on']['runtime']['before'], 'XRT changed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['correctness', 'timing'])
    a = ap.parse_args()
    os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib'
    os.environ['PATH'] = '/opt/xilinx/xrt/bin:' + os.environ['PATH']
    (correctness if a.stage == 'correctness' else timing)()
