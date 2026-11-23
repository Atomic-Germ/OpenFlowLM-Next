"""Predeclared PR4B continuation; no replacement, confirmation or candidate edits."""
import argparse
import os
from pathlib import Path
import shlex
import shutil

import pipeline_capture as C
import pipeline_evidence as E
import pipeline_campaign as P
import fold_dedup_experiment as X

BASE = X.BASE
OUT = BASE / 'continuation-v1'


def tools():
    directory = OUT / 'tools'
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('pipeline_capture.py', 'pipeline_evidence.py', 'pipeline_campaign.py',
                 'fold_dedup_experiment.py', 'fold_dedup_continue.py'):
        shutil.copy2(C.HERE / name, directory / name)
    C.write_json(directory / 'inventory.json', E.snapshot(directory.iterdir()))


def list_capture(arm, prompts, name):
    """Real list requests reset in the CLI; preserve all outputs and live mappings."""
    cli = BASE / 'candidate/build/open_qwen36_cli'
    identity_path = BASE / 'candidate/identity.json'
    identity = E.build_identity(identity_path, cli)
    out = OUT / name
    def inputs():
        return E.provenance_inputs(cli, identity_path, identity, C.MODEL, X.KERNELS) + [Path(__file__)]
    with C.lock_exclusive(C.LOCK):
        X.preflight()
        before = E.snapshot(inputs())
        E.validate_attested_snapshot(before, cli, identity_path, identity)
        out.mkdir(parents=True, exist_ok=False)
        outputs = out / 'outputs'
        outputs.mkdir()
        ids = out / 'ids-list.txt'
        ids.write_text(''.join(','.join(map(str, p)) + '\n' for p in prompts))
        env, removed = C.environment(out / 'unused', None, 'prefill', '0', 'block', X.THREADS,
                                     dedup_input_split=arm == 'on')
        for key in ('OFLM_OPEN_PROFILE', 'OFLM_OPEN_PROFILE_STEPS'):
            env.pop(key)
        cmd = [str(cli), '--model', str(C.MODEL), '--kernels', str(X.KERNELS),
               '--ids-list', str(ids), '--list-out', str(outputs), '--max-ctx', '4096',
               '--max-tokens', '17', '--layers', '64', '--pmode', 'none', '--quiet',
               '--gemm-block', '--block-major', '--decode-route', 'npu']
        invocation = dict(argv=cmd, command=shlex.join(cmd), environment=env, cwd=str(C.ROOT),
                          before=before, build_identity=identity, sanitized_inherited_oflm=removed,
                          arm=arm, prompt_lengths=list(map(len, prompts)))
        C.write_json(out / 'invocation.json', invocation)
        input_before = E.snapshot([ids, out / 'invocation.json'])
        start = P.conditions()
        result = C.capture_process(cmd, env, out / 'run.log', timeout=5400,
                                   executable=before[str(cli)])
        E.require(input_before == E.snapshot(input_before), 'list invocation changed')
        after = E.snapshot(inputs())
        E.require(before == after, 'list source/build/artifact changed')
        manifest = E.read_json(X.KERNELS / 'manifest.json')
        text = (out / 'run.log').read_text()
        E.require('ERROR' not in text and 'TIMEOUT' not in text, 'list failed')
        expected = set()
        generated = []
        for k, prompt in enumerate(prompts):
            E.require(f'prompt {k}: {len(prompt)} tokens' in text, 'missing list completion')
            lf, gf = outputs / f'p{k}_logits.f32', outputs / f'p{k}_gen.txt'
            expected.update([lf.name, gf.name])
            E.require(lf.stat().st_size == len(prompt) * manifest['spec']['real_vocab'] * 4,
                      'incomplete full prompt logits')
            import numpy as np
            values = np.fromfile(lf, dtype='<f4')
            E.require(bool(np.isfinite(values).all()), 'nonfinite list logits')
            ids_out = [int(x) for x in gf.read_text().strip().split(',')]
            E.require(len(ids_out) == 17 and all(0 <= i < C.REAL_VOCAB for i in ids_out),
                      'invalid list generated IDs')
            generated.append(ids_out)
        E.require({p.name for p in outputs.iterdir()} == expected, 'extra list outputs')
        record = dict(**invocation, **result, after=after, input_before=input_before,
                      input_after=E.snapshot(input_before), generated_ids=generated,
                      outputs={p.name: E.fingerprint(p) for p in outputs.iterdir()},
                      conditions_start=start, conditions_end=P.conditions(),
                      state_scope='NOT MEASURED: list path has no per-request state dumps')
        record['raw_hashes'] = {str(p.relative_to(out)): E.sha256(p)
                                for p in out.rglob('*') if p.is_file()}
        C.write_json(out / 'capture.json', record)
        print(name, 'CAPTURED', flush=True)
        return record


def compare_isolation(fresh, aba):
    for suffix in ('logits.f32', 'gen.txt'):
        hashes = [fresh['outputs'][f'p0_{suffix}']['sha256'],
                  aba['outputs'][f'p0_{suffix}']['sha256'],
                  aba['outputs'][f'p2_{suffix}']['sha256']]
        E.require(len(set(hashes)) == 1, 'A/B/A reset/isolation mismatch: ' + suffix)
    E.require(fresh['runtime']['before'] == aba['runtime']['before'], 'list XRT changed')


def isolation():
    OUT.mkdir(parents=True, exist_ok=False)
    tools()
    a, b = C.fixture(20), C.fixture(13, seed=26027)
    E.require(a != b and a[:len(b)] != b, 'prompts must differ')
    records = {}
    for arm in ('off', 'on'):
        fresh = list_capture(arm, [a], f'isolation-{arm}-fresh-a')
        aba = list_capture(arm, [a, b, a], f'isolation-{arm}-aba')
        compare_isolation(fresh, aba)
        records[arm] = aba
    for name in records['off']['outputs']:
        E.require(records['off']['outputs'][name]['sha256'] == records['on']['outputs'][name]['sha256'],
                  'OFF/ON list mismatch: ' + name)
    E.require(records['off']['runtime']['before'] == records['on']['runtime']['before'], 'XRT changed')
    C.write_json(OUT / 'isolation-passed.json', dict(passed=True,
        scope='complete prompt logits and 17 generated IDs; fresh A and A/B/A; both arms',
        state_dumps='NOT MEASURED per request', captures=[str(p) for p in OUT.glob('isolation-*/capture.json')]))


def timing():
    E.require(E.read_json(OUT / 'isolation-passed.json')['passed'], 'isolation not passed')
    E.require(E.read_json(BASE / 'correctness-passed.json')['passed'], 'boundary correctness absent')
    campaign = BASE / 'campaign'
    old_selection = E.read_json(OUT / 'selection.json') if (OUT / 'selection.json').exists() else None
    E.require(old_selection is None, 'continuation already started; no automatic resampling')
    selected = []
    for rep in (0, 1):
        for arm in ('off', 'on'):
            path = campaign / f'prefill-256-{rep}-{arm}'
            rec = E.read_json(path / 'capture.json')
            E.require(rec['raw_hashes'] == {p.name: E.sha256(p) for p in path.iterdir()
                                          if p.is_file() and p.name != 'capture.json'}, 'old raw changed')
            selected.append(dict(phase='prefill', tokens=256, rep=rep, warmup=False, arm=arm,
                                 capture=str(path), sha256=E.sha256(path / 'capture.json'), retained=True))
    selection = dict(protocol='initial five: original pairs 0,1 plus fresh 2,3,4; not confirmation',
        excluded=[dict(capture=str(campaign / f'prefill-256-2-{arm}'),
                       reason='unpaired interrupted attempt' if arm == 'off' else
                       'invalid: final process metadata absent; never reconstructed') for arm in ('off', 'on')],
        samples=selected)
    C.write_json(OUT / 'selection.json', selection)
    for phase, n, max_tokens, reps in [('prefill', 256, 1, [-1, 2, 3, 4]),
                                     ('prefill', 512, 1, range(-1, 5)),
                                     ('prefill', 1024, 1, range(-1, 5)),
                                     ('decode', 32, 33, range(-1, 5))]:
        for rep in reps:
            pair = {}
            for arm in (['off', 'on'] if rep % 2 == 0 else ['on', 'off']):
                name = f'resume-{phase}{n}-rep{rep}-{arm}'
                dest = campaign / name
                start = P.conditions()
                rec = C.run(BASE / 'candidate/build/open_qwen36_cli', phase, n, max_tokens,
                    dest, 4096, 'none', '0' if phase == 'prefill' else 'decode:32',
                    extra_env=X.THREADS, fold_instrumentation=phase == 'prefill',
                    build_identity=BASE / 'candidate/identity.json', kernels=X.KERNELS,
                    dedup_input_split=arm == 'on', launch_check=X.preflight)
                audit = E.audit_capture(dest)
                E.require(audit['valid'], str(audit['errors']))
                E.require(rec['runtime']['before'] == E.read_json(
                    campaign / 'prefill-256-0-off/capture.json')['runtime']['before'], 'original XRT changed')
                C.write_json(OUT / f'{name}-conditions.json', dict(start=start, end=P.conditions()))
                pair[arm] = rec
                selected.append(dict(phase=phase, tokens=n, rep=rep, warmup=rep < 0, arm=arm,
                    capture=str(dest), sha256=E.sha256(dest / 'capture.json'), retained=False))
                C.write_json(OUT / 'selection.json', selection)
                print(name, rec['prompt_wall_ms'], rec['generated_ms_per_token'], flush=True)
            E.require(pair['off']['generated_ids'] == pair['on']['generated_ids'], 'timing IDs changed')
            E.require(pair['off']['runtime']['before'] == pair['on']['runtime']['before'], 'XRT changed')
    C.write_json(OUT / 'timing-completed.json', dict(completed=True, confirmation_collected=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['isolation', 'timing'])
    args = parser.parse_args()
    os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib'
    os.environ['PATH'] = '/opt/xilinx/xrt/bin:' + os.environ['PATH']
    (isolation if args.stage == 'isolation' else timing)()
