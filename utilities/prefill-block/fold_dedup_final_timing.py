"""Fixed 037f2fcb campaign, only after explicit numerical/application passes."""
import argparse
import os
from pathlib import Path
import shutil

import pipeline_capture as C
import pipeline_evidence as E
import pipeline_campaign as P
import fold_dedup_experiment as X


SCENARIOS = [('prefill', 256, 1, 14), ('prefill', 512, 1, 6),
             ('prefill', 1024, 1, 6), ('decode', 32, 33, 6)]


def campaign(root):
    E.require(E.read_json(root / 'numerical-v1/numerical-passed.json')['passed'], 'numerical gate absent')
    E.require(E.read_json(root / 'application-passed.json')['passed'], 'application gate absent')
    out = root / 'timing-v1'
    out.mkdir(exist_ok=False)
    tools = out / 'tools'
    tools.mkdir()
    sidecars = out / 'conditions'
    sidecars.mkdir()
    for name in ('fold_dedup_final_timing.py', 'pipeline_capture.py', 'pipeline_evidence.py',
                 'pipeline_campaign.py', 'fold_dedup_experiment.py'):
        shutil.copy2(C.HERE / name, tools / name)
    plan = C.ROOT / 'specs/open-engine/reports/qwen38-host-fold-final-acceptance-plan.md'
    C.write_json(out / 'policy.json', dict(protocol=E.fingerprint(plan), scenarios=SCENARIOS,
        tool_snapshot=E.snapshot(tools.iterdir()), threads=X.THREADS,
        power='none; unchanged/unverified governor', profiling='layer both arms; fold both prefill arms',
        primary='CLI prefill outer wall, no subtraction', old_observations_pooled=False,
        optional_4k='NOT MEASURED: optional exploratory arm not selected',
        npu_memory='UNKNOWN: no per-process NPU memory counter exposed in basic accel sysfs; RSS is separate'))
    selection = []
    reference_runtime = None
    try:
        for phase, tokens, max_tokens, reps in SCENARIOS:
            for rep in range(-1, reps):
                pair = {}
                for arm in (('original', 'on') if rep % 2 == 0 else ('on', 'original')):
                    build = X.BASE / ('control' if arm == 'original' else 'candidate')
                    dest = out / f'{phase}-{tokens}-{rep}-{arm}'
                    conditions = P.conditions()
                    rec = C.run(build / 'build/open_qwen36_cli', phase, tokens, max_tokens,
                        dest, 4096, 'none', '0' if phase == 'prefill' else 'decode:32',
                        extra_env=X.THREADS, fold_instrumentation=phase == 'prefill',
                        build_identity=build / 'identity.json', kernels=X.KERNELS,
                        dedup_input_split=arm == 'on', launch_check=X.preflight)
                    audit = E.audit_capture(dest)
                    E.require(audit['valid'], str(audit['errors']))
                    if reference_runtime is None:
                        reference_runtime = rec['runtime']['before']
                    E.require(rec['runtime']['before'] == reference_runtime, 'runtime changed')
                    C.write_json(sidecars / f'{dest.name}.json', dict(start=conditions, end=P.conditions(),
                        npu_memory_bytes=None, npu_memory_scope='UNKNOWN; process RSS is not NPU memory'))
                    pair[arm] = rec
                    selection.append(dict(phase=phase, tokens=tokens, rep=rep, warmup=rep < 0,
                        arm=arm, capture=str(dest), capture_sha256=E.sha256(dest / 'capture.json')))
                    C.write_json(out / 'selection.json', dict(samples=selection))
                    print(dest.name, rec['prompt_wall_ms'], rec['generated_ms_per_token'], flush=True)
                E.require(pair['original']['generated_ids'] == pair['on']['generated_ids'],
                          'timing greedy IDs differ')
        C.write_json(out / 'completed.json', dict(completed=True, measured_pairs=32))
    except Exception as exc:
        C.write_json(out / 'stopped.json', dict(error=repr(exc), completed=False,
            instruction='No retry or recovery; inspect retained raw logs; control hardware failure requires operator'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    os.environ['LD_LIBRARY_PATH'] = '/opt/xilinx/xrt/lib'
    os.environ['PATH'] = '/opt/xilinx/xrt/bin:' + os.environ['PATH']
    campaign(args.root.absolute())
