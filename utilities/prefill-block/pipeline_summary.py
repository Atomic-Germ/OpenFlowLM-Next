"""Audit an explicitly selected PR4A campaign and summarize only its new samples."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as S

import pipeline_capture as C
import pipeline_evidence as E
from pipeline_campaign import validate_disabled_log


def stats(values):
    E.require(bool(values) and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values),
              'nonnegative finite samples required')
    return dict(samples=values, median=S.median(values), minimum=min(values),
                maximum=max(values), mean=S.mean(values),
                stdev=S.stdev(values) if len(values) > 1 else None)


def paired(off, on):
    E.require(len(off) == len(on) and len(off) >= 5 and all(x > 0 for x in off),
              'at least five matched positive baseline samples required')
    delta = [b - a for a, b in zip(off, on)]
    return dict(off=stats(off), on=stats(on), paired_delta_ms=delta,
                median_paired_delta_ms=S.median(delta),
                median_paired_change_percent=S.median([100 * (b / a - 1)
                                                       for a, b in zip(off, on)]),
                paired_delta_range_ms=[min(delta), max(delta)])


def load(directory, enabled):
    record = E.read_json(directory / 'capture.json')
    if enabled:
        audit = E.audit_capture(directory)
        E.require(audit['valid'], str(audit['errors']))
    else:
        E.require(record['schema'] == 'pr4a-disabled-capture-v1', 'unknown disabled schema')
        E.require(record['before'] == record['after'] == E.snapshot(record['before']),
                  'disabled snapshot mismatch')
        E.require(record['input_before'] == E.snapshot(record['input_before']),
                  'disabled invocation changed')
        E.validate_runtime(record['runtime'])
        for name, digest in record['raw_hashes'].items():
            E.require(E.sha256(directory / name) == digest, 'disabled raw hash mismatch')
        E.require(not any(k in record['environment'] for k in
                          ['OFLM_OPEN_PROFILE', 'OFLM_OPEN_PROFILE_STEPS', 'OFLM_OPEN_FOLD_PROFILE']),
                  'disabled sample had instrumentation enabled')
        info = dict(real_vocab=C.REAL_VOCAB, block_tokens=256)
        observed = validate_disabled_log((directory / 'run.log').read_text(), record['phase'],
                                         record['real_prompt_tokens'],
                                         len(record['generated_ids']), info)
        for key in ('prompt_wall_ms', 'generated_ids', 'generated_ms_per_token'):
            E.require(observed[key] == record[key], 'disabled metadata/log contradiction')
    return record


def profile(directory):
    rows = E.jsonl(directory / 'layers.jsonl')
    requests = [r for r in rows if r['record'] == 'request']
    layers = [r for r in rows if r['record'] == 'layer']
    n = len(requests)
    result = {key: sum(r[key] for r in requests) / n for key in
              ['total_ms', 'gemm_ms', 'host_ms', 'lmhead_ms', 'dispatch_ms', 'residue_ms']}
    for key in ['dn_conv_ms', 'dn_rule_ms', 'attn_ms', 'prenorm_ms', 'gemm_tile_ms',
                'gemm_tr_ms', 'tail_ms', 'state_ms', 'sync_ms', 'shared_ms']:
        result[key] = sum(r[key] for r in layers) / n
    for kind in ['linear_attention', 'full_attention']:
        result[kind + '_layer_wall_ms'] = sum(r['wall_ms'] for r in layers if r['kind'] == kind) / n
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    selection = E.loads('{"samples":' + (a.root / 'campaign/selection.json').read_text() + '}')['samples']
    samples = []
    measured = {}
    for item in selection:
        directory = Path(item['capture'])
        E.require(E.sha256(directory / 'capture.json') == item['capture_sha256'],
                  'selected capture metadata changed')
        rec = load(directory, item['arm'] == 'on')
        if item['arm'] == 'on':
            sidecar = a.root / 'conditions' / f'{directory.name}.json'
            if not sidecar.exists():
                sidecar = directory.parent / 'conditions' / f'{directory.name}.json'
            cond = E.read_json(sidecar)
        else:
            sidecar = None
            cond = dict(start=rec['conditions_start'], end=rec['conditions_end'])
        row = dict(**item, argv=rec['argv'], environment=rec['environment'],
                   raw_hashes=rec['raw_hashes'],
                   conditions=cond,
                   conditions_sha256=E.sha256(sidecar) if sidecar else None,
                   build_identity_content_sha256=hashlib.sha256(json.dumps(
                       rec['build_identity'], sort_keys=True).encode()).hexdigest(),
                   runtime={p: v['sha256'] for p, v in rec['runtime']['before'].items()},
                   prompt_wall_ms=rec['prompt_wall_ms'], generated_ids=rec['generated_ids'],
                   generated_ms_per_token=rec.get('generated_ms_per_token'))
        samples.append(row)
        if not item['warmup']:
            measured[(item['phase'], item['rep'], item['arm'])] = (rec, directory)
    E.require(len(measured) == 20, 'exactly five measured pairs per phase required')
    overhead = {}
    profiles = {}
    fold = []
    for phase in ['prefill', 'decode']:
        timings = {'off': [], 'on': []}
        profiles[phase] = []
        for rep in range(5):
            off, _ = measured[(phase, rep, 'off')]
            on, directory = measured[(phase, rep, 'on')]
            E.require(off['generated_ids'] == on['generated_ids'], 'pair output mismatch')
            E.require(off['build_identity'] == on['build_identity'], 'pair build mismatch')
            E.require(off['runtime']['before'] == on['runtime']['before'], 'pair runtime mismatch')
            for arm, rec in [('off', off), ('on', on)]:
                timings[arm].append(rec['prompt_wall_ms'] if phase == 'prefill'
                                    else rec['generated_ms_per_token'])
            profiles[phase].append(profile(directory))
            if phase == 'prefill':
                rows = E.jsonl(directory / 'fold.jsonl')[1:]
                E.require(len(rows) == 1, 'expected single physical block')
                f = rows[0]
                f['prompt_wall_ms'] = on['prompt_wall_ms']
                f['fold_share_percent'] = 100 * f['total_ms'] / on['prompt_wall_ms']
                f['split_share_percent'] = 100 * f['split_ms'] / on['prompt_wall_ms']
                f['duplicate_split_assumption_ceiling_percent'] = f['split_share_percent'] / 2
                fold.append(f)
        overhead[phase] = paired(timings['off'], timings['on'])
    scaling = []
    for tokens in [512, 1024]:
        directory = a.root / f'fold-{tokens}-single'
        rec = load(directory, True)
        rows = E.jsonl(directory / 'fold.jsonl')[1:]
        scaling.append(dict(tokens=tokens, repetitions=1, capture=str(directory),
                            capture_sha256=E.sha256(directory / 'capture.json'),
                            raw_hashes=rec['raw_hashes'], prompt_wall_ms=rec['prompt_wall_ms'],
                            fold_rows=rows, fold_total_ms=sum(r['total_ms'] for r in rows)))
    first = measured[('prefill', 0, 'on')][0]
    accounting = [dict(total_ms=p['total_ms'], gemm_bucket_ms=p['gemm_ms'],
                       host_bucket_ms=p['host_ms'], lmhead_ms=p['lmhead_ms'],
                       fold_exclusive_ms=f['total_ms'],
                       dn_ms=p['dn_conv_ms'] + p['dn_rule_ms'],
                       unassigned_ms=p['total_ms'] - p['gemm_ms'] - p['host_ms'] -
                       p['lmhead_ms'] - f['total_ms'])
                  for p, f in zip(profiles['prefill'], fold)]
    result = dict(scope='profiling only; no production optimization',
                  status='NEEDS MORE EVIDENCE',
                  decision_reason='Fold is directly measurable, but duplicate-split savings remain a conditional ceiling, not a measured production gain; allocation/reuse and phase CPU/wait remain unresolved.',
                  source_revision=first['build_identity']['source_revision'],
                  executable_sha256=first['cli']['sha256'],
                  build_identity=first['build_identity'],
                  artifact_snapshot=first['before'],
                  prefill_accounting=accounting,
                  overhead=overhead, profiles=profiles, fold_samples=fold,
                  fold_summary={key: stats([r[key] for r in fold]) for key in
                                ['split_ms', 'copy_ms', 'accumulate_ms', 'total_ms',
                                 'fold_share_percent', 'split_share_percent',
                                 'duplicate_split_assumption_ceiling_percent']},
                  scaling_single_runs=scaling, selected_samples=samples,
                  allocation_reuse='NOT MEASURED separately; resize/assign costs are included in split/copy windows',
                  cpu_wait='NOT MEASURED by phase; dispatch residence overlaps',
                  not_measured=['phase-specific CPU/wait', '4K decode', 'serialization-only cost',
                                'separate allocation counts/time', 'full-state equality',
                                'production candidate speedup', 'HTTP TTFT', 'NPU utilization'])
    C.write_json(a.out, result)
    print(json.dumps(dict(overhead=overhead, fold_summary=result['fold_summary'],
                          profile_medians={phase: {k: S.median([r[k] for r in rows])
                                                   for k in rows[0]} for phase, rows in profiles.items()},
                          scaling=scaling), indent=2))


if __name__ == '__main__':
    main()
