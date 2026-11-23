"""Fixed paired statistics for final acceptance; never changes raw evidence."""
import argparse
import json
from pathlib import Path
import re
import statistics
from datetime import datetime, timezone

import numpy as np
import pipeline_capture as C
import pipeline_evidence as E
from fold_dedup_final_timing import SCENARIOS


def paired(control, candidate, bootstrap=False):
    a, b = np.asarray(control, dtype=float), np.asarray(candidate, dtype=float)
    E.require(a.shape == b.shape and a.ndim == 1 and len(a) > 0, 'invalid pairs')
    E.require(bool(np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all()),
              'nonfinite/nonpositive timing')
    reductions = 100 * (a - b) / a
    median = float(np.median(reductions))
    result = dict(count=len(a), control=list(map(float, a)), candidate=list(map(float, b)),
        reductions_percent=list(map(float, reductions)), median_reduction_percent=median,
        improving_pairs=int((reductions > 0).sum()), median_absolute_difference_ms=float(np.median(a-b)),
        reduction_range_percent=[float(reductions.min()), float(reductions.max())],
        reduction_mad_percent=float(np.median(np.abs(reductions-median))))
    if bootstrap:
        E.require(len(a) == 14, 'primary needs exactly 14 pairs')
        rng = np.random.default_rng(26026)
        medians = np.median(reductions[rng.integers(0, 14, size=(100000, 14))], axis=1)
        result['bootstrap'] = dict(resamples=100000, seed=26026, method='two-sided percentile median',
                                  ci95_percent=list(map(float, np.percentile(medians, [2.5, 97.5]))))
    return result


def summarize(root):
    timing = root / 'timing-v1'
    E.require(E.read_json(timing / 'completed.json')['completed'], 'campaign incomplete')
    selected = E.read_json(timing / 'selection.json')['samples']
    E.require(len(selected) == 72, 'expected 64 measured and eight warmup captures')
    scenarios = []
    for phase, tokens, _, reps in SCENARIOS:
        rows = [s for s in selected if s['phase'] == phase and s['tokens'] == tokens]
        E.require(len(rows) == 2*(reps+1), 'wrong scenario count')
        pairs = []
        for rep in range(reps):
            captures = {}
            for arm in ('original', 'on'):
                entry, = [s for s in rows if s['rep'] == rep and s['arm'] == arm and not s['warmup']]
                directory = Path(entry['capture'])
                E.require(E.sha256(directory/'capture.json') == entry['capture_sha256'], 'capture changed')
                record = E.read_json(directory/'capture.json')
                raw = {p.name:E.sha256(p) for p in directory.iterdir()
                       if p.is_file() and p.name != 'capture.json'}
                E.require(raw == record['raw_hashes'], 'raw inventory changed')
                E.validate_runtime(record['runtime'])
                fold = {}
                if phase == 'prefill':
                    for line in (directory/'fold.jsonl').read_text().splitlines():
                        value = json.loads(line)
                        if value.get('record') == 'fold':
                            for key in ('split_ms', 'copy_ms', 'accumulate_ms', 'total_ms'):
                                fold[key] = fold.get(key, 0.0) + value[key]
                captures[arm] = dict(capture=entry, metric_ms=record[
                    'prompt_wall_ms' if phase == 'prefill' else 'generated_ms_per_token'],
                    rss_bytes=record['sampled_rss_peak_bytes'], hwm_bytes=record['sampled_hwm_peak_bytes'],
                    fold=fold)
            for arm in captures.values():
                E.require(arm['rss_bytes'] > 0 and arm['hwm_bytes'] > 0, 'RSS/HWM not measured')
            baseline, candidate = captures['original'], captures['on']
            memory = {}
            for key in ('rss_bytes', 'hwm_bytes'):
                limit = max(.01 * baseline[key], 32*1024**2)
                memory[key] = dict(control=baseline[key], candidate=candidate[key], limit_bytes=limit,
                                   increase_bytes=candidate[key]-baseline[key],
                                   passed=candidate[key]-baseline[key] <= limit)
            pairs.append(dict(rep=rep, arms=captures, memory=memory))
        stats = paired([p['arms']['original']['metric_ms'] for p in pairs],
                       [p['arms']['on']['metric_ms'] for p in pairs], phase == 'prefill' and tokens == 256)
        memory_passed = all(v['passed'] for p in pairs for v in p['memory'].values())
        if tokens == 256 and phase == 'prefill':
            passed = (stats['median_reduction_percent'] >= 1 and stats['improving_pairs'] >= 11 and
                      stats['bootstrap']['ci95_percent'][0] > 0)
        else:
            passed = stats['median_reduction_percent'] >= -1
        scenarios.append(dict(phase=phase, tokens=tokens, pairs=pairs, statistics=stats,
                              performance_passed=passed, memory_passed=memory_passed))
    return dict(complete=True, scenarios=scenarios,
                decision='MERGE CANDIDATE' if all(s['performance_passed'] and s['memory_passed']
                    for s in scenarios) else 'NO-GO',
                optional_4k='NOT MEASURED; optional exploratory scenario not selected')


def publish(root):
    audit = E.read_json(root/'final-audit.json')
    application = E.read_json(root/'application-passed.json')
    E.require(audit['passed'] and application['passed'], 'final gates absent')
    result = E.read_json(root/'statistics.json')
    policy = E.read_json(root/'timing-v1/policy.json')
    selected = E.read_json(root/'timing-v1/selection.json')['samples']
    metadata, loads, temperatures, available, governors = [], [], [], [], set()
    for row in selected:
        directory = Path(row['capture'])
        record = E.read_json(directory/'capture.json')
        conditions_path = root/'timing-v1/conditions'/f'{directory.name}.json'
        conditions = E.read_json(conditions_path)
        for when in ('start', 'end'):
            c = conditions[when]
            loads.append(float(re.search(r'load average: ([0-9.]+)', c['load']['stdout'])[1]))
            available.append(int(next(s for s in c['memory']['stdout'].splitlines()
                                      if s.startswith('Mem:')).split()[-1]))
            for path, value in c['sysfs'].items():
                if path.endswith('/temp'):
                    temperatures.append(int(value)/1000)
                if path.endswith('/scaling_governor'):
                    governors.add(value)
        metadata.append(dict(row, argv=record['argv'], command=record['command'],
            environment=record['environment'], cli=record['cli'], runtime=record['runtime']['before'],
            process_sample_count=len(record['cpu_observations']),
            sampled_rss_peak_bytes=record['sampled_rss_peak_bytes'],
            sampled_hwm_peak_bytes=record['sampled_hwm_peak_bytes'],
            generated_ids=record['generated_ids'], conditions=E.fingerprint(conditions_path),
            capture=E.fingerprint(directory/'capture.json')))
    first_conditions = E.read_json(root/'timing-v1/conditions/prefill-256-0-original.json')['start']
    old = audit['committed_reports_unchanged']
    E.require(any(p.endswith('qwen38-host-fold-dedup.md') for p in old), 'old report not audited')
    evidence_names = ['final-audit.json', 'independent-audit.json', 'recovery-cross-audit.json',
        'application-passed.json', 'numerical-v1/numerical-passed.json',
        'numerical-v1/isolation-passed.json', 'model-identity-incident.json', 'statistics.json',
        'npu-postcampaign.json', 'pytest-focused.log', 'pytest-full-with-skips.log', 'settings-ci.log']
    for name in ('parent-review.json', 'application-output-inventory.json'):
        if (root / name).exists():
            evidence_names.append(name)
    settings_status = 'NOT RUN: check refuses dirty tree; parent must run after final commits'
    final_settings = root / 'settings-ci-final.log'
    if final_settings.exists():
        E.require('check-settings-ci: OK --' in final_settings.read_text(), 'final settings check failed')
        settings_status = 'PASS: clean committed-source merged-target check'
        evidence_names.append('settings-ci-final.log')
    app = root/'app-buildroot-v1'
    app_paths = [app/p for p in ('source.tar', 'open-kernels-source.tar', 'image.txt',
        'build/src/oflm', 'build/CMakeCache.txt', 'build/build.ninja', 'build/.ninja_deps',
        'build.log', 'build-repaired.log', 'build-headers-repaired.log')]
    result.update(protocol_commit='037f2fcb', campaign_policy=policy,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), evidence_root=str(root),
        machine_conditions=dict(host_xrt_report=first_conditions['npu']['stdout'],
            cpu_governors=sorted(governors), sampled_system_thermal_c_range=[min(temperatures), max(temperatures)],
            load_average_1m_range=[min(loads), max(loads)],
            available_host_ram_bytes_range=[min(available), max(available)],
            npu_power='CLI none; no policy writes; governor unchanged/unverified during campaign; post-campaign platform reports default'),
        timing_capture_metadata=metadata, final_audit=audit,
        application=application, app_build_artifacts=E.snapshot(app_paths),
        evidence_files=E.snapshot(root/p for p in evidence_names),
        initialization_metadata=str(root.parent/'recovery-20261123-1246/initialization-passed.json'),
        initialization_evidence=E.snapshot(root.parent/'recovery-20261123-1246'/p for p in
            ('capacity-result.json', 'initialization-passed.json',
             'control-initialization/capture.json', 'candidate-initialization/capture.json')),
        capacity=dict(required=11, created=11, maximum_measured=False, operator_recovery_performed=False),
        numerical=dict(captures=22, boundaries=[95,96,97,255,256,257,511,512,513],
            logits='A/B/A: every real-vocabulary prompt row; boundaries: last prompt row and 17 continuation rows',
            state_layers=[0,3,62,63], state_scope='complete declared state buffers for those layers only; no all-layer or per-request state-isolation claim',
            schedule='list A/B/A layer-major; boundary captures block-major'),
        npu_memory=dict(bytes=None, status='UNKNOWN', rss_is_npu_memory=False,
            exposed_postcampaign_total_memory_usage='N/A', live_memory_samples='NOT MEASURED: no byte counter exposed by available reports/basic sysfs'),
        validation=dict(focused='7 passed', full='2937 passed, 172 skipped',
            skips='NOT MEASURED for skipped capabilities; exact reasons in pytest-full-with-skips.log',
            merged_settings=settings_status),
        exclusions=['No old timing observations pooled', 'Four warmup pairs excluded exactly once per scenario',
            'Optional 4000-token/32-step route NOT MEASURED: existing decode capture contract disallows block-prefill logs; sequential 4000-token prefill is not practical for this bounded campaign',
            'No full-context, all-layer-state or general model-support acceptance claim'],
        incident='Initial real app run used a directory symlink and shipped metadata triggered automatic download. Config changed, original weight bytes stayed intact. Process stopped; downloaded config, 321347584-byte partial weight and install record retained under download-incident. Configuration reconstructed from pre-normalization backup only after exact original SHA256 match; original inode changed. Later application attempts use isolated small-file copies, local file metadata and a weight-file symlink. Timing begins after restoration; no algorithm/default/kernel change.',
        repository_actions='Evidence/protocol commits only; no push, merge, rebase, reset, reload, reboot, firmware or power-policy writes')
    destination = C.ROOT/'specs/open-engine/reports/qwen38-host-fold-final-acceptance'
    C.write_json(destination.with_suffix('.json'), result)
    lines = ['# Host-fold final acceptance', '', f'**Decision: {result["decision"]}. Defaults remain OFF. No push or merge.**', '',
        'This decision applies to the scoped host-fold optimization on the machine below, not to general model support or a full-context claim.', '',
        '## Machine and fixed protocol', '',
        'Measured on prihlop-laptop2 / EVO14-AI9-STP, AMD Ryzen AI 9 365 (20 logical CPUs), Ubuntu 26.04.1 LTS, Linux 7.0.0-44-generic, glibc 2.43, NPU Strix aie2p 6x8.',
        'XRT 2.26.0 (e9db9ab15f10173f8d2fc93ff92ab4c7eb09d2e6), amdxdna 2.26.0_20260817 (2b71d02b641561a40ab2db1c26a69620794f4e9f), firmware 1.1.2.64.',
        'The [committed protocol](qwen38-host-fold-final-acceptance-plan.md) is fixed by commit 037f2fcb. Earlier incomplete observations are not pooled.',
        'Threads: OMP_NUM_THREADS=8, OMP_WAIT_POLICY=PASSIVE, OPENBLAS_NUM_THREADS=1. CLI power mode is none: no policy write. NPU governor is unchanged/unverified during the campaign; the post-campaign platform report says default.',
        f'Observed CPU governor: {", ".join(sorted(governors))}. Pre/post capture system thermal sensors: {min(temperatures):.1f}–{max(temperatures):.1f} °C; one-minute load: {min(loads):.2f}–{max(loads):.2f}; available host RAM: {min(available)/1024**3:.2f}–{max(available)/1024**3:.2f} GiB. These are boundary observations, not continuous NPU thermals.',
        'Each inference launch holds the model flock, requires at least 35 GiB available host RAM and no unrelated accelerator client. Every capture records actual executable and mapped core/coreutil/xdna library identities.', '',
        '## Identity and pre-timing gates', '',
        '- Capacity: the supplied archived probe created all 11 required manifest contexts. The capacity-result.json and initialization metadata under recovery-20261123-1246 are fingerprinted in the JSON report. Maximum capacity is NOT MEASURED. No operator recovery was needed or performed; runtime availability is not attributed to a reset.',
        '- Parent initialization: unchanged control and candidate each completed 8 prompt tokens plus 16 generated NPU steps across 64 layers, with matching IDs and runtime. Metadata remains in recovery-20261123-1246/initialization-passed.json.',
        '- Fresh numerical evidence: 22 captures. Archived control/candidate each pass fresh A and distinct A/B/A with all real-vocabulary logits for every prompt position and 17 generated IDs. The list schedule is layer-major; block options do not turn that path into block-major.',
        '- Fresh block-major boundaries: 95/96/97, 255/256/257 and 511/512/513. Last-prompt and 17 continuation logit buffers, exact greedy IDs, and state buffers at layers 0/3/62/63 agree byte-for-byte. These sampled layers cover linear convolution/recurrent and attention/KV state. This is not an all-layer-state or per-request buffer-zeroing claim.',
        '- Actual candidate application: rebuilt from c5d20c85 with the existing Docker image and writable copied tokenizer dependencies; kernel export disabled. 1,252 archived source files independently match the staged tree. Both OFF/ON application prompt runs return apple; both candidate-specific servers pass distinct A/B/A requests with 17 completion tokens, exact API choices and consumed-engine-token traces.',
        '- Application control is the same rebuilt candidate binary OFF, allowed for this integration comparison. It is not the primary timing control. The API does not expose full logits/state or an independent final sampled ID; complete 17-ID continuation equality is supplied by the CLI numerical gates.',
        '- Application metadata is isolated scratch metadata describing the validated local files. This is not an unmodified shipped-install acceptance claim. The shipped download metadata differs from this historical local container.', '',
        '### Immutable timing arms', '', '| Arm | Source | Executable SHA-256 |', '|---|---|---|']
    for name in ('control', 'candidate'):
        identity = E.read_json(root.parent/name/'identity.json')
        lines.append(f'| {name} | {identity["source_revision"]} | {identity["executable_sha256"]} |')
    lines += ['', 'Executable paths are recorded in the JSON report. Only candidate sets OFLM_OPEN_FOLD_DEDUP_INPUT_SPLIT=1; unchanged control has no candidate switch. Identical model/kernel content hashes are audited independently.', '',
        '## Fixed timing result', '',
        'Exactly 14 new measured 256-token pairs (seven per order), six each for 512/1024 prefill and short decode, plus one excluded warmup pair per scenario: 72 captures total. No retry, selective omission or extra confirmation pair. Both prefill arms include layer and fold profiling; decode includes the same layer profiling in both arms.',
        'Primary prefill metric is the outer CLI prompt wall, without subtracting serialization or substituting inner timers. Decode is 32 real prompt tokens and 32 generated steps (max-tokens 33).', '',
        '| Scenario | Pairs | Median paired reduction | Wins | Median difference | Result |', '|---|---:|---:|---:|---:|---|']
    for scenario in result['scenarios']:
        s = scenario['statistics']
        status = 'PASS' if scenario['performance_passed'] and scenario['memory_passed'] else 'FAIL'
        lines.append(f'| {scenario["phase"]} {scenario["tokens"]} | {s["count"]} | {s["median_reduction_percent"]:.6f}% | {s["improving_pairs"]}/{s["count"]} | {s["median_absolute_difference_ms"]:.3f} ms | {status} |')
    primary = result['scenarios'][0]['statistics']
    ci = primary['bootstrap']['ci95_percent']
    lines += ['', f'Primary two-sided 95% percentile bootstrap median-reduction interval: **[{ci[0]:.6f}%, {ci[1]:.6f}%]**, 100,000 resamples, seed 26026. Median >=1%, wins >=11/14 and interval lower endpoint >0 all pass.',
        f'Primary paired-reduction range: {primary["reduction_range_percent"][0]:.6f}–{primary["reduction_range_percent"][1]:.6f}%; MAD: {primary["reduction_mad_percent"]:.6f} percentage points.',
        'The fixed paired-median 1% regression limits pass for 512/1024 prefill and short decode. Decode pair 5 individually regresses 1.330377%; it remains included. Its paired median does not regress.', '',
        '### Every measured pair', '', '| Scenario | Pair | Control ms | Candidate ms | Reduction |', '|---|---:|---:|---:|---:|']
    for scenario in result['scenarios']:
        s = scenario['statistics']
        for i, (a, b, reduction) in enumerate(zip(s['control'], s['candidate'], s['reductions_percent'])):
            lines.append(f'| {scenario["phase"]} {scenario["tokens"]} | {i} | {a:.3f} | {b:.3f} | {reduction:.6f}% |')
    lines += ['', '## Direct fold and process memory', '',
        'Each fold entry below is the median per-capture sum of direct fold windows, not an extrapolation. RSS/HWM are sampled whole-process observations and are not total NPU memory.', '',
        '| Prefill tokens | Control split / total fold ms | Candidate split / total fold ms |', '|---:|---:|---:|']
    for scenario in result['scenarios'][:3]:
        medians = {arm: {k:statistics.median(p['arms'][arm]['fold'][k] for p in scenario['pairs'])
                    for k in ('split_ms', 'total_ms')} for arm in ('original', 'on')}
        a, b = medians['original'], medians['on']
        lines.append(f'| {scenario["tokens"]} | {a["split_ms"]:.3f} / {a["total_ms"]:.3f} | {b["split_ms"]:.3f} / {b["total_ms"]:.3f} |')
    lines += ['', '| Scenario | Median control RSS / HWM MiB | Median candidate RSS / HWM MiB | Max paired RSS increase MiB |', '|---|---:|---:|---:|']
    for scenario in result['scenarios']:
        values = {arm:[statistics.median(p['arms'][arm][key]/1024**2 for p in scenario['pairs'])
                       for key in ('rss_bytes', 'hwm_bytes')] for arm in ('original', 'on')}
        a, b = values['original'], values['on']
        increase = max(p['memory']['rss_bytes']['increase_bytes']/1024**2 for p in scenario['pairs'])
        lines.append(f'| {scenario["phase"]} {scenario["tokens"]} | {a[0]:.3f} / {a[1]:.3f} | {b[0]:.3f} / {b[1]:.3f} | {increase:.3f} |')
    lines += ['', 'Every measured RSS/HWM comparison passes max(1% of paired control, 32 MiB). The JSON keeps each observed peak, limit and difference; it does not equate RSS with NPU allocation.',
        'NPU memory bytes: **UNKNOWN**. Available XRT reports expose host/platform/partitions, with total_memory_usage=N/A in the post-campaign partition report. Basic accel sysfs exposes no byte counter. The post-campaign report shows no live hardware contexts; that is not a live per-capture memory measurement. Per-capture runtime mappings, pre/post device/system conditions and all RSS/HWM samples remain in raw captures.', '',
        '## Evidence, environment repair and limits', '',
        result['incident'], '',
        'Configuration restoration SHA-256 is 30cf5c47d75b13934df23683f6f8f660d948d3c8e4e3ca91ad90b2278c536d6b. The new inode is explicitly recorded; initial numerical captures retain the original inode metadata. Content hashes match across initialization, numerical and timing evidence. No old record is rewritten to conceal the change.',
        'Accepted application logs/requests and ancillary files have a separate content inventory sealed at audit time, not reconstructed runtime capture metadata.',
        'The writable Docker build uses the candidate license and harness headers from the same candidate commit. Failed build/client/server attempts, including an invalid inherited proxy and a harness-SIGINT classification error, remain separate from timing and accepted captures. Accepted server captures explicitly record the harness-issued SIGINT exit.',
        'Optional genuine 4000-token plus 32-step decode is **NOT MEASURED**: the existing validated decode-capture contract excludes block-prefill logs, while its sequential 4000-token prefill is not practical for this bounded campaign. No seek/fake context or extrapolation is used.',
        'Full-context, independent API logits/state, complete all-layer/per-request state isolation and continuous NPU-memory/thermal measurements are outside these passing scoped gates. Existing linked closed libraries remain application dependencies; this work neither removes nor replaces them.', '',
        'The [JSON report](qwen38-host-fold-final-acceptance.json) records the raw evidence root, all individual pairs, all 72 capture paths/hashes/commands/environments, observed runtime identities, evidence/build hashes, initialization references and machine conditions. The [duplicate-split experiment report](qwen38-host-fold-dedup.md) describes the separate bounded attempt. Its [JSON](qwen38-host-fold-dedup.json), raw hashes and recovery directory remain unchanged.',
        'Reproduction tools: [numerical driver](../../../utilities/prefill-block/fold_dedup_final.py), [application capture](../../../utilities/prefill-block/fold_dedup_app.py), [application audit](../../../utilities/prefill-block/fold_dedup_app_audit.py), [fixed timing](../../../utilities/prefill-block/fold_dedup_final_timing.py), [statistics/report](../../../utilities/prefill-block/fold_dedup_final_report.py), [final audit](../../../utilities/prefill-block/fold_dedup_final_audit.py). Fresh output paths are mandatory; do not rerun into existing evidence.', '',
        '## Portable checks and repository scope', '',
        'Focused tests: 7 passed. Full portable suite: 2,937 passed, 172 skipped. Skips are NOT MEASURED, with exact capability/server prerequisites retained in pytest-full-with-skips.log; they are not inference passes.',
        f'Merged-target settings check: **{settings_status}**. Only the clean committed-source check counts as a pass; the retained dirty-tree refusal is not acceptance evidence. No new production setting, flag or environment variable is introduced.',
        'Only acceptance tooling, portable tests and these separate reports change. Production optimization code, defaults, kernels and criteria are unchanged. Evidence/protocol commits do not authorize a default promotion, push, merge, rebase, reset, reload, reboot, firmware or power-policy change.', '']
    destination.with_suffix('.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    C.write_json(args.root/'statistics.json', summarize(args.root.absolute()))
    if args.publish:
        publish(args.root.absolute())
