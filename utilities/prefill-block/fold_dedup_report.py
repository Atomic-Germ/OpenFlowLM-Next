"""Inventory the bounded duplicate-split attempt without repairing or replacing raw evidence."""
import argparse
import json
from pathlib import Path
import re
import statistics

import pipeline_capture as C
import pipeline_evidence as E


def blocked_continuation(base, report):
    """Retain accepted history, verify unchanged bytes, and report an unmet gate."""
    base, report = base.absolute(), report.absolute()
    history = base / 'continuation-history-v1'
    previous = E.read_json(history / 'qwen38-host-fold-dedup.json')
    inventory = E.read_json(history / 'original-evidence-inventory.json')
    for path, digest in inventory.items():
        E.require(E.sha256(C.ROOT / path) == digest, 'original evidence changed: ' + path)
    continuation = base / 'continuation-v1'
    E.require(not (continuation / 'isolation-passed.json').exists(),
              'blocked report cannot substitute for a completed isolation audit')
    E.require(not (continuation / 'selection.json').exists(), 'timing unexpectedly started')
    selection = dict(protocol='original initial five; not confirmation or replacement dataset',
                     retained=[], pending=[], excluded=[])
    for rep in (0, 1):
        for arm in ('off', 'on'):
            path = base / 'campaign' / f'prefill-256-{rep}-{arm}' / 'capture.json'
            selection['retained'].append(dict(rep=rep, arm=arm, capture=reference(path)))
    for rep in (2, 3, 4):
        selection['pending'].append(dict(rep=rep,
            off=str(base / 'campaign' / f'resume-prefill256-rep{rep}-off'),
            on=str(base / 'campaign' / f'resume-prefill256-rep{rep}-on'), measured=False))
    for arm in ('off', 'on'):
        path = base / 'campaign' / f'prefill-256-2-{arm}'
        selection['excluded'].append(dict(capture=str(path), counted=False,
            reason='unpaired interrupted attempt' if arm == 'off' else
                   'invalid: final process metadata missing; never reconstructed'))
    selection['warmups'] = dict(original='excluded', restart='one excluded pair pending',
        other_scenarios='one excluded warmup then exactly five pairs each pending')
    C.write_json(continuation / 'predeclared-selection.json', selection)
    failures = []
    for directory in sorted(continuation.glob('isolation-*')):
        E.require(not (directory / 'capture.json').exists(), 'unexpected successful capture')
        failures.append(dict(capture=str(directory), valid=False,
            reason='hardware context creation rc=1, err=-22 before inference; no final metadata',
            raw_files=[reference(p) for p in sorted(directory.rglob('*')) if p.is_file()]))
    E.require(len(failures) == 2, 'expected initial attempt and one bounded retry')
    record = dict(previous)
    record.update(schema='pr4b-host-fold-dedup-blocked-continuation-v2', decision='KEEP DRAFT',
        model_acceptance=False, campaign_complete=False,
        reason='A/B/A correctness prerequisite blocked before inference by HWCTX creation -22 twice',
        original_evidence_unchanged=True, predeclared_selection=selection,
        previous_reports=[reference(history / ('qwen38-host-fold-dedup' + suffix))
                          for suffix in ('.md', '.json')],
        original_inventory=reference(history / 'original-evidence-inventory.json'),
        continuation_failures=failures,
        timer_validation=dict(scope='one-sided inner containment; no upper bound on wrapper overhead',
            allowance_ms='0.5 + 0.0005 * request_count + 1e-9 arithmetic slack',
            outer_summary='integer-ms and integer-ms/token rounding intervals must intersect',
            numerical_thresholds_changed=False, performance_thresholds_changed=False),
        confirmation=dict(collected=False, eligibility='not established: initial five incomplete'),
        continuation_support=[reference(p) for p in sorted(base.glob('continuation-*.log'))
                              if not p.name.startswith('continuation-report-generation')] +
            [reference(p) for p in sorted((continuation / 'tools').iterdir()) if p.is_file()],
        current_tooling=[reference(p) for p in sorted((continuation / 'tools-final-v2').iterdir())
                         if p.is_file()],
        unmeasured=['distinct-prompt reset/isolation', 'complete five-pair primary gate',
                    '512 prefill', '1024 prefill', '32-prompt plus 32-step decode',
                    'around-4K populated-context decode', 'application/server inference',
                    'per-repeat state dumps'])
    for entry in record['timing_captures']:
        if entry.get('valid'):
            entry['audit_scope'] = ('retained original accepted audit; raw bytes verified unchanged; '
                                    'not replayed with the modified validator')
            audit = entry['audit']
            # Full mapping samples and source inventories stay in hashed captures.
            # Keep identities here without duplicating every live-process sample.
            entry['audit'] = {k: audit[k] for k in (
                'valid', 'errors', 'capture_metadata_sha256', 'version', 'phase',
                'real_prompt_tokens', 'generated_tokens', 'command', 'argv', 'binary',
                'raw_hashes') if k in audit}
            entry['audit']['runtime'] = {k: audit['runtime'][k] for k in ('before', 'after')}
        if entry['name'] == 'prefill-256-2-on':
            entry['reason'] = 'invalid: missing final process metadata; structural timer fix does not restore it'
    for arm in ('control', 'candidate'):
        E.build_identity(base / arm / 'identity.json', base / arm / 'build/open_qwen36_cli')
    record['build_identities_reverified'] = True
    record['tests'] = {}
    for label, name in [('focused', 'continuation-focused-final.log'),
                        ('portable', 'continuation-portable-final.log')]:
        path = base / name
        text = path.read_text()
        record['tests'][label] = dict(log=reference(path), summary=text.splitlines()[-1])
    record['tests']['settings_ci'] = dict(status='NOT RUN: dirty-tree refusal; parent commit required',
        log=reference(base / 'continuation-settings-ci.log'))
    settings = base / 'settings-ci-final.log'
    if settings.exists():
        E.require('check-settings-ci: OK --' in settings.read_text(), 'final settings check failed')
        record['tests']['settings_ci'] = dict(status='PASS: clean committed-tree target merge',
                                             log=reference(settings))
    record['report_generator'] = reference(Path(__file__))
    C.write_json(report.with_suffix('.json'), record)
    observation = record['descriptive_only']
    rows = [r for r in record['timing_captures'] if r.get('valid')]
    table = '\n'.join(f"| {r['rep']} | {r['arm']} | {'excluded warmup' if r['warmup'] else 'unpaired, excluded' if r['rep'] == 2 else 'retained'} | "
        f"{r['prefill_ms']:.0f} | {r['fold_ms']['split_ms']:.3f} | {r['fold_ms']['total_ms']:.3f} | "
        f"{r['rss_bytes']/1024**2:.3f} | {r['hwm_bytes']/1024**2:.3f} |" for r in rows)
    old_text = (history / 'qwen38-host-fold-dedup.md').read_text()
    machine_build = old_text.split('## Machine and measurement conditions\n', 1)[1].split(
        '## Hardware correctness before timing', 1)[0]
    machine_build = re.sub(r'\bPR\s*#?\s*\d+[A-Z]?\b', 'host-fold profiling', machine_build)
    machine_build = re.sub(r'`Models/[^`]+` and `candidate/`',
        'the control and candidate archive paths listed in the JSON report', machine_build)
    correctness = old_text.split('## Hardware correctness before timing\n', 1)[1].split(
        '## Timing observations:', 1)[0]
    report.write_text(f'''# Bounded duplicate-split hardware experiment

## Decision: KEEP DRAFT — hardware prerequisite blocked

The initial five-pair campaign is incomplete. A fresh-A native list request and
one bounded retry both fail before inference with hardware-context creation
`DRM_IOCTL_AMDXDNA_CREATE_HWCTX`, error -22 (Invalid argument). The kernel journal
records management opcode 0x2 status 0x2000003 and AIE resource allocation failure.
The device remains visible, runtime status active / D0, with no competing
accelerator client found. The underlying cause is not established; no driver
reset, power-policy change, runtime replacement or candidate tweak is performed.
This is not a numerical mismatch and is not a measured performance NO-GO.

Distinct-prompt correctness is **NOT MEASURED**, so no restart warmup or new timed
sample is collected. The original two accepted pairs remain accepted within
their original scope, including the negative-gain pair. No five-pair speedup,
confirmation eligibility, RSS acceptance or application acceptance is claimed.
All numerical and primary-performance thresholds remain unchanged.

## Machine and measurement conditions
{machine_build}
## Candidate and redundant split rationale

Each paired projection dispatches the same activation twice, against high and
low weight halves. The high-weight half already validates the FP32 input and
forms its BF16 residual. The candidate low-weight half borrows this same
host-owned residual through its two existing dispatches instead of repeating
the split/validation loop. The first validation, four GEMM products, three
output copies and FP32 `(a+b)+(c+d)` grouping remain unchanged. The opt-in stays
OFF by default. Product/sum arithmetic and workspace ownership are unchanged;
there is no separate allocation-strategy refactor.

The default-OFF candidate is committed as c5d20c85; host-fold profiling evidence is committed
as ad867144. The narrow production changes are
[the prepared-input fold helper](../../../src/open_qwen36/prefill_precision.hpp:51),
[the Core opt-in and call site](../../../src/open_qwen36/core.cpp:622), and
[the default-false member](../../../src/open_qwen36/core.hpp:648).
The environment control is classified in
[the registry generator](../../../utilities/generate-registry.py).
The candidate changes duplicate host work, not product inputs, output layout,
dispatch order, grouped FP32 rounding, kernel code or production defaults.

## Accepted full-depth correctness scope
{correctness}
## Distinct-prompt reset/isolation prerequisite

The inspected native `run_list` takes one comma-separated prompt per line via
`--ids-list`, writes to an existing `--list-out` directory, and calls `Core.reset`
before each request. It writes complete real-vocabulary prompt logits at every
position and 17 generated IDs per request. The prescribed fixtures are a
20-token A (seed 26026), a different 13-token B (seed 26027), then A again.
For both OFF and ON the tooling requires p0/p2 complete logits and generated-ID
file hashes to match fresh A, plus complete OFF/ON outputs and runtime identity.
List execution does not expose per-request state dumps; those remain unmeasured.

Only OFF fresh A is attempted, twice, in separate new directories. Both processes
return 1 at hardware-context creation before residency or output. Their actual
argv/environment, token-list inputs, pre-launch artifact snapshots and logs are
retained, not reconstructed into successful capture metadata. No successful live
runtime mapping, RSS/HWM, logits or generated IDs is claimed for these failures.
The initial tooling version is archived outside the raw directories. The JSON
links both attempts and device/kernel-journal diagnostics. No A/B/A pass marker
exists, and the timing entry point refuses to run without that marker.

## Retained timing observations — not the initial five-pair result

Measurements below are the original single-run observations on the machine above,
not new measurements. The primary metric is outer CLI prefill wall, including
profiling and wrapper work, with no subtraction. RSS/HWM are whole-process peaks
sampled at nominal 20 ms, not phase-specific allocation measurements.

| Pair | Arm | Selection | Prefill ms | Fold split ms | Fold total ms | RSS MiB | HWM MiB |
|---|---|---|---:|---:|---:|---:|---:|
{table}

Retained paired reductions: {', '.join(f'{g:.6f}%' for g in observation['paired_percent_reductions'])}.
Their descriptive two-pair median is {observation['paired_median_percent']:.6f}%,
spread {observation['spread_percentage_points']:.6f} percentage points and
{observation['wins']}/2 wins. These do not establish the initial-five target or
authorize confirmation. Decode before/after samples are **NOT MEASURED**.

The explicit selection in JSON retains original pairs 0/1 and reserves exactly
three fresh pairs 2/3/4 under resume-prefill256 names, after one excluded restart
warmup. Original pair 2 OFF remains an excluded unpaired observation. Original
pair 2 ON remains invalid: final process metadata is absent and is never
reconstructed. Its valid timer inequality does not replace missing metadata.
The original warmup, failed raw files, selection and conditions remain unchanged;
the pre-continuation inventory verifies their bytes. Archived report snapshots
under Models preserve the evidence available before continuation.

Five pairs each of 512/1024 prefill and 32-prompt plus 32 generated decode steps,
each after one excluded scenario warmup, remain pending behind correctness.
Around-4K decode is **NOT MEASURED**: genuinely populated context has additional
cost and is not practical while basic hardware-context creation is blocked.
No seek or empty-context substitute is used. No confirmation is collected.

## Structural timer validation

Core request timing ends before profile serialization, fold flush and local
deallocation. CLI prefill wall also includes printing/setup. Inner totals must
therefore fit inside outer wall with only the printed rounding allowance:
sum(inner) - outer <= 0.5 + 0.0005*N + 1e-9 ms. There is no upper bound on
legitimate excluded wrapper work. The integer outer-ms and outer-ms/token
rounding intervals must intersect. Block geometry lines must be after residency
and before the prefill boundary. Exact log/profile timing agreement, layer sums,
finite checks, labels, request/fold counts and generation checks are preserved.
These are scope checks, not relaxed numerical or performance thresholds.

### Timer scope and unusable observation

The retained invalid ON attempt has inner total 29490.778 ms and outer 29493 ms.
Excluded serialization/flush/deallocation/printing work has no 2 ms upper bound,
so symmetric timer equality is not a valid scope requirement. The attempt is
unusable because final process metadata is absent. A valid timer inequality does
not replace missing capture metadata; this observation remains excluded.

## Application/server and verification

Application startup and server inference remain **NOT MEASURED / unmet**. The
prior Docker configure attempts encounter read-only SentencePiece/abseil
dependencies; their original command/log records remain linked in JSON.
No retained build container is found in the read-only container inventory.
The current hardware prerequisite prevents promotion regardless of application
build success, so no unrelated dependency modifications are attempted.
Full 64-layer CLI correctness is scoped evidence, not full-application acceptance.

Pre-hardware focused CPU tests: 322 passed. Final focused result:
{record['tests']['focused']['summary']}. Final portable result:
{record['tests']['portable']['summary']}. Skips are not passes. Settings CI:
{record['tests']['settings_ci']['status']}. Only the clean committed-tree check
counts as a pass; the retained dirty-tree refusal is not acceptance evidence.
Exact test logs, full commands/environments, build IDs, runtime hashes, original
capture references and continuation tool hashes are in JSON and retained local
evidence. The candidate commit c5d20c85 and host-fold profiling commit ad867144 identify
the production and profiling snapshots independently of acceptance tooling. No
further production optimization or default promotion is included in this gate.
''')
    print(report)


def paired_observations(pairs):
    """Descriptive observations only; never substitutes for the fixed five pairs."""
    gains = [100 * (off - on) / off for off, on in pairs]
    return dict(paired_percent_reductions=gains,
                paired_median_percent=statistics.median(gains) if gains else None,
                spread_percentage_points=max(gains) - min(gains) if gains else None,
                wins=sum(g > 0 for g in gains), pairs=len(gains),
                complete_initial_set=len(gains) == 5)


def reference(path):
    path = Path(path)
    return dict(path=str(path.absolute()), sha256=E.sha256(path), bytes=path.stat().st_size)


def generate(base, report):
    base, report = base.absolute(), report.absolute()
    captures, rows = [], []
    correctness = []
    for path in sorted((base / 'correctness').glob('*/capture.json')):
        record = E.read_json(path)
        E.require(record['before'] == record['after'], 'correctness inputs changed')
        E.validate_runtime(record['runtime'])
        for name, digest in record['raw_hashes'].items():
            E.require(E.sha256(path.parent / name) == digest, 'correctness raw hash changed')
        for name, entry in record['binaries'].items():
            E.require(E.sha256(path.parent / name) == entry['sha256'], 'dump changed')
        correctness.append(dict(capture=reference(path), invocation=reference(path.parent / 'invocation.json'),
                                raw_files=[reference(p) for p in sorted(path.parent.iterdir()) if p.is_file()]))
    for directory in sorted((base / 'campaign').glob('prefill-*')):
        if not directory.is_dir():
            continue
        path = directory / 'capture.json'
        refs = [reference(p) for p in sorted(directory.iterdir()) if p.is_file()]
        if not path.exists():
            captures.append(dict(name=directory.name, valid=False, raw_files=refs,
                reason='prefill wall contradicts profile; process result not serialized before validation raised'))
            continue
        record = E.read_json(path)
        audit = E.audit_capture(directory)
        E.require(audit['valid'], str(audit['errors']))
        fold = [r for r in E.jsonl(directory / 'fold.jsonl') if r['record'] == 'fold']
        rep = int(directory.name.split('-')[-2])
        if '--1-' in directory.name:
            rep = -1
        row = dict(name=directory.name, rep=rep, warmup=rep == -1,
                   arm=directory.name.split('-')[-1], prefill_ms=record['prompt_wall_ms'],
                   rss_bytes=record['sampled_rss_peak_bytes'], hwm_bytes=record['sampled_hwm_peak_bytes'],
                   fold_ms={k: sum(r[k] for r in fold) for k in E.FOLD_MS},
                   capture=reference(path), invocation=reference(directory / 'invocation.json'))
        rows.append(row)
        captures.append(dict(**row, valid=True, audit=audit, raw_files=refs))
    pairs = []
    for rep in range(5):
        selected = {r['arm']: r for r in rows if r['rep'] == rep}
        if set(selected) == {'off', 'on'}:
            pairs.append((selected['off']['prefill_ms'], selected['on']['prefill_ms']))
    observation = paired_observations(pairs)
    complete_reps = {r for r in range(5) if len([x for x in rows if x['rep'] == r]) == 2}
    observation['arm_medians'] = {}
    for arm in ('off', 'on'):
        selected = [r for r in rows if r['rep'] in complete_reps and r['arm'] == arm]
        observation['arm_medians'][arm] = {
            key: statistics.median(r[key] for r in selected)
            for key in ('prefill_ms', 'rss_bytes', 'hwm_bytes')}
        observation['arm_medians'][arm]['fold_ms'] = {
            key: statistics.median(r['fold_ms'][key] for r in selected) for key in E.FOLD_MS}
    E.require(not observation['complete_initial_set'], 'this stopped-attempt report requires incomplete campaign')
    policy = E.read_json(base / 'campaign/policy.json')
    focused = re.search(r'(\d+) passed', (base / 'focused-tests-report.log').read_text())[1]
    portable = re.search(r'(\d+) passed, (\d+) skipped', (base / 'all-tests-report.log').read_text())
    runtime = E.read_json(base / 'campaign/prefill-256-0-off/capture.json')['runtime']['before']
    record = dict(schema='pr4b-host-fold-dedup-stopped-attempt-v1', decision='KEEP DRAFT',
        model_acceptance=False, campaign_complete=False,
        reason='Unchanged capture audit rejected initial measured pair 2 ON; no replacement or further timing',
        protocol=reference(C.ROOT / 'specs/open-engine/reports/qwen38-host-fold-dedup-plan.md'),
        policy=policy, mapped_runtime=runtime, correctness=correctness,
        correctness_result=E.read_json(base / 'correctness-passed.json'),
        builds={arm: {name: reference(base / arm / name) for name in
                      ('identity.json', 'before.json', 'after.json', 'build.log', 'sources.tar')}
                for arm in ('control', 'candidate')},
        tests=dict(focused_passed=int(focused), portable_passed=int(portable[1]),
                   portable_skipped=int(portable[2]), settings_ci='BLOCKED: uncommitted tooling'),
        timing_captures=captures, descriptive_only=observation,
        unmeasured=['complete five-pair primary gate', '512 prefill', '1024 prefill',
                    '32-prompt plus 32-step decode', 'around-4K populated-context decode',
                    'application/server inference', 'per-repeat state dumps', 'distinct-request isolation'],
        supporting_files=[reference(p) for p in sorted(base.glob('*.log'))
                          if p.name != 'report-generation.log'] +
            [reference(p) for p in sorted((base / 'campaign/conditions').glob('*.json'))] +
            [reference(base / 'campaign/conditions-relocation.json')] +
            [reference(p) for p in sorted((base / 'application').glob('*.json'))] +
            [reference(p) for p in sorted((base / 'application').glob('*.log'))] +
            [reference(p) for p in sorted((base / 'tool-snapshots').iterdir()) if p.is_file()])
    C.write_json(report.with_suffix('.json'), record)
    table = '\n'.join(f"| {r['rep'] if not r['warmup'] else 'warmup'} | {r['arm']} | "
        f"{r['prefill_ms']:.0f} | {r['fold_ms']['split_ms']:.3f} | {r['fold_ms']['total_ms']:.3f} | "
        f"{r['rss_bytes']/1024**2:.3f} | {r['hwm_bytes']/1024**2:.3f} |" for r in rows)
    report.write_text(f'''# Bounded duplicate-split hardware experiment

## Decision: KEEP DRAFT

The bounded attempt stops at an unchanged evidence audit failure. The full initial
five-pair primary gate is **NOT MEASURED**. No replacement samples, confirmation,
criterion relaxation, further optimization, default promotion, commit, branch
switch, rebase, push or merge occurs. This is not a candidate acceptance claim.
The fixed target remains a >=1.00% median of paired 256-token wall reductions,
four of five wins, bounded variability, the specified other-scenario regression
and RSS limits, and complete numerical/application integration gates.

## Machine and measurement conditions

2026-11-23; EVO14-AI9-STP, AMD Ryzen AI 9 365, Strix aie2p 6x8, 20 logical
CPUs, 97,889,218,560 bytes RAM; Ubuntu 26.04.1, Linux 7.0.0-44-generic,
glibc 2.43. XRT and amdxdna runtime 2.26.0, NPU firmware 1.1.2.64.
Live core/coreutil/xdna mappings are 2.26.0, not inferred from an older CLI label;
their exact paths/hashes appear in the JSON and each successful capture.

Power policy is predeclared `none unchanged`; NPU governor control is unavailable
and **unverifiable**. Observed CPU governor is powersave. OpenMP has eight threads
with PASSIVE waiting; BLAS has one. All timing arms use identical layer profiling
and both prefill arms use direct fold profiling. Primary wall observations include
profiling. No synthetic transfer or utilization inference is used.
Initial load averages are 2.12/2.14/1.91 and thermal readings are 66 C; individual
successful captures retain before/after load, temperature and memory conditions.
Conditions sidecars live outside the sealed raw directories. A relocation manifest
preserves their original/new paths and unchanged content hashes; capture metadata
and raw inventories are not rewritten.
No quiet-machine guarantee is made. Tests and application-build attempts finish
before timing. Every run holds the exclusive model NPU lock, checks >=35 GiB
available before load and requires no existing accelerator client.

## Immutable native builds

Control revision ad8671440c9c3ab3671a00aa1cf9bfe962a210dc and candidate revision
c5d20c85defced034d8f2912953f10a2acf97969 are fresh git archives under
the control and candidate archive paths listed in the JSON report. Each archive's own native
build helper compiles the CLI. The known per-unit AVX2/FMA and block-host
`-ffp-contract=off` flags are unchanged. Source files are read-only after build.
Identity version 1 records full revision, executable digest, argv, nonempty
absolute source/build-input inventories, compiler version, XRT headers/libraries,
and equal before/after input snapshots. Native execution uses installed XRT.
Existing profiling binaries/evidence and production kernels are untouched.

The model is the existing Qwen3.8-27B-NPU2 and the kernel tree is the profiling
95-file production-equivalent tree excluding self-loop. Per-run snapshots bind
model/kernel contents, identities and executable; mappings attest live libraries.
The capture helper used for correctness is preserved by its exact recorded hash
in `tool-snapshots/correctness-pipeline_capture.py`. Timing adds only a locked
launch-preflight callback; both helper versions are retained. Raw metadata is not
rewritten to point at the preserved helper.

## Hardware correctness before timing

All 13 full-64-layer captures pass their scoped comparisons. Real prompt lengths
255, 256 and 257 each use max-tokens 17. Fresh candidate OFF/ON complete captured
logit buffers are byte-exact, including the padded vocabulary (248320 floats),
for the final prompt position and all 16 decode positions; all 17 greedy IDs match.
Candidate OFF also matches the original archived control at 256 tokens.

For each boundary and arm, a separate resident repeat-2 process reproduces both
greedy trajectories. The final repeated logits are byte-exact to fresh logits.
The existing CLI checkpoints/restores before the second request, whose request
path resets state. Dumps reuse logit filenames: only final-repeat logit files
remain, and no separate first-repeat logit retention is claimed.

State samples use manifest layout: layers 0 and 62 include the entire 3,317,760
byte linear allocation (FP32 convolution history followed by recurrence storage,
state_s_off 122880); full-attention layers 3 and 63 include all populated rows
from offset zero, row size 8192, through prompt+16 positions. State samples match
byte-exact OFF/ON and fresh/repeat processes. State dumps occur only after the
first request, not after each repeat. Per-repeat state equality, distinct-prompt
request isolation, every-layer state and independent checkpoint correctness are
**NOT MEASURED**. Thus this is a scoped reset check, not full serving acceptance.
All raw buffers and their sizes/digests are retained in the JSON inventory.

## Timing observations: incomplete, not an acceptance dataset

One warmup pair is excluded. Only measured pairs 0 and 1 are complete. Pair 2
OFF is retained as an unpaired observation. Pair 2 ON completes inference but is
rejected by the existing capture audit: profile total 29490.778 ms versus raw
prefill wall 29493 ms, a 2.222 ms difference above its unchanged 2 ms bound.
Its raw log/profiles/invocation remain, but capture.json, sampled RSS and runtime
observations were not serialized because validation raised. They are not rebuilt
or inferred from other runs. Pair 2 ON is not counted as valid evidence.

All rows below are single-run 256-token prefill observations on the machine above.
RSS/HWM are whole-process sampled peaks, not phase-specific memory or an exact
kernel-accounted maximum. VmRSS/VmHWM are sampled at nominal 20 ms intervals.

| Pair | Arm | Prefill ms | Fold split ms | Fold total ms | Sampled RSS MiB | Sampled HWM MiB |
|---|---|---:|---:|---:|---:|---:|
{table}

For the **two complete measured pairs only**, reductions are
{', '.join(f'{g:.6f}%' for g in observation['paired_percent_reductions'])};
descriptive paired median {observation['paired_median_percent']:.6f}%,
range {observation['spread_percentage_points']:.6f} percentage points,
{observation['wins']}/2 improving. These are not the fixed initial five-pair result
and do not authorize confirmation or a decision based on a difference of medians.
Descriptive OFF/ON arm medians over only those two pairs: prefill
{observation['arm_medians']['off']['prefill_ms']:.3f}/
{observation['arm_medians']['on']['prefill_ms']:.3f} ms, fold total
{observation['arm_medians']['off']['fold_ms']['total_ms']:.3f}/
{observation['arm_medians']['on']['fold_ms']['total_ms']:.3f} ms,
sampled RSS {observation['arm_medians']['off']['rss_bytes']/1024**2:.3f}/
{observation['arm_medians']['on']['rss_bytes']/1024**2:.3f} MiB. These arm medians
are not used to calculate the primary gain.
Direct fold improvement alone is not production-wall acceptance. No conclusion
on the fixed RSS regression gate is possible without the complete campaign.

512/1024-token prefill and 32-prompt+32-generated decode are **NOT MEASURED**:
the evidence gate stops the campaign before them. Around-4K decode is also
**NOT MEASURED**, because this stopped bounded experiment never reaches the
optional genuinely populated long-context case. No seek/empty context is used.

## Application integration and tests

The available image is openflowlm-build:ubuntu26 at
sha256:8cde355cca6838a2d881c79e50e7a660847a30a6cd4c8fdae9b3b5bf9f28038d.
A separate candidate archive is configured with read-only current submodule
dependencies and no accelerator device. The versioned configure attempt is
blocked by SentencePiece FetchContent trying to modify the read-only abseil-cpp
dependency directory. Application build/startup and actual server inference are
**NOT MEASURED**; CLI-only results do not satisfy those gates. Both configure
attempt logs and command argv are preserved.

Focused behavioural tests: {focused} passed, including explicit candidate sanitation,
default-OFF opt-in, RSS observations and compiled candidate arithmetic tests.
Portable suite: {portable[1]} passed, {portable[2]} skipped; final test log is
referenced in JSON. Skips are not passes. The merged-target settings CI wrapper refuses
the uncommitted tooling changes; it is **NOT RUN**, with its refusal retained.
No commit or stash is made to bypass it.

## Reproduction and evidence

Build commands (fresh output directories required):

```sh
python3 utilities/prefill-block/fold_dedup_build.py --revision ad867144 --out "$BUILD_ROOT/control"
python3 utilities/prefill-block/fold_dedup_build.py --revision c5d20c85 --out "$BUILD_ROOT/candidate"
ironvenv/bin/python utilities/prefill-block/fold_dedup_experiment.py correctness
ironvenv/bin/python utilities/prefill-block/fold_dedup_experiment.py timing
```

Do not rerun over these evidence directories. The JSON accompanying this report
contains exact per-capture absolute references, SHA-256 digests, full raw-file
inventories, attestation references, observed runtime hashes, policy and descriptive
individual observations. Each invocation retains complete argv/environment; raw
metadata is unmodified. No further samples or optimization are performed.
''')
    print(report)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    a = ap.parse_args()
    if (a.base / 'continuation-v1').exists():
        blocked_continuation(a.base, a.report)
    else:
        generate(a.base, a.report)
