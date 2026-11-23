"""Fail-closed host-fold profiling evidence validation; never rewrites raw captures or reports.

Build identity is a caller-trusted build-time attestation, not a git HEAD guess:
version=1, executable_sha256, source_revision, build_command (argv), source_files
and build_files (nonempty absolute-path -> sha256 mappings). Source/build files
must still be available, e.g. in an archived build tree, and match the attestation.
The caller is responsible for its authenticity and completeness; hashes cannot
prove that a compiler consumed the attested inputs.

Run this module with capture directories to print independent audits to stdout.
Legacy records remain invalid even when their currently available files match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


class EvidenceError(ValueError):
    """Evidence is missing, inconsistent, or no longer verifiable."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def sha256(path):
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def fingerprint(path):
    path = Path(path).absolute()
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    require(resolved.is_file() and before.st_size > 0, f'missing/empty artifact: {path}')
    digest = sha256(resolved)
    after = resolved.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f'artifact changed while hashing: {path}')
    return dict(path=str(path), resolved=str(resolved), sha256=digest,
                bytes=after.st_size, device=after.st_dev, inode=after.st_ino,
                mtime_ns=after.st_mtime_ns)


def snapshot(paths):
    return {str(Path(p).absolute()): fingerprint(p) for p in sorted(set(map(str, paths)))}


def tree_files(path):
    path = Path(path).absolute()
    if path.is_file():
        return [path]
    require(path.is_dir(), f'missing artifact directory: {path}')
    def walk(directory, ancestors):
        resolved = directory.resolve(strict=True)
        require(resolved not in ancestors, f'cyclic artifact directory: {directory}')
        for child in sorted(directory.iterdir()):
            if child.is_symlink():
                child.resolve(strict=True)
            if child.is_dir():
                yield from walk(child, ancestors | {resolved})
            else:
                require(child.is_file(), f'nonregular artifact: {child}')
                yield child

    files = list(walk(path, set()))
    require(files, f'empty artifact directory: {path}')
    return files


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def loads(text):
    try:
        value = json.loads(text, object_pairs_hook=_object,
                           parse_constant=lambda x: (_ for _ in ()).throw(
                               EvidenceError(f'nonfinite JSON constant: {x}')))
    except (ValueError, TypeError) as exc:
        raise EvidenceError(f'malformed JSON: {exc}') from exc
    require(isinstance(value, dict), 'JSON record must be an object')

    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'nonfinite JSON number')
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def read_json(path):
    return loads(Path(path).read_text())


def jsonl(path):
    text = Path(path).read_text()
    require(text and text.endswith('\n'), f'incomplete JSONL: {path}')
    rows = [loads(line) for line in text.splitlines() if line.strip()]
    require(rows, f'empty JSONL: {path}')
    return rows


def positive_int(value):
    return type(value) is int and value > 0


def timing(record, fields):
    require(all(key in record for key in fields), 'missing timing columns')
    for key, value in record.items():
        if key.endswith('_ms'):
            require(type(value) in (int, float) and math.isfinite(value) and
                    (value >= 0 or key == 'residue_ms'), f'invalid timing: {key}')


REQUEST_MS = ('total_ms', 'gemm_ms', 'host_ms', 'per_token_ms', 'lmhead_ms',
              'setup_ms', 'embed_ms', 'patch_ms', 'dispatch_ms', 'layer_wall_ms', 'residue_ms')
LAYER_MS = ('wall_ms', 'total_ms', 'gemm_ms', 'host_ms', 'per_token_ms', 'lmhead_ms',
            'mid_ms', 'dn_conv_ms', 'dn_rule_ms', 'attn_ms', 'prenorm_ms', 'gemm_tile_ms',
            'gemm_tr_ms', 'tail_ms', 'state_ms', 'sync_ms', 'shared_ms', 'setup_ms',
            'qkv_scatter_ms', 'moe_prep_ms', 'moe_patch_ms', 'moe_run_ms', 'moe_read_ms',
            'dispatch_ms')
FOLD_MS = ('split_ms', 'copy_ms', 'accumulate_ms', 'total_ms')


def manifest_info(manifest, layers):
    spec = manifest.get('spec', {})
    require(isinstance(spec, dict), 'invalid manifest spec')
    kinds = spec.get('layer_types')
    require(positive_int(layers) and positive_int(spec.get('num_layers')) and
            spec['num_layers'] == layers and
            isinstance(kinds, list) and len(kinds) == layers and
            all(isinstance(k, str) and k for k in kinds), 'invalid manifest layer geometry')
    family, spec_hash = manifest.get('family'), manifest.get('spec_hash')
    require(isinstance(family, str) and family and isinstance(spec_hash, str) and
            re.fullmatch(r'sha256:[0-9a-f]{64}', spec_hash), 'missing manifest family/spec identity')
    descriptions = manifest.get('layer_types')
    require(isinstance(descriptions, dict), 'missing manifest layer descriptions')
    widths = set()
    fold_calls = 0
    for kind in kinds:
        desc = descriptions.get(kind)
        require(isinstance(desc, dict) and isinstance(desc.get('gemm_block'), dict),
                'missing manifest block program')
        width = desc['gemm_block'].get('t')
        require(positive_int(width), 'invalid manifest block width')
        widths.add(width)
        block = desc['gemm_block']
        program, ffn = block.get('program'), block.get('ffn_program', [])
        require(isinstance(program, list) and program and isinstance(ffn, list),
                'invalid manifest projection program')
        programs = program + ffn
        require(all(isinstance(s, dict) for s in programs),
                'invalid manifest projection program')
        fold_calls += sum(s.get('op') == 'run' for s in programs)
    require(len(widths) == 1 and all(positive_int(t) for t in widths),
            'missing/inconsistent physical block geometry')
    return dict(family=family, spec_hash=spec_hash, layers=layers,
                kinds=kinds, block_tokens=widths.pop(), fold_calls_per_block=fold_calls)


def kernel_files(root, manifest):
    root = Path(root).absolute()
    contexts, kernels = manifest.get('contexts'), manifest.get('kernels')
    require(isinstance(contexts, dict) and contexts and isinstance(kernels, dict) and kernels,
            'missing kernel/instruction inventory')
    references = list(contexts.values())
    for desc in kernels.values():
        require(isinstance(desc, dict) and desc.get('context') in contexts and
                isinstance(desc.get('insts'), str), 'missing kernel instruction/context')
        references.append(desc['insts'])
    for name in references:
        require(isinstance(name, str) and name and not Path(name).is_absolute() and
                '..' not in Path(name).parts, 'invalid kernel artifact path')
        fingerprint(root / name)
    return tree_files(root)


def build_identity(path, executable):
    require(path is not None, 'caller-supplied build identity is required')
    identity = read_json(path)
    require(type(identity.get('version')) is int and identity['version'] == 1 and
            identity.get('executable_sha256') == fingerprint(executable)['sha256'],
            'build identity does not match executable')
    require(isinstance(identity.get('source_revision'), str) and
            identity['source_revision'].strip() not in ('', 'unknown', 'HEAD'),
            'missing trustworthy source revision')
    command = identity.get('build_command')
    require(isinstance(command, list) and command and
            all(isinstance(a, str) and a for a in command), 'missing build command argv')
    for group in ('source_files', 'build_files'):
        files = identity.get(group)
        require(isinstance(files, dict) and files, f'missing build identity {group}')
        for name, digest in files.items():
            require(Path(name).is_absolute() and isinstance(digest, str) and
                    re.fullmatch(r'[0-9a-f]{64}', digest), f'invalid attested {group}')
            require(fingerprint(name)['sha256'] == digest, f'attested input changed: {name}')
    return identity


def provenance_inputs(cli, identity_path, identity, model, kernels):
    """Inventory is re-derived on both sides, so added/deleted artifacts also fail."""
    model_files = tree_files(model)
    require(any(p.name == 'model.q4nx' or p.suffix == '.gguf' for p in model_files),
            'missing model weights')
    here = Path(__file__).resolve().parent
    manifest = read_json(Path(kernels) / 'manifest.json')
    return ([Path(cli), Path(identity_path), here / 'pipeline_capture.py', Path(__file__)] +
            list(map(Path, identity['source_files'])) + list(map(Path, identity['build_files'])) +
            model_files + kernel_files(kernels, manifest))


def validate_attested_snapshot(before, cli, identity_path, identity):
    require(read_json(identity_path) == identity, 'build identity changed before launch')
    require(before[str(Path(cli).absolute())]['sha256'] == identity['executable_sha256'],
            'snapshotted executable differs from build identity')
    for group in ('source_files', 'build_files'):
        for path, digest in identity[group].items():
            require(before[path]['sha256'] == digest, f'snapshotted attested input changed: {path}')


def validate_profile(path, info, phase, tokens, max_tokens, route):
    require(phase in ('prefill', 'decode') and route in ('block', 'layer-major') and
            (phase == 'prefill' or route == 'block') and positive_int(tokens) and
            positive_int(max_tokens), 'invalid phase/token/route metadata')
    rows = jsonl(path)
    header = rows.pop(0)
    require(header.get('record') == 'header' and type(header.get('version')) is int and
            header['version'] == 1 and
            all(header.get(k) == info[k] for k in ('family', 'spec_hash', 'layers')),
            'missing/incorrect profile header')
    count = info['layers']
    requests = []
    while rows:
        request, rows = rows[0], rows[1:]
        require(request.get('record') == 'request' and type(request.get('layers')) is int and
                request['layers'] == count and
                positive_int(request.get('tokens')) and positive_int(request.get('blocks')),
                'invalid/misplaced request record')
        timing(request, REQUEST_MS)
        require(request['total_ms'] > 0 and request['lmhead_ms'] <= request['total_ms'],
                'invalid request/LM-head duration')
        group, rows = rows[:count], rows[count:]
        require(len(group) == count, 'incomplete layer coverage')
        for index, row in enumerate(group):
            require(row.get('record') == 'layer' and type(row.get('layer')) is int and
                    row['layer'] == index and row.get('what') == request.get('what') and
                    row.get('kind') == info['kinds'][index], 'mismatched/out-of-order layer')
            timing(row, LAYER_MS)
        require(abs(sum(r['wall_ms'] for r in group) - request['layer_wall_ms']) <=
                (count + 1) * .00051, 'layer wall sum contradicts request')
        require(abs(request['total_ms'] - request['layer_wall_ms'] - request['residue_ms'])
                <= .0016, 'request residue contradicts wall')
        requests.append(request)
    if phase == 'decode':
        expected = [('step', 1, 1)] * (max_tokens - 1)
    else:
        width = info['block_tokens']
        expected = ([('gemm-prompt', tokens, math.ceil(tokens / width))] if route == 'layer-major'
                    else [('gemm-block', min(width, tokens - at), 1)
                          for at in range(0, tokens, width)])
    require([(r.get('what'), r['tokens'], r['blocks']) for r in requests] == expected,
            'profile phase/token/block coverage mismatch')
    for index, request in enumerate(requests):
        want = phase == 'decode' or index == len(requests) - 1
        require((request['lmhead_ms'] > 0) == want, 'missing/unexpected LM head')
    return header, requests


def validate_fold(path, info, tokens, route):
    rows = jsonl(path)
    require(rows[0] == {'record': 'header', 'version': 1} and
            type(rows[0].get('version')) is int, 'invalid fold header')
    width = info['block_tokens']
    expected = ([(tokens, math.ceil(tokens / width) * width)] if route == 'layer-major' else
                [(min(width, tokens - at), width) for at in range(0, tokens, width)])
    require(len(rows) - 1 == len(expected), 'incomplete/extra fold schedule')
    for row, (real, physical) in zip(rows[1:], expected):
        require(row.get('record') == 'fold' and row.get('what') ==
                ('gemm-prompt' if route == 'layer-major' else 'gemm-block') and
                type(row.get('real_tokens')) is int and row['real_tokens'] == real and
                type(row.get('block_tokens')) is int and row['block_tokens'] == physical and
                positive_int(row.get('calls')) and
                row['calls'] == info['fold_calls_per_block'] * (physical // width),
                'missing/invalid fold labels or calls')
        timing(row, FOLD_MS)
        require(abs(sum(row[k] for k in FOLD_MS[:3]) - row['total_ms']) <= .0021,
                'fold total contradicts exclusive windows')
    return rows[1:]


NUMBER = r'(\d+(?:\.\d+)?)'
STEP = re.compile(r'\s*step @(\d+) \[(\w+)\]: ' + NUMBER + r' ms \(part0 ' +
                  NUMBER + r', route ' + NUMBER + r', part1 ' + NUMBER +
                  r', lm_head ' + NUMBER + r'; embed ' + NUMBER + r', patch ' +
                  NUMBER + r', dispatches ' + NUMBER + r'\)')
BLOCK = re.compile(r'\s*(gemm-block|gemm-prompt) \[(\d+),(\d+)\) t_real=(\d+): ' +
                   NUMBER + r' ms \(GEMM ' + NUMBER + r', host ' + NUMBER +
                   r', per-token ' + NUMBER + r', lm_head ' + NUMBER + r'\)')


def validate_log(text, phase, tokens, max_tokens, requests, info, route):
    lines = text.splitlines()
    resident = re.findall(r'resident after (\d+(?:\.\d+)?) s', text)
    prefill = re.findall(r'^prefill (\d+) tokens: (\d+) ms \((\d+) ms/token\)$', text, re.M)
    decode = re.findall(r'^decode (\d+) tokens: (\d+) ms/token \((\d+(?:\.\d+)?) tok/s\)$',
                        text, re.M)
    require(len(resident) == 1 and text.count('resident after ') == 1 and
            math.isfinite(float(resident[0])) and float(resident[0]) > 0 and len(prefill) == 1 and
            int(prefill[0][0]) == tokens and int(prefill[0][1]) > 0, 'missing/invalid phase lines')
    require(sum(line.startswith('prefill ') for line in lines) == 1 and
            sum(line.startswith('decode ') for line in lines) == len(decode), 'malformed phase lines')
    generated = [line for line in lines if line.startswith('token ')]
    require(len(generated) == max_tokens and all(re.fullmatch(r'token \d+', s) and
            0 <= int(s.split()[1]) < info['real_vocab'] for s in generated),
            'invalid/incomplete generated token trajectory')
    resident_at = text.index('resident after ')
    resident_line = next(i for i, line in enumerate(lines) if 'resident after ' in line)
    prefill_at = next(i for i, line in enumerate(lines) if line.startswith('prefill '))
    require(resident_at < text.index(lines[prefill_at]), 'load/prefill phase order mismatch')
    # Both summaries round the same outer duration independently to integer ms.
    wall, per_token = map(int, prefill[0][1:])
    slack = 1e-9
    require(max(wall - .5, tokens * (per_token - .5)) <=
            min(wall + .5, tokens * (per_token + .5)) + slack,
            'prefill wall/ms-per-token contradiction')
    events = ['token' if line.startswith('token ') else
              'step' if line.lstrip().startswith('step @') else
              'decode' if line.startswith('decode ') else None for line in lines[prefill_at + 1:]]
    events = [event for event in events if event]
    expected_events = [event for _ in range(max_tokens - 1) for event in ('token', 'step')]
    expected_events += ['token'] + (['decode'] if max_tokens > 1 else [])
    require(events == expected_events and
            not any(line.startswith(('token ', 'decode ')) or line.lstrip().startswith('step @')
                    for line in lines[:prefill_at]), 'generation phase order mismatch')
    steps = []
    blocks = []
    for line in lines:
        if line.lstrip().startswith('step @'):
            match = STEP.fullmatch(line)
            require(match is not None, 'malformed generated step')
            values = match.groups()
            row = dict(zip(('total_ms', 'gemm_ms', 'per_token_ms', 'host_ms', 'lmhead_ms',
                            'embed_ms', 'patch_ms', 'dispatch_ms'), map(float, values[2:])))
            timing(row, REQUEST_MS[:5])
            steps.append(dict(position=int(values[0]), route=values[1], **row))
        if line.lstrip().startswith(('gemm-block ', 'gemm-prompt ')):
            match = BLOCK.fullmatch(line)
            require(match is not None, 'malformed block phase')
            values = match.groups()
            row = dict(zip(('total_ms', 'gemm_ms', 'host_ms', 'per_token_ms', 'lmhead_ms'),
                           map(float, values[4:])))
            timing(row, REQUEST_MS[:5])
            blocks.append(dict(what=values[0], start=int(values[1]), end=int(values[2]),
                               tokens=int(values[3]), **row))
    require(len(steps) == max_tokens - 1 and
            [s['position'] for s in steps] == list(range(tokens, tokens + max_tokens - 1)) and
            all(s['route'] == 'npu' and s['lmhead_ms'] > 0 for s in steps),
            'generated step count/position/route/LM head mismatch')
    require((max_tokens == 1 and not decode) or
            (max_tokens > 1 and len(decode) == 1 and int(decode[0][0]) == max_tokens - 1 and
             int(decode[0][1]) > 0 and float(decode[0][2]) > 0), 'decode summary count mismatch')
    if decode:
        average, rate = float(decode[0][1]), float(decode[0][2])
        require(math.isfinite(average) and math.isfinite(rate) and
                sum(s['total_ms'] for s in steps) / len(steps) <= average + .551,
                'decode summary is shorter than generated steps')
        require(1000 / (rate + .005) <= average + .5 and
                (rate <= .005 or 1000 / (rate - .005) >= average - .5),
                'decode latency/rate contradiction')
    observed = steps if phase == 'decode' else blocks
    if phase == 'decode':
        require(not blocks, 'decode capture contains unexpected block schedule')
    require(all(resident_line < i < prefill_at for i, line in enumerate(lines)
                if line.lstrip().startswith(('gemm-block ', 'gemm-prompt '))),
            'prefill block outside resident/prefill boundaries')
    require(len(observed) == len(requests), 'log/profile request count mismatch')
    for log, request in zip(observed, requests):
        for key in ('total_ms', 'gemm_ms', 'host_ms', 'per_token_ms', 'lmhead_ms'):
            tolerance = .0056 if phase == 'decode' and key == 'per_token_ms' else .0506
            require(abs(log[key] - request[key]) <= tolerance,
                    f'log/profile contradiction: {key}')
        if phase == 'decode':
            for key, tolerance in (('dispatch_ms', .0506), ('embed_ms', .0056), ('patch_ms', .0056)):
                require(abs(log[key] - request[key]) <= tolerance,
                        f'log/profile contradiction: {key}')
    if phase == 'prefill':
        start = 0
        for block, request in zip(blocks, requests):
            end = tokens if route == 'layer-major' else start + info['block_tokens']
            require(block['what'] == request['what'] and block['tokens'] == request['tokens'] and
                    block['start'] == start and block['end'] == end, 'log block geometry mismatch')
            start += request['tokens']
        # Core stops before serialization, fold flush and deallocation; CLI also
        # includes printing/setup. Excluded wrapper work has no upper bound.
        require(math.fsum(r['total_ms'] for r in requests) - wall <=
                .5 + .0005 * len(requests) + slack,
                'prefill wall is shorter than profile')
    return dict(model_load_s=float(resident[0]), prompt_wall_ms=float(prefill[0][1]),
                generated_tokens=len(steps), generated_ids=[int(s.split()[1]) for s in generated],
                generated_ms_per_token=float(decode[0][1]) if decode else None,
                generated_steps=steps)


RUNTIME_NAMES = ('libxrt_core.so', 'libxrt_coreutil.so', 'libxrt_driver_xdna.so')


def validate_runtime(runtime, verify_files=True):
    require(isinstance(runtime, dict), 'invalid runtime provenance object')
    before, after = runtime.get('before', {}), runtime.get('after', {})
    require(isinstance(before, dict) and before and isinstance(after, dict) and before == after,
            'missing/changed live runtime provenance')
    for name in RUNTIME_NAMES:
        require(any(Path(p).name == name or Path(p).name.startswith(name + '.') for p in before),
                f'missing live runtime: {name}')
    require(positive_int(runtime.get('observed_pid')) and
            isinstance(runtime.get('mapping_samples'), list) and runtime['mapping_samples'],
            'runtime mappings were not sampled from the measured process')
    observed = set()
    for sample in runtime['mapping_samples']:
        require(isinstance(sample, dict), 'invalid live mapping sample object')
        require(type(sample.get('at_s')) in (int, float) and math.isfinite(sample['at_s']) and
                sample['at_s'] >= 0 and isinstance(sample.get('paths'), list) and sample['paths'] and
                all(isinstance(p, str) and Path(p).is_absolute() for p in sample['paths']),
                'invalid live mapping sample')
        observed.update(sample['paths'])
    require(observed == set(before), 'live mapping inventory contradicts runtime hashes')
    if verify_files:
        require(snapshot(before) == after, 'runtime artifact changed')


def audit_layer_profile(directory):
    """Inventory the older multi-route format without inventing phase or live maps."""
    record = read_json(directory / 'result.json')
    executable = record.get('cli')
    require(isinstance(executable, str) and executable, 'missing legacy executable path')
    provenance = record.get('provenance', {})
    require(isinstance(provenance, dict) and isinstance(provenance.get(executable, {}), dict),
            'invalid legacy binary provenance')
    binary = dict(path=executable, **provenance.get(executable, {}))
    raw_hashes = {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}
    result = dict(capture=str(directory), valid=False, format='legacy-layer-profile',
                  phase=record.get('phase'), real_prompt_tokens=record.get('prompt_tokens'),
                  binary=binary, runtime=record.get('runtime'), raw_hashes=raw_hashes,
                  source_build_identity=record.get('build_identity'),
                  recorded_artifact_provenance=provenance,
                  cpu_wait='NOT MEASURED', runs=[], errors=[
                      'legacy capture lacks verifiable per-capture snapshots',
                      'missing build identity', 'missing per-capture live runtime mappings'])
    if not result['phase']:
        result['errors'].append('phase was not recorded; directory names are not phase evidence')
    if Path(executable).is_file():
        result['binary_current_sha256'] = sha256(executable)
        if result['binary_current_sha256'] != binary.get('sha256'):
            result['errors'].append('recorded binary was replaced')
    else:
        result['errors'].append('recorded binary is unavailable')
    runs = record.get('runs')
    require(isinstance(runs, dict) and runs, 'missing legacy runs')
    for route, run in runs.items():
        require(isinstance(run, dict), 'invalid legacy run')
        entry = dict(capture=f'{directory}#{route}', route=route, binary=binary,
                     phase=run.get('phase'), real_prompt_tokens=record.get('prompt_tokens'),
                     command=run.get('command'), argv=run.get('argv'),
                     runtime=run.get('runtime'), raw_hashes=raw_hashes,
                     generated_ids=run.get('generated_ids'), profile_requests=[], errors=[])
        if not entry['command']:
            entry['errors'].append('missing recorded command; not reconstructed')
        if not entry['runtime']:
            entry['errors'].append('missing per-run live runtime mappings')
        try:
            require(isinstance(run.get('profile'), str), 'missing raw profile path')
            profile = directory / Path(run['profile']).name
            entry['profile_requests'] = [r for r in jsonl(profile) if r.get('record') == 'request']
            if any(r.get('what') == 'step' and r.get('lmhead_ms') == 0
                   for r in entry['profile_requests']):
                entry['errors'].append('raw step profile contains prompt steps without LM head; '
                                       'not generated-decode evidence')
        except (ValueError, OSError) as exc:
            entry['errors'].append(str(exc))
        result['runs'].append(entry)
    return result


def audit_capture(directory):
    """Inventory each capture as recorded; legacy gaps are errors, never filled in."""
    directory = Path(directory).absolute()
    result = dict(capture=str(directory), valid=False, errors=[], cpu_wait='NOT MEASURED')
    try:
        if not (directory / 'capture.json').exists() and (directory / 'result.json').is_file():
            return audit_layer_profile(directory)
        record = read_json(directory / 'capture.json')
        result.update(capture_metadata_sha256=sha256(directory / 'capture.json'),
                      version=record.get('version'), phase=record.get('phase'),
                      real_prompt_tokens=record.get('real_prompt_tokens'),
                      generated_tokens=record.get('generated_tokens'),
                      command=record.get('command'), argv=record.get('argv'),
                      binary=record.get('cli'), runtime=record.get('runtime'),
                      model=record.get('model'), kernels=record.get('kernels'),
                      source_build_identity=record.get('build_identity'),
                      recorded_raw_hashes=record.get('raw_hashes'))
        raw = [p for p in directory.iterdir() if p.is_file() and p.name != 'capture.json']
        result['raw_hashes'] = {p.name: sha256(p) for p in sorted(raw)}
        cli = record.get('cli', {})
        require(isinstance(cli, dict), 'invalid binary provenance object')
        if cli.get('path') and Path(cli['path']).is_file():
            result['binary_current_sha256'] = sha256(cli['path'])
            if result['binary_current_sha256'] != cli.get('sha256'):
                result['errors'].append('recorded binary was replaced')
        else:
            result['errors'].append('recorded binary is unavailable')
        if record.get('version') != 2:
            result['errors'].append('legacy capture lacks verifiable per-capture snapshots')
        if not record.get('build_identity'):
            result['errors'].append('missing build identity')
        fold = directory / 'fold.jsonl'
        if fold.exists():
            try:
                rows = jsonl(fold)
                require(rows[0] == {'record': 'header', 'version': 1}, 'invalid fold header')
                require(len(rows) > 1, 'empty fold evidence')
                for row in rows[1:]:
                    require(row.get('record') == 'fold' and
                            row.get('what') in ('gemm-block', 'gemm-prompt') and
                            all(positive_int(row.get(k)) for k in
                                ('real_tokens', 'block_tokens', 'calls')),
                            'legacy/raw fold has missing or invalid labels; cannot repair')
                    timing(row, FOLD_MS)
            except (ValueError, OSError) as exc:
                result['errors'].append(str(exc))
        if record.get('version') == 2:
            try:
                require(type(record.get('version')) is int and
                        record.get('tool') == 'prefill-block-pipeline-capture' and
                        type(record.get('fold_instrumentation')) is bool,
                        'invalid capture schema')
                require(record.get('raw_hashes') == result['raw_hashes'], 'raw hash inventory changed')
                before, after = record.get('before', {}), record.get('after', {})
                require(before and before == after and snapshot(before) == after,
                        'source/build/executable/model/kernel provenance changed')
                input_before, input_after = record.get('input_before', {}), record.get('input_after', {})
                require(set(input_before) == {str(directory / 'ids.txt'),
                                               str(directory / 'invocation.json')} and
                        input_before == input_after and snapshot(input_before) == input_after,
                        'missing/changed invocation/token input snapshots')
                identity = build_identity(record.get('build_identity_path'), cli['path'])
                require(identity == record['build_identity'], 'build identity changed')
                expected_inputs = provenance_inputs(cli['path'], record['build_identity_path'], identity,
                                                    record['model']['path'], record['kernels']['path'])
                require(set(before) == set(map(lambda p: str(p.absolute()), expected_inputs)),
                        'incomplete provenance inventory')
                validate_attested_snapshot(before, cli['path'], record['build_identity_path'], identity)
                validate_runtime(record.get('runtime', {}))
                argv = record.get('argv')
                require(isinstance(argv, list) and argv and argv[0] == cli['path'],
                        'missing/mismatched command argv')
                for flag, value in (('--model', record['model']['path']),
                                    ('--kernels', record['kernels']['path']),
                                    ('--ids-file', str(directory / 'ids.txt')),
                                    ('--max-tokens', str(record['max_tokens'])),
                                    ('--layers', str(record['layers']))):
                    require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv) and
                            argv[argv.index(flag) + 1] == value,
                            f'command/capture contradiction: {flag}')
                require(record['manifest_path'] == str(Path(record['kernels']['path']) / 'manifest.json'),
                        'manifest path contradicts kernel inventory')
                import shlex
                require(record.get('command') == shlex.join(argv), 'recorded command changed')
                invocation = read_json(directory / 'invocation.json')
                require(all(record.get(k) == invocation.get(k) for k in
                            ('environment', 'argv', 'command', 'before', 'build_identity', 'phase',
                             'max_tokens', 'cwd', 'sanitized_inherited_oflm')) and
                        invocation.get('tokens') == record['real_prompt_tokens'],
                        'invocation/environment changed')
                manifest = read_json(record['manifest_path'])
                info = manifest_info(manifest, record['layers'])
                physical = record.get('physical_block_tokens')
                require(record.get('manifest_block_tokens') == info['block_tokens'] and
                        type(record.get('manifest_block_tokens')) is int and
                        'physical_block_tokens' in record and
                        ((physical == info['block_tokens'] and type(physical) is int)
                         if record['phase'] == 'prefill' else physical is None) and
                        (not record['fold_instrumentation'] or record['phase'] == 'prefill'),
                        'contradictory capture block/fold metadata')
                info['real_vocab'] = manifest['spec']['real_vocab']
                _, requests = validate_profile(directory / 'layers.jsonl', info, record['phase'],
                                                record['real_prompt_tokens'], record['max_tokens'],
                                                record['route'])
                measured = validate_log((directory / 'run.log').read_text(), record['phase'],
                                        record['real_prompt_tokens'], record['max_tokens'], requests,
                                        info, record['route'])
                require(all(record.get(k) == v for k, v in measured.items()),
                        'capture summaries contradict raw evidence')
                if record['fold_instrumentation']:
                    validate_fold(fold, info, record['real_prompt_tokens'], record['route'])
                else:
                    require(not fold.exists(), 'unexpected fold evidence')
            except (ValueError, OSError, KeyError, TypeError) as exc:
                result['errors'].append(str(exc))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result['errors'].append(str(exc))
    result['valid'] = not result['errors']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path, nargs='+')
    args = parser.parse_args()
    audits = [audit_capture(path) for path in args.captures]
    print(json.dumps(audits, indent=2, allow_nan=False))
    raise SystemExit(0 if all(a['valid'] for a in audits) else 1)
