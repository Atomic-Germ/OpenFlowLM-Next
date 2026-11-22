"""Check the measured draft summary; optionally audit all retained local evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'specs/open-engine/reports/qwen38-q4-gemm-dma.json'


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def positive(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def check_summary(record: dict) -> None:
    method = record['methodology']
    require(method['pairs'] >= 5 and method['warmup_dispatches_per_process'] >= 1 and
            method['measured_dispatches_per_process'] >= 1, 'insufficient repetitions')
    require(method['rel_fro_gate'] == 5e-3 and method['requires_exact_output'] is True,
            'numerical gate or exact-output requirement changed')
    require(record['production_default_dma_batch'] == 1 and record['end_to_end_acceptance'] is False,
            'draft must not claim production promotion or end-to-end acceptance')
    names = [case['name'] for case in record['cases']]
    require(len(names) == len(set(names)) and bool(names), 'duplicate or absent projection cases')
    by_name = {case['name']: case for case in record['cases']}
    for case in record['cases']:
        match = re.fullmatch(r'gemm_n(\d+)_k(\d+)(_low)?', case['name'])
        require(match is not None, 'unrecognized projection shape')
        n, k = map(int, match.group(1, 2))
        sibling = case['name'][:-4] if match[3] else case['name'] + '_low'
        require(sibling in names, 'missing high/low companion')
        reference = 'reference-low.bin' if match[3] else 'reference-high.bin'
        require(set(case['fixtures']) == {'weights.bin', 'activations.bin', reference},
                'incomplete fixture identities')
        require(all(case['fixtures'][key] == by_name[sibling]['fixtures'].get(key)
                    for key in ('weights.bin', 'activations.bin')), 'companions use different inputs')
        require(case['complete'] is True and case['exact_outputs'] is True, 'incomplete numerical case')
        require(math.isfinite(case['rel_fro']) and 0 <= case['rel_fro'] <= method['rel_fro_gate'],
                'invalid numerical result')
        require(0 <= case['summary']['paired_wins'] <= method['pairs'], 'invalid paired-win count')
        for arm, batch in [('before', 1), ('after', 4)]:
            timing, build = case['summary'][arm], case['builds'][arm]
            require(set(build['binaries']) == {'final.xclbin', 'insts.bin', 'insts.elf'},
                    'incomplete binary identities')
            require(all(positive(timing[key]) for key in ('min_ms', 'median_ms', 'max_ms')) and
                    timing['min_ms'] <= timing['median_ms'] <= timing['max_ms'], 'invalid timing range')
            require(build['dma_batch'] == batch and 0 < build['max_text'] < 16384 and
                    0 < build['max_l1_buffer_end'] <= 65536, 'invalid build variant or resource budget')
            expected = 2 * n * k * method['physical_tokens'] / (timing['median_ms'] * 1e9)
            require(math.isclose(case['effective_tflops'][arm], expected, rel_tol=1e-12),
                    'throughput does not match measured latency')
        expected = 100 * (case['summary']['after']['median_ms'] / case['summary']['before']['median_ms'] - 1)
        require(math.isclose(case['summary']['change_percent'], expected, abs_tol=1e-12),
                'percentage does not match measured medians')
        identities = [case['output_sha256'], case['raw_result_sha256'], case['harness_sha256'],
                      *case['fixtures'].values()]
        for build in case['builds'].values():
            identities.extend([build['record_sha256'], *build['binaries'].values()])
        require(all(isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v) for v in identities),
                'invalid SHA-256 identity')


def check_local(record: dict, root: Path) -> dict:
    spec = importlib.util.spec_from_file_location('q4_gemm_comparison_audit', Path(__file__).with_name('compare.py'))
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    checked = {}

    def verify(path: Path, expected: str) -> None:
        path = path if path.is_absolute() else root / path
        if path not in checked:
            with path.open('rb') as stream:
                checked[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
        require(checked[path] == expected, f'evidence hash mismatch: {path}')

    for path, identity in record['sources'].items():
        verify(Path(path), identity)
    dispatches = 0
    for case in record['cases']:
        path = root / case['raw_result']
        verify(path, case['raw_result_sha256'])
        raw = json.loads(path.read_text())
        require(raw['complete'] is True and raw['exact_outputs'] is True, 'incomplete raw comparison')
        require(raw['summary'] == comparison.summary(raw['pairs']) == case['summary'],
                'raw and reported summary differ')
        require(len(raw['pairs']) == record['methodology']['pairs'], 'raw repetition count differs')
        for file, identity in raw['inputs'].items():
            verify(Path(file), identity)
        method = record['methodology']
        for index, pair in enumerate(raw['pairs']):
            order = ['before', 'after'] if index % 2 == 0 else ['after', 'before']
            require(pair['index'] == index and pair['order'] == order, 'raw alternating order differs')
            for arm in order:
                row = pair[arm]
                require(len(row['warmup_ms']) == method['warmup_dispatches_per_process'] and
                        len(row['measured_ms']) == method['measured_dispatches_per_process'],
                        'raw dispatch count differs')
                cfg = Path(row['command'][1])
                samples = comparison.timings(cfg.with_suffix('.log').read_text(),
                                             len(row['warmup_ms']) + len(row['measured_ms']))
                require(samples == row['warmup_ms'] + row['measured_ms'], 'raw log timings differ')
                require(row['output']['sha256'] == case['output_sha256'] and
                        row['output']['rel_fro'] == case['rel_fro'], 'raw output identity or error differs')
                verify(cfg.with_suffix('.bin'), case['output_sha256'])
                dispatches += len(samples)
        cores = {}
        for arm in ('before', 'after'):
            item = case['builds'][arm]
            path = root / item['record']
            verify(path, item['record_sha256'])
            build = json.loads(path.read_text())
            require(build['complete'] is True and build['dma_batch'] == item['dma_batch'],
                    'incomplete or wrong raw build')
            for key in ('sources', 'generated', 'compiler_dependencies', 'binaries'):
                for file, identity in build[key].items():
                    verify(Path(file), identity)
            for file, identity in build['resources']['evidence'].items():
                verify(Path(file), identity)
            resources = build['resources']
            require(len(resources['text']) == len(resources['l1_buffer_end']) == 32 and
                    max(resources['text'].values()) == item['max_text'] and
                    max(resources['l1_buffer_end'].values()) == item['max_l1_buffer_end'],
                    'incomplete or inconsistent per-core resources')
            require({Path(file).name: identity for file, identity in build['binaries'].items()} == item['binaries'],
                    'raw and reported binaries differ')
            cores[arm] = {Path(file).name: identity for file, identity in resources['evidence'].items()
                          if file.endswith('.elf')}
        require(len(cores['before']) == 32 and cores['before'] == cores['after'], 'compute-core images differ')
        for name, identity in case['fixtures'].items():
            require(any(Path(file).name == name and value == identity for file, value in raw['inputs'].items()),
                    'reported fixture identity differs')
        require(any(Path(file).name == 'run_kernel' and value == case['harness_sha256']
                    for file, value in raw['inputs'].items()), 'reported harness identity differs')
    return dict(files_verified=len(checked), dispatches_verified=dispatches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--local-evidence', action='store_true', help='require and verify retained files, logs and binaries')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    check_summary(data)
    result = dict(summary='PASS', local_evidence='NOT CHECKED')
    if args.local_evidence:
        result.update(local_evidence='PASS', **check_local(data, args.root.resolve()))
    print(json.dumps(result))
