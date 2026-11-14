"""Audit full-depth paired-input A/B captures without rerunning the model."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
s = importlib.util.spec_from_file_location('_paired_prefill_base', ROOT / 'utilities/prefill-slice/check.py')
base = importlib.util.module_from_spec(s)
s.loader.exec_module(base)


def identity(before, after, tokens, routes=('sequential', 'hybrid', 'block_major')):
    return {name: base.digest(before / name) == base.digest(after / name)
            for route in routes for name in base.expected_dumps(route, tokens)}


def validate_run(directory):
    meta = json.loads((directory / 'reference.json').read_text())
    report = json.loads((directory / 'report.json').read_text())
    base.check_reference_coverage(meta)
    base.verify(meta['sources'])
    base.verify({str(directory / p): h for p, h in meta['references'].items()})
    if base.digest(directory / 'ids.txt') != meta['ids_sha256']:
        raise ValueError('changed prompt IDs')
    if report['reference_sha256'] != base.digest(directory / 'reference.json'):
        raise ValueError('changed reference metadata')
    if [r['route'] for r in report['routes']] != ['sequential', 'hybrid', 'block_major']:
        raise ValueError('incomplete route coverage')
    n = len(meta['ids'])
    offsets = base.layout(base.spec_from_model_dir(Path(meta['model'])), max_ctx=meta['max_ctx'])
    for r in report['routes']:
        route = r['route']
        names = base.expected_dumps(route, n)
        if set(r['outputs']) != set(names + [route + '-act.bin']):
            raise ValueError('incomplete native output inventory')
        base.verify({str(directory / name): sha for name, sha in r['outputs'].items()})
        log = directory / (route + '.log')
        if base.digest(log) != r['log_sha256']:
            raise ValueError('changed native log')
        rows = []
        for pos in range(n + 1):
            got = f'{route}_p{pos}.bin' if pos < n else f'{route}_t1.bin'
            ref = f'ref_p{pos}.bin' if pos < n else 'ref_cont_logits.bin'
            rows.append(dict(position=pos, **base.logit_score(
                base.read(directory / got, meta['vocab']),
                base.read(directory / ref, meta['vocab']), meta['real_vocab'])))
        if rows != r['logits']:
            raise ValueError('reported logits do not match captured arrays')
        base.check_tokens(log.read_text(), meta['expected_tokens'])
        alias = base.digest(directory / names[-2]) == base.digest(directory / names[n - 1])
        residual = base.final_residual((directory / (route + '-act.bin')).read_bytes(), offsets, meta['hidden'])
        ref = base.read(directory / f"ref_cont_res{meta['layers']-1}.bin", meta['hidden'])
        err = float(np.max(np.abs(residual - ref)) / max(float(np.max(np.abs(ref))), 1e-30))
        passed = alias and err < base.RESIDUAL_MAXREL and all(row['passed'] for row in rows)
        if (r['passed'] != passed or r['continuation_last_residual_maxrel'] != err
                or not r['tokens_match'] or r['last_prompt_alias'] != alias):
            raise ValueError('reported continuation gate differs from captures')
    pairs = {name: base.digest(directory / name) == base.digest(directory / name.replace('hybrid', 'block_major', 1))
             for name in base.expected_dumps('hybrid', n)}
    if pairs != report['hybrid_schedule_identity']:
        raise ValueError('hybrid schedule identity differs from captures')
    if report['passed'] != (all(r['passed'] for r in report['routes']) and all(pairs.values())):
        raise ValueError('inconsistent acceptance status')
    return meta, report


def audit(a):
    out = a.out.resolve()
    result = dict(passed=False, scope='private paired-input GEMM, full-depth short prompt and one continuation')
    try:
        bm, br = validate_run(a.baseline.resolve())
        reports = {}
        for variant in ('control', 'paired'):
            directory = out / variant
            build = json.loads((directory / 'build-evidence.json').read_text())
            if not build['complete'] or build['variant'] != variant:
                raise ValueError('incomplete or wrong native build')
            base.verify(build['sources'])
            base.verify({str(directory / p): h for p, h in build['staged_sources'].items()})
            cli = directory / 'build/open_qwen36_cli'
            if base.digest(cli) != build['cli_sha256']:
                raise ValueError('native CLI differs from build evidence')
            meta, report = validate_run(directory / 'run')
            if meta != bm or report['cli_sha256'] != build['cli_sha256']:
                raise ValueError('A/B references or native build differ')
            reports[variant] = report
        n = len(bm['ids'])
        control_identity = identity(a.baseline.resolve(), out / 'control/run', n)
        sequential_identity = identity(out / 'control/run', out / 'paired/run', n, ('sequential',))
        result.update(control_identity=control_identity, sequential_identity=sequential_identity,
                      baseline_passed=br['passed'], control_passed=reports['control']['passed'],
                      paired_passed=reports['paired']['passed'],
                      reference_sha256=base.digest(a.baseline / 'reference.json'),
                      paired_hybrid_schedule_identity=reports['paired']['hybrid_schedule_identity'])
        result['measurements'] = {
            variant: [dict(route=r['route'], passed=r['passed'],
                           min_correlation=min(x['correlation'] for x in r['logits']),
                           continuation_residual_maxrel=r['continuation_last_residual_maxrel'])
                      for r in report['routes']]
            for variant, report in reports.items()}
        artifacts = [a.baseline / 'report.json', Path(__file__),
                     ROOT / 'utilities/prefill-slice/check.py']
        for variant in reports:
            artifacts += [out / variant / 'build-evidence.json', out / variant / 'run/report.json']
        result['evidence'] = {str(p.resolve()): base.digest(p) for p in artifacts}
        result['passed'] = (all(control_identity.values()) and all(sequential_identity.values())
                            and reports['paired']['passed'])
    except Exception as e:
        result['error'] = str(e)
        raise
    finally:
        (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    raise SystemExit(audit(p.parse_args()))
