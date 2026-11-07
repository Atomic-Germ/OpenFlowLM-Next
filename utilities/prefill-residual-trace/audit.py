"""Audit private native residual traces only after byte-exact baseline logits."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
s = importlib.util.spec_from_file_location('_trace_paired_audit', ROOT/'utilities/prefill-paired-input/audit.py')
paired = importlib.util.module_from_spec(s)
s.loader.exec_module(paired)
base = paired.base
ROUTES = ('sequential', 'hybrid', 'block_major')


def read_trace(path, rows, hidden):
    return base.read(path, rows * hidden).reshape(rows, hidden)


def score_rows(got, reference):
    got, reference = np.asarray(got, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if got.shape != reference.shape or got.ndim != 2 or not got.size:
        raise ValueError('invalid residual shape')
    if not np.isfinite(got).all() or not np.isfinite(reference).all():
        raise ValueError('non-finite residual')
    errors = np.max(np.abs(got-reference), axis=1)
    scales = np.max(np.abs(reference), axis=1)
    return [dict(position=p, maxabs=float(e), maxrel=float(e/s) if s else (0.0 if not e else None),
                 passed=bool(e/s < base.RESIDUAL_MAXREL) if s else bool(e == 0))
            for p, (e,s) in enumerate(zip(errors,scales))]


def require_identity(before, after, tokens):
    checks = paired.identity(before, after, tokens)
    if not all(checks.values()):
        raise ValueError('baseline logit identity failed: ' + ', '.join(k for k,v in checks.items() if not v))
    return checks


def localize(rows, layers, positions):
    """Keep temporal onset and shallowest failing depth separate."""
    index = {(r['route'], r['layer'], r['position']): r for r in rows}
    required = {(route, l, p) for route in ROUTES for l in range(layers) for p in range(positions)}
    if layers <= 0 or positions <= 0 or len(index) != len(rows) or set(index) != required:
        raise ValueError('incomplete or duplicate localization coverage')

    def worst(group):
        # A nonzero error against an exact-zero reference has no finite maxrel.
        return max(group, key=lambda r: float('inf') if r['maxrel'] is None else r['maxrel'])

    def section(group):
        failures = [r for r in group if not r['passed']]
        return dict(passed=len(group)-len(failures), total=len(group),
                    first_failure=failures[0] if failures else None, worst=worst(group))

    result = {}
    for route in ROUTES:
        depth = [[index[route,l,p] for p in range(positions)] for l in range(layers)]
        time = [[index[route,l,p] for l in range(layers)] for p in range(positions)]
        ordered = [r for group in time for r in group]
        failures = [r for r in ordered if not r['passed']]
        result[route] = dict(
            passed=len(ordered)-len(failures), total=len(ordered),
            first_by_position=failures[0] if failures else None,
            first_by_layer=min(failures, key=lambda r:(r['layer'],r['position'])) if failures else None,
            worst=worst(ordered),
            layers=[dict(layer=l, **section(group)) for l,group in enumerate(depth)],
            positions=[dict(position=p, **section(group)) for p,group in enumerate(time)])
    return result


def verify_build(directory, variant):
    evidence = json.loads((directory/'build-evidence.json').read_text())
    if not evidence['complete'] or evidence['variant'] != variant:
        raise ValueError('incomplete or wrong native build')
    base.verify(evidence['sources'])
    base.verify({str(directory/p): h for p,h in evidence['staged_sources'].items()})
    if variant == 'paired-residual-trace':
        if not evidence.get('compiler_dependencies'):
            raise ValueError('missing compiler dependency provenance')
        base.verify(evidence['compiler_dependencies'])
    if base.digest(directory/'build/open_qwen36_cli') != evidence['cli_sha256']:
        raise ValueError('CLI differs from native build')
    if base.digest(directory/'build/CMakeCache.txt') != evidence['cmake_cache_sha256']:
        raise ValueError('changed CMake configuration')
    return evidence


def audit(a):
    result = dict(capture_valid=False, numerical_passed=False,
                  scope='actual per-layer residual after causal prompt positions and one NPU continuation')
    try:
        build = verify_build(a.build.resolve(), 'paired-residual-trace')
        control = verify_build(a.baseline_build.resolve(), 'paired')
        bm, br = paired.validate_run(a.baseline.resolve())
        meta, report = paired.validate_run(a.run.resolve())
        if meta != bm or br['cli_sha256'] != control['cli_sha256'] or report['cli_sha256'] != build['cli_sha256']:
            raise ValueError('baseline/reference/CLI mismatch')
        n, hidden, layers = len(meta['ids']), meta['hidden'], meta['layers']
        result['logit_identity'] = require_identity(a.baseline, a.run, n)
        expected = {f'{route}_res{l}.bin' for route in ROUTES for l in range(layers)}
        actual = {p.name for p in a.run.glob('*_res*.bin') if not p.name.startswith('ref_')}
        if actual != expected:
            raise ValueError('incomplete or unexpected residual trace inventory')
        rows = []
        hashes = {}
        for l in range(layers):
            reference = np.vstack((read_trace(a.run/f'ref_res{l}.bin',n,hidden),
                                   read_trace(a.run/f'ref_cont_res{l}.bin',1,hidden)))
            for route in ROUTES:
                name = f'{route}_res{l}.bin'
                got = read_trace(a.run/name,n+1,hidden)
                rows.extend(dict(route=route, layer=l, **r) for r in score_rows(got,reference))
                hashes[name] = base.digest(a.run/name)
        pairs = {str(l): hashes[f'hybrid_res{l}.bin'] == hashes[f'block_major_res{l}.bin'] for l in range(layers)}
        first = {}
        for route in ROUTES:
            failures = [r for r in rows if r['route']==route and not r['passed']]
            first[route] = min(failures, key=lambda r:(r['position'],r['layer'])) if failures else None
        # This reports the first threshold crossing, not the operation causing it.
        result.update(capture_valid=True, numerical_passed=report['passed'] and all(r['passed'] for r in rows),
                      native_logits_passed=report['passed'], threshold=base.RESIDUAL_MAXREL,
                      first_failing_position_layer=first, rows=rows, trace_hashes=hashes,
                      hybrid_residual_identity=pairs, localization=localize(rows,layers,n+1))
        evidence = [Path(__file__), a.run/'reference.json', a.run/'report.json', a.baseline/'report.json',
                    a.build/'build-evidence.json', a.baseline_build/'build-evidence.json']
        result['evidence'] = {str(p.resolve()):base.digest(p) for p in evidence}
    except Exception as e:
        result.update(capture_valid=False, numerical_passed=False, error=str(e))
        raise
    finally:
        a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return 0 if result['numerical_passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('build','baseline-build','baseline','run','out'):
        p.add_argument('--'+name, type=Path, required=True)
    raise SystemExit(audit(p.parse_args()))
