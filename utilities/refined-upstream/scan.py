"""Causal upstream drift scan of retained control/candidate native residuals."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('_upstream_sublayers', ROOT/'utilities/refined-sublayers/analyze.py')
sublayers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sublayers)
replay, drift = sublayers.replay, sublayers.drift
base, digest, write = replay.base, replay.digest, replay.write
ROLES = ('control', 'candidate')
POSITIONS = (125, 205, 220)
LAYERS = 64
COMPONENTS = (*drift.PARTS, 'propagated', 'total')
DOWNSTREAM_NODES = dict(native='node_native', ideal_reference='node_oracle_reference',
                        mixed='node_current_input', ideal_native='node_input_history', reference='node_reference')
store = sublayers.store


def sources():
    return sublayers.sources() | {str(Path(__file__).resolve()): digest(__file__)}


def coverage(rows):
    expected = [(l, r, p) for l in range(LAYERS) for r in ROLES for p in POSITIONS]
    if [(r['layer'], r['role'], r['position']) for r in rows] != expected:
        raise ValueError('incomplete, duplicate or reordered scan coverage')


def artifact_names():
    return {f'{role}_L{l}_P{p}.npz' for l in range(LAYERS) for role in ROLES for p in POSITIONS}


def reference_identity(got, expected):
    a, b = np.asarray(got), np.asarray(expected)
    if (a.ndim != 1 or not a.size or a.shape != b.shape or not np.isfinite(a).all() or
            not np.isfinite(b).all() or a.astype('<f4').tobytes() != b.astype('<f4').tobytes()):
        raise ValueError('independent reference does not reconstruct bytewise')


def downstream_identity(path, arrays):
    with np.load(path, allow_pickle=False) as saved:
        for current, previous in DOWNSTREAM_NODES.items():
            a, b = np.asarray(arrays[current], np.float64), saved[previous]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError('retained downstream attribution changed: '+current)


def summarize(rows):
    coverage(rows)
    result = {}
    index = {(r['layer'], r['role'], r['position']): r for r in rows}

    def error(row, field):
        value = row['metrics'][field]['maxrel']
        return float('inf') if value is None else value

    def compact(row, field):
        return dict(layer=row['layer'], position=row['position'], maxrel=row['metrics'][field]['maxrel'])

    for role in ROLES:
        selected = [r for r in rows if r['role'] == role]
        result[role] = dict(total=len(selected), passed=sum(r['metrics']['total']['passed'] for r in selected),
                            local_passed=sum(r['metrics']['implementation_gap']['passed'] for r in selected),
                            first_failure={str(p): next((compact(index[l,role,p], 'total') for l in range(LAYERS)
                                                         if not index[l,role,p]['metrics']['total']['passed']), None)
                                           for p in POSITIONS},
                            largest={field: [compact(r, field) for r in
                                             sorted(selected, key=lambda r: error(r, field), reverse=True)[:12]]
                                     for field in COMPONENTS})
    improved = tied = 0
    regressions = []
    for layer in range(LAYERS):
        for p in POSITIONS:
            a, b = index[layer,'control',p], index[layer,'candidate',p]
            av, bv = error(a, 'total'), error(b, 'total')
            improved += bv < av
            tied += bv == av
            if bv > av:
                regressions.append(dict(layer=layer, position=p, control=a['metrics']['total']['maxrel'],
                                        candidate=b['metrics']['total']['maxrel']))
    result.update(improved=improved, tied=tied, regressions=regressions)
    return result


def validate(directory, downstream):
    # Retained numerical FAIL is evidence, never a waiver of current thresholds.
    meta, refs, spec, _, _, _, inputs = replay.baseline(directory)
    previous = json.loads((downstream/'report.json').read_text())
    audit = json.loads((downstream/'audit.json').read_text())
    if (previous.get('complete') is not True or audit.get('complete') is not True or
            previous.get('model_acceptance') is not False or audit.get('model_acceptance') is not False or
            previous['sources'] != sublayers.sources() or
            audit.get('report_sha256') != digest(downstream/'report.json') or audit.get('cases') != 3 or
            any(audit.get(k) != previous[k] for k in ('numerical_passed', 'local_passed'))):
        raise ValueError('requires bound complete downstream analysis audit')
    sublayers.coverage(previous['cases'])
    expected = {n for l,p in replay.TARGETS.items() for n in (f'L{l}_P{p}.npz', f'L{l}-delta.json')}
    if set(previous['artifacts']) != expected:
        raise ValueError('incomplete downstream artifacts')
    capture = Path(previous['capture'])
    prep, captured = sublayers.validate(capture)
    if prep['baseline'] != str(directory) or captured != previous['inputs']:
        raise ValueError('downstream inputs differ from selected native baseline')
    inputs.update(captured)
    inputs.update({str(downstream/n): h for n,h in previous['artifacts'].items()})
    inputs.update({str(downstream/n): digest(downstream/n) for n in ('report.json', 'audit.json')})
    base.verify(inputs)
    return meta, refs, spec, inputs


def evaluate(directory, downstream, meta, refs, spec, out=None, expected=None):
    model = base.Q4NX(Path(meta['model'])/'model.q4nx')
    model.native_q8 = lambda name: False
    cached = sublayers.attribution.Cached(model)
    rows, artifacts = [], {}
    peak = reconstructions = controls = 0
    try:
        embeddings = np.array([model.embed(t, spec.hidden) for t in meta['ids']], dtype='<f4')
        for layer in range(LAYERS):
            print('layer', layer, spec.layer_types[layer], flush=True)
            reference_x = (embeddings if layer == 0 else
                           replay.trace(refs/f'ref_res{layer-1}.bin', 257, spec.hidden))
            reference_y = replay.trace(refs/f'ref_res{layer}.bin', 257, spec.hidden)
            ideal_ref, states = drift.replay(cached, spec, layer, reference_x, POSITIONS)
            for p in POSITIONS:
                reference_identity(ideal_ref[p], reference_y[p])
                reconstructions += 1
            for role in ROLES:
                native_dir = directory/role/'trace'
                xs = (embeddings if layer == 0 else
                      replay.trace(native_dir/f'sequential_res{layer-1}.bin', 258, spec.hidden)[:257])
                native_y = replay.trace(native_dir/f'sequential_res{layer}.bin', 258, spec.hidden)
                ideal_native, native_states = drift.replay(cached, spec, layer, xs, POSITIONS)
                del native_states
                for p in POSITIONS:
                    mixed = drift.on_history(cached, spec, layer, xs[p], states[p], p)
                    metrics, vectors = drift.decompose(native_y[p], ideal_native[p], mixed, ideal_ref[p], reference_y[p])
                    arrays = dict(native=native_y[p].astype(np.float64), ideal_native=ideal_native[p].astype(np.float64),
                                  mixed=mixed.astype(np.float64), ideal_reference=ideal_ref[p].astype(np.float64),
                                  reference=reference_y[p].astype(np.float64), native_input=xs[p],
                                  reference_input=reference_x[p], **vectors)
                    control = role == 'candidate' and replay.TARGETS.get(layer) == p
                    if control:
                        downstream_identity(downstream/f'L{layer}_P{p}.npz', arrays)
                        controls += 1
                    name = f'{role}_L{layer}_P{p}.npz'
                    store(out, expected, name, arrays)
                    if out is not None:
                        artifacts[name] = digest(out/name)
                    row = dict(layer=layer, position=p, role=role, kind=spec.layer_types[layer], reference_exact=True,
                               downstream_control=control, metrics=metrics,
                               input=drift.scan.error_score(xs[p].astype(np.float64)-reference_x[p], reference_x[p]))
                    rows.append(row)
                    print(role, p, 'total', metrics['total']['maxrel'], 'local', metrics['implementation_gap']['maxrel'], flush=True)
            peak = max(peak, sum(w.nbytes for w in cached.cache.values()))
            cached.cache.clear()
            del states
    finally:
        cached.cache.clear()
        model.mm.close()
        model.f.close()
    coverage(rows)
    if reconstructions != 192 or controls != 3:
        raise ValueError('incomplete independent/native control coverage')
    return dict(cases=rows, summary=summarize(rows), reference_reconstructions=reconstructions,
                downstream_controls=controls, peak_cached_weight_bytes=peak), artifacts


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(complete=False, model_acceptance=False, sources=sources())
    try:
        directory, downstream = args.baseline.resolve(), args.downstream.resolve()
        meta, refs, spec, inputs = validate(directory, downstream)
        report.update(baseline=str(directory), downstream=str(downstream), inputs=inputs,
                      scope='CPU causal prefix replay of retained native inputs; local gap includes accumulated implementation state; no native correction or performance acceptance')
        measured, artifacts = evaluate(directory, downstream, meta, refs, spec, out=out)
        base.verify(inputs)
        base.verify(report['sources'])
        report.update(measured, complete=True, artifacts=artifacts,
                      numerical_passed=all(r['metrics']['total']['passed'] for r in measured['cases'] if r['role'] == 'candidate'))
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        write(out/'report.json', report)
    return 0 if report['numerical_passed'] else 1


def audit(args):
    out = args.out.resolve()
    proof = dict(complete=False, model_acceptance=False)
    try:
        report = json.loads((out/'report.json').read_text())
        if report.get('complete') is not True or report.get('model_acceptance') is not False or report['sources'] != sources():
            raise ValueError('incomplete or changed scan')
        coverage(report['cases'])
        if set(report['artifacts']) != artifact_names() or {p.name for p in out.glob('*.npz')} != artifact_names():
            raise ValueError('incomplete scan artifact inventory')
        base.verify({str(out/n): h for n,h in report['artifacts'].items()})
        directory, downstream = Path(report['baseline']), Path(report['downstream'])
        meta, refs, spec, inputs = validate(directory, downstream)
        if report['inputs'] != inputs:
            raise ValueError('scan inputs changed')
        measured, _ = evaluate(directory, downstream, meta, refs, spec, expected=out)
        verdict = all(r['metrics']['total']['passed'] for r in measured['cases'] if r['role'] == 'candidate')
        if any(report[k] != v for k,v in measured.items()) or report['numerical_passed'] != verdict:
            raise ValueError('scan measurements/verdict changed')
        base.verify(inputs)
        base.verify(report['sources'])
        proof.update(complete=True, numerical_passed=verdict, cases=len(measured['cases']),
                     reference_reconstructions=measured['reference_reconstructions'],
                     downstream_controls=measured['downstream_controls'], report_sha256=digest(out/'report.json'))
    except Exception as error:
        proof['error'] = str(error)
        raise
    finally:
        write(out/'audit.json', proof)
    return 0 if proof['numerical_passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action, flags in (('run', ('out', 'baseline', 'downstream')), ('audit', ('out',))):
        command = sub.add_parser(action)
        for flag in flags:
            command.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(globals()[args.action](args))
