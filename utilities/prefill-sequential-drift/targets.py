"""Replay selected long-prefix layers, including the shared embedding at layer zero."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import importlib.util
import numpy as np

s = importlib.util.spec_from_file_location('_selected_drift',Path(__file__).with_name('check.py'))
drift = importlib.util.module_from_spec(s); s.loader.exec_module(drift)
base, digest = drift.base, drift.digest


def parse_targets(values):
    result = {}
    if not values: raise ValueError('at least one layer:positions target is required')
    for value in values:
        try:
            layer, raw = value.split(':'); layer = int(layer)
            positions = tuple(int(p) for p in raw.split(','))
        except (ValueError, AttributeError): raise ValueError('expected layer:position[,position...]') from None
        if (not 0 <= layer < 64 or layer in result or not positions or
                len(set(positions)) != len(positions) or any(not 0 <= p < 257 for p in positions)):
            raise ValueError('duplicate or out-of-range target')
        result[layer] = tuple(sorted(positions))
    return dict(sorted(result.items()))


def target_record(targets):
    return {str(k):list(v) for k,v in targets.items()}


def prefix_inputs(layer, route, model, meta, refs, directory):
    if route not in ('reference',*drift.ROUTES): raise ValueError('unknown input route')
    n, hidden = len(meta['ids']), meta['hidden']
    if layer == 0:
        return np.array([model.embed(t,hidden) for t in meta['ids']],dtype=np.float32)
    if route == 'reference':
        return drift.trace.read_trace(refs/f'ref_res{layer-1}.bin',n,hidden).astype(np.float32)
    return drift.trace.read_trace(directory/f'{route}_res{layer-1}.bin',n+1,hidden).astype(np.float32)[:n]


def check_embedding_identity(layer, vectors):
    if layer == 0 and any(np.any(vectors[k] != 0) for k in ('current_input','input_history','propagated')):
        raise ValueError('layer-zero reference/native embedding trajectories differ')


def summarize(rows, targets):
    required = {(l,p,r) for l,ps in targets.items() for p in ps for r in drift.ROUTES}
    index = {(r['layer'],r['position'],r['route']):r for r in rows}
    if not required or len(index) != len(rows) or set(index) != required:
        raise ValueError('incomplete or duplicate selected cases')
    result = {}
    for route in drift.ROUTES:
        group = [r for r in rows if r['route'] == route]
        result[route] = dict(total=len(group),passed=sum(r['metrics']['total']['passed'] for r in group),
            worst={k:max(group,key=lambda r:float('inf') if r['metrics'][k]['maxrel'] is None else r['metrics'][k]['maxrel'])
                   for k in (*drift.PARTS,'propagated','total')})
    return result


def sources():
    return drift.sources() | {str(Path(__file__).resolve()):digest(__file__)}


def evaluate(directory, meta, refs, targets, out=None, expected=None):
    spec = base.spec_from_model_dir(Path(meta['model']))
    model = base.Q4NX(Path(meta['model'])/'model.q4nx')
    model.hidden = spec.hidden; model.native_q8 = lambda name: False
    plain, cache = model.matmul_w, {}
    def cached(name,o,i):
        key = (name,o,i)
        if key not in cache: cache[key] = plain(name,o,i)
        return cache[key]
    model.matmul_w = cached
    rows, artifacts, count, peak = [], {}, 0, 0
    try:
        for layer, positions in targets.items():
            cache.clear(); print('replaying layer',layer,'positions',positions,flush=True)
            d = directory/'expanded/257'
            ref_inputs = prefix_inputs(layer,'reference',model,meta,refs,d)
            reference = drift.trace.read_trace(refs/f'ref_res{layer}.bin',257,spec.hidden)
            ideal_reference, states = drift.replay(model,spec,layer,ref_inputs,positions)
            for p in positions:
                if ideal_reference[p].astype('<f4').tobytes() != reference[p].astype('<f4').tobytes():
                    raise ValueError(f'CPU reference differs at L{layer}/P{p}')
                count += 1
            for route in drift.ROUTES:
                xs = prefix_inputs(layer,route,model,meta,refs,d)
                native = drift.trace.read_trace(d/f'{route}_res{layer}.bin',258,spec.hidden)
                ideal_native, _ = drift.replay(model,spec,layer,xs,positions)
                for p in positions:
                    mixed = drift.on_history(model,spec,layer,xs[p],states[p],p)
                    metrics, vectors = drift.decompose(native[p],ideal_native[p],mixed,ideal_reference[p],reference[p])
                    check_embedding_identity(layer,vectors)
                    arrays = dict(native=native[p],ideal_native=ideal_native[p],mixed=mixed,
                                  ideal_reference=ideal_reference[p],reference=reference[p],
                                  native_input=xs[p],reference_input=ref_inputs[p],**vectors)
                    name = f'{route}_L{layer}_P{p}.npz'
                    if out is not None:
                        np.savez(out/name,**arrays); artifacts[name] = digest(out/name)
                    else:
                        with np.load(expected/name,allow_pickle=False) as data:
                            if set(data.files) != set(arrays) or any(data[k].dtype != x.dtype or data[k].shape != x.shape or data[k].tobytes() != x.tobytes() for k,x in arrays.items()):
                                raise ValueError('causal re-audit differs: '+name)
                    row = dict(layer=layer,position=p,route=route,metrics=metrics,
                               input=drift.scan.error_score(xs[p].astype(np.float64)-ref_inputs[p],ref_inputs[p]))
                    rows.append(row); print(json.dumps(row),flush=True)
            peak = max(peak,sum(x.nbytes for x in cache.values()))
    finally:
        cache.clear(); model.mm.close(); model.f.close()
    return rows, artifacts, count, peak


def run(a):
    out = a.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    r = dict(complete=False,model_acceptance=False,cases=[])
    try:
        targets = parse_targets(a.target); directory = a.baseline.resolve()
        meta, refs, inputs = drift.validate(directory)
        r.update(baseline=str(directory),inputs=inputs,sources=sources(),targets=target_record(targets),
                 scope='selected CPU causal attribution; layer zero uses shared model embeddings; no native state capture')
        rows, artifacts, count, peak = evaluate(directory,meta,refs,targets,out=out)
        base.verify(r['sources']); base.verify(r['inputs'])
        r.update(complete=True,cases=rows,summary=summarize(rows,targets),artifacts=artifacts,
                 reference_reconstructions=count,peak_cached_weight_bytes=peak)
    except Exception as error: r.update(complete=False,error=str(error)); raise
    finally: (out/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    return 0


def audit(a):
    out = a.out.resolve(); result = dict(complete=False,model_acceptance=False)
    try:
        saved = json.loads((out/'report.json').read_text())
        if not saved['complete'] or saved['sources'] != sources() or saved['model_acceptance'] is not False:
            raise ValueError('incomplete diagnostic or changed sources')
        targets = parse_targets([k+':'+','.join(map(str,v)) for k,v in saved['targets'].items()])
        if saved['targets'] != target_record(targets): raise ValueError('noncanonical target inventory')
        base.verify(saved['sources']); base.verify(saved['inputs'])
        names = {f'{r}_L{l}_P{p}.npz' for l,ps in targets.items() for p in ps for r in drift.ROUTES}
        if set(saved['artifacts']) != names or {p.name for p in out.glob('*.npz')} != names:
            raise ValueError('incomplete selected artifacts')
        base.verify({str(out/name):sha for name,sha in saved['artifacts'].items()})
        directory = Path(saved['baseline']); meta, refs, inputs = drift.validate(directory)
        if inputs != saved['inputs']: raise ValueError('diagnostic input inventory changed')
        rows, _, count, peak = evaluate(directory,meta,refs,targets,expected=out)
        if (rows != saved['cases'] or summarize(rows,targets) != saved['summary'] or
                count != saved['reference_reconstructions'] or peak != saved['peak_cached_weight_bytes']):
            raise ValueError('diagnostic measurements differ')
        base.verify(saved['sources'])
        result.update(complete=True,report_sha256=digest(out/'report.json'),cases=len(rows),reference_reconstructions=count)
    except Exception as error: result['error'] = str(error); raise
    finally: (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','audit']); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--target',action='append',help='layer:position[,position...]; repeat for different layers')
    a = p.parse_args()
    if a.command == 'run' and (a.baseline is None or not a.target): p.error('--baseline and --target required')
    raise SystemExit(run(a) if a.command == 'run' else audit(a))
