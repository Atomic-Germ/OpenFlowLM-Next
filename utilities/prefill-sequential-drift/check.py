"""CPU causal attribution of retained long-prefix sequential drift; no model acceptance."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


boundary = load('_drift_boundaries', ROOT/'utilities/prefill-expanded-cli/boundaries.py')
scan = load('_drift_prefix_scan', ROOT/'utilities/prefill-prefix-scan/check.py')
base, digest = boundary.base, boundary.digest
trace = boundary.runner.original.trace
TARGETS = {33: (240, 241, 242), 60: (37, 38, 39)}
ROUTES = ('sequential', 'hybrid')
PARTS = ('reference_rounding', 'current_input', 'input_history', 'implementation_gap')


def initial_state(spec):
    return (np.zeros((spec.conv_kernel-1, spec.lin_qkv_dim)),
            np.zeros((spec.lin_value_heads, spec.lin_value_dim, spec.lin_value_dim)),
            np.zeros((0, spec.num_kv_heads, spec.head_dim)),
            np.zeros((0, spec.num_kv_heads, spec.head_dim)))


def advance(model, spec, layer, x, state, pos, output):
    conv, recurrent, k, v = state
    if spec.layer_types[layer] == 'full_attention':
        block, k, v = base.oracle.attn_decode(model,spec,layer,x,k,v,pos)
    else:
        block, conv, recurrent = base.oracle.linear_decode(model,spec,layer,x,conv,recurrent)
    result = None
    if output:
        res = x + block
        result = res + base.oracle.ffn(model,spec,layer,res)
    return result, (conv,recurrent,k,v)


def on_history(model, spec, layer, x, state, pos):
    # The oracle mutates recurrent state. Never let a counterfactual alter the
    # independent reference trajectory or another selected position.
    copied = tuple(a.copy() for a in state)
    return advance(model,spec,layer,np.array(x,copy=True),copied,pos,True)[0]


def replay(model, spec, layer, inputs, positions):
    xs = np.asarray(inputs)
    if (xs.ndim != 2 or not xs.size or not np.isfinite(xs).all() or not positions
            or any(type(p) is not int or not 0 <= p < len(xs) for p in positions)
            or len(set(positions)) != len(positions)):
        raise ValueError('invalid causal prefix or target positions')
    state = initial_state(spec); outputs, snapshots = {}, {}
    for pos in range(max(positions)+1):
        selected = pos in positions
        if selected: snapshots[pos] = tuple(a.copy() for a in state)
        y, state = advance(model,spec,layer,xs[pos].copy(),state,pos,selected)
        if selected: outputs[pos] = y
    return outputs, snapshots


def decompose(native, ideal_native, mixed, ideal_reference, reference):
    values = [np.asarray(x,np.float64) for x in (native,ideal_native,mixed,ideal_reference,reference)]
    if (any(x.ndim != 1 or not x.size or not np.isfinite(x).all() for x in values)
            or any(x.shape != values[0].shape for x in values)):
        raise ValueError('requires equal complete finite vectors')
    native, ideal_native, mixed, ideal_reference, reference = values
    vectors = dict(reference_rounding=ideal_reference-reference,
                   current_input=mixed-ideal_reference,
                   input_history=ideal_native-mixed,
                   implementation_gap=native-ideal_native,
                   propagated=ideal_native-ideal_reference, total=native-reference)
    closure = sum(vectors[k] for k in PARTS)-vectors['total']
    bound = 16*np.finfo(np.float64).eps*sum(np.abs(x) for x in values)
    if np.any(np.abs(closure) > bound): raise ValueError('signed attribution vectors do not close')
    return {k: scan.error_score(v,reference) for k,v in vectors.items()}, vectors


def summarize(rows):
    required = {(l,p,r) for l,ps in TARGETS.items() for p in ps for r in ROUTES}
    index = {(r['layer'],r['position'],r['route']):r for r in rows}
    if len(index) != len(rows) or set(index) != required:
        raise ValueError('incomplete or duplicate diagnostic cases')
    result = {}
    for route in ROUTES:
        group = [r for r in rows if r['route'] == route]
        result[route] = dict(total=len(group),passed=sum(r['metrics']['total']['passed'] for r in group),
            worst={k:max(group,key=lambda r:float('inf') if r['metrics'][k]['maxrel'] is None else r['metrics'][k]['maxrel'])
                   for k in (*PARTS,'propagated','total')})
    return result


def sources():
    return boundary.evidence() | {str(Path(p).resolve()):digest(p) for p in
        (__file__, scan.__file__, base.oracle.__file__, ROOT/'open_kernels/model/q4nx.py')}


def validate(directory):
    saved = json.loads((directory/'report.json').read_text())
    audited = json.loads((directory/'audit.json').read_text())
    if (not saved['complete'] or not audited['complete'] or
            audited['report_sha256'] != digest(directory/'report.json') or
            audited['passed'] != saved['passed'] or saved['model_acceptance'] is not False or
            saved['sources'] != boundary.evidence()):
        raise ValueError('unbound or incomplete long-prefix audit')
    prepared = Path(saved['prepared'])
    if saved['preparation_sha256'] != digest(prepared/'preparation.json'):
        raise ValueError('long-prefix preparation changed')
    record, metas, offsets = boundary.verify_preparation(prepared)
    # Validate the audited raw inventory at every length, not only selected rows.
    inputs = {str(directory/name):digest(directory/name) for name in ('report.json','audit.json')}
    inputs[str(prepared/'preparation.json')] = digest(prepared/'preparation.json')
    for variant in boundary.VARIANTS:
        for n in boundary.LENGTHS:
            d = directory/variant/str(n); stored = saved['variants'][variant][str(n)]
            if [r['route'] for r in stored['routes']] != list(boundary.runner.ROUTES):
                raise ValueError('incomplete baseline routes')
            for row in stored['routes']:
                if set(row['outputs']) != set(boundary.runner.original.capture_names(row['route'],n,64)):
                    raise ValueError('incomplete baseline capture inventory')
                inputs.update({str(d/name):sha for name,sha in row['outputs'].items()})
                inputs[str(d/(row['route']+'.log'))] = row['log_sha256']
    base.verify(inputs)
    meta = metas[257]; refs = Path(record['reference_root'])/'257'
    d = directory/'expanded/257'
    measured = boundary.runner.original.audit_variant(d,refs,meta,offsets)
    stored = saved['variants']['expanded']['257']
    if any(stored.get(k) != v for k,v in measured.items()):
        raise ValueError('selected long-prefix metrics differ from captures')
    for name,sha in meta['references'].items(): inputs[str(refs/name)] = sha
    inputs[str(refs/'reference.json')] = digest(refs/'reference.json')
    return meta, refs, inputs


def evaluate(directory, meta, refs, out=None, expected=None):
    spec = base.spec_from_model_dir(Path(meta['model']))
    model = base.Q4NX(Path(meta['model'])/'model.q4nx')
    model.hidden = spec.hidden; model.native_q8 = lambda name: False
    plain, cache = model.matmul_w, {}
    def cached(name, o, i):
        key = (name,o,i)
        if key not in cache: cache[key] = plain(name,o,i)
        return cache[key]
    model.matmul_w = cached
    rows, artifacts, reconstructed, peak = [], {}, 0, 0
    try:
        for layer, positions in TARGETS.items():
            cache.clear()
            ref_inputs = trace.read_trace(refs/f'ref_res{layer-1}.bin',257,spec.hidden).astype(np.float32)
            reference = trace.read_trace(refs/f'ref_res{layer}.bin',257,spec.hidden)
            ideal_reference, states = replay(model,spec,layer,ref_inputs,positions)
            for p in positions:
                if ideal_reference[p].astype('<f4').tobytes() != reference[p].astype('<f4').tobytes():
                    raise ValueError(f'independent CPU reference differs at L{layer}/P{p}')
                reconstructed += 1
            for route in ROUTES:
                d = directory/'expanded/257'
                xs = trace.read_trace(d/f'{route}_res{layer-1}.bin',258,spec.hidden).astype(np.float32)[:257]
                native = trace.read_trace(d/f'{route}_res{layer}.bin',258,spec.hidden)
                ideal_native, _ = replay(model,spec,layer,xs,positions)
                for p in positions:
                    mixed = on_history(model,spec,layer,xs[p],states[p],p)
                    metrics, vectors = decompose(native[p],ideal_native[p],mixed,ideal_reference[p],reference[p])
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
                               input=scan.error_score(xs[p].astype(np.float64)-ref_inputs[p],ref_inputs[p]))
                    rows.append(row); print(json.dumps(row),flush=True)
            peak = max(peak,sum(x.nbytes for x in cache.values()))
    finally:
        cache.clear(); model.mm.close(); model.f.close()
    return rows, artifacts, reconstructed, peak


def run(a):
    out = a.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    r = dict(complete=False,model_acceptance=False,cases=[])
    try:
        directory = a.baseline.resolve(); meta, refs, inputs = validate(directory)
        r.update(baseline=str(directory),inputs=inputs,sources=sources(),targets={str(k):list(v) for k,v in TARGETS.items()},
                 scope='CPU counterfactual histories on captured residuals; implementation gap includes local arithmetic and accumulated native state')
        rows, artifacts, count, peak = evaluate(directory,meta,refs,out=out)
        base.verify(r['sources']); base.verify(r['inputs'])
        r.update(complete=True,cases=rows,summary=summarize(rows),artifacts=artifacts,
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
        base.verify(saved['sources']); base.verify(saved['inputs'])
        expected_names = {f'{r}_L{l}_P{p}.npz' for l,ps in TARGETS.items() for p in ps for r in ROUTES}
        if set(saved['artifacts']) != expected_names or {p.name for p in out.glob('*.npz')} != expected_names:
            raise ValueError('incomplete diagnostic artifacts')
        base.verify({str(out/name):sha for name,sha in saved['artifacts'].items()})
        directory = Path(saved['baseline']); meta, refs, inputs = validate(directory)
        if inputs != saved['inputs'] or saved['targets'] != {str(k):list(v) for k,v in TARGETS.items()}:
            raise ValueError('diagnostic inputs/targets changed')
        rows, _, count, peak = evaluate(directory,meta,refs,expected=out)
        if (rows != saved['cases'] or summarize(rows) != saved['summary'] or
                count != saved['reference_reconstructions'] or peak != saved['peak_cached_weight_bytes']):
            raise ValueError('diagnostic measurements differ')
        result.update(complete=True,report_sha256=digest(out/'report.json'),cases=len(rows),reference_reconstructions=count)
    except Exception as error: result['error'] = str(error); raise
    finally: (out/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','audit']); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    a = p.parse_args()
    if a.command == 'run' and a.baseline is None: p.error('--baseline required')
    raise SystemExit(run(a) if a.command == 'run' else audit(a))
