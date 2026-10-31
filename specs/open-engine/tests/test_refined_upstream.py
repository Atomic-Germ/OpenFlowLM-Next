"""Complete causal coverage and immutable controls for upstream drift tracing."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location('refined_upstream', ROOT/'utilities/refined-upstream/check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_complete_scan_coverage():
    m = load()
    assert m.POSITIONS == (125, 205, 220)
    assert m.LAYERS == 64 and m.ROLES == ('control', 'candidate')
    rows = [dict(layer=l, position=p, role=r) for l in range(64) for r in m.ROLES for p in m.POSITIONS]
    m.coverage(rows)
    for bad in (rows[:-1], rows[::-1], rows+rows[:1]):
        with pytest.raises(ValueError, match='coverage'):
            m.coverage(bad)
    assert len(m.artifact_names()) == 384


def test_reference_identity_preserves_signed_zero_and_shape():
    m = load()
    x = np.array([0., 1.], dtype='<f4')
    m.reference_identity(x, x.copy())
    for bad in (np.array([-0., 1.], dtype='<f4'), x[:1], np.array([np.nan, 1.], dtype='<f4')):
        with pytest.raises(ValueError, match='reference'):
            m.reference_identity(bad, x)


def test_downstream_vectors_are_exact_not_merely_close(tmp_path):
    m = load()
    arrays = dict(native=np.array([2., 4.]), ideal_reference=np.array([1., 3.]),
                  mixed=np.array([1.5, 3.5]), ideal_native=np.array([1.75, 3.75]),
                  reference=np.array([1., 3.]))
    saved = {v:arrays[k] for k,v in m.DOWNSTREAM_NODES.items()}
    path = tmp_path/'target.npz'
    np.savez(path, **saved)
    m.downstream_identity(path, arrays)
    arrays['native'][0] += 1e-12
    with pytest.raises(ValueError, match='downstream'):
        m.downstream_identity(path, arrays)


def test_causal_replay_uses_all_rows_and_isolates_counterfactuals(monkeypatch):
    m = load()
    calls = []
    initial = lambda spec: (np.zeros(1),)

    def advance(model, spec, layer, x, state, pos, output):
        calls.append((pos, output))
        state[0][:] += x.sum()
        return (x+state[0] if output else None), state

    monkeypatch.setattr(m.drift, 'initial_state', initial)
    monkeypatch.setattr(m.drift, 'advance', advance)
    xs = np.arange(6, dtype='<f4').reshape(3, 2)
    outputs, states = m.drift.replay(None, None, 0, xs, (1, 2))
    assert calls == [(0, False), (1, True), (2, True)]
    before = states[1][0].copy()
    m.drift.on_history(None, None, 0, xs[1], states[1], 1)
    assert np.array_equal(states[1][0], before)
    assert np.array_equal(outputs[2], xs[2]+xs.sum())


def test_signed_local_gap_and_zero_reference_keep_existing_gate():
    m = load()
    x = np.array([1., 2.])
    scores, effects = m.drift.decompose(x+.001, x+1., x+.5, x, x)
    np.testing.assert_allclose(effects['propagated']+effects['implementation_gap'], effects['total'])
    assert scores['total']['passed']
    assert not scores['implementation_gap']['passed']
    scores, _ = m.drift.decompose(np.ones(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
    assert scores['total']['maxrel'] is None and not scores['total']['passed']


def test_scan_refuses_existing_output(tmp_path):
    m = load()
    with pytest.raises(FileExistsError):
        m.run(SimpleNamespace(out=tmp_path))


def test_summary_retains_all_regressions_and_infinite_errors():
    m = load()
    rows = []
    for layer in range(m.LAYERS):
        for role in m.ROLES:
            for p in m.POSITIONS:
                total = .01 if layer == 50 else .001
                value = total*2 if role == 'candidate' else total
                metrics = {key: dict(maxrel=value, passed=value < .005) for key in m.COMPONENTS}
                rows.append(dict(layer=layer, position=p, role=role, metrics=metrics,
                                 input=dict(maxrel=.001, passed=True)))
    s = m.summarize(rows)
    assert s['candidate']['total'] == 192 and s['candidate']['passed'] == 189
    assert all(v['layer'] == 50 for v in s['candidate']['first_failure'].values())
    assert len(s['regressions']) == 192
    rows[3]['metrics']['implementation_gap']['maxrel'] = None
    rows[3]['metrics']['implementation_gap']['passed'] = False
    assert m.summarize(rows)['candidate']['largest']['implementation_gap'][0]['layer'] == 0


def test_comparison_requires_identical_coverage():
    m = load()
    with pytest.raises(ValueError, match='coverage'):
        m.summarize([])


def test_array_audit_rejects_changed_bits_or_inventory(tmp_path):
    m = load()
    arrays = dict(native=np.array([0., 1.], dtype='<f8'))
    m.store(tmp_path, None, 'case.npz', arrays)
    m.store(None, tmp_path, 'case.npz', arrays)
    with pytest.raises(ValueError):
        m.store(None, tmp_path, 'case.npz', arrays | {'extra': np.zeros(2)})
    with pytest.raises(ValueError):
        m.store(None, tmp_path, 'case.npz', dict(native=np.array([-0., 1.])))


def test_retained_verdict_requires_complete_consistent_numerical_fail():
    m = load()
    rows = [dict(role='control', metrics={'total':dict(passed=True)}),
            dict(role='candidate', metrics={'total':dict(passed=False)})]
    report = dict(complete=True, model_acceptance=False, numerical_passed=False, cases=rows)
    audit = dict(complete=True, model_acceptance=False, numerical_passed=False,
                 report_sha256='sha', cases=384, reference_reconstructions=192, downstream_controls=3)
    m.retained_verdict(report, audit, 'sha')
    for key, value in (('numerical_passed', True), ('complete', False), ('model_acceptance', None)):
        with pytest.raises(ValueError, match='retained causal audit'):
            m.retained_verdict(report | {key: value}, audit, 'sha')
    for key in ('cases', 'reference_reconstructions', 'downstream_controls', 'report_sha256'):
        with pytest.raises(ValueError, match='retained causal audit'):
            m.retained_verdict(report, audit | {key: -1 if key != 'report_sha256' else 'other'}, 'sha')


def test_retained_sources_map_the_historical_scanner_without_relabelling(tmp_path):
    m = load()
    original, snapshot = tmp_path/'check.py', tmp_path/'scan.py'
    # The snapshot must still carry the original scanner body bytewise; the
    # wrapper replaced check.py and must not be relabelled as that source.
    body = 'scanner'
    original.write_text(body), snapshot.write_text(body)
    expected = {str(original): m.digest(original), str(snapshot): m.digest(snapshot)}
    assert m.retained_sources(expected, original, snapshot) == {
        str(original): str(snapshot), str(snapshot): str(snapshot)}
    with pytest.raises(ValueError, match='historical scanner'):
        m.retained_sources({str(snapshot): m.digest(snapshot)}, original, snapshot)
    snapshot.write_text('changed')
    with pytest.raises(ValueError, match='retained source'):
        m.retained_sources(expected, original, snapshot)
    original.write_text('wrapper')
    with pytest.raises(ValueError, match='retained source'):
        m.retained_sources({str(original): 'unrelated-body-hash'}, original, snapshot)


def test_retained_case_recomputes_signed_arrays_and_rejects_changed_metrics():
    m = load()
    rng = np.random.default_rng(3)
    reference, reference_input = rng.normal(size=8), rng.normal(size=8)
    native_input, native = reference_input+1e-4, reference+2e-3
    native, ideal_native = native.astype(np.float64), (reference+1e-3).astype(np.float64)
    mixed, ideal_reference = (reference+1.5e-3).astype(np.float64), reference.astype(np.float64)
    metrics, effects = m.drift.decompose(native, ideal_native, mixed, ideal_reference, reference.astype(np.float64))
    arrays = dict(native=native, ideal_native=ideal_native, mixed=mixed,
                  ideal_reference=ideal_reference, reference=reference.astype(np.float64),
                  native_input=native_input, reference_input=reference_input, **effects)
    row = dict(metrics=metrics, input=m.drift.scan.error_score(native_input.astype(np.float64)-reference_input, reference_input))
    m.retained_case(arrays, row)
    with pytest.raises(ValueError, match='signed effect'):
        m.retained_case(arrays | {'propagated': effects['propagated']+1e-9}, row)
    with pytest.raises(ValueError, match='metrics'):
        m.retained_case(arrays, row | {'metrics': metrics | {'total': dict(maxrel=0.0, passed=True)}})
    with pytest.raises(ValueError, match='arrays'):
        m.retained_case(dict(arrays, extra=np.zeros(8)), row)


@pytest.mark.parametrize('tamper', ['audit_hash', 'missing_artifact', 'different_baseline', 'changed_inputs'])
def test_downstream_provenance_rejects_tampering(tmp_path, monkeypatch, tamper):
    m = load()
    downstream = tmp_path/'analysis'
    downstream.mkdir()
    baseline = tmp_path/'native'
    refs = tmp_path/'refs'
    monkeypatch.setattr(m.replay, 'baseline', lambda directory: ({}, refs, None, None, None, None, {}))
    monkeypatch.setattr(m.sublayers, 'sources', lambda: {'source': 'hash'})
    artifacts = {n: 'hash' for l,p in m.replay.TARGETS.items()
                 for n in (f'L{l}_P{p}.npz', f'L{l}-delta.json')}
    report = dict(complete=True, model_acceptance=False, numerical_passed=False,
                  local_passed=True, sources={'source': 'hash'}, artifacts=artifacts,
                  capture=str(tmp_path/'capture'), inputs={'capture': 'hash'},
                  cases=[dict(layer=l, position=p) for l,p in m.replay.TARGETS.items()])
    if tamper == 'missing_artifact':
        report['artifacts'].pop(next(iter(artifacts)))
    m.write(downstream/'report.json', report)
    m.write(downstream/'audit.json', dict(complete=True, model_acceptance=False, cases=3,
            numerical_passed=False, local_passed=True,
            report_sha256='changed' if tamper == 'audit_hash' else m.digest(downstream/'report.json')))
    prep = dict(baseline=str(tmp_path/'wrong' if tamper == 'different_baseline' else baseline))
    inputs = {'capture': 'changed' if tamper == 'changed_inputs' else 'hash'}
    monkeypatch.setattr(m.sublayers, 'validate', lambda capture: (prep, inputs))
    with pytest.raises(ValueError):
        m.validate(baseline, downstream)


def test_retained_source_alias_is_exact_and_only_for_scanner(tmp_path):
    m = load()
    old = tmp_path/'check.py'; saved = tmp_path/'scan.py'; other = tmp_path/'other.py'
    old.write_bytes(b'new wrapper'); saved.write_bytes(b'old scanner'); other.write_bytes(b'unchanged')
    expected = {str(old):m.digest(saved), str(other):m.digest(other)}
    assert m.retained_sources(expected, old, saved)[str(old)] == str(saved)
    saved.write_bytes(b'changed scanner')
    with pytest.raises(ValueError): m.retained_sources(expected, old, saved)


def test_retained_arrays_recompute_metrics_and_reject_missing_nodes():
    m = load()
    x = np.array([1., 2.], dtype='<f8')
    arrays = dict(native=x+.001, ideal_native=x+.0005, mixed=x+.0002,
                  ideal_reference=x, reference=x, native_input=x.astype('<f4'),
                  reference_input=x.astype('<f4'))
    metrics, effects = m.drift.decompose(*(arrays[k] for k in ('native','ideal_native','mixed','ideal_reference','reference')))
    arrays.update(effects)
    row = dict(metrics=metrics, input=m.drift.scan.error_score(np.zeros(2), x))
    m.retained_case(arrays, row)
    broken = dict(arrays); broken['total'] = broken['total']+.1
    with pytest.raises(ValueError): m.retained_case(broken, row)
    with pytest.raises(ValueError): m.retained_case({k:v for k,v in arrays.items() if k!='mixed'}, row)


def test_retained_verdict_does_not_turn_numerical_failure_into_pass():
    m = load()
    rows = [dict(role='candidate', metrics={'total':{'passed':False}})]
    report = dict(complete=True, model_acceptance=False, numerical_passed=False, cases=rows)
    proof = dict(complete=True, model_acceptance=False, numerical_passed=False, cases=384,
                 reference_reconstructions=192, downstream_controls=3, report_sha256='bound')
    m.retained_verdict(report, proof, 'bound')
    for key in ('numerical_passed','report_sha256','complete'):
        bad = proof | {key: True if key=='numerical_passed' else 'wrong' if key=='report_sha256' else False}
        with pytest.raises(ValueError): m.retained_verdict(report, bad, 'bound')
