"""Resident reset must retain a bound, independently scored FP32 baseline."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location('post_reset', ROOT/'utilities/prefill-production/post_reset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_fixture(m, tmp_path, monkeypatch):
    plain = [dict(route=r, passed=True) for r in m.old.runner.ROUTES]
    traced = [dict(route=r, first=dict(passed=True, outputs={r:'capture'}), identity={'file':True})
              for r in m.old.runner.ROUTES]
    schedule = {'file':True}
    saved = dict(complete=True, passed=True, model_acceptance=False, sources=m.post.sources(),
                 plain_build='plain', trace_build='trace', plain=plain, routes=traced,
                 schedule_identity=schedule)
    for kind in ('plain', 'trace'):
        (tmp_path/kind).mkdir()
        (tmp_path/kind/'ids.txt').write_text('1,2')
    monkeypatch.setattr(m.post, 'context', lambda _: (dict(ids_sha256=m.digest(tmp_path/'plain/ids.txt')), None, None))
    monkeypatch.setattr(m.post, 'measure', lambda *a: (plain, traced, schedule))
    (tmp_path/'report.json').write_text(json.dumps(saved))
    proof = dict(complete=True, passed=True, model_acceptance=False, report_sha256=m.digest(tmp_path/'report.json'))
    (tmp_path/'audit.json').write_text(json.dumps(proof))
    record = dict(baseline=str(tmp_path), baseline_hashes={n:m.digest(tmp_path/n) for n in ('report.json','audit.json')})
    return record, saved, proof


@pytest.mark.parametrize('fault', ['none','incomplete','failed_audit','unbound','source','measurement','schedule','prompt','replaced_report'])
def test_baseline_requires_bound_fresh_measurements(tmp_path, monkeypatch, fault):
    m = load()
    record, saved, proof = baseline_fixture(m, tmp_path, monkeypatch)
    if fault == 'incomplete': saved['complete'] = False
    elif fault == 'failed_audit': proof['passed'] = False
    elif fault == 'source': saved['sources'] = {}
    elif fault == 'measurement': saved['plain'][0]['passed'] = False
    elif fault == 'schedule': saved['schedule_identity'] = {}
    elif fault == 'prompt': (tmp_path/'trace/ids.txt').write_text('3,4')
    (tmp_path/'report.json').write_text(json.dumps(saved))
    proof['report_sha256'] = 'stale' if fault == 'unbound' else m.digest(tmp_path/'report.json')
    (tmp_path/'audit.json').write_text(json.dumps(proof))
    if fault != 'replaced_report':
        record['baseline_hashes'] = {n:m.digest(tmp_path/n) for n in ('report.json','audit.json')}
    else:
        (tmp_path/'report.json').write_text(json.dumps(saved)+'\n')
    if fault == 'none':
        assert set(m.baseline(record)[3]) == set(m.old.runner.ROUTES)
    else:
        with pytest.raises(ValueError): m.baseline(record)


@pytest.mark.parametrize('fault', ['second_numeric','first_identity','repeat_identity','missing_route'])
def test_reset_audit_rejects_partial_or_failed_replay(tmp_path, monkeypatch, fault):
    m = load()
    rows = [dict(route=r, first={'passed':True}, second={'passed':True},
                 identity={'file':True}, repeat_identity={'file':True}) for r in m.old.runner.ROUTES]
    if fault == 'second_numeric': rows[0]['second']['passed'] = False
    elif fault == 'first_identity': rows[0]['identity']['file'] = False
    elif fault == 'repeat_identity': rows[0]['repeat_identity']['file'] = False
    else: rows.pop()
    record = dict(complete=True, passed=True, model_acceptance=False, sources=m.sources(),
                  routes=rows, schedule_identity={'file':True})
    (tmp_path/'report.json').write_text(json.dumps(record))
    (tmp_path/'audit.json').write_text(json.dumps(dict(complete=True, passed=True)))
    (tmp_path/'ids.txt').write_text('1,2')
    monkeypatch.setattr(m, 'context', lambda _: (dict(ids_sha256=m.digest(tmp_path/'ids.txt')), None, None, {}))
    monkeypatch.setattr(m, 'measure', lambda *a: (rows, {'file':True}))
    with pytest.raises(ValueError): m.audit(SimpleNamespace(out=tmp_path))
    proof = json.loads((tmp_path/'audit.json').read_text())
    assert not proof['complete'] and not proof['passed'] and not proof['model_acceptance']
