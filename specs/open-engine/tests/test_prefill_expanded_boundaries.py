"""Long-prefix validation preserves controls, causal prefixes and numerical failures."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('expanded_boundaries', ROOT/'utilities/prefill-expanded-cli/boundaries.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def test_requires_all_boundary_lengths_and_same_geometry():
    m = load()
    cases = {n: dict(layers=64, hidden=5120, ids=list(range(n)), vocab=10,
                    real_vocab=9, model='m', kernels='k', build_key='b',
                    spec_hash='s', max_ctx=2048, message='prompt') for n in m.LENGTHS}
    m.check_cases(cases)
    with pytest.raises(ValueError): m.check_cases({255: cases[255]})
    cases[257]['ids'][10] = 99
    with pytest.raises(ValueError): m.check_cases(cases)


def make_prefixes(tmp_path, m):
    for n in m.LENGTHS:
        d = tmp_path/str(n); d.mkdir()
        for route in m.runner.ROUTES:
            # Different continuations must not be compared across prompt lengths.
            rows = np.arange(n+1, dtype='<f4'); rows[-1] = -n
            (d/f'{route}_res0.bin').write_bytes(rows.tobytes())
            for p in range(n): (d/f'{route}_p{p}.bin').write_bytes(np.array([p], '<f4').tobytes())
    return {n: dict(layers=1, hidden=1, vocab=1, ids=list(range(n))) for n in m.LENGTHS}


def test_causal_prefixes_exclude_continuation(tmp_path):
    m = load(); metas = make_prefixes(tmp_path, m)
    got = m.native_prefixes(tmp_path, metas)
    assert len(got) == 3*(255+256+2) and all(got.values())


@pytest.mark.parametrize('name', ['hybrid_res0.bin', 'block_major_p10.bin'])
def test_prefix_mismatch_is_reported(tmp_path, name):
    m = load(); metas = make_prefixes(tmp_path, m)
    p = tmp_path/'257'/name
    values = np.frombuffer(p.read_bytes(), dtype='<f4').copy(); values[0] += 1
    p.write_bytes(values.tobytes())
    assert not all(m.native_prefixes(tmp_path, metas).values())


def test_complete_failure_cannot_become_model_acceptance():
    m = load()
    rows = {str(n): {'passed': True} for n in m.LENGTHS}
    assert m.aggregate(rows, {'prefix': True})
    rows['256']['passed'] = False
    assert not m.aggregate(rows, {'prefix': True})
    with pytest.raises(ValueError): m.aggregate({'255': {'passed': True}}, {'prefix': True})
    assert not m.aggregate({str(n): {'passed': True} for n in m.LENGTHS}, {'prefix': False})


def setup_run(m, monkeypatch, tmp_path):
    prepared = tmp_path/'prepared'; prepared.mkdir(); (prepared/'preparation.json').write_text('{}')
    monkeypatch.setattr(m, 'verify_preparation', lambda p: ({}, {n: {} for n in m.LENGTHS}, None))
    monkeypatch.setattr(m.runner, 'device_lock', lambda: nullcontext())
    monkeypatch.setattr(m.base, 'memory_preflight', lambda n: 40*2**30)
    return SimpleNamespace(prepared=prepared, out=tmp_path/'native')


def test_control_mismatch_stops_before_expanded(tmp_path, monkeypatch):
    m = load(); a = setup_run(m, monkeypatch, tmp_path); called = []
    monkeypatch.setattr(m, 'capture_case', lambda variant, *args: called.append(variant) or {'passed': False})
    monkeypatch.setattr(m, 'control_identity', lambda *args: (_ for _ in ()).throw(ValueError('control mismatch')))
    with pytest.raises(ValueError, match='control mismatch'): m.run(a)
    assert called == ['paired']
    r = json.loads((a.out/'report.json').read_text())
    assert not r['complete'] and not r['passed'] and not r['model_acceptance']


def test_low_memory_preserves_failure_without_dispatch(tmp_path, monkeypatch):
    m = load(); a = setup_run(m, monkeypatch, tmp_path); called = []
    monkeypatch.setattr(m.base, 'memory_preflight', lambda n: (_ for _ in ()).throw(RuntimeError('low RAM')))
    monkeypatch.setattr(m, 'capture_case', lambda *args: called.append(args))
    with pytest.raises(RuntimeError, match='low RAM'): m.run(a)
    assert not called
    assert not json.loads((a.out/'report.json').read_text())['complete']
    with pytest.raises(FileExistsError): m.run(a)
