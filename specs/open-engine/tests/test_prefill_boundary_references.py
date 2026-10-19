"""Full-depth boundary references must share causal prefixes, not continuation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def tool():
    s = importlib.util.spec_from_file_location('boundary_refs', ROOT / 'utilities/prefill-slice/boundaries.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


@pytest.mark.parametrize('damage', ['none', 'residual', 'logit', 'ids', 'geometry'])
def test_causal_prefix_checks_every_layer_and_position(tmp_path, damage):
    m = tool()
    short, long = tmp_path / 'short', tmp_path / 'long'
    short.mkdir(); long.mkdir()
    common = dict(layers=64, hidden=2, vocab=4, real_vocab=4, model='m', kernels='k',
                  build_key='b', spec_hash='s', max_ctx=2048, message='fixture')
    sm = dict(common, ids=[1, 2]); lm = dict(common, ids=[1, 2, 3])
    x = np.arange(6, dtype='<f4').reshape(3, 2)
    for layer in range(64):
        x[:2].tofile(short / f'ref_res{layer}.bin')
        y = x.copy()
        if damage == 'residual' and layer == 63: y[1, 0] += 1
        y.tofile(long / f'ref_res{layer}.bin')
    for pos in range(3):
        y = np.arange(4, dtype='<f4') + pos
        if pos < 2: y.tofile(short / f'ref_p{pos}.bin')
        if damage == 'logit' and pos == 0: y[0] += 1
        y.tofile(long / f'ref_p{pos}.bin')
    # Continuation is intentionally different: each boundary has its own state.
    np.array([9, 8], '<f4').tofile(short / 'ref_cont_res63.bin')
    np.array([7, 6], '<f4').tofile(long / 'ref_cont_res63.bin')
    if damage == 'ids': lm['ids'][1] = 0
    if damage == 'geometry': lm['hidden'] = 3
    if damage == 'none':
        got = m.compare_prefix(short, sm, long, lm)
        assert got == dict(residual_arrays=64, logit_vectors=2)
    else:
        with pytest.raises(ValueError): m.compare_prefix(short, sm, long, lm)


def test_slice_comparison_uses_only_prompt_for_different_continuation_tokens(tmp_path):
    m = tool()
    small, full = tmp_path / 'small', tmp_path / 'full'
    small.mkdir(); full.mkdir()
    common = dict(hidden=2, vocab=4, real_vocab=4, model='m', kernels='k', build_key='b',
                  spec_hash='s', max_ctx=2048, message='fixture', ids=[1, 2])
    sm = dict(common, layers=4, expected_tokens=[0, 1])
    fm = dict(common, layers=64, expected_tokens=[2, 3])
    for i in range(4):
        for d in (small, full): np.arange(4, dtype='<f4').tofile(d / f'ref_res{i}.bin')
    assert m.compare_slice(full, fm, small, sm) == 4
    (full / 'ref_res3.bin').write_bytes(b'changed')
    with pytest.raises(ValueError): m.compare_slice(full, fm, small, sm)


def test_failed_boundary_audit_invalidates_old_success(tmp_path):
    m = tool()
    p = tmp_path / 'reference-audit.json'
    p.write_text('{"passed":true}')
    with pytest.raises(FileNotFoundError):
        m.audit(SimpleNamespace(out=tmp_path, slice=tmp_path / 'missing'))
    assert not json.loads(p.read_text())['passed']


@pytest.mark.parametrize('damage', ['none', 'changed', 'missing'])
def test_native_prefix_checks_each_route_without_comparing_continuation(damage):
    m = tool(); reports = {}
    for n in (2, 3):
        routes = []
        for route in ('sequential', 'hybrid', 'block_major'):
            outputs = {f'{route}_p{p}.bin': f'position{p}' for p in range(n)}
            outputs[f'{route}_t1.bin'] = f'continuation{n}'
            routes.append(dict(route=route, outputs=outputs))
        reports[n] = dict(routes=routes)
    if damage == 'changed': reports[3]['routes'][2]['outputs']['block_major_p1.bin'] = 'changed'
    if damage == 'missing': del reports[3]['routes'][0]['outputs']['sequential_p0.bin']
    if damage == 'missing':
        with pytest.raises(KeyError): m.native_prefixes(reports)
    else:
        pairs = m.native_prefixes(reports)
        assert len(pairs) == 6
        assert all(pairs.values()) == (damage == 'none')


def test_native_audit_failure_after_reference_success_is_failed(tmp_path, monkeypatch):
    m = tool()
    common = dict(model='m', kernels='k', max_ctx=2048, build_key='b', spec_hash='s',
                  sources={}, references={}, expected_tokens=[1, 2])
    monkeypatch.setattr(m, 'validate_reference', lambda path, layers, tokens: dict(common))
    monkeypatch.setattr(m, 'compare_prefix', lambda *args: {})
    monkeypatch.setattr(m, 'compare_slice', lambda *args: 4)
    monkeypatch.setattr(m.base, 'spec_from_model_dir', lambda *args: None)
    monkeypatch.setattr(m.base, 'manifest', lambda *args, **kwargs: {})
    monkeypatch.setattr(m.base, 'load_bundle', lambda *args: dict(build_key='b', spec_hash='s'))
    for root in (tmp_path, tmp_path / 'slice'):
        for n in (255, 256, 257):
            d = root / str(n); d.mkdir(parents=True)
            (d / 'reference.json').write_text('{}')
    with pytest.raises(FileNotFoundError):
        m.audit(SimpleNamespace(out=tmp_path, slice=tmp_path / 'slice', cli=tmp_path / 'missing'))
    report = json.loads((tmp_path / 'native-audit.json').read_text())
    assert not report['passed'] and not report['native_acceptance']
