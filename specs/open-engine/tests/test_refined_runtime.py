"""Private full-depth comparison must preserve provenance and numerical failures."""
import copy
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location('refined_runtime', ROOT/'utilities/refined-runtime/check.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def test_derivative_key_binds_role_parent_and_private_build():
    m = load()
    key = m.derivative_key('parent', 'ax', 'candidate')
    assert key.startswith('private-refined-ax:')
    assert len({key, m.derivative_key('other', 'ax', 'candidate'),
                m.derivative_key('parent', 'other', 'candidate'),
                m.derivative_key('parent', 'ax', 'control')}) == 4
    with pytest.raises(ValueError): m.derivative_key('parent', 'ax', 'production')


def test_exact_control_requires_complete_inventory():
    m = load(); names = m.post.old.runner.capture_names('sequential', 2, 4)
    expected = dict.fromkeys(names, 'hash')
    assert len(m.exact_control(expected, expected, 2, 4)) == len(names)
    for bad in ({}, expected | {'extra': 'hash'}, expected | {names[0]: 'wrong'}):
        with pytest.raises(ValueError): m.exact_control(bad, expected, 2, 4)


def test_comparison_keeps_regressions_and_failed_candidate():
    m = load()
    a = dict(passed=False, residuals=[dict(layer=0, position=0, maxrel=.01, passed=False),
                                    dict(layer=0, position=1, maxrel=.001, passed=True)],
             logits=[dict(position=0, correlation=.999, passed=False)])
    b = copy.deepcopy(a); b['residuals'][0]['maxrel'] = .006
    b['residuals'][1]['maxrel'] = .002
    result = m.compare(a, b)
    assert result['residuals']['improved'] == 1
    assert result['residuals']['regressed'] == 1
    assert result['residuals']['regressions'][0]['position'] == 1
    assert result['candidate_passed'] is False
    b['residuals'].pop()
    with pytest.raises(ValueError): m.compare(a, b)


def test_run_never_overwrites_evidence(tmp_path):
    m = load()
    from types import SimpleNamespace
    with pytest.raises(FileExistsError): m.run(SimpleNamespace(out=tmp_path))


@pytest.mark.parametrize('role', ['control', 'candidate'])
def test_bundle_preserves_other_kernels_and_rejects_tampering(tmp_path, monkeypatch, role):
    import json
    m = load(); parent = tmp_path/'parent'; private = tmp_path/'private'; out = tmp_path/'bundle'
    (parent/'ax').mkdir(parents=True); (parent/'lx').mkdir(); private.mkdir()
    for kernel in ('ax', 'lx'):
        for name in ('final.xclbin', 'insts.bin'): (parent/kernel/name).write_bytes((kernel+name).encode())
    (private/'final.xclbin').write_bytes(b'private')
    (private/'insts.bin').write_bytes((parent/'ax/insts.bin').read_bytes())
    (private/'build.json').write_text('{}')
    manifest = dict(build_key='historical', spec_hash='shape', max_ctx_default=2048)
    m.write(parent/'manifest.json', manifest); (parent/'spec.json').write_text('{}')
    hashes = {str(p.relative_to(parent)):m.digest(p) for k in ('ax', 'lx') for p in (parent/k).iterdir()}
    m.write(parent/'toolchain.json', dict(sha256=hashes))
    monkeypatch.setattr(m.ax, 'verify', lambda p: dict(role=role, sha256={n:m.digest(private/n) for n in ('insts.bin','final.xclbin')}))
    monkeypatch.setattr(m.ax, 'load_spec', lambda p: None)
    monkeypatch.setattr(m.base, 'manifest', lambda *a, **kw: {})
    # Geometry and XCLBIN parsing are tested by load_bundle's own suite.
    monkeypatch.setattr(m.base, 'load_bundle', lambda *a: None)
    original = (parent/'manifest.json').read_bytes()
    m.bundle(parent, private, out, role); m.verify_bundle(out, parent, private, role)
    assert (parent/'manifest.json').read_bytes() == original
    assert (out/'ax/final.xclbin').read_bytes() == b'private'
    assert (out/'lx/final.xclbin').read_bytes() == (parent/'lx/final.xclbin').read_bytes()
    selected = json.loads((out/'manifest.json').read_text())
    assert selected['build_key'] != manifest['build_key']
    assert selected | {'build_key': 'historical'} == manifest
    record = json.loads((out/'toolchain.json').read_text()); record['role'] = 'production'
    m.write(out/'toolchain.json', record)
    with pytest.raises(ValueError, match='provenance'): m.verify_bundle(out, parent, private, role)


def test_comparison_preserves_infinite_residual_and_duplicate_rejection():
    m = load()
    a = dict(passed=False, residuals=[dict(layer=0, position=0, maxrel=None, passed=False)],
             logits=[dict(position=0, correlation=.99, passed=False)])
    b = copy.deepcopy(a); b['residuals'][0].update(maxrel=.001, passed=True)
    result = m.compare(a, b)
    assert result['residuals']['improved'] == 1 and result['candidate_passed'] is False
    a['residuals'] *= 2; b['residuals'] *= 2
    with pytest.raises(ValueError, match='coverage'): m.compare(a, b)
