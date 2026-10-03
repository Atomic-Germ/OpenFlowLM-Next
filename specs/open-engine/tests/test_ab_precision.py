"""Real AB replay keeps dot and nonlinear error separate from model acceptance."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def module():
    spec = importlib.util.spec_from_file_location('ab_diagnosis', ROOT/'utilities/diagnose-wide-ab.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_conditional_nonlinear_uses_given_logits():
    m = module()
    dots = np.array([[0., 1.], [0., -2.]], np.float32)
    a, dt = np.array([-1., -.5]), np.array([0., .25])
    got = m.nonlinear(dots, a, dt)
    np.testing.assert_allclose(got[0], np.exp(a*np.logaddexp(0, dots[0]+dt)))
    np.testing.assert_allclose(got[1], 1/(1+np.exp(-dots[1].astype(np.float64))))


def test_compare_requires_dot_accuracy_repeat_and_intact_inputs(tmp_path):
    m = module()
    ref = np.zeros((4,48), np.float32); ref[2:] = .5
    np.savez(tmp_path/'reference.npz', ab=ref, a=np.full(48,-1.), dt=np.zeros(48))
    ref.tofile(tmp_path/'recorded.bin')
    names = ['reference.npz', 'recorded.bin']
    meta = dict(tag='test', head=4, sha256={n:m.sha(tmp_path/n) for n in names})
    (tmp_path/'ab-replay-fixture.json').write_text(json.dumps(meta))
    def captures(value):
        for i in range(2): (tmp_path/f'got{i}.bin').write_bytes(value.tobytes()+m.GUARD)
    got = ref.copy(); got[1,4] = .01; captures(got)
    assert m.compare(tmp_path, True) == 1
    captures(ref)
    assert m.compare(tmp_path, True) == 0
    report = json.loads((tmp_path/'ab-replay-results.json').read_text())
    assert report['diagnostic_only'] and report['model_passed'] is False
    (tmp_path/'got1.bin').write_bytes(got.tobytes()+m.GUARD)
    assert m.compare(tmp_path, True) == 1
    (tmp_path/'got1.bin').write_bytes(ref.tobytes()+bytes(64))
    with pytest.raises(ValueError, match='canary'): m.compare(tmp_path, True)
    (tmp_path/'recorded.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'): m.compare(tmp_path, True)
