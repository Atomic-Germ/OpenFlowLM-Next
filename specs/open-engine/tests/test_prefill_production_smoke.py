"""The integrated smoke gate must never accept incomplete or altered logits."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('production_smoke', ROOT/'utilities/prefill-production/smoke.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def fixture(tmp_path):
    expected = {}
    for name in smoke.names('hybrid', 2):
        p = tmp_path/name
        p.write_bytes(np.array([1., 2., 3.], dtype='<f4').tobytes())
        expected[name] = smoke.digest(p)
    return expected


def test_complete_exact_finite_logits_pass(tmp_path):
    expected = fixture(tmp_path)
    assert smoke.check_outputs(tmp_path, 'hybrid', 2, 3, expected) == expected


@pytest.mark.parametrize('mutation', ['missing', 'short', 'nan', 'different', 'extra'])
def test_invalid_logits_fail(tmp_path, mutation):
    expected = fixture(tmp_path)
    p = tmp_path/'hybrid_p0.bin'
    if mutation == 'missing':
        p.unlink()
    elif mutation == 'extra':
        (tmp_path/'hybrid_p2.bin').write_bytes(p.read_bytes())
    else:
        data = {'short': [1.], 'nan': [1., float('nan'), 3.], 'different': [1., 2., 4.]}[mutation]
        p.write_bytes(np.array(data, dtype='<f4').tobytes())
        # Even a matching control digest cannot bless NaN or the wrong shape.
        if mutation != 'different':
            expected[p.name] = smoke.digest(p)
    with pytest.raises((ValueError, FileNotFoundError)):
        smoke.check_outputs(tmp_path, 'hybrid', 2, 3, expected)


def test_reference_metadata_must_match_audited_preparation(tmp_path):
    import json
    refs = tmp_path/'references'
    refs.mkdir()
    path = refs/'reference.json'
    path.write_text(json.dumps({'ids': [1, 2]}))
    prep = tmp_path/'preparation.json'
    prep.write_text(json.dumps({'artifacts': {str(path): smoke.digest(path)}}))
    sha = smoke.digest(prep)
    assert smoke.read_fixture(tmp_path, sha)['ids'] == [1, 2]
    path.write_text(json.dumps({'ids': [3, 4]}))
    with pytest.raises(ValueError, match='reference metadata'):
        smoke.read_fixture(tmp_path, sha)
