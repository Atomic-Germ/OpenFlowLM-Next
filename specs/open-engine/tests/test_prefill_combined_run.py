"""Fresh combined results must not bypass control or independent residual gates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def tool():
    spec = importlib.util.spec_from_file_location('combined_run', ROOT/'utilities/prefill-combined-cli/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    m = tool()
    meta = dict(ids=[1, 2], layers=2, hidden=2, vocab=3, real_vocab=3, expected_tokens=[2, 2])
    for name in ['ref_p0.bin', 'ref_p1.bin', 'ref_cont_logits.bin', *m.base.expected_dumps('hybrid', 2)]:
        np.array([1, 2, 3], '<f4').tofile(tmp_path/name)
    for layer in range(2):
        np.ones((2, 2), '<f4').tofile(tmp_path/f'ref_res{layer}.bin')
        np.ones((1, 2), '<f4').tofile(tmp_path/f'ref_cont_res{layer}.bin')
        np.ones((3, 2), '<f4').tofile(tmp_path/f'hybrid_res{layer}.bin')
    (tmp_path/'hybrid.log').write_text('token 2\ntoken 2\nDONE\n')
    np.array([1, 1, 0, 0, 0, 0], '<f4').tofile(tmp_path/'hybrid-act.bin')
    offsets = SimpleNamespace(AA_BYTES=24, AA_RES=0, AA_OUT2=8, AA_OUT2B=16)
    return m, meta, offsets


def test_intermediate_residual_failure_cannot_hide_behind_passing_logits(tmp_path):
    m, meta, offsets = fixture(tmp_path)
    report = m.score_route(tmp_path, tmp_path, meta, offsets, 'hybrid')
    assert report['passed'] and len(report['residuals']) == 6
    np.array([[1.02, 1], [1, 1], [1, 1]], '<f4').tofile(tmp_path/'hybrid_res0.bin')
    report = m.score_route(tmp_path, tmp_path, meta, offsets, 'hybrid')
    assert report['logits_passed'] and not report['residuals_passed'] and not report['passed']


@pytest.mark.parametrize('kind', ['missing', 'extra', 'nan'])
def test_incomplete_or_nonfinite_residuals_are_rejected(tmp_path, kind):
    m, meta, offsets = fixture(tmp_path)
    if kind == 'missing':
        (tmp_path/'hybrid_res1.bin').unlink()
    elif kind == 'extra':
        (tmp_path/'hybrid_res2.bin').touch()
    else:
        np.full((3, 2), np.nan, '<f4').tofile(tmp_path/'hybrid_res0.bin')
    with pytest.raises((ValueError, FileNotFoundError)):
        m.score_route(tmp_path, tmp_path, meta, offsets, 'hybrid')


def test_control_mismatch_blocks_candidate(tmp_path):
    m = tool()
    baseline = tmp_path/'baseline'; baseline.mkdir()
    current = tmp_path/'current'; current.mkdir()
    for name in m.capture_names('sequential', 2, 2):
        (baseline/name).write_bytes(b'old')
        (current/name).write_bytes(b'old')
    assert all(m.require_identity(baseline, current, 2, 2, ('sequential',)).values())
    (current/'sequential_res0.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity'):
        m.require_identity(baseline, current, 2, 2, ('sequential',))


def test_failed_preflight_never_retains_success(tmp_path):
    m = tool()
    out = tmp_path/'run'
    with pytest.raises(FileNotFoundError):
        m.run(SimpleNamespace(out=out, prepared=tmp_path/'missing'))
    r = json.loads((out/'report.json').read_text())
    assert not r['complete'] and not r['passed'] and not r['model_acceptance']
    with pytest.raises(FileExistsError):
        m.run(SimpleNamespace(out=out, prepared=tmp_path/'missing'))
