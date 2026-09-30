"""Cancellation regression uses the packed bytes and an analytic exact result."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from q4_1_pack import pool_reference

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def projection_test(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'utilities'))
    spec = importlib.util.spec_from_file_location('projection_test', ROOT/'utilities/test-dense-projection.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('k', [5120, 6144])
def test_cancellation_fixture_has_exact_nonzero_answer(k, projection_test):
    pool, x = projection_test.cancellation_case(128, k)
    expected = np.full(128, -(k // 32) * 2.0**-24, np.float32)
    np.testing.assert_array_equal(pool_reference(pool, x, 128, k, 2), expected)


def test_cancellation_gate_rejects_lost_residual_and_checks_canaries(tmp_path, projection_test):
    p = projection_test
    (tmp_path/'probe-toolchain.json').write_text(json.dumps(dict(projection_k=5120, projection_n=128)))
    for name in ('final.xclbin', 'insts.bin'):
        (tmp_path/name).write_bytes(b'test artifact')
    p.prepare(tmp_path, cancellation=True)
    for i in range(5):
        (tmp_path/f'got{i}.bin').write_bytes((tmp_path/f'ref{i}.bin').read_bytes()+p.GUARD)
    assert p.compare(tmp_path) == 0
    (tmp_path/'got0.bin').write_bytes(np.zeros(128, np.float32).tobytes()+p.GUARD)
    assert p.compare(tmp_path) == 1
    (tmp_path/'got0.bin').write_bytes((tmp_path/'ref0.bin').read_bytes()+bytes(64))
    with pytest.raises(ValueError, match='canary'):
        p.compare(tmp_path)


def test_product_probe_always_includes_cancellation_gate(tmp_path, projection_test):
    (tmp_path/'probe-toolchain.json').write_text(json.dumps(dict(
        projection_k=6144, projection_n=128, product_correction=True)))
    for name in ('final.xclbin', 'insts.bin'):
        (tmp_path/name).write_bytes(b'test artifact')
    projection_test.prepare(tmp_path)
    fixture = json.loads((tmp_path/'projection-fixture.json').read_text())
    assert fixture['inputs'] == 14
    assert fixture['cancellation_inputs'] == list(range(9, 14))
    assert 'load w cancellation-weights.bin' in (tmp_path/'projection.cfg').read_text()
    assert 'cancellation-weights.bin' in fixture['sha256']
    np.testing.assert_array_equal(np.fromfile(tmp_path/'ref9.bin', np.float32),
                                  np.full(128, -192 * 2.0**-24, np.float32))


@pytest.mark.parametrize('args', [[], ['--scope','ffn'], ['--scope','projection']])
def test_product_mode_requires_residual_tables(args, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'utilities/probe-qwen35-wide.py'),
                             '--product-correction', '--out', str(tmp_path/'build'), *args],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'requires a corrected projection or FFN' in result.stderr
    assert not (tmp_path/'build').exists()
