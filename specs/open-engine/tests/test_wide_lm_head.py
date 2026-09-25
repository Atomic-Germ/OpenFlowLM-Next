"""The LM-head oracle and production pool order must agree for the whole K."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from ml_dtypes import bfloat16
from recipes import pack

ROOT = Path(__file__).resolve().parents[3]


def oracle():
    spec = importlib.util.spec_from_file_location(
        'lm_head_fixture', ROOT / 'open_kernels/designs/lm_head_q8/make_test.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def packed_matrix(k, n=256):
    """Encode the documented file raster, then use the real production pack op."""
    rng = np.random.default_rng(2963)
    codes = rng.integers(-128, 128, (n, k), dtype=np.int8)
    scales = rng.uniform(.001, .02, (n, k // 32)).astype(bfloat16)
    expected = codes.astype(np.float64) * np.repeat(scales.astype(np.float64), 32, axis=1)
    chunks = []
    for row in range(0, n, 32):
        for col in range(0, k, 256):
            s = scales[row:row+32, col//32:col//32+8].T.copy().tobytes()
            c = codes[row:row+32, col:col+256].reshape(2, 16, 256).transpose(0, 2, 1).copy().tobytes()
            chunks.append(s + c)
    raw = b''.join(chunks)
    class Model:
        def raw(self, name): return raw
    pool = np.empty(len(raw), np.uint8)
    pack.apply_op(dict(op='lmhead_q8', tensor='lm_head.weight', dst=0,
                       in_dim=k, chunk_bytes=8704), Model(), 0, pool)
    return pool, expected


@pytest.mark.parametrize('k', [2048, 2560, 4096, 5120])
def test_oracle_uses_actual_width_and_production_pool(k):
    pool, dense = packed_matrix(k)
    x = np.zeros(k, bfloat16)
    x[[0, 255, 256, 2047, k-1]] = [1, -2, .5, 2, -1]
    expected = (dense @ x.astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(oracle().reference(pool, x, 256, batch=13), expected)


def test_batched_oracle_matches_independent_dense_and_rejects_wrong_lengths():
    pool, dense = packed_matrix(5120)
    xs = np.random.default_rng(8).normal(0, .5, (3, 5120)).astype(bfloat16)
    mod = oracle()
    np.testing.assert_allclose(mod.references(pool, xs, 256, batch=13),
                               (xs.astype(np.float64) @ dense.T).astype(np.float32), rtol=1e-6, atol=1e-6)
    for bad in (pool[:-1], np.concatenate([pool, pool])):
        with pytest.raises(ValueError): mod.references(bad, xs, 256)
    with pytest.raises(ValueError): mod.references(pool, xs[:, :-1], 256)


def test_hardware_fixture_roundtrip_and_output_canary(tmp_path):
    spec = importlib.util.spec_from_file_location('lm_probe', ROOT / 'utilities/test-wide-lm-head.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for file in ('final.xclbin', 'insts.bin'):
        (tmp_path / file).write_bytes(b'test artifact')
    mod.prepare(tmp_path, 256, 128, 1)
    import json
    meta = json.loads((tmp_path / 'lm-fixture.json').read_text())
    for i in range(len(meta['cases'])):
        (tmp_path / f'got{i}.bin').write_bytes((tmp_path / f'ref{i}.bin').read_bytes() + mod.GUARD)
    assert mod.compare(tmp_path) == 0
    with (tmp_path / 'got0.bin').open('r+b') as f:
        f.seek(-1, 2)
        f.write(b'\0')
    with pytest.raises(ValueError, match='canary'):
        mod.compare(tmp_path)
