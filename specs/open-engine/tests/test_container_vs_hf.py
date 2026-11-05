"""A source-weight check must not pass empty coverage or non-finite tensors."""
import json
import sys

import numpy as np
import pytest

import container_vs_hf as check


def test_missing_requested_layers_fails(tmp_path, monkeypatch, capsys):
    (tmp_path / 'config.json').write_text(json.dumps(dict(num_attention_heads=2, head_dim=2)))
    monkeypatch.setattr(check, 'Q4NX', lambda p: object())
    monkeypatch.setattr(check, 'Shard', lambda p: type('Empty', (), {'has': lambda self, name: False})())
    monkeypatch.setattr(sys, 'argv', ['container_vs_hf.py', '--model-dir', str(tmp_path),
                                    '--hf-shard', str(tmp_path / 'empty'), '--layers', '0,3'])
    assert check.main() == 1
    assert 'ALL MATCH' not in capsys.readouterr().out


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_tensor_never_matches(bad):
    assert not check.matches(np.array([1., bad]), np.array([1., 2.]))
    assert not check.matches(np.array([1., 2.]), np.array([1., bad]))


def test_constant_and_scalar_tensors_require_finite_equality():
    assert check.matches(np.ones(4), np.ones(4))
    assert check.matches(np.array([2.]), np.array([2.]))
    assert not check.matches(np.ones(4), np.ones(4) * 2)
    assert not check.matches(np.array([2.]), np.array([3.]))
    assert not check.matches(np.ones(4), np.ones(3))


def test_quantized_projection_keeps_correlation_gate():
    x = np.arange(12, dtype=np.float64)
    assert check.matches(x + np.sin(x) * 0.01, x)
    assert not check.matches(-x, x)
