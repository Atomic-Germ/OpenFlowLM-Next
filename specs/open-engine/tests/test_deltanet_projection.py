"""Projection diagnostics must retain the unrounded primary record gate."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'utilities'))
import deltanet_projection as mod
from deltanet_inputs import records_reference


def fixture(perturb=False):
    s = SimpleNamespace(hidden=2, norm_eps=1e-6, lin_key_heads=1,
                        lin_value_heads=1, lin_key_dim=128, lin_value_dim=128,
                        lin_qkv_dim=384, conv_kernel=4)
    rng = np.random.default_rng(38)
    w = dict(norm=np.ones(2), qkv=rng.normal(0, .2, (384, 2)),
             conv=np.ones((4, 384))*.25, alpha=np.zeros((1, 2)),
             beta=np.zeros((1, 2)), a=-np.ones(1), dt=np.zeros(1))
    inputs = [np.array([1., 2.]), np.array([2., -1.])]
    history = np.zeros((3, 384)); captures = []
    for t, x in enumerate(inputs):
        xn = x / np.sqrt(np.mean(x*x)+s.norm_eps)
        qkv = w['qkv'] @ xn
        if perturb and t == 0:
            qkv[128] += 1
        records, history = records_reference(s, xn, qkv, history, w)
        captures.append(dict(xn=xn, qkv=qkv, records=records))
    return s, w, inputs, captures


def test_exact_projection_variants_match_primary():
    rows = mod.compare_entry(*fixture())
    assert all(r['primary_records_passed'] for r in rows)
    assert all(m['maxrel'] == 0 for r in rows
               for variant in r['observed_vs_variant'].values() for m in variant.values())


def test_diagnostic_match_does_not_hide_primary_failure_or_causal_history():
    rows = mod.compare_entry(*fixture(perturb=True))
    assert all(not r['primary_records_passed'] for r in rows)
    for row in rows:
        assert row['variant_vs_primary']['norm_only']['k']['maxrel'] == 0
        assert row['variant_vs_primary']['qkv_only']['k']['maxrel'] > .005
        assert row['observed_vs_variant']['captured_entry']['k']['maxrel'] == 0


def test_projection_requires_complete_multitoken_inputs():
    s, w, inputs, captures = fixture()
    with pytest.raises(ValueError, match='coverage'):
        mod.compare_entry(s, w, inputs[:1], captures[:1])
    with pytest.raises(ValueError, match='coverage'):
        mod.compare_entry(s, w, inputs, captures[:1])


def test_projection_rejects_nonfinite_capture():
    s, w, inputs, captures = fixture()
    captures[0]['qkv'][0] = np.inf
    with pytest.raises(ValueError):
        mod.compare_entry(s, w, inputs, captures)


def test_cli_rejects_unvalidated_projection_quantization(monkeypatch, tmp_path):
    spec = SimpleNamespace(family='qwen35', num_layers=1, hidden=2,
                           layer_types=('linear_attention',), quant_of=lambda role: 'q8')
    meta = dict(build_key='key', spec_hash='spec', max_ctx=2048, hidden=2, layers=1)
    monkeypatch.setattr(mod, 'spec_from_model_dir', lambda path: spec)
    monkeypatch.setattr(mod, 'load_reference', lambda path: meta)
    monkeypatch.setattr(mod, 'manifest', lambda *a, **kw: meta)
    monkeypatch.setattr(mod, 'load_bundle', lambda *a: meta)
    with pytest.raises(ValueError, match='Q4_1'):
        mod.check(tmp_path, tmp_path, tmp_path, 0)
