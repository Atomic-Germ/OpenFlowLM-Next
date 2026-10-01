"""Distinguish activation rounding from down-GEMV error without changing oracles."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def diagnosis(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'utilities'))
    spec = importlib.util.spec_from_file_location('ffn_diagnosis', ROOT/'utilities/diagnose-wide-ffn.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rounding_crossing_is_separated_from_correct_down_math(diagnosis):
    # Gate=0 and arbitrary up give an exactly zero reference activation.
    # A changed device activation must affect the conditional down reference,
    # while the independent FFN reference remains zero.
    def project(name, x):
        if name == 'up': return np.ones(4, np.float32)
        if name == 'gate': return np.zeros(4, np.float32)
        return x.astype(bfloat16).astype(np.float32).copy()
    h = np.array([0, 0.25, 0, 0], np.float32)
    result, refs = diagnosis.analyze(project, np.ones(4, bfloat16), h, h.copy())
    assert result['h_bf16_differences'] == 1
    assert result['conditional_down']['maxabs'] == 0
    assert result['ffn']['maxabs'] == .25
    assert result['first_h_differences'][0]['index'] == 1
    np.testing.assert_array_equal(refs['h'], np.zeros(4))
    np.testing.assert_array_equal(refs['down_from_device_h'], h)


def test_nonfinite_and_wrong_sized_captures_are_rejected(diagnosis, tmp_path):
    p = tmp_path/'capture.bin'
    p.write_bytes(np.array([np.nan], np.float32).tobytes()+diagnosis.GUARD)
    with pytest.raises(ValueError, match='nonfinite'):
        diagnosis.capture(p, 4, np.float32)
    p.write_bytes(np.array([1], np.float32).tobytes()+bytes(64))
    with pytest.raises(ValueError, match='canary'):
        diagnosis.capture(p, 4, np.float32)
    with pytest.raises(ValueError, match='size'):
        diagnosis.capture(p, 8, np.float32)


def test_trace_preserves_band_and_up_gate_order(diagnosis):
    raw = np.arange(3*2*64, dtype=np.float32)
    u, g = diagnosis.trace_arrays(raw, 192)
    np.testing.assert_array_equal(u, np.concatenate([raw[i:i+64] for i in (0,128,256)]))
    np.testing.assert_array_equal(g, np.concatenate([raw[i:i+64] for i in (64,192,320)]))
    with pytest.raises(ValueError, match='trace size'):
        diagnosis.trace_arrays(raw[:-1], 192)


def test_trace_padding_moves_only_the_canary(diagnosis):
    raw = bytes(range(16))+diagnosis.GUARD
    padded = diagnosis.pad_activation(raw, 32)
    assert padded[:16] == raw[:16] and padded[-64:] == diagnosis.GUARD
    assert len(padded)==96
    assert np.isnan(np.frombuffer(padded[16:32],np.float32)).all()
    with pytest.raises(ValueError): diagnosis.pad_activation(raw, 8)
    with pytest.raises(ValueError): diagnosis.pad_activation(bytes(80),32)


def test_strict_trace_flag_cannot_be_silently_ignored(tmp_path):
    p = subprocess.run([sys.executable,str(ROOT/'utilities/diagnose-wide-ffn.py'),
                        '--out',str(tmp_path),'--require-exact-h-bf16'],capture_output=True,text=True)
    assert p.returncode == 2 and 'requires --compare-trace' in p.stderr


def test_block_carry_requires_product_correction(tmp_path):
    p = subprocess.run([sys.executable,str(ROOT/'utilities/probe-qwen35-wide.py'),
                        '--block-carry','--out',str(tmp_path/'absent')],capture_output=True,text=True)
    assert p.returncode == 2 and 'requires --product-correction' in p.stderr
    assert not (tmp_path/'absent').exists()


def test_trace_replay_checks_repeat_and_immutable_references(diagnosis, monkeypatch, tmp_path):
    m = diagnosis
    monkeypatch.setattr(m,'H',4)
    monkeypatch.setattr(m,'F',64)
    arena = np.zeros(68,np.float32).tobytes()+m.GUARD
    trace = np.zeros(128,np.float32).tobytes()+m.GUARD
    for name in ('act','got0','got1'): (tmp_path/f'{name}.bin').write_bytes(arena)
    for name in ('trace0','trace1'): (tmp_path/f'{name}.bin').write_bytes(trace)
    (tmp_path/'source-fo.bin').write_bytes(np.zeros(4,np.float32).tobytes()+m.GUARD)
    np.savez(tmp_path/'reference.npz',up=np.zeros(64),gate=np.zeros(64),h=np.zeros(64))
    meta = dict(tag='test',act_bytes=272,h_offset=0,out_offset=256,
                sha256={n:m.sha(tmp_path/n) for n in ('act.bin','source-fo.bin','reference.npz')})
    (tmp_path/'trace-fixture.json').write_text(json.dumps(meta))
    r = m.compare_trace(tmp_path)
    assert r['h_bf16_differences']==0 and r['repeat_exact']
    assert r['matches_recorded_h'] and r['matches_recorded_fo']
    (tmp_path/'trace1.bin').write_bytes(np.ones(128,np.float32).tobytes()+m.GUARD)
    assert not m.compare_trace(tmp_path)['repeat_exact']
    (tmp_path/'reference.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='fixture changed'): m.compare_trace(tmp_path)
