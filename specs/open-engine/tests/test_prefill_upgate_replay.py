"""Isolated GEMM replay must reproduce captured paired arithmetic before diagnosis."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    s = importlib.util.spec_from_file_location(
        'upgate_replay', ROOT / 'utilities/prefill-upgate-replay/check.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_split_keeps_original_fp32_high_input_and_fp32_low_subtraction():
    m = load()
    x = np.array([1.003, -.7123, 0, 100.1], np.float32)
    high, low = m.split_inputs(x)
    np.testing.assert_array_equal(high, x)
    np.testing.assert_array_equal(low, x - m.proj.bf16_to_f32(m.proj.f32_to_bf16(x)))
    assert low.dtype == np.float32


@pytest.mark.parametrize('bad', [np.array([np.nan]), np.array([np.inf]), np.array([])])
def test_bad_split_input_refused(bad):
    with pytest.raises(ValueError):
        load().split_inputs(bad)


def test_control_requires_fp32_add_and_exact_capture_identity():
    m = load()
    high = np.zeros((4, 256), np.float32)
    low = high.copy()
    high[:, 7] = [1, -2, 2**24, -2**24]
    low[:, 7] = [.1, -.1, 1, -1]
    capture = np.add(high[:, 7], low[:, 7], dtype=np.float32)
    out = m.control(high, low, capture, 7)
    np.testing.assert_array_equal(out, capture)
    capture[0] = np.nextafter(capture[0], np.float32(np.inf))
    with pytest.raises(ValueError, match='identity'):
        m.control(high, low, capture, 7)


@pytest.mark.parametrize('bad', ['shape', 'nan', 'padding', 'slot', 'capture'])
def test_control_refuses_bad_dump_and_inactive_columns(bad):
    m = load()
    hi = np.zeros((4, 256), np.float32)
    lo = hi.copy()
    capture = np.ones(4, np.float32)
    hi[:, 7] = capture
    slot = 7
    if bad == 'shape':
        lo = lo[:, :255]
    elif bad == 'nan':
        hi[0, 7] = np.nan
    elif bad == 'padding':
        hi[0, 6] = 1e-30
        lo[0, 6] = -1e-30  # Even cancelling inactive outputs must fail.
    elif bad == 'slot':
        slot = -1
    else:
        capture = capture[:3]
    with pytest.raises(ValueError):
        m.control(hi, lo, capture, slot)


def test_projection_checks_keep_up_and_gate_separate():
    m = load()
    ref = np.array([1e6, 1e6, 1., 1.])
    got = ref.copy()
    got[3] += .1
    result = m.projection_metrics(got, ref)
    assert result['up']['passed']
    assert not result['gate']['passed']


def test_builder_refuses_existing_output_without_touching_evidence(tmp_path):
    m = load()
    p = tmp_path / 'build-evidence.json'
    p.write_text('preserve')
    with pytest.raises(FileExistsError):
        m.build(SimpleNamespace(out=tmp_path))
    assert p.read_text() == 'preserve'


def test_failed_run_invalidates_stale_report(tmp_path):
    m = load()
    p = tmp_path / 'report.json'
    p.write_text('{"control_valid":true}')
    with pytest.raises(FileNotFoundError):
        m.run(SimpleNamespace(out=tmp_path, build=tmp_path / 'missing'))
    result = json.loads(p.read_text())
    assert not result['control_valid'] and not result['model_acceptance']
