"""Autoregressive control must select device logits and gather the selected row."""
import importlib.util
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_native_greedy_and_embedding_reject_bad_data(tmp_path):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('g++ required for native decode control test')
    binary = tmp_path/'control'
    subprocess.run([compiler, '-std=c++17', '-O1', '-fsanitize=undefined,bounds',
                    '-I'+str(ROOT/'open_kernels/harness'),
                    str(ROOT/'specs/open-engine/tests/fixtures/decode_control_test.cpp'),
                    '-o', str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary), str(tmp_path/'embedding.bin')], check=True,
                   capture_output=True, text=True)


def test_token_feedback_uses_logits_and_preserves_states():
    from recipes.wide_decode import head_commands, embedding_commands
    commands = head_commands(5120,248320,'cold-0')
    assert 'run ln y zero finalw finalres finalxn' in commands
    assert 'run lm lmw finalxn logits' in commands
    assert commands.index('run lm lmw finalxn logits') < commands.index('greedy logits 248320 token')
    assert 'dump token cold-0-next.bin 4' in commands
    assert embedding_commands(5120,248320,'cold-1') == [
        'dump token cold-1-token.bin 4',
        'embed x embedding.bin token 248320 5120']
    assert not any('state' in c or 'cache' in c or c.startswith('load token') for c in commands)


@pytest.mark.parametrize('h,n',[(0,128),(5120,0),(-1,248320)])
def test_decode_rejects_invalid_sizes(h,n):
    from recipes.wide_decode import head_commands
    with pytest.raises(ValueError): head_commands(h,n,'test')


def probe():
    spec = importlib.util.spec_from_file_location('wide_decode_probe',ROOT/'utilities/test-wide-decode.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_logits_gate_requires_exact_shape_finite_and_same_argmax():
    gate = probe().logits_metric
    ref = np.array([1.,2.,3.,4.],np.float32)
    assert gate(ref,ref)['passed']
    assert not gate(ref[:-1],ref)['passed']
    assert not gate(np.array([1,2,3,np.nan]),ref)['passed']
    assert not gate(np.array([1,2,4,3]),ref)['passed']
    assert not gate(np.ones(4),np.ones(4))['passed']
    # High correlation alone must not bless a different token at a near tie.
    assert not gate(np.array([1,2,4.00001,4]),np.array([1,2,4,4.00001]))['passed']
