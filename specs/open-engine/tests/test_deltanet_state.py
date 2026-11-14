"""Persistent DeltaNet checks must carry their own state and cover every byte."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import deltanet_state as mod
from q4nx import f32_to_bf16
from recipes.spec import LINEAR, FULL


def records(heads=2):
    r = np.zeros((heads, 512), np.float64)
    r[:, 0] = 1
    r[:, 128] = 2
    r[:, 257] = 3
    r[:, 384:386] = [.5, .25]
    return r


def test_fp64_recurrence_orientation_decay_and_carry():
    r = records()
    s = np.zeros((2, 128, 128))
    first, o = mod.step_reference(s, r)
    assert first.dtype == np.float64
    assert first[0, 0, 1] == .75 and first[0, 1, 0] == 0
    assert o[0, 1] == 1.5 / np.sqrt(128)
    second, _ = mod.step_reference(first, r)
    assert second[0, 0, 1] == 1.03125
    assert not s.any()  # the oracle must not mutate its caller's state


def sequence():
    r = records(); s = np.zeros((2, 128, 128)); conv = np.zeros((3, 384), np.uint16)
    captures = []
    for t in range(3):
        qkv = np.linspace(-1.01, 2.01 + t, 384, dtype=np.float32)
        conv = np.vstack([conv[1:], f32_to_bf16(qkv)])
        s, o = mod.step_reference(s, r)
        captures.append(dict(records=r.copy(), state=np.pad(s, ((0, 0), (0, 12), (0, 0))),
                             output=o.copy(), conv=conv.copy(), qkv=qkv))
    return captures


def test_three_tokens_have_independent_state_and_head_diagnostics():
    rows = mod.compare_sequence(sequence())
    assert len(rows) == 3 and all(r['passed'] for r in rows)
    assert all(len(r['heads']) == 2 for r in rows)
    assert all(r['head_diagnostics_passed'] for r in rows)


def test_each_token_cannot_restart_or_adopt_captured_state():
    caps = sequence()
    caps[1]['state'] = caps[0]['state'].copy()
    rows = mod.compare_sequence(caps)
    assert rows[0]['passed'] and not rows[1]['passed'] and rows[2]['passed']
    # A corrupted previous snapshot must not change the third reference.
    assert rows[2]['state']['maxrel'] == 0


@pytest.mark.parametrize('damage', ['conv', 'padding', 'output', 'last_head'])
def test_corruption_at_later_positions_fails(damage):
    caps = sequence()
    if damage == 'conv': caps[2]['conv'][0, -1] ^= 1
    if damage == 'padding': caps[2]['state'][-1, -1, -1] = 1e-12
    if damage == 'output': caps[2]['output'][-1, -1] = 1
    if damage == 'last_head': caps[2]['state'][-1, 0, 1] += .1
    rows = mod.compare_sequence(caps)
    assert not rows[2]['passed']


def test_head_local_failure_remains_visible_under_whole_tensor_pass():
    caps = sequence()
    for c in caps:
        c['records'][1, 257] *= 1e-6
        c['state'][1] *= 1e-6; c['output'][1] *= 1e-6
    caps[2]['output'][1, 1] *= 1.001
    r = mod.compare_sequence(caps)[2]
    assert r['passed'] and not r['head_diagnostics_passed']
    assert not r['heads'][1]['output']['passed']


@pytest.mark.parametrize('damage', ['nan', 'decay', 'beta', 'shape', 'reserved'])
def test_invalid_records_are_rejected(damage):
    r = records()
    if damage == 'nan': r[0, 1] = np.nan
    if damage == 'decay': r[0, 384] = -1
    if damage == 'beta': r[0, 385] = 1.1
    if damage == 'shape': r = r[:, :-1]
    if damage == 'reserved': r[-1, -1] = 1
    with pytest.raises(ValueError): mod.step_reference(np.zeros((2, 128, 128)), r)


def test_zero_reference_and_strict_bound():
    z = np.zeros(4)
    assert mod.metric(z, z)['passed']
    assert not mod.metric(z + 1e-20, z)['passed']
    assert not mod.metric(np.ones(4), np.zeros(3))['passed']
    assert not mod.metric(np.array([1., 1e-4]), np.array([1., 0.]))['passed']


def test_cosine_is_a_separate_gate():
    ref = np.zeros(16384); ref[0] = 1
    got = ref.copy(); got[1:] = 9e-5
    result = mod.metric(got, ref)
    assert result['maxrel'] < 1e-4 and not result['passed']


def test_conv_shift_preserves_signed_zero_bits():
    caps = sequence()
    caps[0]['qkv'][0] = -0.
    caps[0]['conv'][-1, 0] = 0  # numerically equal but the sign bit was lost
    assert not mod.compare_sequence(caps)[0]['conv_bit_exact']


@pytest.fixture
def geometry():
    spec = SimpleNamespace(family='qwen35', hidden=2, num_layers=4, layer_types=[LINEAR]*3+[FULL],
                           lin_key_dim=128, lin_value_dim=128, lin_key_heads=1, lin_value_heads=2,
                           lin_qkv_dim=512, conv_kernel=4)
    offsets = SimpleNamespace(A_QKV=0, A_VEC=2048, A_O=6144, A_BYTES=7168,
                              S_ROWS=140, S_HEAD_BYTES=140*128*4, STATE_S_OFF=3*512*2,
                              STATE_BYTES=3*512*2+2*140*128*4)
    return spec, offsets


def encode(c, spec, offsets):
    # Adapt the channel count without altering the recurrence example.
    qkv = np.pad(c['qkv'], (0, spec.lin_qkv_dim-c['qkv'].size))
    conv = np.pad(c['conv'], ((0, 0), (0, spec.lin_qkv_dim-c['conv'].shape[1])))
    act = qkv.astype('<f4').tobytes()+c['records'].astype('<f4').tobytes()+c['output'].astype('<f4').tobytes()
    state = conv.astype('<u2').tobytes()+c['state'].astype('<f4').tobytes()
    assert len(act) == offsets.A_BYTES and len(state) == offsets.STATE_BYTES
    return act, state


@pytest.mark.parametrize('damage', ['act_short', 'act_long', 'state_short', 'state_long', 'conv_nan', 'state_nan', 'qkv_nan'])
def test_binary_reader_rejects_incomplete_or_nonfinite_buffers(geometry, damage):
    spec, off = geometry
    act, state = map(bytearray, encode(sequence()[0], spec, off))
    if damage == 'act_short': act.pop()
    if damage == 'act_long': act += b'x'
    if damage == 'state_short': state.pop()
    if damage == 'state_long': state += b'x'
    if damage == 'conv_nan': state[:2] = b'\xc0\x7f'
    if damage == 'state_nan': state[-4:] = np.float32(np.nan).tobytes()
    if damage == 'qkv_nan': act[:4] = np.float32(np.nan).tobytes()
    with pytest.raises(ValueError): mod.read_boundaries(act, state, spec, off)


@pytest.mark.parametrize('field,value', [('S_ROWS',127), ('S_HEAD_BYTES',1), ('STATE_S_OFF',1),
                                        ('A_VEC',-4), ('A_O',3), ('A_QKV',999999)])
def test_binary_reader_rejects_invalid_layout(geometry, field, value):
    spec, off = geometry
    act, state = encode(sequence()[0], spec, off)
    setattr(off, field, value)
    with pytest.raises(ValueError): mod.read_boundaries(act, state, spec, off)


@pytest.fixture
def bundle(tmp_path, monkeypatch, geometry):
    spec, off = geometry
    meta = dict(hidden=2, vocab=32, layers=4, tokens=3, max_ctx=8, spec_hash='spec', build_key='build')
    (tmp_path/'decode_reference.json').write_text(json.dumps(meta))
    (tmp_path/f'zstate_{LINEAR}.bin').write_bytes(bytes(off.STATE_BYTES))
    for l in range(3):
        for t, c in enumerate(sequence()):
            act, state = encode(c, spec, off)
            suffix = '' if t == 0 else f'_t{t}'
            (tmp_path/f'y_act{l}{suffix}.bin').write_bytes(act)
            (tmp_path/f'y_state{l}{suffix}.bin').write_bytes(state)
    monkeypatch.setattr(mod, 'spec_from_model_dir', lambda _: spec)
    monkeypatch.setattr(mod, 'load_reference', lambda _: meta)
    monkeypatch.setattr(mod, 'manifest', lambda *a, **k: {})
    monkeypatch.setattr(mod, 'load_bundle', lambda *a: dict(spec_hash='spec', build_key='build'))
    monkeypatch.setattr(mod, 'layout', lambda *a, **k: off)
    return tmp_path, meta


def test_complete_checker_hashes_all_positions_layers_and_initial_state(bundle):
    path, _ = bundle
    result = mod.check(path, path, path)
    assert result['passed'] and len(result['results']) == 9
    assert result['layers'] == [0, 1, 2] and result['tokens'] == 3
    assert len(result['capture_sha256']) == 19
    assert result['conditioning'] == 'captured DeltaNet records and QKV projection outputs'


@pytest.mark.parametrize('damage', ['missing', 'build', 'hidden', 'layers', 'single_token', 'initial', 'warm_initial'])
def test_checker_fails_closed(bundle, damage):
    path, meta = bundle
    if damage == 'missing': (path/'y_state2_t2.bin').unlink()
    if damage == 'build': meta['build_key'] = 'different'
    if damage == 'hidden': meta['hidden'] += 1
    if damage == 'layers': meta['layers'] = 5
    if damage == 'single_token': meta['tokens'] = 1
    if damage == 'initial': (path/f'zstate_{LINEAR}.bin').write_bytes(b'bad')
    if damage == 'warm_initial':
        p = path/f'zstate_{LINEAR}.bin'; data = bytearray(p.read_bytes()); data[-1] = 1; p.write_bytes(data)
    with pytest.raises((ValueError, OSError)): mod.check(path, path, path)


def test_checker_reports_numeric_failure_without_discarding_coverage(bundle):
    path, _ = bundle
    p = path/'y_state2_t2.bin'; data = bytearray(p.read_bytes())
    data[-4:] = np.float32(1e-20).tobytes(); p.write_bytes(data)
    r = mod.check(path, path, path)
    assert not r['passed'] and len(r['results']) == 9
    assert sum(not x['passed'] for x in r['results']) == 1
    assert not r['results'][-1]['padding_zero']
