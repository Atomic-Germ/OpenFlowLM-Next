"""Independent checks for the isolated Q4 projection precision experiment."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'utilities/qkv-precision'))
from qkv_reference import activation_table, project, unpack_chunks


def chunks(codes, scale=.25, minimum=-2.):
    """One constant-metadata tile, encode by explicit documented row/lane law."""
    tile = np.zeros((1, 5120), np.uint8)
    tile[0, :512].view('<u2')[:] = np.float32(scale).view(np.uint32) >> 16
    tile[0, 512:1024].view('<u2')[:] = np.float32(minimum).view(np.uint32) >> 16
    for r in range(32):
        for k in range(256):
            i = (r//16)*4096 + k*16 + r%16
            tile[0, 1024+i//2] |= int(codes[r, k]) << (4*(i%2))
    return tile


def test_integer_table_rounding_saturation_and_zero():
    x = np.zeros(256, np.float32)
    x[:32] = np.nextafter(np.float32(2), np.float32(0))
    x[32:64] = -x[:32]
    x[64:68] = [1, 2.5/16384, 3.5/16384, -2.5/16384]
    t = activation_table(x)
    assert np.all(t['xi'][0] == 32767)
    assert np.all(t['xi'][1] == -32768)
    assert list(t['xi'][2, :4]) == [16384, 2, 4, -2]
    assert np.all(t['shift'][3:] == 126)


def test_unpack_layout_matches_distinct_rows_blocks_and_lanes():
    codes = np.arange(32*256).reshape(32, 256) % 13
    q, d, m = unpack_chunks(chunks(codes), 256)
    np.testing.assert_array_equal(q.reshape(32, 256), codes)
    np.testing.assert_array_equal(d, .25)
    np.testing.assert_array_equal(m, -2.)


def test_exact_integer_input_matches_independent_dense_product():
    codes = np.random.default_rng(3).integers(0, 16, (32, 256))
    x = np.arange(256, dtype=np.float32) % 17 - 8
    tile = chunks(codes)
    for mode in ('original', 'quantized'):
        got = project(tile, x, sum_mode=mode)
        np.testing.assert_array_equal(got, (.25*codes-2) @ x)


def test_consistent_sum_removes_constant_weight_cancellation_error():
    # All effective weights are zero. Using original x sums with quantized
    # nibble products creates a spurious output; consistent sums cancel it.
    tile = chunks(np.full((32, 256), 8))
    x = np.tile(np.array([1, .10003, .20002, .30001], np.float32), 64)
    assert np.max(abs(project(tile, x, sum_mode='original'))) > 1e-5
    np.testing.assert_array_equal(project(tile, x, sum_mode='quantized'), 0)


@pytest.mark.parametrize('x', [np.ones(255), np.full(256, np.nan), np.full(256, np.inf)])
def test_reject_invalid_activations(x):
    with pytest.raises(ValueError):
        activation_table(x)


def test_reject_malformed_weights_and_unknown_sum_mode():
    with pytest.raises(ValueError):
        unpack_chunks(np.zeros((1, 5119), np.uint8), 256)
    with pytest.raises(ValueError):
        project(chunks(np.zeros((32, 256), int)), np.ones(256), sum_mode='typo')


def test_control_uses_byte_exact_production_headers():
    from qkv_build import sources, GEMV
    generated = sources('control')
    for name in ('gemv_q4.h', 'gemv_tab.h'):
        assert generated[name] == (GEMV/name).read_text()


def test_candidate_changes_only_direct_sum_not_quantization_or_gemv():
    from qkv_build import sources
    control, candidate = sources('control'), sources('consistent')
    assert control['gemv_q4.h'] == candidate['gemv_q4.h']
    anchor = 'inline void gemv_q4_prep_f32_direct_blocks('
    assert control['gemv_tab.h'].split(anchor)[0] == candidate['gemv_tab.h'].split(anchor)[0]
    assert 'aie::to_float<float>(aie::load_v<32>(xi + kb * 32), s)' in candidate['gemv_tab.h']


def test_private_build_refuses_source_drift():
    from qkv_build import consistent_header
    with pytest.raises(ValueError, match='anchor'):
        consistent_header('unrecognized header')


def test_private_build_refuses_stale_or_missing_generated_files(tmp_path):
    from qkv_build import sources, verify_generated
    for name, text in sources('control').items():
        (tmp_path/name).write_text(text)
    assert verify_generated(tmp_path, 'control')
    (tmp_path/'prep.cc').write_text('stale')
    with pytest.raises(ValueError, match='stale'):
        verify_generated(tmp_path, 'control')


def test_unknown_build_mode_is_rejected():
    from qkv_build import sources
    with pytest.raises(ValueError, match='mode'):
        sources('typo')


def test_projection_score_preserves_failed_primary_gate_and_control_identity():
    from qkv_replay import acceptance
    fields = {'k': {'passed': False}, 'q': {'passed': True}}
    # A control reproducing a known failure is a valid control, but the primary
    # failure must remain visible. Candidate equality cannot excuse that failure.
    assert acceptance('control', True, True, fields, {'passed': True}) == (True, False)
    assert acceptance('consistent', True, True, fields, {'passed': True}) == (False, False)
    assert acceptance('control', False, True, fields, {'passed': True}) == (False, False)
    assert acceptance('consistent', False, False, {'k': {'passed': True}}, {'passed': True}) == (False, True)


def test_table_serialization_matches_known_abi_offsets():
    from qkv_replay import table_bytes
    data = table_bytes(np.ones(256, np.float32), 'original')
    assert len(data) == 576
    np.testing.assert_array_equal(np.frombuffer(data, '<i2', count=256), 16384)
    np.testing.assert_array_equal(np.frombuffer(data, '<i4', count=8, offset=512), 14)
    np.testing.assert_array_equal(np.frombuffer(data, '<u2', count=8, offset=544), 0x4200)
    np.testing.assert_array_equal(np.frombuffer(data, '<u2', count=8, offset=560), 0)


def test_original_sum_rounding_is_bounded_but_candidate_is_byte_exact():
    from qkv_replay import table_bytes, validate_table
    x = np.ones(256, np.float32)
    data = bytearray(table_bytes(x, 'original'))
    np.frombuffer(data, '<u2', count=8, offset=560)[0] = 0x3580  # 2**-20
    assert validate_table(x, data, 'original')['passed']
    assert not validate_table(x, data, 'quantized')['passed']
    np.frombuffer(data, '<u2', count=8, offset=560)[0] = 0x3f80  # 1.0 is excessive
    assert not validate_table(x, data, 'original')['passed']
    with pytest.raises(ValueError, match='size'):
        validate_table(x, data[:-1], 'original')
