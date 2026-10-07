"""Decode acceptance must reject incomplete captures and numerical failures."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'open_kernels/model/compare_decode.py'


def put(out, name, values):
    np.asarray(values, dtype=np.float32).tofile(out / name)


def seal(out, layers=2, tokens=2):
    files = {}
    for t in range(tokens):
        suffix = '' if t == 0 else f'_t{t}'
        for name in [f'ref_logits{suffix}.bin', *[f'ref_res{l}{suffix}.bin' for l in range(layers)]]:
            files[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    meta = dict(version=1, layers=layers, tokens=tokens, hidden=4, vocab=5, max_ctx=4096,
                spec_hash='test-spec', build_key='test-build', seed_token=1,
                routing='independent', references=files)
    (out / 'decode_reference.json').write_text(json.dumps(meta))


@pytest.fixture
def bundle(tmp_path):
    for t in range(2):
        suffix = '' if t == 0 else f'_t{t}'
        for prefix in ('ref', 'y'):
            put(tmp_path, f'{prefix}_logits{suffix}.bin', [1, 3, 2, 4, 0])
            for l in range(2):
                put(tmp_path, f'{prefix}_res{l}{suffix}.bin', [1, -2, 3, 4])
    seal(tmp_path)
    return tmp_path


def run(out, *args):
    return subprocess.run([sys.executable, str(SCRIPT), '--out', str(out), *args],
                          capture_output=True, text=True)


def rejected(result, diagnostic):
    assert result.returncode != 0, result.stdout
    assert diagnostic in result.stdout + result.stderr
    assert 'Traceback' not in result.stderr


def test_complete_decode_passes(bundle):
    result = run(bundle)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'token 1:' in result.stdout  # omitted --tokens checks the whole bundle
    assert result.stdout.rstrip().endswith('PASS')


@pytest.mark.parametrize('name', ['y_logits.bin', 'ref_logits.bin', 'y_res1_t1.bin',
                                 'ref_res1_t1.bin'])
def test_missing_capture_fails(bundle, name):
    (bundle / name).unlink()
    rejected(run(bundle, '--tokens', '2'), name)


@pytest.mark.parametrize('size', [0, 4, 6])
def test_wrong_logit_count_fails(bundle, size):
    put(bundle, 'y_logits.bin', [1, 3, 2, 4, 0, 2][:size])
    rejected(run(bundle), 'y_logits.bin')


def test_both_arrays_truncated_still_fail_declared_shape(bundle):
    for prefix in ('ref', 'y'):
        put(bundle, f'{prefix}_logits.bin', [1, 3, 2, 4])
    seal(bundle)
    rejected(run(bundle), 'ref_logits.bin')


def test_partial_float_is_not_silently_ignored(bundle):
    p = bundle / 'y_logits.bin'
    p.write_bytes(p.read_bytes() + b'x')
    rejected(run(bundle), 'y_logits.bin')


def test_missing_entire_last_layer_fails(bundle):
    for prefix in ('ref', 'y'):
        (bundle / f'{prefix}_res1_t1.bin').unlink()
    rejected(run(bundle, '--tokens', '2'), 'ref_res1_t1.bin')


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('name', ['y_logits.bin', 'ref_logits.bin', 'y_res0.bin', 'ref_res0.bin'])
def test_nonfinite_capture_fails(bundle, name, value):
    values = np.fromfile(bundle / name, np.float32)
    values[0] = value
    put(bundle, name, values)
    seal(bundle)
    rejected(run(bundle), name)


def test_residual_error_is_an_acceptance_gate(bundle):
    put(bundle, 'y_res0.bin', [-1, 2, -3, -4])
    rejected(run(bundle), 'layer 0')


@pytest.mark.parametrize('delta,passes', [(0.019, True), (0.021, False)])
def test_residual_normalized_error_bound(bundle, delta, passes):
    put(bundle, 'y_res0.bin', [1 + delta, -2, 3, 4])  # scale = 4, bound 0.005
    result = run(bundle)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr


@pytest.mark.parametrize('delta,passes', [(0, True), (1e-30, False)])
def test_zero_reference_residual_has_no_division_by_zero_escape(bundle, delta, passes):
    put(bundle, 'ref_res0.bin', [0, 0, 0, 0])
    put(bundle, 'y_res0.bin', [delta, 0, 0, 0])
    seal(bundle)
    result = run(bundle)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr


@pytest.mark.parametrize('token_count', ['0', '-1', '1', '3'])
def test_requested_token_count_cannot_hide_or_exceed_declared_run(bundle, token_count):
    rejected(run(bundle, '--tokens', token_count), 'tokens')


def test_changed_reference_fails_hash_check(bundle):
    put(bundle, 'ref_res0.bin', [1, -2, 3, 4.001])
    rejected(run(bundle), 'hash')


def test_missing_metadata_fails_with_regeneration_instruction(bundle):
    (bundle / 'decode_reference.json').unlink()
    rejected(run(bundle), 'make_decode.py')


@pytest.mark.parametrize('field,value', [('version', 2), ('layers', 0), ('tokens', -1),
                                        ('hidden', True), ('vocab', 0)])
def test_invalid_metadata_fails(bundle, field, value):
    path = bundle / 'decode_reference.json'
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    rejected(run(bundle), field)


def test_reference_index_cannot_omit_expected_layer(bundle):
    path = bundle / 'decode_reference.json'
    data = json.loads(path.read_text())
    del data['references']['ref_res1.bin']
    path.write_text(json.dumps(data))
    rejected(run(bundle), 'references')


def test_unequal_argmax_fails_even_with_high_correlation(bundle):
    values = [0, 1, 2, 3, 3.0001]
    for prefix in ('ref', 'y'):
        put(bundle, f'{prefix}_logits.bin', values)
    seal(bundle)
    put(bundle, 'y_logits.bin', [0, 1, 2, 3.0001, 3])
    rejected(run(bundle), 'token 0: FAIL')


def test_generator_pins_manifest_dimensions_and_reference_hashes(bundle):
    from decode_reference import save_reference, load_reference, check_reference
    manifest = dict(max_ctx_default=4096, layout=dict(hidden=4), globals=dict(logits=20),
                    layers=['linear_attention', 'full_attention'],
                    spec_hash='fixture-spec', build_key='fixture-build')
    save_reference(bundle, manifest, 2, 2, 1, 'independent')
    meta = load_reference(bundle)
    assert (meta['layers'], meta['tokens'], meta['hidden'], meta['vocab']) == (2, 2, 4, 5)
    assert (meta['spec_hash'], meta['build_key'], meta['seed_token']) == ('fixture-spec', 'fixture-build', 1)
    assert len(meta['references']) == 6
    check_reference(bundle, manifest, 2, 2, 1, 'independent')
    with pytest.raises(ValueError, match='tokens'):
        check_reference(bundle, manifest, 2, 1, 1, 'independent')
    with pytest.raises(ValueError, match='seed_token'):
        check_reference(bundle, manifest, 2, 2, 2, 'independent')
    put(bundle, 'ref_logits.bin', [0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match='hash'):
        check_reference(bundle, manifest, 2, 2, 1, 'independent')


@pytest.mark.parametrize('size', [0, 3, 5])
def test_wrong_residual_count_fails(bundle, size):
    put(bundle, 'y_res1_t1.bin', [1, -2, 3, 4, 5][:size])
    rejected(run(bundle), 'y_res1_t1.bin')


def test_residual_exact_boundary_is_rejected(bundle):
    put(bundle, 'ref_res0.bin', [125, 0, 0, 0])
    put(bundle, 'y_res0.bin', [125, 0.625, 0, 0])
    seal(bundle)
    rejected(run(bundle), 'layer 0')


@pytest.mark.parametrize('values', [[-3, -1, -2, 4, 0], [4, 4, 4, 4, 4]])
def test_logit_correlation_remains_a_gate(bundle, values):
    put(bundle, 'y_logits.bin', values)
    rejected(run(bundle), 'token 0: FAIL')


def test_reference_writer_does_not_accept_incomplete_arrays(bundle):
    from decode_reference import save_reference
    manifest = dict(max_ctx_default=4096, layout=dict(hidden=4), globals=dict(logits=20), layers=['a', 'b'],
                    spec_hash='test-spec', build_key='test-build')
    put(bundle, 'ref_res1.bin', [1, 2, 3])
    with pytest.raises(ValueError, match='ref_res1.bin'):
        save_reference(bundle, manifest, 2, 2, 1, 'independent')


def test_make_decode_publishes_contract_only_after_complete_reference(tmp_path, monkeypatch):
    """Exercise generator -> comparator wiring without model weights or an NPU."""
    from types import SimpleNamespace
    import make_decode as make
    import replica_dense
    from decode_reference import load_reference

    out = tmp_path / 'decode'
    spec = SimpleNamespace(num_layers=1, layer_types=('dense',), hidden=4, family='qwen3',
                           num_kv_heads=1, head_dim=4, real_vocab=5)
    manifest = dict(
        max_ctx_default=4096, layout=dict(hidden=4, pool_bytes=4, kv_row=4, ptab_row=4), layers=['dense'],
        spec_hash='test-spec', build_key='test-build',
        pack=dict(pool_bytes=4, chunk_bytes=4, lm_head=dict(pool_bytes=4),
                  embed={}, norm=dict(tensor='norm')),
        globals=dict(xres=16, zero=16, normw=8, logits=20), tail=[],
        contexts=dict(dx='dx/final.xclbin'),
        kernels=dict(dx=dict(context='dx', build='dx')),
        builds=dict(dx=dict(build_dir='test')),
        layer_types=dict(dense=dict(pack=dict(pool=[], consts=[]),
                                   buffers=dict(consts=4, act=4, state=dict(kind='kv', row=4)),
                                   program=[dict(op='run', kernel='dx', args=['xres'])])))
    class Weights:
        def __init__(self, path):
            pass

        def bf16(self, name):
            return np.ones(4)

        def embed(self, token, hidden):
            return np.arange(hidden, dtype=np.float64) + token

    monkeypatch.setattr(make, 'spec_from_model_dir', lambda p: spec)
    monkeypatch.setattr(make, 'manifest', lambda s, ctx: manifest)
    monkeypatch.setattr(make, 'Q4NX', Weights)
    for fn in ('build_layer_pool', 'build_consts', 'build_lmhead_pool'):
        monkeypatch.setattr(make.PK, fn, lambda *a: np.zeros(4, np.uint8))
    monkeypatch.setattr(replica_dense, 'dense_decode', lambda q, s, l, x, k, v, t, **kw: (x, k, v))
    monkeypatch.setattr(replica_dense, 'final_logits', lambda q, s, x: (x, np.arange(5, dtype=np.float64)))
    argv = ['make_decode.py', '--out', str(out), '--layers', '1', '--tokens', '2', '--token', '1']
    monkeypatch.setattr(sys, 'argv', argv)
    assert make.main() == 0
    meta = load_reference(out)
    assert (meta['layers'], meta['tokens'], meta['vocab']) == (1, 2, 5)
    for name in meta['references']:
        (out / name.replace('ref_', 'y_', 1)).write_bytes((out / name).read_bytes())
    assert run(out).returncode == 0
    pinned = (out / 'decode_reference.json').read_bytes()
    monkeypatch.setattr(sys, 'argv', argv + ['--cfg-only'])
    assert make.main() == 0
    assert (out / 'decode_reference.json').read_bytes() == pinned

    def interrupted(*args, **kwargs):
        assert not (out / 'decode_reference.json').exists()
        raise RuntimeError('reference interrupted')

    monkeypatch.setattr(replica_dense, 'dense_decode', interrupted)
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(RuntimeError, match='reference interrupted'):
        make.main()
    rejected(run(out), 'make_decode.py')


def test_cfg_only_cannot_change_reference_context(bundle):
    from decode_reference import check_reference
    manifest = dict(max_ctx_default=8192, layout=dict(hidden=4), globals=dict(logits=20),
                    layers=['a', 'b'], spec_hash='test-spec', build_key='test-build')
    with pytest.raises(ValueError, match='max_ctx'):
        check_reference(bundle, manifest, 2, 2, 1, 'independent')
