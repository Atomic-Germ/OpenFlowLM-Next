"""Production entry norm must keep FP32 low bits and declare the scratch ABI."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from recipes.load import load_spec
from recipes.qwen35 import recipe
from recipes.manifest import manifest
from attention_projections import read_boundaries
from decode_bundle import load_bundle

ROOT = Path(__file__).resolve().parents[3]


def spec(size='27b'):
    return load_spec(ROOT/f'open_kernels/recipes/specs/qwen35-{size}.json')


def test_attention_entry_stream_and_scratch_preserve_fp32():
    r = recipe(spec()); l = r.layout
    assert r.attn.XN_F32 and r.attn.XN_ELEMS == 5
    assert l.AA_QG-l.AA_XN == 5120*4
    assert l.AA_BYTES == 274432
    assert l.A_QKV-l.A_XN == 5*4096  # linear entry also preserves FP32
    assert r.ffn.XN_ELEMS == 3


def test_other_geometry_retains_entry_abi():
    r = recipe(spec('9b'))
    assert not r.attn.XN_F32 and r.attn.XN_ELEMS == 2
    assert r.layout.AA_QG-r.layout.AA_XN == 8192
    assert 'attn_entry_dtype' not in manifest(spec('9b'))['layout']


def test_manifest_rejects_retained_entry_abi(tmp_path):
    m = manifest(spec())
    assert m['layout']['attn_entry_dtype'] == 'float32'
    old = json.loads(json.dumps(m)); old['layout'].pop('attn_entry_dtype')
    old['layer_types']['full_attention']['buffers']['act'] -= 8192
    (tmp_path/'manifest.json').write_text(json.dumps(old))
    (tmp_path/'toolchain.json').write_text('{}')
    with pytest.raises(ValueError, match='recipe'):
        load_bundle(tmp_path, m)


def test_entry_capture_reads_unrounded_fp32(tmp_path):
    o = SimpleNamespace(AA_BYTES=24, AA_XN=0, AA_OUT=8, AA_RES=16)
    values = np.array([1.001, -2.003, 3, 4, 5, 6], np.float32)
    p = tmp_path/'act'; p.write_bytes(values.tobytes())
    r = read_boundaries(p, o, 2, norm_dtype='float32')
    np.testing.assert_array_equal(r['xn'], values[:2].astype(np.float64))
    np.testing.assert_array_equal(r['res'], values[4:].astype(np.float64))


def test_entry_capture_rejects_unknown_dtype(tmp_path):
    with pytest.raises(ValueError, match='dtype'):
        read_boundaries(tmp_path/'act', None, 2, norm_dtype='float16')
