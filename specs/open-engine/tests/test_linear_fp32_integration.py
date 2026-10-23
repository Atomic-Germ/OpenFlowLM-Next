"""Production DeltaNet declares entry/history ABI and preserves legacy geometry."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest
from recipes.load import load_spec
from recipes.qwen35 import recipe,glue_side_fills
from recipes.manifest import manifest
from decode_bundle import load_bundle
from deltanet_inputs import read_entry

ROOT=Path(__file__).resolve().parents[3]
def spec(size='27b'):return load_spec(ROOT/f'open_kernels/recipes/specs/qwen35-{size}.json')


def test_wide_linear_fp32_layout_and_contract():
    s=spec();r=recipe(s);l=r.layout;m=manifest(s)
    assert r.linear.XN_F32 and r.linear.CONV_F32 and r.linear.XN_SIDE_ELEMS==5
    assert l.A_QKV-l.A_XN==20480 and l.A_BYTES==393216
    assert l.STATE_S_OFF==122880 and l.STATE_BYTES==3317760
    assert m['layout']['linear_entry_dtype']=='float32'
    assert m['layout']['linear_conv_dtype']=='float32'
    assert m['layer_types']['linear_attention']['gemm_block']['conv_dtype']=='float32'
    assert glue_side_fills(s)==17  # throttled Pipeline(3), not simultaneous BDs


def test_legacy_linear_geometry_unchanged():
    r=recipe(spec('9b'));m=manifest(spec('9b'))
    assert not r.linear.XN_F32 and not r.linear.CONV_F32
    assert r.layout.STATE_S_OFF==3*8192*2
    assert 'linear_conv_dtype' not in m['layout']


def test_retained_linear_abi_is_rejected(tmp_path):
    m=manifest(spec());old=json.loads(json.dumps(m))
    old['layout'].pop('linear_conv_dtype');old['layout'].pop('linear_entry_dtype')
    old['layer_types']['linear_attention']['buffers']['state']['bytes']-=61440
    (tmp_path/'manifest.json').write_text(json.dumps(old));(tmp_path/'toolchain.json').write_text('{}')
    with pytest.raises(ValueError,match='recipe'):load_bundle(tmp_path,m)


def test_entry_reader_retains_fp32_low_bits():
    from types import SimpleNamespace
    a=np.array([1.001,-2.003],np.float32);l=SimpleNamespace(A_XN=0,A_BYTES=8)
    np.testing.assert_array_equal(read_entry(a.tobytes(),l,2,norm_dtype='float32'),a.astype(np.float64))
    with pytest.raises(ValueError,match='dtype'):read_entry(a.tobytes(),l,2,norm_dtype='float16')


def test_checkers_select_only_declared_storage_types():
    from deltanet_state import linear_dtypes
    assert linear_dtypes(manifest(spec()))==('float32','fp32')
    assert linear_dtypes(manifest(spec('9b')))==('bfloat16','bf16')
    for field in ('linear_entry_dtype','linear_conv_dtype'):
        with pytest.raises(ValueError,match='dtype'):linear_dtypes({'layout':{field:'float16'}})


def test_new_precision_does_not_repack_weights():
    from recipes.qwen35 import pack_plan
    from recipes import qwen36moe as m
    s=spec();before=pack_plan(s)
    # Only the storage selector is disabled; weight schema must be identical.
    from unittest.mock import patch
    with patch.object(m,'linear_fp32',return_value=False):old=pack_plan(s)
    assert before==old
