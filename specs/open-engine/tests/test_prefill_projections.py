"""Per-projection gates must reject partial dumps and errors hidden by batching."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('prefill_projections',ROOT/'utilities/prefill-projections/check.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_tiling_matches_explicit_kernel_indices():
    x=np.arange(256*256,dtype=np.float32).reshape(256,256)/128
    tiled=m.tile_input(x)
    for token,k in [(0,0),(31,63),(32,64),(127,128),(255,255)]:
        i=(((k//64*8+token//32)*8+k%64//8)*4+token%32//8)*64+k%8*8+token%8
        assert tiled[i]==m.f32_to_bf16(x)[token,k]


def test_exact_shape_finite_and_token_coverage_required():
    ref=np.ones((8,3)); got=np.zeros((8,256));got[:,[0,127,255]]=ref
    assert m.score(got,ref,ref,[0,127,255],{'up':[0,4],'gate':[4,8]})['passed']
    for bad in [got[:-1],got[:,:255],np.full_like(got,np.nan)]:
        with pytest.raises(ValueError):m.score(bad,ref,ref,[0,127,255],{'all':[0,8]})
    with pytest.raises(ValueError):m.score(got,ref,ref,[0,0,255],{'all':[0,8]})


def test_small_role_failure_not_hidden_by_large_projection():
    ref=np.ones((8,3));ref[:6]*=10000;got=np.zeros((8,256));got[:,[0,127,255]]=ref
    got[7,255]+=1
    score=m.score(got,ref,ref,[0,127,255],{'large':[0,6],'small':[6,8]})
    assert not score['passed']
    assert not score['roles']['small']['fp32_input'][2]['passed']


def test_bf16_control_does_not_replace_unrounded_input_gate():
    raw=np.ones((8,3));rounded=raw*1.02;got=np.zeros((8,256));got[:,[0,127,255]]=rounded
    s=m.score(got,raw,rounded,[0,127,255],{'all':[0,8]})
    assert not s['passed'] and s['roles']['all']['bf16_input'][0]['passed']


def test_zero_columns_and_complete_roles_are_required():
    ref=np.ones((8,3));got=np.zeros((8,256));got[:,[0,127,255]]=ref
    got[0,1]=1e-20
    assert not m.score(got,ref,ref,[0,127,255],{'all':[0,8]})['passed']
    with pytest.raises(ValueError):m.score(got,ref,ref,[0,127,255],{'partial':[0,7]})


def test_linear_output_input_is_bf16_not_fp32():
    from types import SimpleNamespace
    off=SimpleNamespace(A_BYTES=16,A_OG=0,A_OUT=8)
    values=np.array([1.25,-2.5,.125,4],np.float32)
    data=m.f32_to_bf16(values).tobytes()+np.full(2,123,np.float32).tobytes()
    np.testing.assert_array_equal(m.activation_region(data,off,'A_','OG',4),values)
    with pytest.raises(ValueError):m.activation_region(data[:-1],off,'A_','OG',4)


def test_all_five_manifest_shapes_and_separate_roles_are_covered():
    from recipes.load import load_spec
    from recipes.manifest import manifest
    s=load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json')
    cases=m.cases(manifest(s,max_ctx=2048))
    assert len(cases)==8
    assert len({c['kernel'] for c in cases})==5
    assert set(next(c for c in cases if c['tag']=='L3_entry')['roles'])=={'q','k','v','gate'}
    assert set(next(c for c in cases if c['tag']=='L0_entry')['roles'])=={'q','k','v','z'}
    assert all(sum(hi-lo for lo,hi in c['roles'].values())==c['n'] for c in cases)


def test_changed_input_hash_is_rejected(tmp_path):
    p=tmp_path/'input.bin';p.write_bytes(b'good');hashes={str(p):m.digest(p)}
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='provenance mismatch'):m.verify(hashes)


def test_failed_run_cannot_leave_old_success(tmp_path):
    from types import SimpleNamespace
    import json
    (tmp_path/'report.json').write_text('{"passed":true}')
    with pytest.raises(FileNotFoundError):m.run(SimpleNamespace(out=tmp_path,harness=tmp_path/'missing'))
    assert not json.loads((tmp_path/'report.json').read_text())['passed']
