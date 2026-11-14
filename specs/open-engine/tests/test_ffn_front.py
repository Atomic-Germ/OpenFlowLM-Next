"""Independent FFN front reference must not hide errors behind BF16 inputs."""
from types import SimpleNamespace

import numpy as np
import pytest

from ffn_front import compare_front, read_front, metric
from q4nx import f32_to_bf16, bf16_to_f32


def fixture():
    res = np.array([1., -2.])
    norm = np.array([.7, 1.1])
    up = np.array([[1., 2.], [-3., 1.], [2., 1.]])
    gate = np.array([[2., -1.], [1., 2.], [-1., 1.]])
    x = res / np.sqrt(np.mean(res**2) + 1e-6) * norm
    g = gate @ x
    h = (up @ x) * g / (1 + np.exp(-g))
    return [res, norm, up, gate, x, h, 1e-6]


def test_exact_front_passes():
    result = compare_front(*fixture())
    assert result['passed']
    assert set(result['metrics']) == {'norm', 'front_h', 'activation_from_captured_norm'}
    assert all(m['maxrel'] < 1e-14 for m in result['metrics'].values())


def test_rounded_input_cannot_replace_primary_reference():
    res = np.ones(2); norm = np.array([1., 1.001])
    x = bf16_to_f32(f32_to_bf16((res / np.sqrt(1+1e-6)*norm).astype(np.float32)))
    result = compare_front(res, norm, np.array([[1., -1.]]), np.ones((1, 2)), x, np.zeros(1), 1e-6)
    assert not result['passed']
    assert not result['metrics']['front_h']['passed']
    assert result['metrics']['activation_from_captured_norm']['passed']
    assert result['norm_bf16_rounding']['mismatched_values'] == 0
    assert result['rounded_norm_diagnostic']['passed']
    assert not result['norm_rounding_effect']['passed']


@pytest.mark.parametrize('index', [4, 5])
def test_wrong_norm_or_activation_fails(index):
    args = fixture(); args[index] = args[index] + 1
    assert not compare_front(*args)['passed']


@pytest.mark.parametrize('index', range(6))
@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nonfinite_refused(index, value):
    args = fixture(); args[index].flat[0] = value
    with pytest.raises(ValueError, match='finite'):
        compare_front(*args)


@pytest.mark.parametrize('index', range(6))
def test_shape_mismatch_refused(index):
    args = fixture(); args[index] = args[index][:-1]
    with pytest.raises(ValueError, match='shape'):
        compare_front(*args)


@pytest.mark.parametrize('eps', [0, -1, np.nan, np.inf])
def test_invalid_epsilon_refused(eps):
    args = fixture(); args[-1] = eps
    with pytest.raises(ValueError, match='epsilon'):
        compare_front(*args)


def test_stable_silu_large_negative_gate():
    args = [np.ones(2), np.ones(2), np.ones((1, 2)), np.full((1, 2), -1e4),
            np.ones(2)/np.sqrt(1+1e-6), np.zeros(1), 1e-6]
    with np.errstate(over='raise', invalid='raise'):
        assert compare_front(*args)['passed']


def test_strict_bound_and_zero_reference():
    assert not metric(np.array([201.]), np.array([200.]))['passed']
    assert metric(np.zeros(1), np.zeros(1))['passed']
    assert not metric(np.array([1e-30]), np.zeros(1))['passed']


@pytest.mark.parametrize('full', [False, True])
def test_mixed_precision_capture_offsets_and_truncation(tmp_path, full):
    prefix = 'AA_' if full else 'A_'; base = 8 if full else 0
    layout = SimpleNamespace(**{prefix+'BYTES': base+24, prefix+'RES':base,
                               prefix+'XM':base+8, prefix+'H':base+12})
    data = bytearray(base+24)
    data[base:base+8] = np.array([1., -2.], np.float32).tobytes()
    data[base+8:base+12] = f32_to_bf16(np.array([.5, -.25], np.float32)).tobytes()
    data[base+12:] = np.array([3., 4., 5.], np.float32).tobytes()
    p=tmp_path/'act';p.write_bytes(data)
    r=read_front(p,layout,2,3,full)
    np.testing.assert_array_equal(r['xm'], [.5,-.25])
    np.testing.assert_array_equal(r['res'], [1,-2])
    np.testing.assert_array_equal(r['h'], [3,4,5])
    p.write_bytes(data[:-1])
    with pytest.raises(ValueError, match='bytes'):
        read_front(p,layout,2,3,full)


def test_region_bounds_and_nonfinite(tmp_path):
    p=tmp_path/'act';p.write_bytes(bytes(24))
    layout=SimpleNamespace(A_BYTES=24,A_RES=0,A_XM=8,A_H=20)
    with pytest.raises(ValueError, match='outside'):
        read_front(p,layout,2,3,False)
    layout.A_H=12
    data=bytearray(24);data[8:12]=np.array([0x7fc0,0],np.uint16).tobytes();p.write_bytes(data)
    with pytest.raises(ValueError, match='finite'):
        read_front(p,layout,2,3,False)


@pytest.fixture
def tiny_fixture(tmp_path, monkeypatch):
    import ffn_front as mod
    from recipes.spec import LINEAR, FULL
    res,norm,up,gate,x,h,eps=fixture()
    offsets=SimpleNamespace(A_BYTES=24,A_RES=0,A_XM=8,A_H=12,
                            AA_BYTES=32,AA_RES=8,AA_XM=16,AA_H=20)
    meta=dict(spec_hash='spec',build_key='export',max_ctx=4,hidden=2,layers=2,tokens=2)
    spec=SimpleNamespace(family='qwen35',hidden=2,intermediate=3,num_layers=2,
                         norm_eps=eps,layer_types=[LINEAR,FULL])
    monkeypatch.setattr(mod,'spec_from_model_dir',lambda _:spec)
    monkeypatch.setattr(mod,'manifest',lambda *a,**k:meta.copy())
    monkeypatch.setattr(mod,'load_bundle',lambda *a:meta.copy())
    monkeypatch.setattr(mod,'load_reference',lambda _:meta.copy())
    monkeypatch.setattr(mod,'layout',lambda *a,**k:offsets)
    monkeypatch.setattr(mod,'Q4NX',lambda _:SimpleNamespace(bf16=lambda _:norm,
        matmul_w=lambda name,*a:up if 'up_proj' in name else gate))
    (tmp_path/'decode_reference.json').write_text('{}')
    for layer in range(2):
        for suffix in ('','_t1'):
            data=bytes(8*layer)+res.astype(np.float32).tobytes()+f32_to_bf16(x.astype(np.float32)).tobytes()+h.astype(np.float32).tobytes()
            (tmp_path/f'y_act{layer}{suffix}.bin').write_bytes(data)
    return mod,tmp_path,meta


def test_every_layer_type_and_position_checked(tiny_fixture):
    mod,out,meta=tiny_fixture
    result=mod.check(out,out,out,[0,1])
    assert result['passed'] and len(result['results'])==4
    assert len(result['capture_sha256'])==4
    (out/'y_act1_t1.bin').unlink()
    with pytest.raises(OSError):
        mod.check(out,out,out,[0,1])


@pytest.mark.parametrize('layers',[[],[-1],[2],[0,0]])
def test_invalid_coverage_refused(tiny_fixture,layers):
    mod,out,meta=tiny_fixture
    with pytest.raises(ValueError,match='indices'):
        mod.check(out,out,out,layers)


@pytest.mark.parametrize('key',['spec_hash','build_key'])
def test_stale_export_refused(tiny_fixture,monkeypatch,key):
    mod,out,meta=tiny_fixture
    monkeypatch.setattr(mod,'load_bundle',lambda *a:dict(meta,**{key:'other'}))
    with pytest.raises(ValueError,match=key):
        mod.check(out,out,out,[0])


def test_cli_rejects_numerical_failure(tiny_fixture,monkeypatch,capsys):
    import json
    mod,out,meta=tiny_fixture
    p=out/'y_act0_t1.bin';data=bytearray(p.read_bytes());data[12:]=np.full(3,100.,np.float32).tobytes();p.write_bytes(data)
    monkeypatch.setattr('sys.argv',['ffn_front','--model-dir',str(out),'--kernel-dir',str(out),'--out',str(out),'--layers','0,1'])
    assert mod.main()==1
    r=json.loads(capsys.readouterr().out)
    assert not r['passed'] and len(r['results'])==4
    assert not r['results'][1]['metrics']['front_h']['passed']


def test_empty_shapes_refused():
    with pytest.raises(ValueError,match='shape'):
        compare_front(np.zeros(0),np.zeros(0),np.zeros((0,0)),np.zeros((0,0)),np.zeros(0),np.zeros(0),1e-6)
