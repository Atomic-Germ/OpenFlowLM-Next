"""Independent final norm/head gates must not hide BF16 cancellation."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from lm_head import norm_reference, compare_tail, project_head, head_blocks, read_norm
from q4nx import f32_to_bf16, bf16_to_f32, CHUNK_Q8


def test_unrounded_norm_and_exact_head():
    x=np.array([1.,-2.,3.]); n=np.array([.5,1.,2.])
    h=norm_reference(x,n,1e-6); w=np.array([[1,2,3],[-3,1,2],[0,1,-1.]])
    ref,_=project_head(h[None],[(0,w)],3)
    assert compare_tail(h,h,ref[0],ref[0],ref[0],ref[0])['passed']
    np.testing.assert_allclose(h,x/np.sqrt(np.mean(x*x)+1e-6)*n)


def test_rounded_conditioned_pass_cannot_hide_composed_failure():
    h=norm_reference(np.ones(2),np.array([1.,1.001]),1e-6)
    rounded=bf16_to_f32(f32_to_bf16(h.astype(np.float32))).astype(np.float64)
    w=np.array([[1,-1],[2,-2],[-1,1.]])
    ref=w@h; got=w@rounded
    r=compare_tail(h,rounded,got,ref,got,got)
    assert not r['passed'] and not r['composed']['passed']
    assert r['conditioned']['passed'] and r['rounded_diagnostic']['passed']


def test_argmax_is_required_even_for_small_numeric_error():
    h=np.ones(2); ref=np.array([1.,1.0001,0.]); got=np.array([1.0002,1.,0.])
    r=compare_tail(h,h,got,ref,ref,ref)
    assert r['composed']['maxrel']<.005 and not r['composed']['passed']


def test_zero_reference_requires_exact_zero():
    h=np.ones(2); z=np.zeros(3)
    assert compare_tail(h,h,z,z,z,z)['passed']
    assert not compare_tail(h,h,z+1e-9,z,z,z)['passed']


@pytest.mark.parametrize('damage',['gap','overlap','short','width','nan','empty'])
def test_streamed_projection_rejects_bad_coverage_or_weights(damage):
    blocks=[(0,np.ones((2,3))),(2,np.ones((2,3)))]
    if damage=='gap':blocks[1]=(3,blocks[1][1])
    if damage=='overlap':blocks[1]=(1,blocks[1][1])
    if damage=='short':blocks.pop()
    if damage=='width':blocks[1]=(2,np.ones((2,2)))
    if damage=='nan':blocks[0][1][0,0]=np.nan
    if damage=='empty':blocks=[]
    with pytest.raises(ValueError):project_head(np.ones((2,3)),blocks,4)


def test_projection_hash_and_result_independent_of_block_size():
    x=np.arange(6.).reshape(2,3);w=np.arange(15.).reshape(5,3)
    a,ha=project_head(x,[(0,w)],5)
    b,hb=project_head(x,[(0,w[:2]),(2,w[2:])],5)
    np.testing.assert_allclose(a,x@w.T);np.testing.assert_array_equal(a,b);assert ha==hb


@pytest.mark.parametrize('eps',[0,-1,np.nan,np.inf])
def test_invalid_norm_epsilon(eps):
    with pytest.raises(ValueError):norm_reference(np.ones(2),np.ones(2),eps)


def test_norm_reader_size_finite_and_dtype(tmp_path):
    p=tmp_path/'norm';v=np.array([1.25,-2.5],np.float32);p.write_bytes(f32_to_bf16(v).tobytes())
    np.testing.assert_array_equal(read_norm(p,2),v)
    p.write_bytes(b'bad')
    with pytest.raises(ValueError):read_norm(p,2)
    p.write_bytes(f32_to_bf16(np.array([np.nan,0],np.float32)).tobytes())
    with pytest.raises(ValueError):read_norm(p,2)


def test_q8_stream_reads_rows_and_k_tiles_without_full_tensor_copy():
    # Four file tiles: two row groups and two K tiles, with different scale/code.
    chunks=[]
    for scale,code in [(1.,1),(2.,-2),(.5,3),(4.,-1)]:
        chunks.append(f32_to_bf16(np.full(256,scale,np.float32)).tobytes()+np.full(8192,code,np.int8).tobytes())
    raw=b'prefix'+b''.join(chunks)
    m=SimpleNamespace(mm=raw,data_base=6,tensors={'lm_head.weight':{'data_offsets':[0,len(raw)-6]}},chunk_bytes_of=lambda _:CHUNK_Q8)
    rows=list(head_blocks(m,512,64,block_rows=32));assert [r[0] for r in rows]==[0,32]
    np.testing.assert_array_equal(rows[0][1][:,:256],1.)
    np.testing.assert_array_equal(rows[0][1][:,256:],-4.)
    np.testing.assert_array_equal(rows[1][1][:,:256],1.5)
    np.testing.assert_array_equal(rows[1][1][:,256:],-4.)
    with pytest.raises(ValueError):list(head_blocks(m,512,96))
    with pytest.raises(ValueError):list(head_blocks(m,512,64,block_rows=1))


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    import lm_head as mod
    spec=SimpleNamespace(family='qwen35',hidden=2,num_layers=4,vocab=3,real_vocab=3,norm_eps=1e-6)
    meta=dict(hidden=2,vocab=3,layers=4,tokens=2,max_ctx=8,spec_hash='spec',build_key='build')
    (tmp_path/'decode_reference.json').write_text(json.dumps(meta))
    w=np.array([[1.,2.],[-1.,1.],[2.,-1.]])
    for t in range(2):
        suffix='' if not t else '_t1';x=np.array([1.+t,-2.],np.float32)
        x.tofile(tmp_path/f'y_res3{suffix}.bin')
        hn=bf16_to_f32(f32_to_bf16(norm_reference(x,np.ones(2),1e-6).astype(np.float32)))
        f32_to_bf16(hn).tofile(tmp_path/f'y_norm{suffix}.bin')
        (w@hn).astype(np.float32).tofile(tmp_path/f'y_logits{suffix}.bin')
    monkeypatch.setattr(mod,'spec_from_model_dir',lambda _:spec)
    monkeypatch.setattr(mod,'load_reference',lambda _:meta)
    monkeypatch.setattr(mod,'manifest',lambda *a,**k:{})
    monkeypatch.setattr(mod,'load_bundle',lambda *a:dict(meta,globals={'hn':4}))
    monkeypatch.setattr(mod,'Q4NX',lambda _:SimpleNamespace(bf16=lambda _:np.ones(2)))
    monkeypatch.setattr(mod,'head_blocks',lambda *a:[(0,w)])
    return mod,tmp_path,meta


def test_checker_complete_positions_and_provenance(bundle):
    mod,p,_=bundle;r=mod.check(p,p,p)
    assert r['passed'] and len(r['results'])==2 and len(r['capture_sha256'])==6
    assert set(r['weight_sha256'])=={'model.norm.weight','lm_head.weight'}
    assert len(r['fixture_sha256'])==64


@pytest.mark.parametrize('name',['y_res3.bin','y_norm.bin','y_logits.bin','y_norm_t1.bin'])
def test_missing_boundary_refused(bundle,name):
    mod,p,_=bundle;(p/name).unlink()
    with pytest.raises(OSError):mod.check(p,p,p)


@pytest.mark.parametrize('name',['y_res3.bin','y_norm_t1.bin','y_logits.bin'])
def test_malformed_boundary_refused(bundle,name):
    mod,p,_=bundle;(p/name).write_bytes(b'bad')
    with pytest.raises(ValueError):mod.check(p,p,p)


@pytest.mark.parametrize('field',['build_key','spec_hash'])
def test_export_fixture_identity_refused(bundle,monkeypatch,field):
    mod,p,meta=bundle
    monkeypatch.setattr(mod,'load_bundle',lambda *a:dict(meta,**{field:'other','globals':{'hn':4}}))
    with pytest.raises(ValueError,match='identity'):mod.check(p,p,p)


@pytest.mark.parametrize('field,value',[('layers',5),('hidden',3),('vocab',4)])
def test_fixture_dimension_mismatch_refused(bundle,field,value):
    mod,p,meta=bundle;meta[field]=value
    with pytest.raises(ValueError,match='dimensions'):mod.check(p,p,p)


def test_cli_numerical_failure_is_nonzero(bundle,monkeypatch,capsys):
    mod,p,_=bundle;np.zeros(3,np.float32).tofile(p/'y_logits_t1.bin')
    monkeypatch.setattr(mod.sys,'argv',['lm_head','--model-dir',str(p),'--kernel-dir',str(p),'--out',str(p)])
    assert mod.main()==1
    assert not json.loads(capsys.readouterr().out)['passed']
