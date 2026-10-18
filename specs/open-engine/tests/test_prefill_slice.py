"""Causal prefill checks must not accept missing positions or divergent continuation."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
s=importlib.util.spec_from_file_location('prefill_slice',ROOT/'utilities/prefill-slice/check.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_complete_prompt_and_continuation_coverage():
    assert m.expected_dumps('seq',3)==['seq_p0.bin','seq_p1.bin','seq_p2.bin','seq_t0.bin','seq_t1.bin']
    with pytest.raises(ValueError):m.expected_dumps('seq',0)


def test_logits_require_shape_finite_correlation_and_argmax():
    ref=np.arange(12,dtype=float)
    assert m.logit_score(ref,ref,10)['passed']
    bad=ref.copy();bad[8]=bad[9]+.001
    assert not m.logit_score(bad,ref,10)['passed']
    for bad in [ref[:-1],np.full(12,np.nan),np.zeros(12)]:
        with pytest.raises(ValueError):m.logit_score(bad,ref,10)


def test_continuation_rejects_different_token_despite_matching_logits():
    with pytest.raises(ValueError,match='continuation'):
        m.check_tokens('token 7\ntoken 8\n',[6,8])
    m.check_tokens('prefill 3 tokens\ntoken 6\ntoken 8\n',[6,8])
    with pytest.raises(ValueError):m.check_tokens('token 6\n',[6,8])


def test_read_rejects_partial_nonfinite_and_changed_reference(tmp_path):
    p=tmp_path/'x';np.arange(4,dtype='<f4').tofile(p);h=m.digest(p)
    np.testing.assert_array_equal(m.read(p,4,h),np.arange(4))
    with pytest.raises(ValueError):m.read(p,3)
    np.ones(4,dtype='<f4').tofile(p)
    with pytest.raises(ValueError):m.read(p,4,h)
    np.full(4,np.nan,dtype='<f4').tofile(p)
    with pytest.raises(ValueError):m.read(p,4)


def test_last_residual_adds_both_down_partials():
    from types import SimpleNamespace
    l=SimpleNamespace(AA_BYTES=48,AA_RES=0,AA_OUT2=16,AA_OUT2B=32)
    a=np.arange(12,dtype='<f4').tobytes()
    np.testing.assert_array_equal(m.final_residual(a,l,4),np.array([12,15,18,21]))
    with pytest.raises(ValueError):m.final_residual(a[:-4],l,4)


def test_failed_run_invalidates_prior_success(tmp_path):
    from types import SimpleNamespace
    import json
    (tmp_path/'report.json').write_text('{"passed":true}')
    with pytest.raises(FileNotFoundError):m.run(SimpleNamespace(out=tmp_path,cli=tmp_path/'missing'))
    assert not json.loads((tmp_path/'report.json').read_text())['passed']


def test_changed_source_is_rejected(tmp_path):
    p=tmp_path/'source';p.write_text('before');hashes={str(p):m.digest(p)}
    p.write_text('after')
    with pytest.raises(ValueError,match='provenance mismatch'):m.verify(hashes)


def test_batched_q8_head_matches_existing_scalar_reference():
    from types import SimpleNamespace
    from q4nx import Q4NX,f32_to_bf16
    rng=np.random.default_rng(8);hidden=512;vocab=64
    chunks=np.zeros((vocab//32*hidden//256,8704),np.uint8)
    chunks[:,:512]=f32_to_bf16(np.full((len(chunks),256),.015625,np.float32)).view(np.uint8)
    chunks[:,512:]=rng.integers(-127,128,(len(chunks),8192),dtype=np.int16).astype(np.int8).view(np.uint8)
    model=SimpleNamespace(raw=lambda name:chunks.tobytes(),chunk_bytes_of=lambda name:8704,hidden=hidden)
    xs=rng.standard_normal((3,hidden)).astype(np.float32)
    want=np.stack([Q4NX.lmhead_logits(model,x,block=2) for x in xs])
    got=m.head_batch(model,xs,vocab)
    np.testing.assert_allclose(got,want,rtol=2e-5,atol=2e-5)
    with pytest.raises(ValueError):m.head_batch(model,xs[:,:-1],vocab)
