"""Depth diagnostics retain per-position gates, including isolated late failures."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
s=importlib.util.spec_from_file_location('prefill_depth',ROOT/'utilities/prefill-slice/depth.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_depths_must_end_at_full_attention_and_be_unique():
    assert m.depths('4,8,16,32,48,60')==[4,8,16,32,48,60]
    for bad in ('','0','3','65','4,4','8,4'):
        with pytest.raises(ValueError):m.depths(bad)


def test_one_bad_position_fails_despite_matching_last_token():
    ref=np.tile(np.arange(24,dtype=float),(20,1));got=ref.copy()
    got[17,0]+=10
    result=m.score_rows(got,ref,24)
    assert not result['passed']
    assert [r['position'] for r in result['logits'] if not r['passed']]==[17]
    assert all(r['argmax']==r['reference_argmax'] for r in result['logits'])
    assert m.score_rows(ref,ref,24)['passed']


def test_missing_or_nonfinite_positions_are_rejected():
    ref=np.tile(np.arange(24,dtype=float),(20,1))
    for got in (ref[:-1],np.full(ref.shape,np.nan)):
        with pytest.raises(ValueError):m.score_rows(got,ref,24)


def test_failed_scan_cannot_leave_prior_success(tmp_path):
    import json
    from types import SimpleNamespace
    (tmp_path/'report.json').write_text('{"passed":true}')
    with pytest.raises(FileNotFoundError):m.run(SimpleNamespace(out=tmp_path,cli=tmp_path/'missing'))
    assert not json.loads((tmp_path/'report.json').read_text())['passed']


@pytest.mark.parametrize('wrong_token',[False,True])
def test_scan_keeps_all_depths_when_one_gate_fails(tmp_path,monkeypatch,wrong_token):
    import json
    from types import SimpleNamespace as NS
    meta=dict(depths=[4,8],ids=list(range(20)),vocab=24,real_vocab=24,
              model='model',kernels='kernels',max_ctx=2048,sources={},references={})
    ref=np.tile(np.arange(24,dtype='<f4'),(20,1))
    for d in meta['depths']:
        name=f'ref_depth{d}.bin';ref.tofile(tmp_path/name)
        meta['references'][name]=m.base.digest(tmp_path/name)
    (tmp_path/'ids.txt').write_text('ids');meta['ids_sha256']=m.base.digest(tmp_path/'ids.txt')
    (tmp_path/'reference.json').write_text(json.dumps(meta));(tmp_path/'cli').write_text('fixture')
    monkeypatch.setattr(m.base,'memory_preflight',lambda *args:40*2**30)
    monkeypatch.setattr(m.base,'spec_from_model_dir',lambda *args:None)
    monkeypatch.setattr(m.base,'manifest',lambda *args,**kwargs:None)
    monkeypatch.setattr(m.base,'load_bundle',lambda *args:None)
    def native(cmd,stdout,**kwargs):
        d=int(cmd[cmd.index('--layers')+1]);prefix=cmd[cmd.index('--dump-logits')+1]
        got=ref.copy()
        if d==8:got[17,0]+=10
        for i,row in enumerate(got):row.tofile(f'{prefix}_p{i}.bin')
        got[-1].tofile(f'{prefix}_t0.bin')
        stdout.write(f'token {22 if wrong_token and d==4 else 23}\n')
    monkeypatch.setattr(m.subprocess,'run',native)
    assert m.run(NS(out=tmp_path,cli=tmp_path/'cli'))==1
    r=json.loads((tmp_path/'report.json').read_text())
    assert [c['depth'] for c in r['cases']]==[4,8]
    assert r['cases'][0]['passed']==(not wrong_token)
    assert [x['position'] for x in r['cases'][1]['logits'] if not x['passed']]==[17]
