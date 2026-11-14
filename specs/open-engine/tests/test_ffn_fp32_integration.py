"""Wider norm scratch must be explicit in the ABI and retain the other dense recipes."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from recipes.load import load_spec
from recipes.qwen35 import recipe
from recipes.manifest import manifest
from decode_bundle import load_bundle
from ffn_front import read_front

ROOT=Path(__file__).resolve().parents[3]

def spec(size):return load_spec(ROOT/f'open_kernels/recipes/specs/qwen35-{size}.json')


def test_split_ffn_stream_reserves_fp32_norm_without_overlap():
    r=recipe(spec('27b'));f=r.ffn;l=r.layout
    assert f.XM_F32 and f.XM_ELEMS==5 and f.XN_ELEMS==3
    for p in ('A_','AA_'):
        assert getattr(l,p+'H')-getattr(l,p+'XM')==5*4096
        assert getattr(l,p+'BYTES')>=getattr(l,p+'OUT2B')+5120*4


def test_unsplit_dense_keeps_bf16_norm():
    r=recipe(spec('9b'))
    assert not r.ffn.XM_F32 and r.ffn.XM_ELEMS==2
    assert 'ffn_norm_dtype' not in manifest(spec('9b'))['layout']


def test_manifest_declares_new_norm_format_and_refuses_old_layout(tmp_path):
    m=manifest(spec('27b'))
    assert m['layout']['ffn_norm_dtype']=='float32'
    old=json.loads(json.dumps(m));old['layout'].pop('ffn_norm_dtype')
    for layer in old['layer_types'].values():layer['buffers']['act']-=8192
    (tmp_path/'manifest.json').write_text(json.dumps(old));(tmp_path/'toolchain.json').write_text('{}')
    with pytest.raises(ValueError,match='recipe'):load_bundle(tmp_path,m)


@pytest.mark.parametrize('full',[False,True])
def test_front_reader_keeps_fp32_low_bits(tmp_path,full):
    p='AA_' if full else 'A_'
    offsets=SimpleNamespace(**{p+'BYTES':24,p+'RES':0,p+'XM':8,p+'H':16})
    a=np.array([1,2,1.001,-2.003,4,5],np.float32);file=tmp_path/'act';file.write_bytes(a.tobytes())
    r=read_front(file,offsets,2,2,full,norm_dtype='float32')
    np.testing.assert_array_equal(r['xm'],a[2:4])


def test_front_reader_refuses_unknown_precision(tmp_path):
    p=tmp_path/'act';p.write_bytes(bytes(24))
    with pytest.raises(ValueError,match='dtype'):
        read_front(p,SimpleNamespace(A_BYTES=24,A_RES=0,A_XM=8,A_H=16),2,2,False,norm_dtype='guess')
