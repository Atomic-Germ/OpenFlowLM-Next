"""Full-model scheduling and Q4NX preflight without allocating model weights."""
from dataclasses import replace
import json
from pathlib import Path
import struct

import pytest
from recipes.spec import ModelSpec, LINEAR, FULL
from recipes import qwen35

FIXTURE = Path(__file__).parent/'fixtures/config_qwen38_27b.json'


@pytest.fixture
def geometry(monkeypatch):
    monkeypatch.setenv('OPEN_KERNELS_UNVALIDATED','1')
    s = ModelSpec.from_hf_config(json.loads(FIXTURE.read_text()))
    return s,qwen35.layout(s,max_ctx=257)


def test_full_schedule_keeps_all_64_states_and_streams_weights(geometry):
    from recipes.wide_slice import layer_types,layer_commands
    s,l = geometry
    kinds = layer_types(s,layers=64)
    assert (len(kinds),kinds.count(LINEAR),kinds.count(FULL)) == (64,48,16)
    count = 0
    for i,kind in enumerate(kinds):
        commands = layer_commands(s,l,i,256,257,layers=64)
        assert f'load pool layer{i}/pool.bin' in commands
        assert f'load const layer{i}/const.bin' in commands
        state = f'state{i}' if kind==LINEAR else f'cache{i}'
        assert any(state in c.split() for c in commands)
        assert not any(f'state{j}' in c.split() or f'cache{j}' in c.split()
                       for c in commands for j in range(64) if j!=i)
        count += sum(c.startswith('run ') for c in commands)
    assert count == 592
    assert layer_commands(s,l,7,0,257) == layer_commands(s,l,7,0,257,layers=64)


def test_full_schedule_rejects_truncated_or_wrong_geometry(geometry):
    from recipes.wide_slice import layer_types,layer_commands
    s,l = geometry
    for bad in (replace(s,num_layers=63),replace(s,layer_types=s.layer_types[:8]),
                replace(s,layer_types=(LINEAR,)*64)):
        with pytest.raises(ValueError): layer_types(bad,layers=64)
    for count in (0,7,32,65):
        with pytest.raises(ValueError): layer_types(s,layers=count)
    with pytest.raises(ValueError): layer_commands(s,l,64,0,257,layers=64)


def model_header(s):
    from recipes.wide_model import tensor_requirements
    header,offset = {},0
    for name,r in tensor_requirements(s).items():
        size = r['bytes']
        header[name] = dict(dtype=r['dtype'],shape=r['shape'],data_offsets=[offset,offset+size])
        offset += size
    return header,offset


def test_model_contract_includes_last_layer_and_both_q_gate_halves(geometry):
    from recipes.wide_model import tensor_requirements,validate_header
    s,_ = geometry
    header,size = model_header(s)
    r = tensor_requirements(s)
    assert r['model.layers.63.self_attn.q_proj.weight']['shape'] == [7680,5120]
    assert r['model.layers.62.linear_attn.ssm_alpha_proj.bf16.weight']['shape'] == [48,5120]
    assert r['model.embed_tokens.weight']['shape'] == [248320,5120]
    assert r['lm_head.weight']['shape'] == [155200,8704]
    assert validate_header(s,header,size)['ready_for_packing']
    del header['model.layers.63.self_attn.k_proj.weight']
    with pytest.raises(ValueError,match='63.*k_proj'): validate_header(s,header,size)


def test_model_preflight_rejects_bad_dtype_shape_truncation_and_overlap(geometry):
    from recipes.wide_model import validate_header
    s,_ = geometry
    for field,value in [('dtype','F16'),('shape',[248319,5120]),('data_offsets',[0,1])]:
        header,size = model_header(s)
        header['model.embed_tokens.weight'][field] = value
        with pytest.raises(ValueError): validate_header(s,header,size)
    header,size = model_header(s)
    with pytest.raises(ValueError): validate_header(s,header,size-1)
    names=list(header)
    header[names[1]]['data_offsets'] = header[names[0]]['data_offsets']
    with pytest.raises(ValueError): validate_header(s,header,size)


def test_header_reader_rejects_missing_or_malformed_container(tmp_path):
    from recipes.wide_model import read_header
    with pytest.raises(FileNotFoundError): read_header(tmp_path/'absent.q4nx')
    file=tmp_path/'model.q4nx'
    for raw in (b'',struct.pack('<Q',2**40),struct.pack('<Q',20)+b'{}'):
        file.write_bytes(raw)
        with pytest.raises(ValueError): read_header(file)
    raw=json.dumps({'__metadata__':{'format':'test'}}).encode()
    file.write_bytes(struct.pack('<Q',len(raw))+raw+b'1234')
    header,size=read_header(file)
    assert size==4 and header['__metadata__']['format']=='test'


def test_preflight_accepts_native_block_grid_and_rejects_transposed_grid(geometry):
    from recipes.wide_model import validate_header
    s,_=geometry
    header,size=model_header(s)
    header['model.layers.0.mlp.up_proj.weight']['shape']=[544,20,5120]
    header['lm_head.weight']['shape']=[7760,20,8704]
    assert validate_header(s,header,size)['ready_for_packing']
    header['model.layers.0.mlp.up_proj.weight']['shape']=[20,544,5120]
    with pytest.raises(ValueError): validate_header(s,header,size)
