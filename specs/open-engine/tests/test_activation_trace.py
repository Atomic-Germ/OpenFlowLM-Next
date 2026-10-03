import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from ml_dtypes import bfloat16


def test_activation_oracle_keeps_fp32_before_bf16_boundary():
    path=Path(__file__).resolve().parents[3]/'utilities/trace-wide-activation.py'
    spec=importlib.util.spec_from_file_location('activation_trace',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    g=np.array([-.11944595724344254],np.float32)
    u=np.array([.18203915655612946],np.float32)
    ref=m.reference(g,u)
    assert ref[0]==np.float32(-.010223387740552425)
    assert ref.astype(bfloat16)[0]!=bfloat16(-.010223388671875)


def test_activation_trace_guards_inputs_outputs_and_never_claims_acceptance(tmp_path):
    path=Path(__file__).resolve().parents[3]/'utilities/trace-wide-activation.py'
    spec=importlib.util.spec_from_file_location('activation_trace',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source=tmp_path/'source';source.mkdir();kernel=tmp_path/'kernel';kernel.mkdir()
    for name in ('final.xclbin','insts.bin'):(kernel/name).write_bytes(b'fixture')
    # Trace ABI is interleaved 64-row up/gate bands, not two whole tensors.
    bands=np.zeros((272,2,64),np.float32);bands[:,0]=.2;bands[:,1]=-.1
    (source/'trace0.bin').write_bytes(bands.tobytes()+m.GUARD)
    h=m.reference(np.full(17408,-.1,np.float32),np.full(17408,.2,np.float32))
    (source/'got0.bin').write_bytes(h.tobytes()+m.GUARD)
    (source/'trace-fixture.json').write_text(json.dumps(dict(act_bytes=h.nbytes,h_offset=0,
        sha256={'trace0.bin':m.sha(source/'trace0.bin')})))
    out=tmp_path/'out';m.prepare(source,kernel,out,749)
    inputs=np.fromfile(out/'input.bin',np.float32).reshape(2,32)
    np.testing.assert_array_equal(inputs[0],np.full(32,-.1,np.float32))
    np.testing.assert_array_equal(inputs[1],np.full(32,.2,np.float32))
    fake=np.zeros((10,32),np.float32);fake[9]=h[:32]
    for i in range(2):(out/f'got{i}.bin').write_bytes(fake.tobytes()+m.GUARD)
    assert m.compare(out)==1
    r=json.loads((out/'results.json').read_text())
    assert r['repeat_exact'] and r['production_matches_recorded'] and r['bf16_differences']==0
    assert r['diagnostic_only'] and not r['passed']
    (out/'got1.bin').write_bytes(fake.tobytes()+bytes(64))
    with pytest.raises(ValueError,match='canary'):m.compare(out)
    (out/'input.bin').write_bytes(bytes(256))
    with pytest.raises(ValueError,match='changed fixture'):m.compare(out)
