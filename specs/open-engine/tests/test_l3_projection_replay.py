"""Exact native controls must precede interpretation of isolated projections."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name='replay'):
    s = importlib.util.spec_from_file_location('l3_projection_'+name,
        ROOT/'utilities/l3-projection-replay'/f'{name}.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def test_current_generated_production_arithmetic():
    m = load('build')
    from recipes.spec import ModelSpec
    spec = ModelSpec.from_hf_config(json.loads((Path(__file__).parent/'fixtures/config_qwen38_27b.json').read_text()))
    sources = m.generated(spec)
    assert sources['dense_prep_f32.cc'].startswith('#define GEMV_Q4_CONSISTENT_SUM 1')
    assert 'gemv_q4_prep_f32_direct_blocks(' in sources['dense_prep_f32.cc']
    assert '#define GEMV_PER_CALL 1' in sources['gemv_q4_gy.cc']
    for name in ('gemv_q4.h', 'gemv_tab.h'):
        assert sources[name] == (ROOT/'open_kernels/designs/gemv_q4'/name).read_text()


def test_program_poisons_every_output_before_every_dispatch(tmp_path):
    m = load(); lines = m.program(tmp_path,tmp_path/'build').splitlines()
    assert lines.count('run proj w x y tab') == 240
    for p in range(240):
        i = lines.index(f'load x {tmp_path}/x{p}.bin')
        assert lines[i+1:i+4] == [f'load y {tmp_path}/poison-y.bin',
            f'load tab {tmp_path}/poison-tab.bin','run proj w x y tab']
        assert f'dump y {tmp_path}/y{p}.bin 57344' in lines
        assert f'dump tab {tmp_path}/table{p}.bin 11520' in lines


def test_identity_retains_individual_role_and_one_ulp_failure():
    m = load(); ref = np.arange(14336,dtype='<f4'); got = ref.copy()
    assert all(m.identity(got,ref).values())
    got[6144] = np.nextafter(got[6144],np.float32(np.inf))
    assert m.identity(got,ref) == dict(q=True,gate=False,k=True,v=True)


@pytest.mark.parametrize('fault',['shape','dtype','nan'])
def test_identity_rejects_invalid_capture(fault):
    m = load(); a = np.ones(14336,dtype='<f4'); b = a.copy()
    if fault=='shape': b=b[:-1]
    elif fault=='dtype': b=b.astype('<f8')
    else: b[0]=np.nan
    with pytest.raises(ValueError): m.identity(a,b)


def test_table_requires_all_fields_byte_exact():
    m=load(); x=np.linspace(-1,1,5120,dtype='<f4')
    data=m.expected_table(x)
    assert len(data)==11520 and m.table_identity(x,data)
    for off in (0,10240,10880,11200):
        bad=bytearray(data);bad[off]^=1
        assert not m.table_identity(x,bytes(bad))
    with pytest.raises(ValueError): m.table_identity(x,data[:-1])


def test_summary_requires_all_positions_and_keeps_failed_control():
    m=load(); rows=[dict(position=p,identity=dict.fromkeys(m.ROLES,True),table_exact=True) for p in range(240)]
    result=m.summarize(rows)
    assert result['control_valid'] and result['exact_roles']==960 and result['exact_tables']==240
    rows[125]['identity']['q']=False
    assert not m.summarize(rows)['control_valid']
    with pytest.raises(ValueError):m.summarize(rows[:-1])
    rows[-1]['position']=0
    with pytest.raises(ValueError):m.summarize(rows)


def test_reader_rejects_stale_extra_output(tmp_path):
    m=load()
    for p in range(240):
        (tmp_path/f'y{p}.bin').write_bytes(b'')
        (tmp_path/f'table{p}.bin').write_bytes(b'')
    m.inventory(tmp_path)
    (tmp_path/'table240.bin').write_bytes(b'')
    with pytest.raises(ValueError,match='inventory'):m.inventory(tmp_path)


def test_dependencies_require_both_current_private_headers(tmp_path):
    m=load('build'); generated=tmp_path/'generated';generated.mkdir()
    for n in ('gemv_q4.h','gemv_tab.h'): (generated/n).write_text(n)
    dep=tmp_path/'prep.o.d'
    dep.write_text('prep.o: '+str(generated/'gemv_q4.h'))
    with pytest.raises(ValueError,match='header'):m.dependencies([dep],generated)
    dep.write_text('prep.o: '+' '.join(str(p) for p in generated.iterdir()))
    hashes=m.dependencies([dep],generated)
    assert str(dep) in hashes and len(hashes)==3


@pytest.mark.parametrize('failed',[False,True])
def test_audit_preserves_failed_controls_and_rejects_forged_permission(tmp_path,monkeypatch,failed):
    m=load();summary=dict(control_valid=not failed,table_valid=True)
    record=dict(complete=True,model_acceptance=False,preparation_sha256='hash',harness='runner',
        harness_sha256='hash',harness_evidence_sha256='hash',log_sha256='hash',
        summary=summary,rows=[],artifacts={'y0.bin':'hash'},interpretation_allowed=not failed)
    (tmp_path/'report.json').write_text(json.dumps(record))
    monkeypatch.setattr(m,'verified',lambda out:{})
    monkeypatch.setattr(m,'digest',lambda path:'hash')
    monkeypatch.setattr(m.native,'verify_harness',lambda path:'hash')
    monkeypatch.setattr(m,'score',lambda out:(summary,[],record['artifacts']))
    assert m.audit(SimpleNamespace(out=tmp_path))==int(failed)
    proof=json.loads((tmp_path/'audit.json').read_text())
    assert proof['complete'] and not proof['model_acceptance']
    record['interpretation_allowed']=failed
    (tmp_path/'report.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='measurements'):m.audit(SimpleNamespace(out=tmp_path))


def test_run_refuses_stale_captures(tmp_path):
    m=load();(tmp_path/'table125.bin').write_bytes(b'old')
    with pytest.raises(ValueError,match='overwrite'):m.run(SimpleNamespace(out=tmp_path))


def test_score_refuses_incomplete_output_even_with_full_inventory(tmp_path):
    m=load()
    for p in range(240):
        (tmp_path/f'y{p}.bin').write_bytes(b'')
        (tmp_path/f'table{p}.bin').write_bytes(b'')
    np.zeros(14336,dtype='<f4').tofile(tmp_path/'ref0.bin')
    np.zeros(5120,dtype='<f4').tofile(tmp_path/'x0.bin')
    with pytest.raises(ValueError):m.score(tmp_path)


def test_score_rejects_trailing_partial_float(tmp_path):
    m=load()
    for p in range(240):
        (tmp_path/f'y{p}.bin').write_bytes(b'')
        (tmp_path/f'table{p}.bin').write_bytes(b'')
    a=np.zeros(14336,dtype='<f4');a.tofile(tmp_path/'ref0.bin')
    (tmp_path/'y0.bin').write_bytes(a.tobytes()+b'\x00')
    x=np.zeros(5120,dtype='<f4');x.tofile(tmp_path/'x0.bin')
    (tmp_path/'table0.bin').write_bytes(m.expected_table(x))
    with pytest.raises(ValueError,match='size'):m.score(tmp_path)
