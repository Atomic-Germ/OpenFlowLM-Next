"""Private FP32 post output must widen every producer/consumer without moving old ABI."""
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[3]


def load():
    s=importlib.util.spec_from_file_location('post_build',ROOT/'utilities/deltanet-post-fp32/build.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def spec(m):return m.load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json')


def test_append_scratch_and_double_post_stream():
    m=load();sp=spec(m);old=m.recipe(sp).layout;g=m.geometry(sp,'fp32')
    assert g['post_offset']==old.A_BYTES
    assert g['act_bytes']==old.A_BYTES+6144*4
    assert g['post_elements']==6 and g['state_bytes']==old.STATE_BYTES
    assert m.geometry(sp,'control')['post_offset']==old.A_OG


def test_candidate_and_control_compile_the_shared_post_source():
    m=load();original=(m.POST/'post.cc').read_text()
    assert m.post_source('fp32')==m.post_source('control')==original
    assert 'POST_F32=True' in m.helper('fp32')
    assert 'POST_F32=False' in m.helper('control')


def test_private_design_connects_fp32_scratch_and_prep():
    m=load();code=m.design(spec(m),'fp32')
    assert 'OG_ELEMS = 6' in code
    assert 'OUT_PC, "linear_out", fp32=D.POST_F32)' in code
    assert 'POST_F32=True' in code
    assert 'import private_xlayer as XL' in code
    compile(code,'candidate','exec')
    control=m.design(spec(m),'control')
    assert 'POST_F32=False' in control
    compile(control,'control','exec')


def test_only_post_fifo_and_kernel_argument_widen():
    m=load();code=m.helper('fp32')
    assert code.count('POST_F32=True')==1
    assert 'arg_types=[u8_4k, u8_4k, nw_ty, post_type]' in code
    assert 'ObjectFifo(g["u8_4k" if D.POST_F32 else "u8_2k"], name="pout", depth=2)' in code
    assert 'ObjectFifo(g["u8_2k"], name="gout", depth=3)' in code


def test_generated_entries_retain_production_precision():
    m=load();g=m.generated(spec(m),'fp32')
    assert '#define GEMV_Q4_CONSISTENT_SUM 1' in g['dense_prep_f32.cc']
    assert 'gemv_q4_prep_f32_direct_blocks' in g['dense_prep_f32.cc']
    assert 'dnx_factored.h' in g['dnx_row.cc']


@pytest.mark.parametrize('mode',['unknown','bf16',None])
def test_refuse_unknown_mode(mode):
    with pytest.raises(ValueError):load().post_source(mode)


def test_refuse_changed_source_anchor():
    with pytest.raises(ValueError):load().replace_once('other','missing','new')
    with pytest.raises(ValueError):load().replace_once('same same','same','new')


def test_compiler_dependency_audit_handles_renamed_post_tu(tmp_path):
    m=load();gen=tmp_path/'generated';prj=tmp_path/'final.prj';gen.mkdir();prj.mkdir()
    for obj,source in [('post_fn','post'),('dense_prep_f32','dense_prep_f32'),('dense_prep','dense_prep'),('dnx_row','dnx_row'),('dnx_ofin','dnx_ofin')]:
        (gen/(source+'.cc')).write_text('// '+source)
        copied=prj/(obj+'.cc');copied.write_text('// '+source)
        headers=[m.LAYER/'dnx_factored.h',m.LAYER/'dnx_precise.h',m.LAYER.parent/'gemv_q4/gemv_tab.h']
        (prj/(obj+'.o.d')).write_text(f'{obj}.o: '+str(copied)+' '+' '.join(map(str,headers)))
    assert m.dependencies(tmp_path)
    (prj/'post_fn.cc').write_text('// stale')
    with pytest.raises(ValueError,match='stale compiler'):m.dependencies(tmp_path)
