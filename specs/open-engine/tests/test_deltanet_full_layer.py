"""Private full-layer recurrence experiments must preserve the production design."""
import sys
from pathlib import Path
import pytest
from recipes.load import load_spec

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'utilities/deltanet-replay'))
import full_layer_common as m


def spec():return load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json')


def test_full_layer_requires_current_fp32_production_geometry():
    g=m.geometry(spec())
    assert g['act_bytes']==393216 and g['state_bytes']==3317760
    assert g['scratch_floats']==1280 and g['rows']==10


def test_private_sources_change_only_recurrence_headers():
    import recurrence_common as single
    sources=m.generated_sources(spec(),shared=False)
    for name,code in m.production_sources(spec()).items():assert sources[name]==code
    assert sources['dnx.h']==single.sources('factored')['dnx.h']
    assert sources['precise.h']==single.sources('factored')['precise.h']


def test_full_layer_design_keeps_streams_and_stack():
    source=(m.LAYER/'lx.py').read_text()
    result=m.render_design(spec(),shared=False)
    assert 'K = private_kernels(X, inc, t,' in result
    assert 'X.dn_body(win, yout, B, K)' in result
    assert result.count('X.dn_body(win, yout, B, K)')==source.count('X.dn_body(win, yout, B, K)')
    with pytest.raises(ValueError,match='anchor'):m.render_design(spec(),False,source.replace('K = X.kernels(inc, t)','broken'))


def test_shared_helpers_keep_arithmetic_but_use_linker_deduplication():
    normal=m.generated_sources(spec(),False);shared=m.generated_sources(spec(),True)
    for name in normal:
        expected=normal[name].replace('static __attribute__((noinline))','inline __attribute__((noinline))') if name in ('dnx.h','precise.h') else normal[name]
        assert shared[name]==expected
    assert shared['dnx.h']!=normal['dnx.h'] and shared['precise.h']!=normal['precise.h']


def test_full_layer_replay_carries_only_its_own_state(tmp_path):
    from full_layer_replay import config
    g={'act_bytes':393216,'state_bytes':3317760,'hidden':5120}
    bundle={'layout':{'pool_bytes':1024},'layer_types':{'linear_attention':{'buffers':{'consts':512}}}}
    from recipes.spec import LINEAR
    bundle['layer_types'][LINEAR]=bundle['layer_types'].pop('linear_attention')
    text=config(bundle,g,[0,25],3,tmp_path,tmp_path,tmp_path,tmp_path)
    assert text.count('load state ')==2 and text.count('run lx ')==6
    assert 'ref_' not in text and 'load xres ' in text
    assert text.count('dump state ')==6


def test_capture_identity_detects_signed_zero_and_missing_outputs(tmp_path):
    import numpy as np
    from full_layer_replay import exact_capture
    a=tmp_path/'a';b=tmp_path/'b';np.zeros(16,'<f4').tofile(a)
    data=np.zeros(16,'<f4');data[0]=-0.;data.tofile(b)
    assert not exact_capture(a,b,64)
    b.write_bytes(b'short')
    with pytest.raises(ValueError):exact_capture(a,b,64)


def test_copied_compiler_sources_resolve_private_headers_first(tmp_path):
    # IRON copies .cc files into final.prj: source-relative includes no longer work.
    class X:
        HERE=m.LAYER
        def kernels(self,inc,t):return inc
    x=X();inc=m.private_kernels(x,['system',str(m.LAYER)],{},tmp_path)
    assert inc.index(str(tmp_path))<inc.index(str(m.LAYER))
    assert x.HERE==m.LAYER


def test_compiled_dependencies_must_include_both_private_headers(tmp_path):
    from build_full_layer import verify_dependencies
    generated=tmp_path/'generated';generated.mkdir();prj=tmp_path/'final.prj';prj.mkdir()
    for name in ('vcopy','pass1','delta','row','ofin'):
        (prj/f'dnx_{name}.o.d').write_text(f'object: {generated}/dnx.h {generated}/precise.h\n')
    assert len(verify_dependencies(tmp_path))==5
    (prj/'dnx_row.o.d').write_text(f'object: {m.LAYER}/dnx.h\n')
    with pytest.raises(ValueError,match='private header'):verify_dependencies(tmp_path)


def test_prepare_cannot_overwrite_production_fixture(tmp_path):
    from full_layer_replay import prepare
    saved=tmp_path/'y_state0.bin';saved.write_bytes(b'production')
    with pytest.raises(ValueError,match='separate'):prepare(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path,tmp_path)
    assert saved.read_bytes()==b'production'
