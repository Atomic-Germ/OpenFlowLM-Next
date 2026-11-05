"""Private consistent-sum must preserve the current full-layer recurrence and ABI."""
import importlib.util
import sys
from pathlib import Path
import pytest
from recipes.load import load_spec

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'utilities/qkv-full-layer'))
import consistent_lx as m


def spec():
    return load_spec(ROOT / 'open_kernels/recipes/specs/qwen35-27b.json')


def test_private_sources_preserve_current_production_wrappers():
    path = ROOT / 'open_kernels/designs/layer_x/gen_kernels.py'
    loader = importlib.util.spec_from_file_location('_qkv_test_gen', path)
    gen = importlib.util.module_from_spec(loader); loader.loader.exec_module(gen)
    from recipes.qwen35 import recipe
    expected = gen.files(recipe(spec()))
    actual = m.generated_sources(spec())
    for name, code in expected.items():
        assert actual[name] == code
    assert '"dnx_factored.h"' in actual['dnx_row.cc']
    assert 'dnx.h' not in actual and 'precise.h' not in actual
    assert actual['gemv_q4.h'] == (m.GEMV / 'gemv_q4.h').read_text()
    assert actual['gemv_tab.h'] != (m.GEMV / 'gemv_tab.h').read_text()


def test_design_changes_only_source_routing_and_cache_identity():
    original = (m.LAYER / 'lx.py').read_text()
    design = m.render_design(spec())
    assert 'K = private_kernels(X, inc, t,' in design
    assert design.count('X.dn_body(win, yout, B, K)') == original.count('X.dn_body(win, yout, B, K)')
    with pytest.raises(ValueError, match='anchor'):
        m.render_design(spec(), original.replace('K = X.kernels(inc, t)', 'changed'))


def test_private_search_path_and_restoration(tmp_path):
    class X:
        HERE = m.LAYER
        def kernels(self, inc, t):
            assert self.HERE == tmp_path
            assert inc[0] == str(tmp_path)
            raise RuntimeError('compile failure')
    x = X()
    with pytest.raises(RuntimeError):
        m.private_kernels(x, ['system'], {}, tmp_path)
    assert x.HERE == m.LAYER


def test_dependency_audit_rejects_production_table_and_old_recurrence(tmp_path):
    prj = tmp_path / 'final.prj'; prj.mkdir()
    for obj, headers in m.DEPENDENCIES.items():
        (prj / (obj + '.o.d')).write_text('object: ' + ' '.join(str((tmp_path/'generated' if private else m.LAYER)/header) for header, private in headers))
    assert len(m.verify_dependencies(tmp_path)) == len(m.DEPENDENCIES)
    path = prj / 'dense_prep_f32.o.d'
    original = path.read_text()
    path.write_text(original.replace(str(tmp_path/'generated'), str(m.GEMV)))
    with pytest.raises(ValueError, match='header'): m.verify_dependencies(tmp_path)
    path.write_text(original)
    (prj/'dnx_row.o.d').write_text('object: ' + str(m.LAYER/'dnx.h'))
    with pytest.raises(ValueError, match='header'): m.verify_dependencies(tmp_path)


def test_changed_generated_source_is_rejected(tmp_path):
    for name, code in m.generated_sources(spec()).items(): (tmp_path/name).write_text(code)
    assert m.verify_generated(tmp_path, spec())
    (tmp_path/'gemv_tab.h').write_text((m.GEMV/'gemv_tab.h').read_text())
    with pytest.raises(ValueError, match='stale'): m.verify_generated(tmp_path, spec())


def test_control_identity_does_not_hide_numerical_failure():
    from consistent_replay import accepted
    row = dict(control_identical=True, entry_identical=True, isolated_qkv_identical=True,
               delta=dict(passed=False, recurrence=dict(head_diagnostics_passed=True)),
               residual=dict(passed=True), front=dict(passed=True), down=dict(passed=True),
               projections=dict(passed=True))
    assert accepted('control', row)
    assert not accepted('consistent', row)
    row['delta']['passed'] = True
    assert accepted('consistent', row)
    row['delta']['recurrence']['head_diagnostics_passed'] = False
    assert not accepted('consistent', row)


@pytest.mark.parametrize('field', ['residual', 'front', 'down', 'projections'])
def test_candidate_must_pass_every_affected_branch(field):
    from consistent_replay import accepted
    row = dict(entry_identical=True, isolated_qkv_identical=True,
               delta=dict(passed=True, recurrence=dict(head_diagnostics_passed=True)),
               residual=dict(passed=True), front=dict(passed=True), down=dict(passed=True),
               projections=dict(passed=True))
    row[field]['passed'] = False
    assert not accepted('consistent', row)


def test_preparation_cannot_erase_a_production_parent(tmp_path):
    from consistent_replay import separate_output
    source = tmp_path/'production'; source.mkdir()
    with pytest.raises(ValueError, match='separate'): separate_output(tmp_path, [source])
    with pytest.raises(ValueError, match='separate'): separate_output(source/'private', [source])
    separate_output(tmp_path/'private', [source])


def test_provenance_requires_original_pool_hash(tmp_path):
    from consistent_replay import verify_original_inputs
    path = tmp_path/'pool'; path.write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='original'): verify_original_inputs({str(path): m.digest(path)}, {})
    with pytest.raises(ValueError, match='original'): verify_original_inputs({str(path): m.digest(path)}, {str(path): 'wrong'})


def test_projection_gate_detects_nonzero_output_for_zero_weights():
    from types import SimpleNamespace
    import numpy as np
    from consistent_replay import projections
    s = SimpleNamespace(hidden=2, lin_value_width=128, norm_eps=1e-6)
    offsets = SimpleNamespace(A_Z=0, A_OUT=512, A_RES=520, A_OG=528)
    class Weights:
        def matmul_w(self, name, rows, cols): return np.zeros((rows, cols))
        def bf16(self, name): return np.ones(128 if 'ssm_norm' in name else 2)
    x = np.array([2., 3.]); c = dict(xn=x, output=np.zeros((1,128)))
    raw = bytearray(784)
    np.frombuffer(raw, '<f4', count=2, offset=520)[:] = x
    assert projections(raw, offsets, s, Weights(), 0, x, c)['passed']
    np.frombuffer(raw, '<f4', count=128)[0] = 1.
    assert not projections(raw, offsets, s, Weights(), 0, x, c)['passed']
