"""Compiled production selection must be audited, not inferred from recipe flags."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[3]
loader=importlib.util.spec_from_file_location('consistent_audit',ROOT/'utilities/audit-consistent-sum.py')
m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m)


def fixture(tmp_path):
    for name in ('lx','ax'):
        prj=tmp_path/name/'final.prj';prj.mkdir(parents=True)
        (prj/'dense_prep_f32.cc').write_text('#define GEMV_Q4_CONSISTENT_SUM 1\nvoid prep(){gemv_q4_prep_f32_direct_blocks();}\n')
        for obj,headers in m.dependencies(name).items():
            (prj/(obj+'.o.d')).write_text('object: '+' '.join(str(p) for p in headers)+'\n')
    return tmp_path


def test_checks_actual_compiler_sources_and_dependency_paths(tmp_path):
    root=fixture(tmp_path)
    assert m.audit_compilation(root)
    (root/'ax/final.prj/dense_prep_f32.cc').write_text('void prep(){gemv_q4_prep_f32_direct_blocks();}\n')
    with pytest.raises(ValueError,match='selection'):m.audit_compilation(root)


def test_rejects_private_table_even_when_name_matches(tmp_path):
    root=fixture(tmp_path)
    (root/'lx/final.prj/dense_prep_f32.o.d').write_text('object: /private/gemv_tab.h /private/gemv_q4.h\n')
    with pytest.raises(ValueError,match='header'):m.audit_compilation(root)


def test_rejects_legacy_recurrence(tmp_path):
    root=fixture(tmp_path)
    (root/'lx/final.prj/dnx_row.o.d').write_text('object: '+str(m.LAYER/'dnx.h'))
    with pytest.raises(ValueError,match='header'):m.audit_compilation(root)
