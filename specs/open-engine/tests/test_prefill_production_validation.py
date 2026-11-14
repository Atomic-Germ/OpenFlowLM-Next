"""Integrated traces preserve shared arithmetic and cannot bless partial resets."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location('production_'+name, ROOT/f'utilities/prefill-production/{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('repeat', [False, True])
def test_trace_changes_only_capture_and_serial_completion(repeat):
    m = load('build')
    cpp = (ROOT/'src/open_qwen36/core.cpp').read_text()
    cli = (ROOT/'src/open_qwen36/cli.cpp').read_text()
    got_cpp, got_cli = m.instrument(cpp, cli, repeat)
    start = cpp.index('const float* Core::gemm_run(')
    end = cpp.index('const float* Core::gemm_dispatch(', start)
    assert cpp[start:end] in got_cpp
    request = cli[cli.index('std::vector<int> request('):cli.index('\n}  // namespace', cli.index('std::vector<int> request('))]
    assert request in got_cli and 'core.reset();' in request
    assert 'residual_trace::writer.append' in got_cpp
    assert ('request(core, again_args)' in got_cli) == repeat
    with pytest.raises(ValueError):
        m.instrument(got_cpp, got_cli, repeat)


def test_reset_capture_names_are_complete_and_exact():
    m = load('validate')
    expected = {n: str(i) for i,n in enumerate(m.runner.capture_names('hybrid', 2, 3))}
    actual = {n.replace('hybrid', 'hybrid_repeat', 1):h for n,h in expected.items()}
    assert all(m.identity(actual, expected, 'hybrid_repeat', 'hybrid', 2, 3).values())
    for name in ('hybrid_repeat_res2.bin', 'hybrid_repeat_t1.bin', 'hybrid_repeat-act.bin'):
        bad = dict(actual); bad[name] = 'changed'
        with pytest.raises(ValueError):
            m.identity(bad, expected, 'hybrid_repeat', 'hybrid', 2, 3)
    actual.pop('hybrid_repeat_res0.bin')
    with pytest.raises(ValueError):
        m.identity(actual, expected, 'hybrid_repeat', 'hybrid', 2, 3)


@pytest.mark.parametrize('fault', ['missing', 'extra', 'numerical', 'identity'])
def test_aggregate_cannot_hide_route_or_precision_failures(fault):
    m = load('validate')
    rows = [dict(route=r, first={'passed': True}, identity={'x': True}) for r in m.runner.ROUTES]
    assert m.aggregate(rows, False)
    if fault == 'missing': rows.pop()
    elif fault == 'extra': rows.append(rows[0])
    elif fault == 'numerical': rows[0]['first']['passed'] = False
    else: rows[0]['identity']['x'] = False
    with pytest.raises(ValueError): m.aggregate(rows, False)


def test_identical_repeat_does_not_hide_failed_first_request():
    m = load('validate')
    rows = [dict(route=r, first={'passed': True}, second={'passed': True},
                 identity={'x': True}, repeat_identity={'x': True}) for r in m.runner.ROUTES]
    assert m.aggregate(rows, True)
    rows[0]['first']['passed'] = False
    with pytest.raises(ValueError): m.aggregate(rows, True)


def test_capture_inventory_rejects_extra_logits_and_missing_layers(tmp_path):
    m = load('validate')
    for name in m.runner.capture_names('hybrid', 2, 3):
        (tmp_path/name).touch()
    m.capture_inventory(tmp_path, 'hybrid', 2, 3)
    extra = tmp_path/'hybrid_p2.bin'
    extra.touch()
    with pytest.raises(ValueError): m.capture_inventory(tmp_path, 'hybrid', 2, 3)
    extra.unlink()
    (tmp_path/'hybrid_res2.bin').unlink()
    with pytest.raises(ValueError): m.capture_inventory(tmp_path, 'hybrid', 2, 3)


def test_plain_control_has_no_instrumentation_or_repeat_patch():
    m=load('build');cpp='unmodified core';cli='unmodified CLI'
    assert m.instrument(cpp,cli,False,plain=True)==(cpp,cli)
    with pytest.raises(ValueError,match='plain'):
        m.instrument(cpp,cli,True,plain=True)
