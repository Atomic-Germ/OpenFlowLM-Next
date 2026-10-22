"""The combined private CLI must correct only validated projection roles."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from recipes.load import load_spec
from recipes.manifest import manifest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / 'utilities/prefill-combined-cli'


def load():
    spec = importlib.util.spec_from_file_location('combined_builder', TOOL / 'build.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_policy_uses_roles_and_rejects_incompatible_geometry(tmp_path):
    data = manifest(load_spec(ROOT / 'open_kernels/recipes/specs/qwen35-27b.json'))
    fixture = tmp_path / 'manifest.json'
    fixture.write_text(json.dumps(data))
    source = tmp_path / 'policy.cpp'
    source.write_text(r'''
#include "policy.hpp"
#include <functional>
#include <stdexcept>
using namespace open_qwen36;
void require(bool x) { if (!x) throw std::runtime_error("policy check failed"); }
void refuses(std::function<void()> fn) {
    bool failed = false;
    try { fn(); } catch (const std::runtime_error&) { failed = true; }
    require(failed);
}
int main(int argc, char** argv) {
    auto m = Manifest::load(argv[1]);
    auto linear = m.layer_types.at("linear_attention").gemm_block;
    auto full = m.layer_types.at("full_attention").gemm_block;
    combined::validate(linear, 5120); combined::validate(full, 5120);
    auto select = [&](const GemmBlockProgram& g, const Step& s, size_t k, size_t n) {
        return combined::low_kernel(g, s, 256, k, n);
    };
    require(select(linear, linear.program[0], 5120, 16384) == "private_entry_low");
    require(select(linear, linear.program[1], 6144, 5120) == "private_out_low");
    for (const auto* g : {&linear, &full}) {
        require(select(*g, g->ffn_program[0], 5120, 34816) == "private_ffn_low");
        require(select(*g, g->ffn_program[1], 17408, 5120).empty());
    }
    require(select(full, full.program[0], 5120, 14336).empty());
    require(select(full, full.program[1], 6144, 5120).empty());
    auto bad = linear;
    bad.program[0].args[0] = "wrong_weight";
    refuses([&] { combined::validate(bad, 5120); });
    bad = linear; bad.program[1].split = true;
    refuses([&] { combined::validate(bad, 5120); });
    bad = linear; bad.value_heads = 32;
    refuses([&] { combined::validate(bad, 5120); });
    bad = linear; bad.out_split = true;
    refuses([&] { combined::validate(bad, 5120); });
    bad = linear; bad.ff = 8192;
    refuses([&] { combined::validate(bad, 5120); });
    bad = linear; bad.ffn_program.clear();
    refuses([&] { combined::validate(bad, 5120); });
    refuses([&] { combined::validate(linear, 2560); });
    refuses([&] { select(linear, linear.program[0], 5120, 8192); });
    refuses([&] { combined::low_kernel(linear, linear.program[0], 128, 5120, 16384); });
    Step wrong = linear.program[0]; wrong.args.clear();
    refuses([&] { select(linear, wrong, 5120, 16384); });
}
''')
    exe = tmp_path / 'policy'
    subprocess.run(['c++', '-std=c++17', '-O0', '-I'+str(TOOL), '-I'+str(ROOT/'src'),
                    '-I'+str(ROOT/'src/include'), '-I'+str(ROOT/'open_kernels/harness'),
                    str(source), str(ROOT/'src/open_qwen36/manifest.cpp'),
                    '-o', str(exe)], check=True)
    subprocess.run([str(exe), str(fixture)], check=True)


def test_patch_preserves_dispatch_and_exact_control_wrappers():
    m = load()
    cpp = (ROOT/'src/open_qwen36/core.cpp').read_text()
    hpp = (ROOT/'src/open_qwen36/core.hpp').read_text()
    lows = {role: Path('/tmp')/role for role in ('ffn', 'entry', 'out')}
    assert m.patch(cpp, hpp, 'paired', lows) == m.parent.paired.patch(cpp, hpp)
    assert m.patch(cpp, hpp, 'weighted', lows) == m.parent.patch(cpp, hpp, lows['ffn'])
    c, h = m.patch(cpp, hpp, 'combined', lows)
    start = cpp.index('const float* Core::gemm_run(')
    end = cpp.index('\nvoid Core::gemm(', start)
    assert cpp[start:end].replace('Core::gemm_run(', 'Core::gemm_run_once(', 1) in c
    assert 'stage-major is not supported' in c
    assert 'combined::low_kernel' in c and 'combined::validate' in c
    for role in lows:
        assert f'/tmp/{role}/final.xclbin' in c
    with pytest.raises(ValueError):
        m.patch(c, h, 'combined', lows)


def test_existing_build_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        load().build(SimpleNamespace(out=tmp_path))


def test_kernel_geometry_is_checked_before_native_build(tmp_path, monkeypatch):
    m = load()
    record = {'environment': dict(GQP_N='16384', GQP_K='5120', GQP_T='256',
                                 GQP_FMT='q4_1', GQP_BREUSE='0')}
    monkeypatch.setattr(m.parent.kernels, 'verify_build', lambda *args: record)
    assert m.verify_kernel(tmp_path, 'entry') == record
    with pytest.raises(ValueError):
        m.verify_kernel(tmp_path, 'out')
    record['environment']['GQP_FMT'] = 'q4_k'
    with pytest.raises(ValueError):
        m.verify_kernel(tmp_path, 'entry')


def preparer():
    spec = importlib.util.spec_from_file_location('combined_preparer', TOOL/'prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_failure_is_not_a_success_and_cannot_overwrite(tmp_path):
    out = tmp_path/'prepared'
    a = SimpleNamespace(out=out, builds=tmp_path/'missing')
    with pytest.raises(FileNotFoundError):
        preparer().prepare(a)
    record = json.loads((out/'preparation.json').read_text())
    assert not record['complete'] and not record['native_validated'] and not record['model_acceptance']
    with pytest.raises(FileExistsError):
        preparer().prepare(a)


def test_manifest_geometry_gate_accepts_actual_vocab_but_rejects_role_changes():
    m = preparer()
    spec = load_spec(ROOT/'open_kernels/recipes/specs/qwen35-27b.json')
    data = manifest(spec)
    assert m.check_manifest(data, spec) == data
    data['layer_types']['linear_attention']['gemm_block']['program'][1]['args'][0] = 'go_w'
    with pytest.raises(ValueError, match='manifest'):
        m.check_manifest(data, spec)
