"""Private paired-input GEMM experiment: rounding and device-buffer ownership."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / 'utilities/prefill-paired-input'


def builder():
    spec = importlib.util.spec_from_file_location('prefill_paired_build', TOOL / 'build.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_patch_preserves_single_dispatch_body_and_rejects_drift():
    m = builder()
    cpp = (ROOT / 'src/open_qwen36/core.cpp').read_text()
    hpp = (ROOT / 'src/open_qwen36/core.hpp').read_text()
    patched, header = m.patch(cpp, hpp)
    start = cpp.index('const float* Core::gemm_run(')
    end = cpp.index('\nvoid Core::gemm(', start)
    original = cpp[start:end].replace('Core::gemm_run(', 'Core::gemm_run_once(', 1)
    assert original in patched
    assert 'paired_gemm::Workspace' in header
    assert 'gemm_run_once' in header
    assert 'stage-major is not supported by the paired-input experiment' in patched
    with pytest.raises(ValueError):
        m.patch(cpp.replace('Core::gemm_run(', 'Core::renamed('), hpp)
    with pytest.raises(ValueError):
        m.patch(patched, header)


def test_pair_reconstructs_input_and_preserves_reused_device_output(tmp_path):
    # A fake GEMM rounds with the exact production converter and reuses its output
    # memory, like XRT. It records both inputs and detects accidental host writes.
    source = r'''
#include "paired_gemm.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace open_qwen36;
void require(bool ok) { if (!ok) throw std::runtime_error("paired GEMM contract"); }
int main() {
    paired_gemm::Workspace work;
    const std::vector<float> x = {1.00390625f, 1.01171875f, -1.00390625f,
        -1.01171875f, 0.f, -0.f, 0.0001234567f, 123.4567f};
    const auto original = x;
    std::array<float, 2> device{}, last{};
    std::vector<std::vector<float>> inputs;
    auto dispatch = [&](const float* p) -> const float* {
        require(device == last); // never dirty the mapped device buffer
        inputs.emplace_back(p, p + x.size());
        device = {};
        for (size_t i = 0; i < x.size(); ++i) {
            float v = bf16_to_f32(f32_to_bf16(p[i]));
            device[0] += v; device[1] += v * (i % 2 ? -0.5f : 0.25f);
        }
        last = device;
        return device.data();
    };
    const float* result = work.run(x.data(), x.size(), 2, dispatch);
    require(inputs.size() == 2 && inputs[0] == x && x == original);
    require(result != device.data() && device == last);
    for (size_t i = 0; i < x.size(); ++i) {
        float hi = bf16_to_f32(f32_to_bf16(x[i]));
        require(inputs[1][i] == x[i] - hi);
        double reconstructed = double(hi) + bf16_to_f32(f32_to_bf16(inputs[1][i]));
        require(std::abs(reconstructed - x[i]) <= std::abs(double(x[i])) * 8e-6);
    }
    for (size_t j = 0; j < 2; ++j) {
        double expected = 0, single = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            double w = j == 0 ? 1 : (i % 2 ? -0.5 : 0.25);
            expected += w * x[i]; single += w * bf16_to_f32(f32_to_bf16(x[i]));
        }
        require(std::abs(result[j] - expected) < std::abs(single - expected) * 0.01);
    }
    // A second call overwrites rather than accumulating the previous request.
    const std::array<float,2> saved = {result[0], result[1]};
    result = work.run(x.data(), x.size(), 2, dispatch);
    require(inputs.size() == 4 && result[0] == saved[0] && result[1] == saved[1]);
    // These are finite model activations; reject overflow/NaN before any dispatch.
    for (float bad : {INFINITY, NAN, 3.402823466e38f}) {
        const size_t before = inputs.size();
        bool rejected = false;
        try { work.run(&bad, 1, 2, dispatch); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected && inputs.size() == before);
    }
}
'''
    cpp = tmp_path / 'test.cpp'
    cpp.write_text(source)
    exe = tmp_path / 'test'
    subprocess.run(['c++', '-std=c++17', '-O2', '-ffp-contract=off',
                    '-I', str(TOOL), '-I', str(ROOT / 'src'), str(cpp), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def auditor():
    spec = importlib.util.spec_from_file_location('prefill_paired_audit', TOOL / 'audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mutation', ['missing', 'changed', 'none'])
def test_identity_requires_every_position_and_alias(tmp_path, mutation):
    m = auditor()
    before, after = tmp_path / 'before', tmp_path / 'after'
    before.mkdir(); after.mkdir()
    for route in ('sequential', 'hybrid', 'block_major'):
        for name in m.base.expected_dumps(route, 2):
            (before / name).write_bytes(b'original')
            (after / name).write_bytes(b'original')
    target = after / 'hybrid_p0.bin'
    if mutation == 'missing': target.unlink()
    if mutation == 'changed': target.write_bytes(b'different')
    if mutation == 'missing':
        with pytest.raises(FileNotFoundError): m.identity(before, after, 2)
    else:
        result = m.identity(before, after, 2)
        assert len(result) == 12
        assert all(result.values()) == (mutation == 'none')


def test_audit_error_replaces_stale_success(tmp_path):
    from types import SimpleNamespace
    import json
    m = auditor()
    report = tmp_path / 'comparison.json'
    report.write_text('{"passed": true}')
    with pytest.raises(FileNotFoundError):
        m.audit(SimpleNamespace(out=tmp_path, baseline=tmp_path / 'missing'))
    assert not json.loads(report.read_text())['passed']


@pytest.mark.parametrize('damage', ['none', 'score', 'nonfinite', 'route', 'residual'])
def test_audit_recomputes_gates_from_arrays(tmp_path, monkeypatch, damage):
    import json
    from types import SimpleNamespace
    import numpy as np
    m = auditor()
    base = m.base
    monkeypatch.setattr(base, 'spec_from_model_dir', lambda *args: None)
    monkeypatch.setattr(base, 'layout', lambda *args, **kwargs:
                        SimpleNamespace(AA_BYTES=24, AA_RES=0, AA_OUT2=8, AA_OUT2B=16))
    logits = np.arange(6, dtype='<f4')
    residual = np.array([1, 2], dtype='<f4')
    refs = {}
    for layer in range(4):
        for part in ('', 'cont_'):
            name = f'ref_{part}res{layer}.bin'
            residual.tofile(tmp_path / name)
            refs[name] = base.digest(tmp_path / name)
    for name in ('ref_p0.bin', 'ref_cont_logits.bin'):
        logits.tofile(tmp_path / name)
        refs[name] = base.digest(tmp_path / name)
    (tmp_path / 'ids.txt').write_text('1')
    meta = dict(layers=4, ids=[1], hidden=2, vocab=6, real_vocab=6, model='fixture',
                max_ctx=2048, expected_tokens=[5, 5], sources={}, references=refs,
                ids_sha256=base.digest(tmp_path / 'ids.txt'))
    (tmp_path / 'reference.json').write_text(json.dumps(meta))
    report = dict(passed=True, reference_sha256=base.digest(tmp_path / 'reference.json'), routes=[])
    for route in ('sequential', 'hybrid', 'block_major'):
        names = base.expected_dumps(route, 1)
        for name in names:
            row = logits.copy()
            if damage == 'nonfinite' and name == 'hybrid_p0.bin': row[0] = np.nan
            row.tofile(tmp_path / name)
        act = np.concatenate((residual, np.zeros(4, dtype='<f4')))
        if damage == 'residual' and route == 'hybrid': act[0] += 1
        act.tofile(tmp_path / (route + '-act.bin'))
        log = tmp_path / (route + '.log')
        log.write_text('token 5\ntoken 5\n')
        rows = [dict(position=p, **base.logit_score(logits, logits, 6)) for p in range(2)]
        if damage == 'score' and route == 'hybrid': rows[0]['correlation'] = 0.9
        report['routes'].append(dict(route=route, passed=True, log_sha256=base.digest(log),
            tokens_match=True, last_prompt_alias=True, continuation_last_residual_maxrel=0., logits=rows,
            outputs={name: base.digest(tmp_path / name) for name in names + [route + '-act.bin']}))
    report['hybrid_schedule_identity'] = {name: True for name in base.expected_dumps('hybrid', 1)}
    if damage == 'route': report['routes'].pop()
    (tmp_path / 'report.json').write_text(json.dumps(report))
    if damage == 'none':
        assert m.validate_run(tmp_path)[1]['passed']
    else:
        with pytest.raises(ValueError): m.validate_run(tmp_path)
