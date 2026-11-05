"""Executable contracts for per-layer profiling and its measurement drivers."""
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'utilities/prefill-block' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load('layer_profile')
H = load('profile_overhead')


def fixture(what='gemm-block', tokens=256):
    header = dict(record='header', version=1, layers=2, family='qwen35', spec_hash='test')
    request = dict(record='request', what=what, tokens=tokens, blocks=1, layers=2,
                   total_ms=12., lmhead_ms=1., layer_wall_ms=10., residue_ms=2.)
    rows = [dict(record='layer', what=what, layer=i, kind='linear_attention',
                 **{c: 0. for c in P.LAYER_COLS + P.HOST_COLS}) for i in range(2)]
    for row in rows:
        row.update(wall_ms=5., gemm_ms=3., host_ms=1.)
    return header, [request], rows


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'phase', 'nan', 'header'])
def test_rejects_incomplete_or_invalid_profile(damage):
    h, reqs, rows = fixture()
    if damage == 'missing': rows.pop()
    if damage == 'duplicate': rows[1]['layer'] = 0
    if damage == 'phase': rows[1]['what'] = 'step'
    if damage == 'nan': rows[0]['wall_ms'] = float('nan')
    if damage == 'header': h = None
    with pytest.raises(ValueError):
        P.validate_profile(h, reqs, rows)


def test_multi_block_totals_and_unaccounted_time():
    h, reqs, rows = fixture()
    result = P.summarise(dict(route='block', wall_s=1, profile='', header=h,
                             requests=reqs * 2, layers=rows * 2))
    assert result['tokens'] == 512
    assert result['block'] == 2
    assert result['stage_unaccounted_ms'] == 6


@pytest.mark.parametrize('route,force,what', [('block', '0', 'gemm-block'),
                                           ('layer-major', '1', 'gemm-prompt')])
def test_route_is_forced_and_verified(tmp_path, monkeypatch, route, force, what):
    monkeypatch.setattr(P.O, 'LOCK', tmp_path / 'lock')
    def fake_run(cmd, **kw):
        assert kw['env']['OFLM_OPEN_LAYER_MAJOR'] == force
        h, reqs, rows = fixture(what)
        Path(kw['env']['OFLM_OPEN_PROFILE']).write_text('\n'.join(map(json.dumps, [h, *reqs, *rows])))
        return SimpleNamespace(returncode=0, stdout='', stderr='NPU power mode: performance')
    monkeypatch.setattr(P.subprocess, 'run', fake_run)
    P.run([1] * 256, route, layers=2, max_ctx=4096, max_tokens=1, steps=4,
          pmode='performance', workdir=tmp_path, timeout=1)


def test_wrong_observed_route_is_rejected():
    h, reqs, rows = fixture()
    with pytest.raises(ValueError):
        P.validate_route('layer-major', reqs, tokens=256, steps=4, max_tokens=1)


def test_decode_budget_covers_prompt_and_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(H.O, 'LOCK', tmp_path / 'lock')
    import subprocess
    def fake_run(cmd, **kw):
        assert int(kw['env']['OFLM_OPEN_PROFILE_STEPS']) == 39
        h, _, _ = fixture('step', 1)
        h['layers'] = 64
        records = [h]
        for i in range(39):
            req = dict(record='request', what='step', tokens=1, blocks=1, layers=64,
                       total_ms=65., lmhead_ms=1. if i >= 31 else 0.)
            records.append(req)
            for layer in range(64):
                records.append(dict(record='layer', what='step', layer=layer, kind='linear',
                                    **{c: 0. for c in P.LAYER_COLS + P.HOST_COLS}))
        Path(kw['env']['OFLM_OPEN_PROFILE']).write_text('\n'.join(map(json.dumps, records)))
        return SimpleNamespace(returncode=0, stdout='token 1\n' * 8,
                               stderr='NPU power mode: performance\ndecode 7 tokens: 65 ms/token (15.38 tok/s)')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    result = H.one([1] * 32, 'decode', True, tmp_path, 'on', 'performance', 1)
    assert result['profiled_decode_steps'] == 7


@pytest.mark.parametrize('what', ['gemm-block', 'gemm-prompt', 'step'])
def test_jsonl_parser_requires_request_group_order(tmp_path, what):
    h, reqs, rows = fixture(what)
    path = tmp_path / 'profile.jsonl'
    path.write_text('\n'.join(map(json.dumps, [h, *reqs, *rows])))
    assert P.read_jsonl(path) == (h, reqs, rows)
    path.write_text('\n'.join(map(json.dumps, [h, *reqs, *reqs, *rows, *rows])))
    with pytest.raises(ValueError):
        P.read_jsonl(path)


def test_actual_cpp_profile_methods_and_step_label(tmp_path):
    """Compile the real profiling methods without XRT or model weights."""
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    source = (ROOT / 'src/open_qwen36/core.cpp').read_text()
    header = (ROOT / 'src/open_qwen36/core.hpp').read_text()
    structs = '\n'.join(re.search(r'struct ' + name + r' \{.*?\n\};', header, re.S)[0]
                        for name in ('StepTiming', 'LayerTiming'))
    methods = source[source.index('static LayerTiming diff_timing('):source.index('\nxrt::hw_context& Core::context(')]
    start = source.index('    if (profile_ && profile_steps_left_ > 0)')
    step_hook = source[start:source.index('\n    }', start) + 6]
    code = r'''
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now()-t).count(); }
''' + structs + r'''
struct Core {
    FILE* profile_ = stdout;
    bool profile_first_ = true, profile_active_ = false;
    std::string profile_what_;
    size_t profile_tokens_ = 0, profile_blocks_ = 0;
    int profile_pending_ = -1, profile_steps_left_ = 2, nl_ = 2;
    struct Manifest { std::string family="qwen35", spec_hash="test"; } man_;
    struct Type { std::string name="linear_attention"; } type;
    std::vector<Type*> types_ = {&type, &type};
    StepTiming timing_, profile_base_;
    std::vector<LayerTiming> profile_rows_;
    Clock::time_point profile_t0_;
    void profile_begin(const char*, size_t, size_t);
    void profile_layer(const char*, int);
    void profile_switch(int);
    void profile_close();
    void profile_end();
    void sampled_step(bool want_logits) {
''' + step_hook + r'''
        for(int layer=0; layer<2; ++layer) {
            timing_.part0_ms += 3.;
            profile_layer("linear_attention", layer);
        }
        timing_.total_ms = 10.;
        profile_end();
    }
};
''' + methods + '\nint main() { Core core; core.sampled_step(false); core.sampled_step(true); }\n'
    path = tmp_path / 'profile.cpp'
    path.write_text(code)
    binary = tmp_path / 'profile'
    subprocess.run([compiler, '-std=c++17', str(path), '-o', str(binary)], check=True, capture_output=True)
    output = subprocess.check_output([str(binary)], text=True)
    profile = tmp_path / 'profile.jsonl'
    profile.write_text(output)
    _, reqs, rows = P.read_jsonl(profile)
    assert [r['what'] for r in reqs] == ['step', 'step']
    assert [r['layer'] for r in rows] == [0, 1, 0, 1]
    assert [r['gemm_ms'] for r in rows] == [3., 3., 3., 3.]


@pytest.mark.parametrize('off,on', [([], []), ([1], [1, 2]), ([1], [float('nan')]), ([0], [1])])
def test_overhead_rejects_invalid_timing_samples(off, on):
    with pytest.raises(ValueError):
        H.verdict(off, on)


@pytest.mark.parametrize('log', ['', 'NPU power mode: performance (xrt-smi failed; mode is whatever it was)'])
def test_failed_power_setting_cannot_pass(log):
    with pytest.raises(ValueError):
        P.require_power_mode(log, 'performance')
    assert P.require_power_mode(log, 'none').startswith('left unchanged')


@pytest.mark.parametrize('parent', [False, True])
def test_overhead_excludes_warmup_and_separates_parent_gate(tmp_path, monkeypatch, parent):
    import sys
    out = tmp_path / 'out'
    argv = ['profile_overhead.py', '--out', str(out)]
    if parent:
        argv += ['--baseline-cli', str(tmp_path / 'parent')]
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(P, 'provenance', lambda paths: {})
    monkeypatch.setattr(H.P, 'provenance', lambda paths: {})
    # main resolves these defaults; restore the shared oracle module after the test.
    for key in ('CLI', 'MODEL', 'KERNELS'):
        monkeypatch.setattr(H.O, key, getattr(H.O, key))
    calls = []
    def one(ids, what, enabled, workdir, tag, pmode, timeout, cli):
        calls.append(tag)
        ms = 900 if '-1' in tag else 101 if enabled else 100
        return dict(generated_ids=[7] * 8, engine_ms=ms, profiled=enabled, tag=tag)
    monkeypatch.setattr(H, 'one', one)
    assert H.main() == 0
    result = json.loads((out / 'result.json').read_text())
    row = result['rows'][0]
    assert row['off'] == [100] * 5
    assert row['on'] == [101] * 5
    assert sum(s['warmup'] for s in row['samples']) == (3 if parent else 2)
    assert calls.index('on1') < calls.index('off1')
    assert calls.index('off2') < calls.index('on2')
    assert (result['disabled_path_regression'] == 'PASS') == parent
