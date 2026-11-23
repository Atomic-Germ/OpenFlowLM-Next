"""CPU-only execution of the production fold, lifecycle and synthetic probe.

The route bodies are stubbed; the real scoped profiler and fold run unchanged.
Hardware route/state acceptance remains a separate, unmeasured gate.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / 'utilities/prefill-block/host_fold_probe.hpp'

CPP = r'''
#include "open_qwen36/fold_profile.hpp"
#include "host_fold_probe.hpp"
#include <cstdlib>
#include <new>
#include <string>

static bool count_allocations = false;
static size_t allocations = 0;
void* operator new(std::size_t n) {
    if (count_allocations) ++allocations;
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

struct Clock {
    using duration = std::chrono::nanoseconds;
    using time_point = std::chrono::time_point<Clock>;
    static inline long long tick = 0, reads = 0;
    static time_point now() { ++reads; tick += 1000000; return time_point(duration(tick)); }
};

struct Fold {
    open_qwen36::prefill_precision::Workspace ws;
    fold_probe::Fixture f = fold_probe::make_fixture(17, 8, 4, "rounding");
    const float* run(open_qwen36::prefill_precision::FoldPhaseTiming* timing) {
        return ws.run<Clock>(f.x.data(), f.x.size(), f.a.size(),
            [&](bool w, const float* x) {
                Clock::tick += 100000000;  // Dispatch residence is outside fold windows.
                bool low = x != f.x.data();
                return (w ? (low ? f.d : f.c) : (low ? f.b : f.a)).data();
            }, timing);
    }
};

void unit(open_qwen36::FoldProfile& p, const char* phase, size_t real, size_t physical,
          int folds, int exit_mode = 0) {
    bool request_active = true;
    struct CheckReset {
        bool& active;
        ~CheckReset() { if (active) std::abort(); }
    } check{request_active};
    open_qwen36::FoldProfile::Scope scope(p, phase, real, physical, &request_active);
    Fold f;
    for (int i = 0; i < folds; ++i) {
        f.run(p.timing());
        if (exit_mode == 1) throw std::runtime_error("dispatch failed");
        if (exit_mode == 2) return;  // No completion: suppress a partial success row.
    }
    scope.complete();
    if (exit_mode == 3) throw std::runtime_error("tail failed");
}

int main(int argc, char** argv) {
    const std::string mode = argv[1];
    if (mode == "lifecycle") {
        open_qwen36::FoldProfile p;
        p.attach(stdout);
        // Dense and hybrid bodies use the same scope at their shared entry.
        for (int route : {1, 2}) {
            for (int request = 0; request < 2; ++request) {
                unit(p, "gemm-block", 256, 256, route);
                unit(p, "gemm-block", 1, 256, route);
                unit(p, "gemm-prompt", 257, 512, route * 2);
                for (int failure : {1, 2, 3}) {
                    try { unit(p, "gemm-block", 17, 256, route, failure); }
                    catch (const std::runtime_error&) {}
                    if (p.timing()) return 10;
                }
                unit(p, "gemm-block", 255, 256, route);
            }
        }
        if (p.timing()) return 11;
        return 0;
    }
    if (mode == "invalid") {
        open_qwen36::FoldProfile p; p.attach(stdout);
        for (int i = 0; i < 4; ++i) {
            bool rejected = false;
            try {
                open_qwen36::FoldProfile::Scope bad(p, i == 0 ? "" : "gemm-block",
                    i == 1 ? 0 : 257, i == 2 ? 256 : 512);
                if (i == 3) {
                    open_qwen36::FoldProfile::Scope nested(p, "gemm-block", 1, 256);
                }
            } catch (const std::exception&) { rejected = true; }
            if (!rejected || p.timing()) return 12;
        }
        unit(p, "gemm-block", 1, 256, 1);
        return 0;
    }
    if (mode == "disabled") {
        Fold f; f.run(nullptr);  // Warm the inference workspace before counting allocations.
        open_qwen36::FoldProfile p;
        Clock::reads = 0; allocations = 0; count_allocations = true;
        {
            open_qwen36::FoldProfile::Scope off(p, "gemm-block", 1, 256);
            if (p.timing()) return 13;
            f.run(p.timing());
            off.complete();
        }
        count_allocations = false;
        std::printf("%lld %zu\n", Clock::reads, allocations);
        return Clock::reads || allocations ? 14 : 0;
    }
    if (mode == "exclusive") {
        Fold f;
        open_qwen36::prefill_precision::FoldPhaseTiming t;
        Clock::reads = 0;
        f.run(&t);
        std::printf("%lld %lld %.0f %.0f %.0f\n", Clock::reads, t.calls,
            t.split_ms, t.copy_ms, t.accumulate_ms);
        return Clock::reads == 16 && t.calls == 1 && t.split_ms == 2 &&
            t.copy_ms == 3 && t.accumulate_ms == 3 ? 0 : 15;
    }
    if (mode == "numeric") {
        for (const char* kind : {"random", "rounding", "cancellation"}) {
            auto f = fold_probe::make_fixture(17, 8, 4, kind);
            fold_probe::BaselineState b; fold_probe::CandidateState c;
            std::vector<float> bo, co;
            if (!fold_probe::run_baseline(b, f, &bo) ||
                !fold_probe::run_candidate(c, f, &co) ||
                !fold_probe::equal_outputs(bo, co)) return 20;
            // Real dispatches overwrite one mapped output; host sums must not dirty it.
            open_qwen36::prefill_precision::Workspace mapped_ws;
            std::vector<float> mapped(f.a.size());
            auto dispatch = [&](bool w, const float* x) {
                bool low = x != f.x.data();
                const auto& product = w ? (low ? f.d : f.c) : (low ? f.b : f.a);
                std::copy(product.begin(), product.end(), mapped.begin());
                return mapped.data();
            };
            auto result = mapped_ws.run(f.x.data(), f.x.size(), f.a.size(), dispatch);
            if (std::memcmp(result, bo.data(), bo.size() * sizeof(float)) ||
                !fold_probe::equal_outputs(mapped, f.d)) return 25;
            open_qwen36::prefill_precision::FoldPhaseTiming timing;
            result = mapped_ws.run(f.x.data(), f.x.size(), f.a.size(), dispatch, &timing);
            if (std::memcmp(result, bo.data(), bo.size() * sizeof(float))) return 26;
            allocations = 0; count_allocations = true;
            fold_probe::run_baseline(b, f, nullptr);
            fold_probe::run_candidate(c, f, nullptr);
            count_allocations = false;
            if (allocations) return 27;
            for (size_t i = 0; i < bo.size(); ++i) {
                volatile float ab = f.a[i] + f.b[i], cd = f.c[i] + f.d[i];
                float oracle = ab + cd;
                if (std::memcmp(&oracle, &bo[i], sizeof(float))) return 21;
            }
            // Actual full-buffer corruption, including an element beyond the first.
            co.back() = std::nextafter(co.back(), std::numeric_limits<float>::infinity());
            if (fold_probe::equal_outputs(bo, co)) return 22;
            co = bo; co.pop_back();
            if (fold_probe::equal_outputs(bo, co)) return 23;
        }
        for (const char* kind : {"nan", "inf", "negative_inf", "overflow"}) {
            auto c = fold_probe::compare(fold_probe::make_fixture(17, 8, 4, kind));
            if (c.baseline_accepted || c.candidate_accepted || c.outputs_equal || !c.agree)
                return 24;
        }
        return 0;
    }
    return 99;
}
'''


def compile_probe(tmp_path, header=None):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable; CPU fold behaviour NOT MEASURED')
    include = PROBE.parent
    if header is not None:
        include = tmp_path
        (include / PROBE.name).write_text(header)
    exe = tmp_path / 'fold-probe'
    result = subprocess.run(
        [compiler, '-std=c++17', '-O2', '-ffp-contract=off', f'-I{include}',
         f'-I{ROOT / "src"}', '-x', 'c++', '-', '-o', str(exe)],
        input=CPP, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return exe


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    return compile_probe(tmp_path_factory.mktemp('fold-behaviour'))


def run_probe(probe, mode):
    return subprocess.run([str(probe), mode], text=True, capture_output=True, timeout=30)


def test_scoped_lifecycle_multiple_blocks_padding_routes_requests_and_errors(probe):
    import json
    result = run_probe(probe, 'lifecycle')
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows.pop(0) == dict(record='header', version=1)
    expected = []
    for route in (1, 2):
        for _ in range(2):
            expected.extend([('gemm-block', 256, 256, route), ('gemm-block', 1, 256, route),
                             ('gemm-prompt', 257, 512, route * 2), ('gemm-block', 255, 256, route)])
    assert [(r['what'], r['real_tokens'], r['block_tokens'], r['calls']) for r in rows] == expected
    for row in rows:
        assert row['total_ms'] == 8 * row['calls']


def test_invalid_metadata_nested_scopes_rejected_and_recovery_is_clean(probe):
    result = run_probe(probe, 'invalid')
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 2


def test_disabled_profiler_reads_no_clock_and_allocates_no_records(probe):
    result = run_probe(probe, 'disabled')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '0 0'


def test_exclusive_windows_exclude_dispatch_residence_and_nested_work(probe):
    result = run_probe(probe, 'exclusive')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '16 1 2 3 3'


def test_exact_outputs_adversarial_association_nonfinite_and_real_corruption(probe):
    result = run_probe(probe, 'numeric')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('mutation', ['left_association', 'right_association', 'corrupt_last'])
def test_temporary_candidate_mutations_are_rejected(tmp_path, mutation):
    # The source is the mutation artifact here, not a proxy for behaviour: every
    # temporary mutant must compile and fail the SAME executable numerical gate.
    header = PROBE.read_text()
    anchor = 'for (size_t i = 0; i < outputs; ++i) st.out[i] += st.low_sum[i];'
    assert header.count(anchor) == 1, 'candidate mutation anchor changed'
    replacements = {
        'left_association': 'for (size_t i = 0; i < outputs; ++i) { volatile float ab = f.a[i] + f.b[i]; volatile float abc = ab + f.c[i]; st.out[i] = abc + f.d[i]; }',
        'right_association': 'for (size_t i = 0; i < outputs; ++i) { volatile float cd = f.c[i] + f.d[i]; volatile float bcd = f.b[i] + cd; st.out[i] = f.a[i] + bcd; }',
        'corrupt_last': anchor + '\nst.out.back() = std::nextafter(st.out.back(), std::numeric_limits<float>::infinity());',
    }
    mutant = compile_probe(tmp_path, header.replace(anchor, replacements[mutation]))
    result = run_probe(mutant, 'numeric')
    assert result.returncode == 20, (mutation, result.stdout, result.stderr)


def test_benchmark_describes_verified_manifest_shapes_without_running_campaign(tmp_path):
    import re
    from recipes.load import load_spec
    from recipes.manifest import manifest

    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable; benchmark geometry NOT MEASURED')
    exe = tmp_path / 'ceiling'
    compiled = subprocess.run(
        [compiler, '-std=c++17', '-O2', f'-I{ROOT / "src"}',
         str(ROOT / 'utilities/prefill-block/host_fold_ceiling.cpp'), '-o', str(exe)],
        text=True, capture_output=True, timeout=120)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(exe), '--describe'], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    data = manifest(load_spec(ROOT / 'open_kernels/recipes/specs/qwen35-27b.json'))
    described = [line.split() for line in lines[1:]]
    for kind, name in [('linear', 'linear_attention'), ('full', 'full_attention')]:
        block = data['layer_types'][name]['gemm_block']
        expected = []
        for step in block['program'] + block['ffn_program']:
            match = re.fullmatch(r'gemm_n(\d+)_k(\d+)', step['kernel'])
            assert match, step
            expected.append((int(match[1]), int(match[2]), block['t']))
        assert [(int(n), int(k), int(t)) for route, _, n, k, t in described if route == kind] == expected
