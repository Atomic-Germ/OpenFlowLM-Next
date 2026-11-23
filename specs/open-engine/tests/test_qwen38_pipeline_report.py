"""Historical PR4 summary consistency and the shared synthetic probe.

These checks do not validate legacy raw captures against the current capture gate;
test_pipeline_capture covers that fail-closed evidence contract separately.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'specs/open-engine/reports/qwen38-deltanet-pipeline-performance.json'
PROBE = ROOT / 'utilities/prefill-block/host_fold_probe.hpp'


def data():
    return json.loads(REPORT.read_text())


def test_report_scope_and_exclusions():
    d = data()
    assert d['scope'].startswith('profiling only')
    excluded = ' '.join(d['excluded'])
    assert 'activation-reuse' in excluded and 'dequant' in excluded
    assert d['baseline_commit'].startswith('9d77431a')
    assert d['status'] in ('GO', 'NEEDS MORE EVIDENCE', 'NO-GO')


def test_stage_decomposition_is_internally_consistent():
    d = data()
    stages = d['prefill_stages_ms']
    total = stages['request_total_ms']
    parts = stages['lmhead_ms'] + stages['gemm_ms'] + stages['host_ms'] + stages['unattributed_ms']
    assert abs(parts - total) <= 1.0
    assert stages['dn_conv_ms'] + stages['dn_rule_ms'] <= stages['mid_ms'] * 1.05 + 1
    assert stages['mid_ms'] <= stages['host_ms'] * 1.05 + 1
    assert d['prefill_median_wall_ms'] == pytest.approx(
        sorted(c['prompt_wall_ms'] for c in d['prefill'])[len(d['prefill']) // 2])


def test_corrected_probe_is_byte_equivalent_and_rejects_nan():
    """Compile the shared probe and check behaviour, not comments."""
    cxx = shutil.which('g++')
    if cxx is None:
        pytest.skip('no C++ compiler; behavioural probe NOT MEASURED')
    src = r'''
#include "host_fold_probe.hpp"
#include <cstdio>
#include <cstring>
int main() {
    for (const char* kind : {"random", "rounding"}) {
        auto f = fold_probe::make_fixture(5, 8, 4, kind);
        std::string why;
        if (!fold_probe::equivalent(f, &why)) { std::printf("FAIL %s: %s\n", kind, why.c_str()); return 1; }
    }
    // NaN must be rejected by BOTH arms: agreement, not acceptance.
    auto bad = fold_probe::make_fixture(5, 8, 4, "nan");
    auto cmp = fold_probe::compare(bad);
    if (cmp.baseline_accepted || cmp.candidate_accepted || !cmp.agree) {
        std::printf("FAIL: nan handling baseline=%d candidate=%d agree=%d\n",
                    cmp.baseline_accepted, cmp.candidate_accepted, cmp.agree);
        return 2;
    }
    // Deliberately corrupt the last candidate output: full-buffer comparison must fail.
    auto f = fold_probe::make_fixture(5, 8, 4, "random");
    fold_probe::BaselineState b; fold_probe::CandidateState c;
    std::vector<float> bo, co;
    if (!fold_probe::run_baseline(b, f, &bo)) return 3;
    if (!fold_probe::run_candidate(c, f, &co)) return 4;
    if (bo.size() != co.size() || std::memcmp(bo.data(), co.data(), bo.size() * sizeof(float))) {
        std::printf("FAIL: arms differ\n"); return 5;
    }
    co.back() = std::nextafter(co.back(), std::numeric_limits<float>::infinity());
    if (fold_probe::equal_outputs(bo, co)) return 6;
    return 0;
}
'''
    import tempfile
    cleanup = tempfile.TemporaryDirectory(prefix='pr4a-probe-')
    work = Path(cleanup.name)
    exe = work / 'probe.bin'
    proc = subprocess.run([cxx, '-O2', '-std=c++20', f'-I{ROOT}', f'-I{ROOT / "src"}',
                           f'-I{ROOT / "utilities/prefill-block"}', '-x', 'c++',
                           '-o', str(exe), '-'], input=src, text=True,
                          capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    proc = subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_paired_summary_is_recomputed_from_retained_samples():
    d = data()
    fold = d['host_fold']
    b, c = fold['samples_baseline_ms'], fold['samples_candidate_ms']
    assert len(b) == len(c) >= 5
    delta = [y - x for x, y in zip(b, c)]
    pct = [100 * (y / x - 1) for x, y in zip(b, c)]
    med = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else 0.5 * (
        sorted(v)[len(v) // 2 - 1] + sorted(v)[len(v) // 2])
    assert fold['median_baseline_ms'] == pytest.approx(med(b), abs=0.01)
    assert fold['median_candidate_ms'] == pytest.approx(med(c), abs=0.01)
    assert fold['median_paired_delta_ms'] == pytest.approx(med(delta), abs=0.01)
    assert fold['median_paired_change_percent'] == pytest.approx(med(pct), abs=0.01)
    assert fold['wins'] == sum(y < x for x, y in zip(b, c))
    assert fold['numerically_equivalent'] is True
    assert fold['warmup_pairs_excluded'] == 1


def test_decode_samples_are_generation_steps_with_lmhead():
    d = data()
    dec = d['decode']
    assert dec['phase'] == 'generated-token decode'
    assert dec['prompt_steps_excluded'] is True
    assert dec['samples'], 'no generated-token samples'
    # A generated step runs the LM head; prompt steps in the old capture did not.
    assert all(s['lmhead_ms'] > 0 for s in dec['samples'])
    assert all(s['total_ms'] > 0 for s in dec['samples'])


def test_unmeasured_is_stated_and_old_capture_is_marked_invalid():
    d = data()
    assert d['invalid_captures'], 'old decode capture must be retained and marked'
    assert all(c['reason'] for c in d['invalid_captures'])
    assert any('TTFT' in item for item in d['not_measured'])
