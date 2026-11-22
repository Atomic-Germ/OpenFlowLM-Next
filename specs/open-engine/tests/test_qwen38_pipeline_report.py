"""The PR #4 report must not claim more than its measurements support."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'specs/open-engine/reports/qwen38-deltanet-pipeline-performance.json'


def data():
    return json.loads(REPORT.read_text())


def test_report_is_profiling_only_and_excludes_kernel_experiments():
    d = data()
    assert d['status'] == 'GO'
    assert d['scope'].startswith('profiling only')
    excluded = ' '.join(d['excluded'])
    assert 'activation-reuse' in excluded and 'dequant' in excluded
    assert d['baseline_commit'].startswith('9d77431a')


def test_stage_decomposition_is_internally_consistent():
    for capture in data()['prefill']:
        total = capture['request_total_ms']
        parts = (capture['lmhead_ms'] + capture['part0_ms'] + capture['part1_ms']
                 + capture['unattributed_ms'])
        assert abs(parts - total) <= 1.0
        assert capture['verdict'] == 'PASS'
        assert all(check['ok'] for check in capture['checks'])
        stages = capture['host_stages']
        # mid nests conv/rule; host nests mid. Never add a subset to its parent.
        assert stages['dn_conv_ms'] + stages['dn_rule_ms'] <= stages['mid_ms'] * 1.05 + 1
        assert stages['mid_ms'] <= capture['part1_ms'] * 1.05 + 1


def test_deltanet_is_not_reported_as_the_bottleneck():
    d = data()
    dn = d['deltanet_ms']
    assert dn['total'] == dn['conv'] + dn['rule'] + dn['state']
    assert dn['verdict'] == 'not the bottleneck'
    first = d['ranked_candidates'][0]
    assert 'fold' in first['name'] and first['percent_prefill'] > dn['percent_prefill']


def test_ceiling_is_a_host_only_bound_with_arithmetic_preserving_claim():
    c = data()['ceiling']
    assert c['baseline_ms'] > c['candidate_ms'] > 0
    assert c['saved_ms'] == round(c['baseline_ms'] - c['candidate_ms'], 1)
    assert 0 < c['saved_percent'] < 100
    src = (ROOT / 'utilities/prefill-block/host_fold_ceiling.cpp').read_text()
    # The candidate must state the preserved association explicitly.
    assert '(A+B) + (C+D)' in src
    assert 'bf16_to_f32' in src


def test_decode_is_not_claimed_as_cpu_bound_and_full_model_gates_are_listed():
    d = data()
    assert any('TTFT' in item for item in d['not_measured'])
    assert any('decode' in item for item in d['not_measured'])
    assert 'queue residence' in d['part0_definition']
    assert d['prefill'][0]['part1_ms'] > 0
