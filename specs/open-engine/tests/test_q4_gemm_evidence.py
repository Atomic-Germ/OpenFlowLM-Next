"""Validate the measured JSON artifact without requiring local benchmark captures."""
import copy
import hashlib
import importlib.util
import json
import random
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('q4_gemm_verify', ROOT / 'utilities/q4-gemm/verify.py')
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


@pytest.fixture
def report():
    return json.loads((ROOT / 'specs/open-engine/reports/qwen38-q4-gemm-dma.json').read_text())


def test_shipped_summary_has_paired_shapes_without_claiming_complete_acceptance(report):
    TOOL.check_summary(report)
    shapes = ['14336_k5120', '16384_k5120', '34816_k5120', '5120_k17408', '5120_k6144']
    assert {c['name'] for c in report['cases']} == {
        f'gemm_n{shape}{suffix}' for shape in shapes for suffix in ('', '_low')}
    assert all(c['summary']['paired_wins'] == report['methodology']['pairs'] for c in report['cases'])
    assert report['status'].startswith('DRAFT')
    assert report['validation']['full_model'].startswith('PASS: 64-layer captured logits/state')
    assert report['acceptance']['integration']['server_aba'].startswith('PASS for both private arms')
    assert report['acceptance']['runtime_verification'].startswith('Benchmark: XRT 2.26.0 both arms')
    server = report['acceptance']['server_candidate']
    assert server['verdict'] == 'PASS' and server['runtime_comparable'] is True
    assert server['same_answer_between_arms'] is True and server['arm_separation']['manifest_same'] is True
    assert len(server['arm_separation']['changed']) == 10
    for arm in ('before', 'after'):
        assert server['arms'][arm]['graded_pass'] == 11
        assert server['arms'][arm]['gemm_programs'] == 10
        assert all('.2.25.' in p for p in server['arms'][arm]['libraries'])
    provenance = report['acceptance']['runtime_provenance']
    assert provenance['benchmark_comparable'] is True
    assert provenance['benchmark_runtime'] == 'XRT 2.26.0 at /opt/xilinx/xrt/lib for both arms'
    assert provenance['server'] == 'XRT 2.25.00 distro libraries for both arms'
    scope = report['acceptance']['scope_review']
    assert scope['option'].startswith('B:') and scope['pr2_tip_ancestor_of_upstream'] is False
    assert scope['rebase_performed'] is False and scope['review_target'] == '836072db'
    # Counts move with every review commit, so the record stores the structural
    # fact (is the base in upstream?) and lets git answer the numbers.
    import subprocess
    check = subprocess.run(['git', 'merge-base', '--is-ancestor', scope['review_target'],
                            'upstream/main'], cwd=ROOT, capture_output=True)
    if check.returncode not in (0, 1):
        pytest.skip('git history not available')
    assert scope['verified']['pr2_tip_in_upstream'] == (check.returncode == 0)
    assert scope['verified']['pr2_tip_in_upstream'] is False
    diagnosis = report['validation']['context_diagnosis']
    assert diagnosis['firmware_status'] == '0x2000003'
    assert diagnosis['firmware_status_name'] == 'AIE2_STATUS_MGMT_ERT_NOAVAIL'
    assert diagnosis['observed_ceiling_healthy'] == 16
    assert diagnosis['observed_ceiling_degraded'] == 8
    assert diagnosis['ceiling_is_stable'] is False
    assert diagnosis['firmware_max_proven'] is False
    assert diagnosis['capacity_returned_after_destroy'] is True
    assert diagnosis['leak_after_64_churn_cycles'] is False
    assert diagnosis['process_exit_leak'] is None
    assert diagnosis['valid_control_substitute'] is False
    assert diagnosis['trigger_identified'] is False
    assert diagnosis['minimal_reproducer'] is None
    assert diagnosis['exact_trigger_operation'] is None
    assert diagnosis['recovery_validated'] is False
    assert diagnosis['pre_reboot_observation']['held'] == [16, 16, 16]
    assert 'create_ctx_resp' in diagnosis['root_cause_experiment']


def test_fresh_boot_phase_a_preserves_draft_and_does_not_invent_a_trigger(report):
    diagnosis = report['validation']['context_diagnosis']
    phase = diagnosis['phase_a_fresh_boot']
    assert phase['status'] == 'COMPLETED_NO_DECREASE'
    assert phase['start_boot_age_s'] <= 900
    assert phase['initial_capacity_controls'] == [16, 16, 16]
    assert phase['post_process_capacity'] == [16] * 12
    assert len(phase['processes']) == 12
    assert all(p['returncode'] == 0 and p['runtime_verified'] and
               p['capacity_before'] == p['capacity_after'] == 16 for p in phase['processes'])
    assert len({p['pid'] for p in phase['processes']}) == 12
    assert phase['kernel_windows_available'] == 27
    assert phase['kernel_error_count'] == 0
    assert phase['all_trajectories_match'] is True and len(phase['generated_ids']) == 16
    assert len(phase['log_identities']) == 36
    assert all(len(h) == 64 for h in phase['log_identities'].values())
    assert phase['post_idle_observation']['capacity'] == [16, 16, 16]
    assert phase['full_model_acceptance'] is False
    assert diagnosis['trigger_identified'] is False
    assert diagnosis['exact_trigger_operation'] is None
    assert diagnosis['minimal_reproducer'] is None
    assert diagnosis['recovery_validated'] is False
    assert report['end_to_end_acceptance'] is False
    assert report['production_default_dma_batch'] == 1


@pytest.mark.parametrize('field,value', [('end_to_end_acceptance', True),
                                        ('production_default_dma_batch', 4)])
def test_unvalidated_promotion_is_rejected(report, field, value):
    report[field] = value
    with pytest.raises(ValueError, match='production promotion'):
        TOOL.check_summary(report)


@pytest.mark.parametrize('mutation', [
    lambda r: r['cases'][0]['summary'].__setitem__('change_percent', -50),
    lambda r: r['cases'][0]['effective_tflops'].__setitem__('after', 99),
    lambda r: r['cases'][0]['summary']['after'].__setitem__('median_ms', float('nan')),
    lambda r: r['cases'][0]['summary'].__setitem__('paired_wins', 6),
    lambda r: r['cases'][0].__setitem__('rel_fro', 0.006),
    lambda r: r['methodology'].__setitem__('rel_fro_gate', 0.01),
    lambda r: r['cases'][0]['builds']['after'].__setitem__('max_text', 16384),
    lambda r: r['cases'][0]['builds']['after'].__setitem__('dma_batch', 1),
    lambda r: r['cases'][0]['fixtures'].clear(),
    lambda r: r['cases'][0]['fixtures'].__setitem__('weights.bin', '0' * 64),
    lambda r: r['cases'][0].__setitem__('harness_sha256', 'missing'),
    lambda r: r['cases'][0]['builds']['before']['binaries'].pop('insts.bin'),
    lambda r: r['cases'].append(copy.deepcopy(r['cases'][0])),
    lambda r: r['cases'].pop(1),
])
def test_inconsistent_numbers_incomplete_pairs_or_weakened_gates_are_rejected(report, mutation):
    mutation(report)
    with pytest.raises(ValueError):
        TOOL.check_summary(report)


def test_local_audit_fails_closed_for_missing_or_changed_source_evidence(tmp_path):
    pytest.importorskip('numpy', reason='local audit uses the comparison driver')
    record = {'sources': {'missing': '0' * 64}, 'cases': []}
    with pytest.raises(FileNotFoundError):
        TOOL.check_local(record, tmp_path)
    (tmp_path/'missing').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        TOOL.check_local(record, tmp_path)


def test_initial_and_confirmation_campaigns_are_not_mixed(report):
    """The artifact is the reported evidence; campaigns must retain their own numbers."""
    rows = [r for r in report['acceptance']['prefill'] if r['tokens'] == 1024]
    assert len(rows) == 2
    initial, confirmation = rows
    assert initial['verdict'] == 'FAIL' and initial['pairs_faster'] == '4/5'
    assert confirmation['verdict'] == 'PASS' and confirmation['pairs_faster'] == '14/14'
    for row in rows:
        expected = 100 * (row['after_ms'] / row['before_ms'] - 1)
        assert row['change_pct'] == pytest.approx(expected, abs=0.005)
    assert confirmation['before_ms'] != initial['before_ms']


def test_confirmation_statistics_reproduce_from_retained_paired_timings(report):
    confirmation = report['acceptance']['confirmation']
    stats = confirmation['stats']
    before, after = stats['before_prefill_ms'], stats['after_prefill_ms']
    assert len(before) == len(after) == stats['pairs'] == 14
    assert all(TOOL.positive(v) for v in before + after)
    deltas = [a - b for b, a in zip(before, after)]
    relative = [100 * (a / b - 1) for b, a in zip(before, after)]
    assert stats['per_pair_delta_ms'] == deltas
    assert stats['per_pair_rel_pct'] == pytest.approx(relative, abs=0.0005)
    assert stats['median_delta_ms'] == statistics.median(deltas)
    assert stats['median_rel_pct'] == pytest.approx(statistics.median(relative), abs=0.0005)
    assert stats['faster'] == sum(d < 0 for d in deltas) == 14
    rng = random.Random(stats['bootstrap_seed'])
    bootstrap = sorted(statistics.median(rng.choices(deltas, k=14))
                       for _ in range(stats['bootstrap_resamples']))
    assert stats['ci95_median_delta_ms'] == [bootstrap[i] for i in stats['bootstrap_indices']]
    assert stats['median_rel_pct'] <= -1 and stats['faster'] >= 12
    assert max(stats['ci95_median_delta_ms']) < 0
    assert stats['decode_change_pct'] <= 3 and stats['rss_change_pct'] <= 3
    row = next(r for r in report['acceptance']['prefill']
               if r.get('campaign') == 'fixed 14-pair confirmation')
    assert row['before_ms'] == statistics.median(before)
    assert row['after_ms'] == statistics.median(after)
    assert row['paired_median_change_pct'] == stats['median_rel_pct']


def test_acceptance_documents_match_their_recorded_hashes(report):
    acceptance = report['acceptance']
    for item in (acceptance, acceptance['confirmation']):
        assert hashlib.sha256((ROOT / item['report']).read_bytes()).hexdigest() == item['report_sha256']
    confirmation = acceptance['confirmation']
    assert hashlib.sha256((ROOT / confirmation['criterion']).read_bytes()).hexdigest() == confirmation['criterion_sha256']
