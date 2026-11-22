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
    assert report['end_to_end_acceptance'] is False
    assert report['production_default_dma_batch'] == 1
    server = report['acceptance']['server_candidate']
    assert server['verdict'] == 'PASS' and server['runtime_comparable'] is True
    assert server['same_answer_between_arms'] is True
    assert len(server['arm_separation']['changed']) == 10
    assert server['capacity_validation'].startswith('NOT ESTABLISHED')
    for arm in ('before', 'after'):
        assert server['arms'][arm]['gemm_programs'] == 10
        assert len(server['arms'][arm]['library_identity']) >= 2
    assert report['acceptance']['runtime_verification'].startswith('Historical benchmark maps NOT RECORDED')
    scope = report['acceptance']['scope_review']
    assert scope['measurement_base'] == report['baseline_commit']
    assert scope['review_target'] == scope['intended_base']
    assert scope['prerequisites_merged'] is True and scope['rebase_performed'] is True


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
    assert hashlib.sha256((ROOT / acceptance['report']).read_bytes()).hexdigest() == acceptance['report_sha256']
    confirmation = acceptance['confirmation']
    assert hashlib.sha256((ROOT / confirmation['criterion']).read_bytes()).hexdigest() == confirmation['criterion_sha256']
