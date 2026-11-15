"""Validate the measured JSON artifact without requiring local benchmark captures."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('q4_gemm_verify', ROOT / 'utilities/q4-gemm/verify.py')
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


@pytest.fixture
def report():
    return json.loads((ROOT / 'specs/open-engine/reports/qwen38-q4-gemm-dma.json').read_text())


def test_shipped_summary_has_all_five_paired_shapes_and_remains_a_draft(report):
    TOOL.check_summary(report)
    shapes = ['14336_k5120', '16384_k5120', '34816_k5120', '5120_k17408', '5120_k6144']
    assert {c['name'] for c in report['cases']} == {
        f'gemm_n{shape}{suffix}' for shape in shapes for suffix in ('', '_low')}
    assert all(c['summary']['paired_wins'] == report['methodology']['pairs'] for c in report['cases'])
    assert report['status'].startswith('DRAFT')
    assert report['validation']['full_model'].startswith('NOT MEASURED')
    diagnosis = report['validation']['context_diagnosis']
    assert diagnosis['firmware_status'] == '0x2000003'
    assert diagnosis['firmware_status_name'] == 'AIE2_STATUS_MGMT_ERT_NOAVAIL'
    assert diagnosis['observed_ceiling_healthy'] == 16
    assert diagnosis['observed_ceiling_degraded'] == 8
    assert diagnosis['ceiling_is_stable'] is False
    assert diagnosis['firmware_max_proven'] is False
    assert diagnosis['capacity_returned_after_destroy'] is True
    assert diagnosis['leak_after_64_churn_cycles'] is False
    assert diagnosis['process_exit_leak'] is False
    assert diagnosis['valid_control_substitute'] is False
    assert 'create_ctx_resp' in diagnosis['root_cause_experiment']


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
