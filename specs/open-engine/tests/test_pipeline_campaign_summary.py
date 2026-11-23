"""Portable behavioural checks for new-capture summaries; never runs hardware."""
import importlib.util
import math
from pathlib import Path
import statistics
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'utilities/prefill-block'
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('pipeline_summary', HERE / 'pipeline_summary.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_stats_preserve_all_samples_and_variability():
    values = [1, 4, 3, 2, 10]
    got = summary.stats(values)
    assert got['samples'] == values
    assert got['median'] == 3
    assert got['minimum'] == 1 and got['maximum'] == 10
    assert got['stdev'] == pytest.approx(statistics.stdev(values))


@pytest.mark.parametrize('values', [[], [math.nan], [math.inf], [-1], [True], ['1']])
def test_invalid_stats_fail_closed(values):
    with pytest.raises(summary.E.EvidenceError):
        summary.stats(values)


def test_pairs_are_not_a_difference_of_medians():
    got = summary.paired([10, 20, 30, 40, 50], [11, 25, 28, 39, 100])
    assert got['paired_delta_ms'] == [1, 5, -2, -1, 50]
    assert got['median_paired_delta_ms'] == 1
    assert got['median_paired_change_percent'] == pytest.approx(10)


@pytest.mark.parametrize('off,on', [([], []), ([1] * 4, [2] * 4),
                                  ([1] * 5, [2] * 6), ([0] * 5, [1] * 5)])
def test_incomplete_pairs_rejected(off, on):
    with pytest.raises(summary.E.EvidenceError):
        summary.paired(off, on)


GOOD = '''resident after 1.0 s
  gemm-block [0,256) t_real=256: 10.0 ms (GEMM 5.0, host 2.0, per-token 0.0, lm_head 1.0)
prefill 256 tokens: 10 ms (0 ms/token)
token 7
DONE
'''


def test_disabled_log_validates_real_raw_labels():
    got = summary.validate_disabled_log(GOOD, 'prefill', 256, 1,
                                        dict(real_vocab=100, block_tokens=256))
    assert got['prompt_wall_ms'] == 10
    assert got['generated_ids'] == [7]


@pytest.mark.parametrize('text', ['', GOOD.replace('DONE', 'ERROR: timeout'),
                                  GOOD.replace('token 7', 'token 107'),
                                  GOOD.replace('t_real=256', 't_real=255'),
                                  GOOD.replace('prefill 256', 'prefill 255'),
                                  GOOD.replace('token 7\n', ''),
                                  GOOD.replace('10.0 ms', 'nan ms'),
                                  GOOD.replace('DONE', 'token 8\nDONE')])
def test_disabled_log_rejects_missing_or_contradictory_evidence(text):
    with pytest.raises(summary.E.EvidenceError):
        summary.validate_disabled_log(text, 'prefill', 256, 1,
                                      dict(real_vocab=100, block_tokens=256))
