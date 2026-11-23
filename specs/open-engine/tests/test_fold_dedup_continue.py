"""Behavioural CPU gates for the distinct-prompt continuation tooling."""
import copy
import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'utilities/prefill-block'))
R = importlib.import_module('fold_dedup_continue')


def records():
    fresh = dict(outputs={f'p0_{suffix}': dict(sha256=suffix)
                          for suffix in ('logits.f32', 'gen.txt')}, runtime=dict(before={'lib': 'hash'}))
    aba = copy.deepcopy(fresh)
    for suffix in ('logits.f32', 'gen.txt'):
        aba['outputs'][f'p2_{suffix}'] = dict(sha256=suffix)
    return fresh, aba


def test_identical_complete_a_outputs_pass():
    R.compare_isolation(*records())


@pytest.mark.parametrize('request_index', [0, 2])
@pytest.mark.parametrize('suffix', ['logits.f32', 'gen.txt'])
def test_changed_a_outputs_fail(request_index, suffix):
    fresh, aba = records()
    aba['outputs'][f'p{request_index}_{suffix}']['sha256'] = 'changed'
    with pytest.raises(R.E.EvidenceError, match='mismatch'):
        R.compare_isolation(fresh, aba)


def test_changed_live_runtime_fails():
    fresh, aba = records()
    aba['runtime']['before']['lib'] = 'changed'
    with pytest.raises(R.E.EvidenceError, match='XRT'):
        R.compare_isolation(fresh, aba)


def test_timing_cannot_start_without_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        R.timing()
    assert not (tmp_path / 'selection.json').exists()
