"""DMA planner tests require neither the IRON toolchain nor an NPU."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'gemm_dma_schedule', ROOT / 'open_kernels/designs/gemm_q4_prefill/dma_schedule.py')
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


@pytest.mark.parametrize('rows', [1, 2, 3, 7, 20, 56, 64, 136])
@pytest.mark.parametrize('tokens', [1, 2, 4, 8])
@pytest.mark.parametrize('batch', [1, 2, 4])
def test_batches_preserve_every_fifo_visit_and_do_not_pad_tail(rows, tokens, batch):
    groups = PLAN.dma_batches(rows, tokens, batch)
    visits = [visit for group in groups for visit in group]
    assert visits == [(row, token) for row in range(rows) for token in range(tokens)]
    assert len(groups) == (rows * tokens + batch - 1) // batch
    assert all(0 < len(group) <= batch for group in groups)
    assert all(len(group) == batch for group in groups[:-1])


def test_legacy_schedule_is_one_complete_visit_per_task_group():
    assert PLAN.dma_batches(2, 2) == (((0, 0),), ((0, 1),), ((1, 0),), ((1, 1),))


@pytest.mark.parametrize('rows,tokens,batch', [(0, 1, 1), (1, 0, 1), (-1, 1, 1),
                                             (1, 1, 0), (1, 1, 3), (1, 1, 8)])
def test_unbounded_or_invalid_plans_are_refused(rows, tokens, batch):
    with pytest.raises(ValueError):
        PLAN.dma_batches(rows, tokens, batch)
