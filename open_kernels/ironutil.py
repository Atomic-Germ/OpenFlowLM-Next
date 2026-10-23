r"""Shared helpers for phlegm's IRON designs.

- include_dirs(): aie_kernels include paths for ExternalFunction.
- Pipeline: issue fills/drains on a shim channel with at most `inflight`
  outstanding, awaiting the oldest before issuing more. A shim DMA channel's
  start queue holds 4 BDs; pushing more silently drops them and the core waits
  forever (designs/deltanet found this the hard way). Every transfer goes
  through a TaskGroup with wait=True so it can be awaited in issue order.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from aie.iron import TaskGroup
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import config


def include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    inc.append(str(Path(__file__).parent / "include"))      # vecmath.h
    return inc


def piece_tap(pool_bytes: int, off: int, band_stride: int, n: int, nbands: int, tile: int):
    """One K piece of `nbands` consecutive FF-wide down bands.

    The down GEMV reduces over FF, and at a wide FF its activation table does
    not fit a main core, so the reduction runs in K pieces (recipes/qwen36moe.py
    down_split). Inside a band, chunk c covers k-tile c / 2 (gemv_q4.h's band
    law), so a piece is a CONTIGUOUS run of chunks there: `n` bytes at `off`
    inside each band, the bands `band_stride` apart. Three real DMA dims with a
    half-chunk inner wrap, so the innermost transfer stays under 4 KB.

    One copy, two designs: the dense recipe (designs/dense/dx.py) and the MoE
    tail (designs/layer_x/xcommon.py) split the down GEMV the same way, and a
    second copy of this arithmetic would drift -- the copy that drifts is the
    one no shipped model exercises.
    """
    inner = tile // 2
    assert n % inner == 0, (n, inner)
    return TensorAccessPattern((1, pool_bytes), off, [1, nbands, n // inner, inner], [0, band_stride, inner, 1])


class Pipeline:
    """Throttled DMA issue. Keyed by the fifo endpoint (one shim channel each)."""

    def __init__(self, inflight: int = 3):
        self.inflight = inflight
        self.queues: dict[int, deque] = {}

    def _q(self, ep) -> deque:
        return self.queues.setdefault(id(ep), deque())

    def _issue(self, ep, fn):
        q = self._q(ep)
        if len(q) >= self.inflight:
            q.popleft().finish()
        tg = TaskGroup()
        fn(tg)
        q.append(tg)

    def fill(self, prod, tensor, tap):
        self._issue(prod, lambda tg: prod.fill(tensor, tap=tap, wait=True, group=tg))

    def drain(self, cons, tensor, tap):
        self._issue(cons, lambda tg: cons.drain(tensor, tap=tap, wait=True, group=tg))

    def finish_oldest(self, *eps):
        """Await only the OLDEST outstanding transfer of each endpoint given -- for a
        consumer that needs the first region a channel drains but not the ones after it."""
        for ep in eps:
            q = self._q(ep)
            if q:
                q.popleft().finish()

    def finish(self, *eps):
        """Await everything issued (or, with endpoints given, only their queues)."""
        qs = [self._q(ep) for ep in eps] if eps else list(self.queues.values())
        for q in qs:
            while q:
                q.popleft().finish()
