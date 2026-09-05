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
from aie.utils import config


def include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    inc.append(str(Path(__file__).parent / "include"))      # vecmath.h
    return inc


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

    def finish(self, *eps):
        """Await everything issued (or, with endpoints given, only their queues)."""
        qs = [self._q(ep) for ep in eps] if eps else list(self.queues.values())
        for q in qs:
            while q:
                q.popleft().finish()
