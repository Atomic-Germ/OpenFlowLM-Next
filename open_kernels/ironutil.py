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


def _set_bd_id(task, bd_id: int) -> None:
    """Pin the ``aie.dma_bd`` inside a configure task to a fixed buffer-descriptor
    index.

    The on-device router's control packets address a descriptor by its PHYSICAL
    shim register (``0x1D000 + 0x20*bd``), so the descriptor it is going to retarget
    must have a name the router knows at compile time. aiecc's
    ``aie-assign-runtime-sequence-bd-ids`` pass honors a user-set ``bd_id``
    (AIEAssignRuntimeSequenceBDIDs.cpp: "First, honor all the user-specified BD
    IDs"), so pinning the routed descriptors is legal -- and the pass refuses a
    pin that collides with a live descriptor, so a wrong reservation fails loudly
    at build time rather than corrupting at run time.
    """
    from aie import ir

    attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(bd_id))
    op = task.task.operation
    for region in op.regions:
        for block in region.blocks:
            for inner in block.operations:
                if inner.name == "aie.dma_bd":
                    inner.attributes["bd_id"] = attr


def configure_only_fill(prod, tensor, tap, bd_id: int | None = None):
    """Write a shim MM2S descriptor (``dma_configure_task_for``) for this transfer
    but do NOT push it to the channel's task queue.

    This is the first half of the on-device expert routing the fused whole-layer
    MoE needs (benchmarks/PLAN-on-device-routing-integration-2026-09-19.md): the
    routed-expert fills become descriptors the instruction stream writes but
    never enqueues, and the router helper core -- through control packets on the
    shim's own TileControl port (designs/expert_fetch) -- rewrites their DDR
    address and enqueues them, so lx0+lx1 fuse into ONE instruction stream and 40
    per-ctx ELFs batch into ONE xrt::runlist submit.

    IRON's ``DMATask.resolve`` does both halves (``shim_dma_single_bd_task`` then
    ``dma_start_task``); the split exists, it just has no front door. We take the
    unmanaged path (``managed=False``: emit the BD, no TaskGroup) with
    ``dma_start_task`` stubbed out for the duration of the resolve, so the
    descriptor is written and nothing is enqueued. The transfer is NOT tracked --
    nothing here is awaited or freed, because nothing was started; a later
    control packet owns it.
    """
    from aie.iron.runtime import dmatask as dm

    class _ConfigureOnly(dm.DMATask):
        def resolve(self, loc=None, ip=None):
            orig = dm.dma_start_task
            dm.dma_start_task = lambda *a, **k: None
            try:
                super().resolve(loc, ip)
            finally:
                dm.dma_start_task = orig
            if bd_id is not None:
                _set_bd_id(self, bd_id)

    orig_cls = dm.DMATask
    dm.DMATask = _ConfigureOnly
    try:
        return prod.fill(tensor, tap=tap, managed=False)
    finally:
        dm.DMATask = orig_cls


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

    def configure(self, prod, tensor, tap, bd_id: int | None = None):
        """Configure a routed-expert descriptor WITHOUT enqueueing it.

        No ``dma_start_task``, no TaskGroup, no queue slot: the descriptor is
        written into the instruction stream and waits for a control packet on the
        shim's TileControl port to retarget + enqueue it (see
        ``configure_only_fill``). ``bd_id`` pins the descriptor's physical index
        so the router core can address it. Returns the unmanaged Task so the caller
        can return the descriptor to the tile's 16-BD pool once the core has
        consumed the transfer (``free``) -- a descriptor is active from configure
        until free, enqueued or not.

        It still goes through ``_issue``, so the oldest managed fill's TaskGroup is
        finished (awaited and its descriptor returned) when the channel is at capacity:
        without that the MoE header fills hold three descriptors for the whole block, and
        a wave's pinned routed descriptors plus its control descriptors no longer fit a
        tile's 16."""
        box: dict = {}

        def _fn(_tg):
            box["t"] = configure_only_fill(prod, tensor, tap, bd_id=bd_id)

        self._issue(prod, _fn)
        return box["t"]

    def free(self, task):
        """Return a ``configure``d descriptor to the pool (``dma_free_task``).

        Only legal once the core has consumed the transfer: the descriptor is
        recycled here, so freeing before the control packet enqueued it would let
        the next configure overwrite the BD the packet is about to retarget."""
        task.free()

    def drain(self, cons, tensor, tap):
        self._issue(cons, lambda tg: cons.drain(tensor, tap=tap, wait=True, group=tg))

    def finish(self, *eps):
        """Await everything issued (or, with endpoints given, only their queues)."""
        qs = [self._q(ep) for ep in eps] if eps else list(self.queues.values())
        for q in qs:
            while q:
                q.popleft().finish()
