r"""Probe: can IRON build a shim MM2S -> the SAME shim's TileControl packet stream?

The fused whole-layer MoE's on-device routing needs exactly one thing that has no IRON
example anywhere in this tree: a host-issued shim MM2S transfer whose payload is a
TileControl packet stream, routed into the shim's OWN TileControl port (the raw-MLIR
spikes in this directory prove the mechanism; see
benchmarks/RESULTS-expert-fetch-on-device-routing-spike-2026-09-19.md). Everything
else about the design is already expressible -- the routed descriptors are configured
without being started (ironutil.configure_only_fill) and pinned (_, _set_bd_id).

This design is that route and nothing else: it stamps a shim MM2S BD with packet id 1
and adds PacketFlow(pkt_id=1, shim DMA:0 -> shim TileControl:0). There is no compute to
run, so a PASS is BUILD_OK -- the question is only whether aiecc accepts the route.

Build: python build_design.py designs/expert_fetch/ondv_flow_probe.py <out>
"""

from __future__ import annotations

from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    DMAChannelDir,
    WireBundle,
)
from aie.dialects.aie import EndOp  # pyright: ignore[reportAttributeAccessIssue]
from aie.dialects.aiex import (  # pyright: ignore[reportMissingImports]
    bds,
    dma_configure_task,
    dma_start_task,
    shim_dma_bd,
)
from aie.iron import CompileTime, In, PacketFlow, Program, Runtime
from aie.iron.device import Tile

HERE = Path(__file__).parent

# five words: hdr(BD w1, 2 beats), addr_lo, addr_hi, hdr(queue, 1 beat), 0x80000000|bd
CTRL_WORDS = 5
PKT_ID = 1


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def probe(ctrl: In, *, srchash: CompileTime[int] = 0):
    ctrl_ty = np.ndarray[(8,), np.dtype[np.uint32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)

    def sequence(a_ctrl):
        # a raw shim MM2S task: this per-task packet stamping is the dialect-level
        # primitive the ObjectFifo path does not expose (packet_switch's own note)
        t = dma_configure_task(shim.op, DMAChannelDir.MM2S, 0)
        with bds(t) as bd:
            with bd[0]:
                shim_dma_bd(a_ctrl.op, offset=0, sizes=[1, 1, 1, CTRL_WORDS], strides=[0, 0, 0, 1],
                            packet=(0, PKT_ID))
                EndOp()
        dma_start_task(t)

    rt = Runtime(sequence, [ctrl_ty])
    rt.add_flow(PacketFlow(pkt_id=PKT_ID, src=shim, dst=shim, src_port=WireBundle.DMA, src_channel=0,
                           dst_port=WireBundle.TileControl, dst_channel=0,
                           shim_symbol="ctrl_shim_alloc"))
    return Program(iron.get_current_device(), rt).resolve_program()


DESIGN = probe
SPECIALIZE = {"srchash": 0}
