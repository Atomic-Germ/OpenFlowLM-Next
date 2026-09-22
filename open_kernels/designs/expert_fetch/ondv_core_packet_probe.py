r"""Probe: does a CORE-side packet-stamped BD reach a shim's TileControl and retarget a
configured-but-not-enqueued descriptor?

This is the corrected on-device-routing shape, and the one thing the fused design cannot yet
show. The existing probes (`ondv_flow_probe.py`, `ondv_flow_cross_probe.py`) only prove IRON
EMITS the route -- they have nothing to observe, and they therefore cannot tell "the packet
was applied" from "the packet was dropped". The raw-MLIR spikes in this directory DO observe
it, but they are hand-written `aiex.npu.blockwrite/write32/maskwrite32/address_patch/sync`
programs, which fault on the current driver while every high-level-op design here runs (see
benchmarks/RESULTS-ondv-fused-layer-descriptor-milestone-2026-09-22.md).

So this probe re-derives the mechanism with the ops that work:

  * a slab descriptor on the shim's MM2S ch1 is CONFIGURED and never started by the host
    (ironutil.configure_only_fill), so nothing arrives unless something enqueues it;
  * a core holds the retarget+enqueue control words and sends them with ITS OWN MM2S ch1 as
    a packet-stamped BD (TileDma + Bd(packet=(0, 1))) aimed at the shim's TileControl;
  * the shim's controller_id is the placer's (pkt_id 15) -- so the probe also answers whether
    a TileControl filters on the packet id, by being run twice (packet id 15 vs 1).

out[0] == 4096 (the sum of an all-ones slab) means the packet was APPLIED: the descriptor was
retargeted and enqueued with no host round-trip.
"""

from __future__ import annotations

import os
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    DMAChannelDir,
    WireBundle,
)
from aie.iron import (  # noqa: F401
    Acquire,
    Bd,
    Buffer,
    CompileTime,
    DmaChannel,
    In,
    Lock,
    Out,
    PacketFlow,
    Program,
    Release,
    Runtime,
    TileDma,
    Worker,
)
from aie.iron.device import Tile

HERE = Path(__file__).parent
PKT_ID = int(os.environ.get("ONDV_PROBE_PKT_ID", "15"))   # 15 is the placer's controller_id
WORDS = 4096


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def probe(slab_src: In, out: Out, *, srchash: CompileTime[int] = 0):
    slab_ty = np.ndarray[(WORDS,), np.dtype[np.uint32]]
    out_ty = np.ndarray[(4,), np.dtype[np.uint32]]
    shim = Tile(0, 0, tile_type=AIETileType.ShimNOCTile)
    core = Tile(0, 2)

    # The core's control words: five i32 = hdr(w1,2 beats), addr_lo, addr_hi, hdr(queue,1), enq
    ctrl = Buffer(np.ndarray[(8,), np.dtype[np.uint32]], name="ctrlw")
    # the slab the core sums once the descriptor fires
    slab = Buffer(slab_ty, name="slabbuf")
    res = Buffer(out_ty, name="resbuf")
    prod = Lock(core, 0, init=1, name="prod")
    cons = Lock(core, 1, init=0, name="cons")

    # The core builds the words, hands them to its own MM2S ch1 as ONE packet-stamped BD.
    dma = TileDma(
        tile=core,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=1,
                bds=[Bd(buffer=ctrl, acquires=[Acquire(prod)], releases=[Release(cons)],
                        packet=(0, PKT_ID))],
            ),
        ],
    )

    rt = Runtime(lambda a_slab, a_out: None, [slab_ty, out_ty])
    rt.add_flow(
        PacketFlow(pkt_id=PKT_ID, src=core, src_port=WireBundle.DMA, src_channel=1,
                   dst=shim, dst_port=WireBundle.TileControl, dst_channel=0)
    )
    rt.add_tile_dma(dma)
    rt.add_lock(prod)
    rt.add_lock(cons)
    return Program(iron.get_current_device(), rt).resolve_program()


DESIGN = probe
SPECIALIZE = {"srchash": 0}
