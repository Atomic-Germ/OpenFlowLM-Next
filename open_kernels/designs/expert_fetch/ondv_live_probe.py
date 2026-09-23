r"""Liveness probe for the on-device routing mechanism -- the ONE thing the fused design
cannot yet show.

The existing probes (`ondv_flow_probe.py`, `ondv_flow_cross_probe.py`,
`ondv_core_packet_probe.py`) only prove IRON EMITS the route: they have nothing to observe,
so they cannot tell "the packet was applied" from "it was dropped". This one observes it:

  * the shim's slab descriptor (all-zeros DDR -> a core buffer) is CONFIGURED and never
    enqueued (`Pipeline.configure`), so the core's acquire blocks forever unless something
    pushes it;
  * the core builds the retarget+enqueue words (`ondv_live_words`, the encoding in
    designs/router/ondv_ctrl.h) and sends them with ITS OWN MM2S ch1 as a packet-stamped
    BD aimed at its own column's shim TileControl;
  * the packet retargets that descriptor at `ones` (the harness writes the address into
    `cfg` with `poolbase`) and pushes it.

Observables: `out[0] == 1` means the descriptor fired AND carried the slab the packet
pointed it at -- the mechanism works. `out == 0` with a clean exit means the packet was
dropped and something else pushed the descriptor. A timeout means it was never pushed and
the core is still blocked, which is the failure the fused design hits today.

Run it twice with LP_PKT=15 (the placer's controller_id for shims) and LP_PKT=1 to see
whether a TileControl filters on the packet id.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    DMAChannelDir,
    WireBundle,
)
from aie.iron import (
    Acquire,
    Bd,
    Release,
    Buffer,
    CompileTime,
    DmaChannel,
    In,
    Lock,
    ObjectFifo,
    Out,
    PacketFlow,
    Program,
    Runtime,
    TileDma,
    Worker,
)
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

WS = 4096                                   # slab words (16 KB)
NEL = int(os.environ.get("LP_NEL", "1"))    # fifo elements per descriptor fill (1 = probe, 8 = fused 81920-B band)
BD = int(os.environ.get("LP_BD", "8"))      # the pinned descriptor the packet addresses
QUEUE = int(os.environ.get("LP_QUEUE", "0x1D21C"), 16)   # the shim MM2S queue that owns it
PKT = int(os.environ.get("LP_PKT", "15"))   # 15 is the placer's controller_id for shims
CORE_CH = int(os.environ.get("LP_CORE_CH", "1"))
CORE_ROW = int(os.environ.get("LP_ROW", "2"))   # the emitter core's row (2 = probe, 4/5 = fused design)
CHAIN = int(os.environ.get("LP_CHAIN", "1"))   # number of chained packet BDs (1 = probe, 9 = fused)
MAT = int(os.environ.get("LP_MAT", "1"))     # matrices (15-word blocks) per packet BD (fused: 2-3)
SHARED = int(os.environ.get("LP_SHARED", "0"))   # 1 = also enqueue a managed fill on the SAME shim channel


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def live_probe(zero: In, cfg: In, go: In, ones: In, out: Out, *, srchash: CompileTime[int] = 0):
    slab_ty = np.ndarray[(WS // NEL,), np.dtype[np.uint32]]
    cfg_ty = np.ndarray[(2,), np.dtype[np.uint32]]
    go_ty = np.ndarray[(4,), np.dtype[np.uint32]]
    out_ty = np.ndarray[(2,), np.dtype[np.uint32]]
    ctrl_ty = np.ndarray[(16 * MAT,), np.dtype[np.uint32]]
    inc = include_dirs()
    f_words = ExternalFunction("ondv_live_words", source_file=str(HERE / "ondv_live_words.cc"),
                               arg_types=[go_ty, ctrl_ty, np.int32, np.int32], include_dirs=inc,
                               compile_flags=[f"-DLP_MAT={MAT}"])
    f_out = ExternalFunction("ondv_live_out", source_file=str(HERE / "ondv_live_out.cc"),
                             arg_types=[slab_ty, out_ty, np.int32], include_dirs=inc)

    shim = Tile(0, 0, tile_type=AIETileType.ShimNOCTile)
    core = Tile(0, CORE_ROW)
    ctrl = Buffer(ctrl_ty, name="ctrlw", tile=core)
    slab = Buffer(slab_ty, name="slab", tile=core)
    pktlk = [Lock(core, init=0, name=f"pktlk{i}") for i in range(CHAIN)]
    pktdn = Lock(core, init=0, name="pktdn")   # the packet BD releases it when it has fired

    of_slab = ObjectFifo(slab_ty, name="slabf", depth=CHAIN * NEL * MAT + SHARED)
    of_in = ObjectFifo(go_ty, name="inf", depth=2)   # [addr_lo addr_hi ...] then the enable
    of_out = ObjectFifo(out_ty, name="outf", depth=1)

    def core_body(goin, ctrl, slabin, outprod, pktdn, fw, fo, *locks):
        c = goin.acquire(1)          # element 1: the retarget address (written by poolbase)
        g = goin.acquire(1)          # element 2: the enable, which the control program only
                                     # fills AFTER the configure -- so the packet cannot
                                     # precede the descriptor it retargets
        fw(c, ctrl, BD, QUEUE)
        goin.release(2)
        for lk in locks:
            lk.release(1)            # arm BD i, wait for it to fire
            pktdn.acquire(1)
        for _ in range(CHAIN * NEL * MAT + SHARED - 1):   # every element of every re-push must arrive
            s = slabin.acquire(1)
            slabin.release(1)
        s = slabin.acquire(1)        # the last delivery, for the out word
        o = outprod.acquire(1)
        fo(s, o, WS // NEL)
        outprod.release(1)
        slabin.release(1)

    worker = Worker(core_body,
                    fn_args=[of_in.cons(), ctrl, of_slab.cons(), of_out.prod(),
                             pktdn, f_words, f_out, *pktlk],
                    tile=core, stack_size=0x1800)

    bds = []
    for i in range(CHAIN):
        bds.append(Bd(buffer=ctrl, length=60 * MAT,
                      acquires=[Acquire(pktlk[i])], releases=[Release(pktdn)],
                      next=(i + 1) if i + 1 < CHAIN else 0))
    dma = TileDma(tile=core, channels=[
        DmaChannel(direction=DMAChannelDir.MM2S, channel=CORE_CH, bds=bds),
    ])

    def sequence(a_zero, a_cfg, a_go, a_ones, c_out, slabp, inpf, outc):
        p = Pipeline(3)
        # the routed descriptor: CONFIGURED, never enqueued -- the core blocks on it
        if SHARED:
            p.fill(slabp, a_zero, TensorAccessPattern((1, WS), 0, [1, 1, 1, WS], [0, 0, 0, 1]))
        p.configure(slabp, a_zero, TensorAccessPattern((1, WS), 0, [1, 1, 1, WS], [0, 0, 0, 1]), bd_id=BD)
        # element 1: the retarget address (the harness's poolbase wrote it into cfg)
        p.fill(inpf, a_cfg, TensorAccessPattern((1, 4), 0, [1, 1, 1, 4], [0, 0, 0, 1]))
        # element 2: the enable, after the configure, so the packet cannot precede it
        p.fill(inpf, a_go, TensorAccessPattern((1, 4), 0, [1, 1, 1, 4], [0, 0, 0, 1]))
        p.drain(outc, c_out, TensorAccessPattern((1, 2), 0, [1, 1, 1, 2], [0, 0, 0, 1]))
        p.finish()

    rt = Runtime(sequence, [slab_ty, cfg_ty, go_ty, slab_ty, out_ty,
                            of_slab.prod(tile=shim), of_in.prod(tile=shim),
                            of_out.cons(tile=Tile(1, 0))])
    rt.add_tile_dma(dma)
    for lk in pktlk:
        rt.add_lock(lk)
    rt.add_lock(pktdn)
    rt.add_flow(PacketFlow(pkt_id=PKT, src=core, src_port=WireBundle.DMA, src_channel=CORE_CH,
                           dst=shim, dst_port=WireBundle.TileControl, dst_channel=0,
                           keep_pkt_header=True))
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = live_probe
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
