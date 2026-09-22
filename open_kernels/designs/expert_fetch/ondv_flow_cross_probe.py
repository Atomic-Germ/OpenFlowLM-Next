r"""Probe: can a shim's MM2S packet stream be routed to ANOTHER shim tile's TileControl?

If yes, ONE control stream serves all eight columns (each with its own pkt_id), which
matters because every column's two MM2S channels are already spent (lni/w0 on (0,0),
w1/x on (1,0), ...). Build-only.
"""
from __future__ import annotations
from pathlib import Path
import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
from aie.dialects.aie import EndOp
from aie.dialects.aiex import bds, dma_configure_task, dma_start_task, shim_dma_bd
from aie.iron import CompileTime, In, PacketDest, PacketFlow, Program, Runtime
from aie.iron.device import Tile

@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def probe(ctrl: In, *, srchash: CompileTime[int] = 0):
    ctrl_ty = np.ndarray[(8,), np.dtype[np.uint32]]
    src = Tile(col=1, row=0, tile_type=AIETileType.ShimNOCTile)
    dst = Tile(col=3, row=0, tile_type=AIETileType.ShimNOCTile)

    def sequence(a_ctrl):
        t = dma_configure_task(src.op, DMAChannelDir.MM2S, 1)
        with bds(t) as bd:
            with bd[0]:
                shim_dma_bd(a_ctrl.op, offset=0, sizes=[1, 1, 1, 5], strides=[0, 0, 0, 1], packet=(0, 1))
                EndOp()
        dma_start_task(t)

    rt = Runtime(sequence, [ctrl_ty])
    rt.add_flow(PacketFlow(pkt_id=1, src=src, dst=dst, src_port=WireBundle.DMA, src_channel=1,
                           dst_port=WireBundle.TileControl, dst_channel=0, shim_symbol="ctrlw_shim_alloc"))
    return Program(iron.get_current_device(), rt).resolve_program()

DESIGN = probe
SPECIALIZE = {"srchash": 0}
