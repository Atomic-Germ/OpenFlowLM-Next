r"""dit_conv: 3x3 (or 1x1) convolution as implicit GEMM, the 3x3 window read out of the memtile.

    Y[y, x, co] = bias[co] + sum_{ky, kx, ci} X[y + ky - 1, x + kx - 1, ci] W[co, ci, ky, kx]

X and Y are NHWC bf16 in zero-bordered buffers ((H+2) rows of `pitch` pixels, the image
at row 1, column 1); W is bfp16ebs8, packed by conv_pack.py (pack_conv). Built for FLUX.2
[klein]'s VAE decoder (.claude/plans/image-diffusion-phase5-vae.md).

Why not dit_gemm streams: the VAE's convs have 128-512 output channels, and dit_gemm
spreads N over 8 core columns of 128 (N % 1024); and a window read through shim
patterns fetches every activation 9 times from DDR.

Per column (8, independent), per tile: the column's 4 cores take 4 consecutive output
rows x the same 128 pixels x the same 128 output channels, each keeping its 128 x 128 C
tile in L1 over the whole K walk (cin chunk of 64, ky, kx) with dit_gemm's microkernel
(mm_dit.cc: bf16 A converted to bfp16 in-core, bfp16 B, C re-rounded to bf16 every 64
of K).

  A (activations), explicit DMA -- an ObjectFifo link's MM2S length is its L2 object's,
  and the window reads each band element up to 9 times:
    shim -> memtile  a band: 6 input rows x 130 px x one 64-channel chunk (the memtile
                     S2MM stores it chunk-major, [8 ch-block][6][130][8]), ping-pong
    memtile -> core r  its own MM2S channel; one BD per ky reads
                     [kx 3][quarter 4][ch-block 8][32 px x 8 ch] at row r + ky, i.e.
                     the 9 taps' 32 x 64 A quarters in the column-block order the core's
                     S2MM lays out as 8x8 blocks (dit_gemm's a_l2l1 pattern).
    Locks: a band's full lock is released 12 (4 readers x 3 ky BDs); each BD takes 1.
  B (weights): ObjectFifo shim -> memtile -> the column's 4 cores; per tile one bias
    object (conv.cc) then the pack_b tiles of the tile's 128 output channels.
  C: ObjectFifo join of the 4 cores at the memtile, one shim drain per tile.

Per memtile: 6 S2MM (A, B, 4 C) + 6 MM2S (4 A, B, C) -- the whole channel budget; L2
2 x 97.5 KB band + 2 x 9 KB B + 2 x 128 KB C = 469 KB of 512.

taps (compile time): 9 = 3x3; 1 = 1x1 (the band is the 4 output rows x 128 px). One
xclbin per tap count; every conv of a model with that tap count is an instruction
stream over it (runtime parameters: tiles per core, K steps per tile).

The spec (JSON, env DC_SPEC for build_design.py):
    {"H", "W": the grid the conv reads (the source grid when "up"), "Cin" (% 64),
     "Cout" (% 128), "up": 4 output phases (py, px) on the source grid, written to
     output pixel (2y + py, 2x + px) -- a nearest-2x upsample then a 3x3 conv (pack_conv
     builds the phases' weights), or with taps = 1 any per-phase 1x1 map (the VAE's
     unpatchify: pack_conv_phases),
     "x": {"off", "pitch", "border"}, "y": {...}: element offset, row pitch in pixels, and
     border 1 (default) if the image starts at row 1, column 1 of a zero-bordered buffer,
     0 for a plain [H*W, C] tensor (taps = 1 inputs only),
     "w_off": first packed B element, "sizes": {"X", "W", "Y"} in elements}
W % 128 == 0, or W = 64 or 32: a core then computes 128 px of which the first W are
real; the drain sends the other 128/W - 1 parts past the output row's right border, so
such a conv's output needs pitch >= 128/W (Wout + 2) (the default), and a 3x3 conv's
input band reads 130 px of its pitch.

    DC_TAPS=9 DC_SPEC='{"H": 32, "W": 128, "Cin": 64, "Cout": 128}' \
        python build_design.py designs/dit_conv/dit_conv.py <out>
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.dialects.aiex import dma_free_task, dma_start_task, shim_dma_single_bd_task
from aie.dialects.aiex import v8bfp16ebs8
from aie.iron import (
    Acquire, Bd, Buffer, CompileTime, DmaChannel, ExternalFunction, Flow, In, Lock,
    ObjectFifo, Out, Program, Release, Runtime, StreamDims, TaskGroup, TileDma, Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils import config as _aie_config

HERE = Path(__file__).resolve().parent
_CONV_SRC = HERE / "conv.cc"
_MM_SRC = HERE.parent / "dit_gemm" / "mm_dit.cc"
_AIE_KERNELS_INC = Path(_aie_config.cxx_header_path()) / "aie_kernels"

M_T, K_T, N_T = 128, 64, 128     # per-core C tile (pixels x output channels) and K step
N_COLS, N_ROWS = 8, 4
CB = K_T // 8                    # 8-channel blocks per 64-channel chunk
Q_PX = M_T // 4                  # pixels per A quarter
MAX_INFLIGHT = 4                 # tile groups in flight
RTP_TILES, RTP_KB = range(2)
# Memtile MM2S channel of each band reader. aie2p memtiles give channels 0/2/4 BDs 0-23
# and 1/3/5 BDs 24-47; the C join's output alone takes 8 (2 buffers x 4 parts), so the
# readers' 6 BDs each go 3 even + 1 odd: even 18 + A in 2 + C in 4 = 24, odd 6 + B 2 + 2
# + C out 8 + C in 4 = 22.
READER_CH = [0, 2, 4, 1]
RTP_LEN = 4


def band_geom(taps: int) -> tuple[int, int, int]:
    """(rows, pixels, ky BDs) of one band."""
    if taps == 9:
        return N_ROWS + 2, M_T + 2, 3
    assert taps == 1, taps
    return N_ROWS, M_T, 1


def resolve_spec(spec: dict, taps: int) -> dict:
    """Defaults and derived values; shared with make_test.py and the exporter."""
    s = dict(spec)
    H, W, cin, cout = s["H"], s["W"], s["Cin"], s["Cout"]
    up = bool(s.get("up"))
    assert cin % K_T == 0 and cout % N_T == 0, (cin, cout)
    assert W % M_T == 0 or W in (32, 64), W
    assert H % N_ROWS == 0, H
    parts = max(1, M_T // W)
    Ho, Wo = (2 * H, 2 * W) if up else (H, W)
    x = {"off": 0, "pitch": parts * (W + 2), "border": 1, **s.get("x", {})}
    y = {"off": 0, "pitch": parts * (Wo + 2), "border": 1, **s.get("y", {})}
    assert x["border"] or taps == 1, "a 3x3 conv reads a zero-bordered input"
    assert y["border"], "outputs are written into zero-bordered buffers"
    n_cc = cin // K_T
    n_kb = n_cc * taps
    n_vct = (4 if up else 1) * (cout // N_T)
    b_per_tile = (1 + n_kb) * 1024                 # v8bfp16ebs8 elements (9 bytes)
    sizes = {"X": x["off"] + (H + 2 * x["border"] + (parts - 1)) * x["pitch"] * cin,
             "W": s.get("w_off", 0) + n_vct * b_per_tile,
             "Y": y["off"] + (Ho + 2) * y["pitch"] * cout, **s.get("sizes", {})}
    s.update(up=up, parts=parts, Ho=Ho, Wo=Wo, x=x, y=y, n_cc=n_cc, n_kb=n_kb, n_vct=n_vct,
             b_per_tile=b_per_tile, sizes=sizes, w_off=s.get("w_off", 0), taps=taps)
    return s


def tiles(s: dict) -> list[tuple[int, int, int]]:
    """(virtual output-channel tile, row block, x block), in issue order; tile i runs on
    column i % 8. A virtual tile is phase * (Cout/128) + channel tile when "up"."""
    n_xb = max(1, s["W"] // M_T)
    t = [(v, rb, xb) for v in range(s["n_vct"]) for rb in range(s["H"] // N_ROWS)
         for xb in range(n_xb)]
    assert len(t) % N_COLS == 0, f"{len(t)} tiles is not a multiple of {N_COLS}"
    return t


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def dit_conv(
    X: In, Wt: In, Y: Out, *,
    taps: CompileTime[int] = 9,
    spec: CompileTime[str] = "{}",
    stack_size: CompileTime[int] = 0xF00,
):
    s = resolve_spec(json.loads(spec), taps)
    m, k, n = M_T, K_T, N_T
    BR, BP, NKY = band_geom(taps)
    n_kx = 3 if taps == 9 else 1
    n_reads = N_ROWS * NKY
    band_len = CB * BR * BP * 8
    plane = BR * BP * 8                            # one 8-channel block of the band

    A_l1_ty = np.ndarray[(m // 4, k), np.dtype[bfloat16]]
    B_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[bfloat16]]
    C_l2_ty = np.ndarray[(N_ROWS * m, n), np.dtype[bfloat16]]
    band_ty = np.ndarray[(band_len,), np.dtype[bfloat16]]
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]

    flags = [f"-DDIM_M={m}", f"-DDIM_K={k}", f"-DDIM_N={n}", f"-I{_AIE_KERNELS_INC}"]
    bias_kernel = ExternalFunction("dit_conv_bias", source_file=str(_CONV_SRC),
                                   arg_types=[B_ty, C_l1_ty], compile_flags=flags)
    matmul_kernel = ExternalFunction("dit_matmul_quarter", source_file=str(_MM_SRC),
                                     arg_types=[A_l1_ty, B_ty, C_l1_ty],
                                     compile_flags=flags + ["-DMATMUL_ONLY"])

    # the core's S2MM lays each 32 x 64 quarter out as 8x8 blocks (dit_gemm's a_l2l1)
    a_core_sizes, a_core_strides = [k // 8, m // 4 // 8, 64], [64, 8 * k, 1]
    c_l2l3_dims: StreamDims = [(m // 8, 8 * n), (8, 8), (n // 8, 64), (8, 1)]

    flows, locks, dmas = [], [], []
    B_l3l2, C_l2l3, workers = [], [], []
    rtp = [[Buffer(rtp_ty, name=f"rtp_{c}_{r}", initial_value=np.zeros(RTP_LEN, np.int32),
                   use_write_rtp=True) for r in range(N_ROWS)] for c in range(N_COLS)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(N_ROWS)] for _ in range(N_COLS)]

    def core_fn(a0, a1, a_prod, a_cons, in_b, out_c, bias, matmul, my_rtp, barrier):
        barrier.wait_for_value(1)
        n_tiles = my_rtp[RTP_TILES]
        n_kb = my_rtp[RTP_KB]
        for _ in range_(n_tiles):
            c = out_c.acquire(1)
            b = in_b.acquire(1)
            bias(b, c)
            in_b.release(1)
            for _ in range_(n_kb):
                b = in_b.acquire(1)
                for q in range(4):
                    a_cons.acquire(1)
                    matmul(a0 if q % 2 == 0 else a1, b, c)
                    a_prod.release(1)
                in_b.release(1)
            out_c.release(1)
        barrier.release_with_value(1)

    for col in range(N_COLS):
        shim = Tile(col=col, row=0, tile_type=AIETileType.ShimNOCTile)
        mem = Tile(col=col, row=1, tile_type=AIETileType.MemTile)
        cores = [Tile(col=col, row=2 + r, tile_type=AIETileType.CoreTile)
                 for r in range(N_ROWS)]

        # A: shim -> band (chunk-major in L2) -> 4 windowed readers -> cores
        band = [Buffer(band_ty, tile=mem, name=f"band_{col}_{i}") for i in range(2)]
        full = [Lock(tile=mem, init=0, name=f"band_full_{col}_{i}") for i in range(2)]
        free = [Lock(tile=mem, init=n_reads, name=f"band_free_{col}_{i}") for i in range(2)]
        locks += full + free
        chans = [DmaChannel(DMAChannelDir.S2MM, 0, [
            Bd(band[i], length=band_len, sizes=[BR, BP, CB, 8], strides=[BP * 8, 8, plane, 1],
               acquires=[Acquire(free[i], n_reads)], releases=[Release(full[i], n_reads)],
               next=1 - i) for i in range(2)])]
        read_sizes = ([n_kx] if n_kx > 1 else []) + [4, CB, Q_PX * 8]
        read_strides = ([8] if n_kx > 1 else []) + [Q_PX * 8, plane, 1]
        for r in range(N_ROWS):
            bds = [Bd(band[i], offset=(r + ky) * BP * 8, length=n_kx * 4 * CB * Q_PX * 8,
                      sizes=read_sizes, strides=read_strides,
                      acquires=[Acquire(full[i], 1)], releases=[Release(free[i], 1)],
                      next=(i * NKY + ky + 1) % (2 * NKY))
                   for i in range(2) for ky in range(NKY)]
            chans.append(DmaChannel(DMAChannelDir.MM2S, READER_CH[r], bds))
            flows.append(Flow(src=mem, dst=cores[r], src_channel=READER_CH[r], dst_channel=0))
        dmas.append(TileDma(mem, chans))
        flows.append(Flow(src=shim, dst=mem, src_channel=0, dst_channel=0,
                          shim_symbol=f"A_{col}"))

        # B: bias object + weight tiles, broadcast to the column
        fb = ObjectFifo(B_ty, name=f"B_L3L2_{col}", depth=2)
        B_l3l2.append(fb)
        B_l2l1 = fb.cons().forward(tile=mem, obj_type=B_ty, name=f"B_L2L1_{col}", depth=2)

        # C: the 4 cores join at the memtile, row-major out
        fc = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=2, dims_to_stream=c_l2l3_dims)
        C_l2l3.append(fc)
        parts = fc.prod().join([m * n * r for r in range(N_ROWS)], tile=mem,
                               obj_types=[C_l1_ty] * N_ROWS,
                               names=[f"C_L1L2_{col}_{r}" for r in range(N_ROWS)],
                               depths=[1] * N_ROWS)

        for r in range(N_ROWS):
            a_buf = [Buffer(A_l1_ty, name=f"a_{col}_{r}_{i}", tile=cores[r]) for i in range(2)]
            a_prod = Lock(tile=cores[r], init=2, name=f"a_prod_{col}_{r}")
            a_cons = Lock(tile=cores[r], init=0, name=f"a_cons_{col}_{r}")
            locks += [a_prod, a_cons]
            dmas.append(TileDma(cores[r], [DmaChannel(DMAChannelDir.S2MM, 0, [
                Bd(a_buf[i], length=(m // 4) * k, sizes=a_core_sizes, strides=a_core_strides,
                   acquires=[Acquire(a_prod)], releases=[Release(a_cons)], next=1 - i)
                for i in range(2)])]))
            workers.append(Worker(
                core_fn,
                [a_buf[0], a_buf[1], a_prod, a_cons, B_l2l1.cons(), parts[r].prod(),
                 bias_kernel, matmul_kernel, rtp[col][r], barriers[col][r]],
                tile=cores[r], stack_size=stack_size))

    X_ty = np.ndarray[(s["sizes"]["X"],), np.dtype[bfloat16]]
    W_ty = np.ndarray[(s["sizes"]["W"],), np.dtype[v8bfp16ebs8]]
    Y_ty = np.ndarray[(s["sizes"]["Y"],), np.dtype[bfloat16]]
    all_tiles = tiles(s)
    n_groups = len(all_tiles) // N_COLS
    cin, cout = s["Cin"], s["Cout"]
    px_, py_ = s["x"]["pitch"], s["y"]["pitch"]
    n_ct = cout // N_T

    def a_fill(t):
        _, rb, xb = t
        y0, x0 = rb * N_ROWS, xb * M_T
        if taps == 1:                   # the band is the output rows themselves
            y0, x0 = y0 + s["x"]["border"], x0 + s["x"]["border"]
        return (s["x"]["off"] + (y0 * px_ + x0) * cin,
                [s["n_cc"], BR, BP, K_T], [K_T, px_ * cin, cin, 1])

    def c_drain(t):
        v, rb, xb = t
        y0, x0 = rb * N_ROWS, xb * M_T
        ct = v % n_ct
        if s["up"]:
            py, px = divmod(v // n_ct, 2)
            row0, col0, rs, ps = 2 * y0 + py + 1, 2 * x0 + px + 1, 2 * py_ * cout, 2 * cout
        else:
            row0, col0, rs, ps = y0 + 1, x0 + 1, py_ * cout, cout
        off = s["y"]["off"] + (row0 * py_ + col0) * cout + ct * N_T
        if s["parts"] > 1:               # parts 1.. go past the right border, Wo + 1 apart
            return off, [N_ROWS, s["parts"], s["W"], N_T], [rs, (s["Wo"] + 1) * cout, ps, 1]
        return off, [N_ROWS, M_T, N_T], [rs, ps, 1]

    def sequence(x, w, y, B_prods, C_conses):
        for col in range(N_COLS):
            for r in range(N_ROWS):
                rtp[col][r][RTP_TILES] = n_groups
                rtp[col][r][RTP_KB] = s["n_kb"]
        for col in range(N_COLS):
            for r in range(N_ROWS):
                barriers[col][r].set(1)

        inflight: list[TaskGroup] = []
        for g in range(n_groups):
            tg = TaskGroup()
            for col in range(N_COLS):
                t = all_tiles[g * N_COLS + col]
                off, sz, st = a_fill(t)
                task = shim_dma_single_bd_task(f"A_{col}", x.op, offset=off, sizes=sz,
                                               strides=st)
                dma_start_task(task)
                tg._actions.append((dma_free_task, [task]))
                B_prods[col].fill(w, offset=s["w_off"] + t[0] * s["b_per_tile"],
                                  sizes=[1, 1, 1, s["b_per_tile"]], strides=[0, 0, 0, 1],
                                  group=tg, wait=False)
            for col in range(N_COLS):
                off, sz, st = c_drain(all_tiles[g * N_COLS + col])
                C_conses[col].drain(y, offset=off, sizes=sz, strides=st, group=tg, wait=True)
            inflight.append(tg)
            if len(inflight) == MAX_INFLIGHT:
                inflight.pop(0).finish()
                inflight.pop(0).finish()
        for tg in inflight:
            tg.finish()

    rt = Runtime(sequence, [X_ty, W_ty, Y_ty,
                            [f.prod() for f in B_l3l2], [f.cons() for f in C_l2l3]])
    for f in flows:
        rt.add_flow(f)
    for lk in locks:
        rt.add_lock(lk)
    for d in dmas:
        rt.add_tile_dma(d)
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# build_design.py convention
DESIGN = dit_conv
SPECIALIZE = dict(taps=int(os.environ.get("DC_TAPS", 9)),
                  spec=os.environ.get("DC_SPEC", '{"H": 32, "W": 128, "Cin": 64, "Cout": 128}'))
