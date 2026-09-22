"""Reference for the on-device-routing control words (the fused whole-layer MoE).

The router helper core retargets + enqueues the routed-expert descriptors by writing
TileControl packets into a DDR bounce buffer; a shim MM2S channel streams that buffer
into the shim's own TileControl port (designs/expert_fetch, all three spikes PASS).
This module is the executable statement of the packet encoding the router kernel must
emit -- checked here against the words the spike actually sent.

Encoding (recovered from core_ddr_bounce_retarget.mlir / ddr_bounce_fetch.mlir):

    hdr = (stream_id << 24) | (opcode << 22) | ((beats - 1) << 20) | address
    bit 31 = parity, 1 iff popcount(bits[30:0]) is EVEN          (opcode 0 = write)

A retarget + enqueue of descriptor `bd` at DDR byte address `addr` is five words:

    hdr(BD_w1, 2 beats), addr_lo, addr_hi, hdr(queue, 1 beat), 0x80000000 | bd

The last word's leading 1 is the QUEUE's hardware start flag, not a parity bit -- the
reason 0x80000002 pairs with a header whose own parity bit is 0.

Shim register map (both spikes, reconfirmed on this box):
    BD n registers 0x1D000 + 0x20*n: w0 len @+0, w1 addr_low @+4, w2 addr_high @+8
    MM2S ch0 ctrl 0x1D210 / queue 0x1D214; MM2S ch1 ctrl 0x1D218 / queue 0x1D21C

Run `python3 ondv_ctrl_ref.py` to check the self-test against the spike's words.
"""

from __future__ import annotations

BD_BASE = 0x1D000
BD_STRIDE = 0x20
BD_W1 = 0x04                      # addr_low; +4 is addr_high (w2)
MM2S_QUEUE = {0: 0x1D214, 1: 0x1D21C}

# the descriptors the fused path pins (xcommon.ONDV_BD_UP/GATE/DOWN)
BD_UP, BD_GATE, BD_DOWN = 8, 9, 10


def popcount(x: int) -> int:
    return bin(x & 0xFFFFFFFF).count("1")


def hdr(address: int, beats: int, stream_id: int = 0, opcode: int = 0) -> int:
    """A TileControl packet header for `beats` words at `address`, parity filled in."""
    assert beats >= 1
    word = ((stream_id & 0x7F) << 24) | ((opcode & 0x3) << 22) | ((beats - 1) & 0x3) << 20 | (address & 0xFFFFF)
    if popcount(word) % 2 == 0:   # bit31 = 1 iff popcount(bits[30:0]) is even
        word |= 0x80000000
    return word


def bd_w1_reg(bd: int) -> int:
    return BD_BASE + BD_STRIDE * bd + BD_W1


def retarget_enqueue(bd: int, addr: int, channel: int = 1) -> list[int]:
    """The five words that rewrite descriptor `bd`'s DDR address and push it."""
    low, high = (addr & 0xFFFFFFFF), ((addr >> 32) & 0xFFFF)
    return [hdr(bd_w1_reg(bd), 2), low, high, hdr(MM2S_QUEUE[channel], 1), 0x80000000 | bd]


# ---- the 35B recipe's MoE geometry and the whole-stream generator -------------
# Mirrors designs/router/ondv_ctrl.h; designs/router/ondv_ctrl_test.c checks the two
# agree word for word (they are deliberately independent transcriptions).
ONDV_CORES, ONDV_ROUTED = 8, 8
ONDV_SPP, ONDV_CPS = 4, 2
ONDV_STRIPE, ONDV_PAIR = 163840, 10240
ONDV_UP_BYTES, ONDV_DOWN_CORE = 655360, 81920
ONDV_POOL_DOWN = 335544320
# each column's w fifo's MM2S channel, from the built design's shim_dma_allocation table
ONDV_QUEUE = [0x1D21C, 0x1D214, 0x1D21C, 0x1D21C, 0x1D21C, 0x1D214, 0x1D214, 0x1D214]
ONDV_STREAM_WORDS = 8 * 8 * 13                # slots x columns x (3 addr packets + 1 push)


def up_off(expert: int, c: int) -> int:
    return (2 * ONDV_SPP * expert + 2 * (c // ONDV_CPS)) * ONDV_STRIPE + (c % ONDV_CPS) * ONDV_PAIR


def down_off(expert: int, c: int) -> int:
    return ONDV_POOL_DOWN + expert * ONDV_UP_BYTES + c * ONDV_DOWN_CORE


def column_words(queue: int, up: int, gate: int, down: int) -> list[int]:
    """13 words for one (slot, column): three address packets then ONE push packet (the
    three pushes share the task-queue register, so one 3-beat write enqueues BD 8/9/10)."""
    out = []
    for bd, addr in ((BD_UP, up), (BD_GATE, gate), (BD_DOWN, down)):
        out += [hdr(bd_w1_reg(bd), 2), addr & 0xFFFFFFFC, (addr >> 32) & 0xFFFF]
    out += [hdr(queue, 3), 0x80000000 | BD_UP, 0x80000000 | BD_GATE, 0x80000000 | BD_DOWN]
    return out


def emit_stream(idx: list[int], base: int) -> list[int]:
    """The full stream: out[(k*8 + c)*13 + ...] = up addr(3), gate addr(3), down addr(3), push(4)."""
    out = []
    for k in range(ONDV_ROUTED):
        for c in range(ONDV_CORES):
            up = up_off(idx[k], c)
            out += column_words(ONDV_QUEUE[c], base + up, base + up + ONDV_STRIPE,
                                base + down_off(idx[k], c))
    return out


def _selftest() -> None:
    # The documented rule, checked against the two spikes whose comments compute it
    # by hand: core_ctrlpkt.mlir 'hdr(0x1D214,1): popcount(0x1D214)=7 (odd) -> bit31=0'
    # and minpkt.mlir 'hdr(0x1D204): popcount 6 (even) -> parity bit 1 -> 0x8001D204'.
    assert hdr(0x1D214, 1) == 0x0001D214
    assert hdr(0x1D204, 1) == 0x8001D204
    # core_ddr_bounce_retarget.mlir's second word reads 0x801D21C, but the rule gives
    # 0x8001D21C (popcount(0x1D21C)=8, even) -- a typo in that spike's constant, not a
    # second encoding: a dropped leading zero turns bit31's parity into bit27.
    assert hdr(0x1D21C, 1) == 0x8001D21C
    # the retarget sequence with the corrected header
    got = retarget_enqueue(bd=2, addr=0, channel=1)
    want = [0x0011D044, 0x00000000, 0x00000000, 0x8001D21C, 0x80000002]
    assert got == want, [hex(w) for w in got]
    # and with a real address (the spike's idx=3 slab: base + 3*16384)
    got3 = retarget_enqueue(bd=2, addr=3 * 16384, channel=1)
    assert got3[1] == 3 * 16384 and got3[2] == 0
    assert got3[0] == 0x0011D044 and got3[3] == 0x8001D21C
    # the pinned routed descriptors: BD8's w1 is 0x1D104 on every column's shim
    assert bd_w1_reg(BD_UP) == 0x1D104 and bd_w1_reg(BD_GATE) == 0x1D124 and bd_w1_reg(BD_DOWN) == 0x1D144
    # parity is not a constant: BD8/w1 (popcount 7) -> 0, BD9/w1 (popcount 8) -> 1
    assert hdr(bd_w1_reg(BD_UP), 2) & 0x80000000 == 0
    assert hdr(bd_w1_reg(BD_GATE), 2) & 0x80000000 != 0
    print("ondv_ctrl_ref: self-test PASS (both spikes' hand-computed headers reproduced)")


if __name__ == "__main__":
    import struct
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "dump":
        # dump <path> <base> <idx0,..,idx7> -- the expected stream for the C test
        path, base = sys.argv[2], int(sys.argv[3], 0)
        idxs = [int(x, 0) for x in sys.argv[4].split(",")]
        ws = emit_stream(idxs, base)
        with open(path, "wb") as f:
            f.write(struct.pack(f"<{len(ws)}I", *[w & 0xFFFFFFFF for w in ws]))
        print(f"ondv_ctrl_ref: wrote {len(ws)} words to {path}")
    else:
        _selftest()
